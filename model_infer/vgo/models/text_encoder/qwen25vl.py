import math
import re

import torch
import torch.nn as nn
import torchvision  # noqa: F401
from einops import rearrange
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from vgo.data.processor.caption import split_string_in_quotation_and_special_tokens
from vgo.data.processor.prefix_util import PrefixSetting
from vgo.utils.accel import is_npu
from vgo.utils.dist_utils import reshard_module, unshard_module
from vgo.utils.image_utils import to_pil_image, to_tensor
from vgo.utils.transformers_compat import Qwen2_5_VLForConditionalGeneration


class QwenLLMUtil:
    vit_mean: torch.Tensor = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
    vit_std: torch.Tensor = torch.Tensor([0.26862954, 0.26130258, 0.27577711])

    @staticmethod
    def convert_image_to_qwen_vit_input(image: torch.Tensor):
        device = image.device
        dtype = image.dtype
        # On Ascend, the temporary tensors created by normalization + rearrange can
        # exceed device memory even when the final ViT input fits. Do the reshape on
        # CPU first and move only the packed tensor back to NPU.
        work_device = torch.device("cpu") if is_npu() else device
        work_dtype = torch.float32 if is_npu() else dtype
        image = image.to(device=work_device, dtype=work_dtype)
        vit_mean = QwenLLMUtil.vit_mean.to(work_device, work_dtype)
        vit_std = QwenLLMUtil.vit_std.to(work_device, work_dtype)
        image = (image - vit_mean[:, None, None]) / vit_std[:, None, None]

        merge_size = 2
        patch_size = 14
        image = image[None].repeat(2, 1, 1, 1)  # repeat image : CxHxW -> 2xCxHxW
        H, W = image.shape[-2:]

        image = rearrange(
            image,
            "(gT pT) C (gH mH pH) (gW mW pW) -> (gT gH gW) (mH mW C pT pH pW)",
            pT=2,
            mH=merge_size,
            mW=merge_size,
            pH=patch_size,
            pW=patch_size,
        )
        image = image.reshape(H // patch_size * W // patch_size, -1)
        image = image.to(device=device, dtype=dtype)
        thw = torch.tensor([[1, H // patch_size, W // patch_size]], device=device, dtype=torch.int32)
        return image, thw


def convert_image_to_qwen_vit_input(image: torch.Tensor):
    return QwenLLMUtil.convert_image_to_qwen_vit_input(image)


def _load_qwen25vl_processor(model_path: str, *, min_pixels: int, max_pixels: int):
    """
    transformers 的 `Qwen2_5_VLProcessor` 默认会加载 video_processor（AutoVideoProcessor），
    进而强依赖 torchvision。Ascend 环境下 torchvision 往往不可用/不可导入，因此这里通过临时
    修改 Processor 的 `attributes`，只加载 image_processor + tokenizer。
    """

    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

    orig_attributes = list(Qwen2_5_VLProcessor.attributes)
    try:
        Qwen2_5_VLProcessor.attributes = ["image_processor", "tokenizer"]
        return Qwen2_5_VLProcessor.from_pretrained(
            model_path,
            use_fast=False,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    finally:
        Qwen2_5_VLProcessor.attributes = orig_attributes


class Qwen25VL7B_Embedder(torch.nn.Module):
    def __init__(
        self,
        model_path,
        max_length=2048,
        dtype=torch.bfloat16,
        device="cuda",
        llm_image_min_token: int | None = None,
        llm_image_max_token: int | None = None,
        enable_splitids: bool = False,
        splitids_prob: float = 1.0,
        enable_lq_lora: bool = False,
        out_embedding_layer_index: int = -1,
    ):
        super().__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device
        self.enable_splitids = enable_splitids
        self.splitids_prob = float(splitids_prob)
        if not 0.0 <= self.splitids_prob <= 1.0:
            raise ValueError(f"splitids_prob should be in [0, 1], got {self.splitids_prob}")

        attn_implementation = "sdpa" if is_npu() else "flash_attention_2"

        self.out_embedding_layer_index = out_embedding_layer_index

        print(f"Loading Qwen from {model_path=}")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(self.device)  # type: ignore

        self.model.requires_grad_(False)

        PrefixSetting.set_PREFIX_MODE("qwen-image")

        assert llm_image_min_token is not None
        assert llm_image_max_token is not None

        self.processor = _load_qwen25vl_processor(
            model_path,
            min_pixels=llm_image_min_token * 28 * 28,
            max_pixels=llm_image_max_token * 28 * 28,
        )

        self.llm_vit_min_tokens = llm_image_min_token
        self.llm_vit_max_tokens = llm_image_max_token
        self.llm_vit_image_factor = 28

        self.t2i_prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.t2i_prompt_template_encode_start_idx = 34

        self.edit_prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"  # noqa: E501
        self.edit_prompt_template_encode_start_idx = 64

        self.is_shard = False

    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def apply_fsdp(self, world_mesh: DeviceMesh, use_hsdp: bool = True):
        logger.info(f"对 Text Encoder 采用 FSDP 以节省显存，{use_hsdp=}")
        shard_model(self.model, world_mesh=world_mesh, use_hsdp=use_hsdp)
        self.is_shard = True

    def reshard_module(self):
        if self.is_shard:
            reshard_model(self.model)

    def unshard_module(self):
        if self.is_shard:
            unshard_model(self.model)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    @staticmethod
    def special_token_strings(tokenizer):
        tokens = []
        tokens.extend(tokenizer.additional_special_tokens)
        tokens.extend([t.content for t in tokenizer.added_tokens_decoder.values()])
        tokens = [re.escape(x) for x in tokens]
        return tokens

    # from https://github.com/huggingface/transformers/blob/4a03044ddbe41fe6b237fe813d23ba5bab8b23bc/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py#L55
    def resize_to_llm_factor_within_minmax(self, image: torch.Tensor):
        origin_height, origin_width = image.shape[-2:]
        min_area = self.llm_vit_min_tokens * self.llm_vit_image_factor * self.llm_vit_image_factor
        max_area = self.llm_vit_max_tokens * self.llm_vit_image_factor * self.llm_vit_image_factor
        origin_area = origin_height * origin_width

        target_width = round(origin_width / self.llm_vit_image_factor) * self.llm_vit_image_factor
        target_height = round(origin_height / self.llm_vit_image_factor) * self.llm_vit_image_factor
        if origin_area > max_area:
            scale = math.sqrt(origin_area / max_area)
            target_width = math.floor(origin_width / scale / self.llm_vit_image_factor) * self.llm_vit_image_factor
            target_height = math.floor(origin_height / scale / self.llm_vit_image_factor) * self.llm_vit_image_factor
        elif origin_area < min_area:
            scale = math.sqrt(min_area / origin_area)
            target_width = math.ceil(origin_width * scale / self.llm_vit_image_factor) * self.llm_vit_image_factor
            target_height = math.ceil(origin_height * scale / self.llm_vit_image_factor) * self.llm_vit_image_factor

        pil_image = to_pil_image(image)
        pil_image = pil_image.resize((target_width, target_height))

        return to_tensor(pil_image).to(image.device, image.dtype)

    def t2i_forward_one(self, prompt):
        dtype = self.dtype
        device = self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.t2i_prompt_template_encode

        drop_idx = PrefixSetting().PREFIX_TOKEN_LEN_DICT["t2i"]
        txt = [template.format(e) for e in prompt]

        text_split_list, _ = split_string_in_quotation_and_special_tokens(
            txt[0],
            prefix_len=PrefixSetting().PREFIX_STRING_LEN_DICT["t2i"],
            special_tokens=self.special_token_strings(self.processor.tokenizer),
        )
        token_list = []
        for text_each in text_split_list:
            inputs = self.processor(
                text=text_each,
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            token_each = inputs.input_ids
            token_list.append(token_each)
        input_ids = torch.cat(token_list, dim=1).to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[self.out_embedding_layer_index][0, drop_idx:]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def edit_forward_one(self, prompt, image):
        dtype = self.dtype
        device = self.device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
        if isinstance(image, list):
            base_img_prompt = ""
            for i, _ in enumerate(image):
                base_img_prompt += img_prompt_template.format(i + 1)
        elif image is not None:
            base_img_prompt = img_prompt_template.format(1)
        else:
            base_img_prompt = ""

        template = self.edit_prompt_template_encode

        drop_idx = PrefixSetting().PREFIX_TOKEN_LEN_DICT["edit"]
        txt = [template.format(base_img_prompt + e) for e in prompt]

        text_split_list, _ = split_string_in_quotation_and_special_tokens(
            txt[0],
            prefix_len=PrefixSetting().PREFIX_STRING_LEN_DICT["edit"],
            special_tokens=self.special_token_strings(self.processor.tokenizer),
        )
        token_list = []
        for text_each in text_split_list:
            inputs = self.processor(
                text=text_each,
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            token_each = inputs.input_ids
            token_list.append(token_each)
        input_ids = torch.cat(token_list, dim=1).to(self.device)

        all_pixel_values = []
        all_image_grid_thw = []
        for img_i in image:
            # input pixel values should be in [0, 1]
            pixel_values, image_grid_thw = convert_image_to_qwen_vit_input(
                self.resize_to_llm_factor_within_minmax((img_i + 1) * 0.5)
            )
            all_pixel_values.append(pixel_values)
            all_image_grid_thw.append(image_grid_thw)
        all_pixel_values = torch.cat(all_pixel_values, dim=0)
        all_image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        image_pad_token_index = torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device)[
            input_ids[0] == 151655
        ]
        image_pad_token_index = image_pad_token_index[:, None].repeat(1, 2)
        image_pad_token_index[:, 1] += 1
        image_pad_token_index = image_pad_token_index.flatten().cpu()
        input_ids = list(input_ids.tensor_split(image_pad_token_index, dim=1))
        vision_lens = (all_image_grid_thw[:, 1] * all_image_grid_thw[:, 2] // 4).tolist()
        for img_idx in range(len(image)):
            assert input_ids[img_idx * 2 + 1].shape[1] == 1
            input_ids[img_idx * 2 + 1] = input_ids[img_idx * 2 + 1].repeat(
                1,
                vision_lens[img_idx],  # type: ignore
            )
        input_ids = torch.cat(input_ids, dim=1)

        input_ids = input_ids.contiguous()
        all_pixel_values = all_pixel_values.contiguous().to(input_ids.device)
        all_image_grid_thw = all_image_grid_thw.contiguous().to(input_ids.device, torch.int64)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=None,
            pixel_values=all_pixel_values,
            image_grid_thw=all_image_grid_thw,
            output_hidden_states=True,
        )
        prompt_embeds = outputs.hidden_states[self.out_embedding_layer_index][0, drop_idx:]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def forward(
        self,
        texts: list[str],
        ref_images: list[list[torch.Tensor]],
        task_type: list[str],
    ):
        out_prompt_embeds = []
        out_prompt_mebeds_lens = []
        for prompt_i, ref_images_i, task_type_i in zip(texts, ref_images, task_type):
            if task_type_i == "t2i":
                prompt_embeds = self.t2i_forward_one(prompt_i)
            elif task_type_i == "edit":
                prompt_embeds = self.edit_forward_one(prompt_i, ref_images_i)
            else:
                raise ValueError(f"Unregonized task type {task_type_i}")
            out_prompt_embeds.append(prompt_embeds)
            out_prompt_mebeds_lens.append(prompt_embeds.shape[0])

        out_prompt_embeds = torch.cat(out_prompt_embeds)
        return out_prompt_embeds, out_prompt_mebeds_lens


def shard_model(model: Qwen2_5_VLForConditionalGeneration, world_mesh: DeviceMesh, use_hsdp: bool = True):
    fsdp_mesh = world_mesh if use_hsdp else world_mesh._flatten()

    def apply_fully_shard(layer, **fsdp_kwargs):
        if isinstance(layer, nn.ModuleList):
            for sublayer in layer:
                fully_shard(sublayer, **fsdp_kwargs)
        else:
            fully_shard(layer, **fsdp_kwargs)

    apply_fully_shard(model.language_model.layers, reshard_after_forward=False, mesh=fsdp_mesh)
    apply_fully_shard(model.visual.blocks, reshard_after_forward=False, mesh=fsdp_mesh)


def reshard_model(model: Qwen2_5_VLForConditionalGeneration):
    reshard_module(model.language_model.layers)
    reshard_module(model.visual.blocks)


def unshard_model(model: Qwen2_5_VLForConditionalGeneration):
    unshard_module(model.language_model.layers)
    unshard_module(model.visual.blocks)
