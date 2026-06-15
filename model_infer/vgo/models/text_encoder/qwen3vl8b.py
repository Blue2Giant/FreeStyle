import math

import torch
import torch.nn as nn
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torchvision.transforms import functional as F
from vgo.utils.dist_utils import reshard_module, unshard_module
from vgo.utils.transformers_compat import AutoProcessor, Qwen3VLForConditionalGeneration


class Qwen3VL8B_Embedder(torch.nn.Module):
    def __init__(
        self,
        model_path,
        max_length=2048,
        dtype=torch.bfloat16,
        device="cuda",
        llm_image_min_token: int | None = None,
        llm_image_max_token: int | None = None,
        enable_lq_lora: bool = False,
        out_embedding_layer_index: int = -1,
    ):
        super().__init__()
        self.max_length = max_length
        self.dtype = dtype
        self.device = device

        attn_implementation = "flash_attention_2"

        self.out_embedding_layer_index = out_embedding_layer_index

        print(f"Loading Qwen from {model_path=}")

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
        ).to(self.device)  # type: ignore

        self.model.requires_grad_(False)

        assert llm_image_min_token is not None
        assert llm_image_max_token is not None

        self.processor = AutoProcessor.from_pretrained(
            model_path,
        )

        self.t2i_prompt = "Describe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:"  # noqa: E501
        self.t2i_prompt_template_encode_start_idx = 34

        self.edit_prompt = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."  # noqa: E501
        self.edit_prompt_template_encode_start_idx = 64
        self.img_area = 384 * 384

        self.is_shard = False

    def hidden_size(self) -> int:
        return self.model.config.text_config.hidden_size

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

    def calculate_dimensions(self, target_area, aspect_ratio):
        width = int(math.sqrt(target_area * aspect_ratio))
        height = int(math.sqrt(target_area / aspect_ratio))
        return width, height

    def t2i_forward_one(self, prompt):
        dtype = torch.bfloat16
        device = self.device
        prompt = prompt if isinstance(prompt, str) else prompt[0]

        drop_idx = self.t2i_prompt_template_encode_start_idx

        user_content = []

        messages = []
        user_content.append({"type": "text", "text": prompt})
        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.t2i_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        messages.append(msgs)

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.device)

        model_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "output_hidden_states": True,
        }
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            model_kwargs["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = inputs.image_grid_thw

        outputs = self.model(**model_kwargs)
        prompt_embeds = outputs.hidden_states[self.out_embedding_layer_index][0, drop_idx:]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds

    def edit_forward_one(self, prompt, image):
        dtype = torch.bfloat16
        device = self.device
        prompt = prompt if isinstance(prompt, str) else prompt[0]

        user_content = []
        messages = []
        if image:
            for img in image:
                pil_image = F.to_pil_image((img + 1) * 0.5)

                width, height = self.calculate_dimensions(self.img_area, pil_image.width / pil_image.height)
                pil_image = pil_image.resize((width, height))
                user_content.append({"type": "image", "image": pil_image})

        user_content.append({"type": "text", "text": prompt})
        msgs = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.edit_prompt}],
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
        messages.append(msgs)

        drop_idx = self.edit_prompt_template_encode_start_idx

        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", padding=True
        ).to(self.device)

        model_kwargs = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "output_hidden_states": True,
        }
        if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
            model_kwargs["pixel_values"] = inputs.pixel_values
        if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = inputs.image_grid_thw

        outputs = self.model(**model_kwargs)
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


def shard_model(model: Qwen3VLForConditionalGeneration, world_mesh: DeviceMesh, use_hsdp: bool = True):
    fsdp_mesh = world_mesh if use_hsdp else world_mesh._flatten()

    def apply_fully_shard(layer, **fsdp_kwargs):
        if isinstance(layer, nn.ModuleList):
            for sublayer in layer:
                fully_shard(sublayer, **fsdp_kwargs)
        else:
            fully_shard(layer, **fsdp_kwargs)

    apply_fully_shard(model.language_model.layers, reshard_after_forward=False, mesh=fsdp_mesh)
    apply_fully_shard(model.visual.blocks, reshard_after_forward=False, mesh=fsdp_mesh)


def reshard_model(model: Qwen3VLForConditionalGeneration):
    reshard_module(model.language_model.layers)
    reshard_module(model.visual.blocks)


def unshard_model(model: Qwen3VLForConditionalGeneration):
    unshard_module(model.language_model.layers)
    unshard_module(model.visual.blocks)
