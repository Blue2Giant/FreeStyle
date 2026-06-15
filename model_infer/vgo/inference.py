import itertools
import math
import os
import time
from pathlib import Path

import fire
import megfile
import numpy as np
import PIL.Image
import torch
import torch.distributed.tensor
from einops import rearrange, repeat
from safetensors import safe_open
from safetensors.torch import load_file
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import functional as F

from vgo.data.processor.image import default_pil_resize
from vgo.models.modules import LayerNormAutoCast
from vgo.models.modules.varlen_ops import VarLenConfig, cat_seq, split_seq_by_len_list
from vgo.models.text_encoder.qwen3vl8b import Qwen3VL8B_Embedder
from vgo.models.text_encoder.qwen25vl import Qwen25VL7B_Embedder
from vgo.models.transformers.model import DiTParams, VarLenDiT
from vgo.models.vae.qwen_autoencoder import AutoencoderKLQwenImage
from vgo.scheduler import sampling
from vgo.utils.common_utils import combine_list, convert_precision
from vgo.utils.dist_utils import ParallelDims, device_module, device_type, init_distributed


def generate_image_position_qwen_image_ids(h_feat, w_feat, bs, t_offset, device):
    img_ids = torch.zeros(h_feat, w_feat, 3, device=device, dtype=torch.float32)
    img_ids[..., 0] = t_offset
    img_ids[..., 1] = img_ids[..., 1] + (
        torch.arange(h_feat, device=device, dtype=torch.float32)[:, None] - (h_feat - h_feat // 2)
    )
    img_ids[..., 2] = img_ids[..., 2] + (
        torch.arange(w_feat, device=device, dtype=torch.float32)[None, :] - (w_feat - w_feat // 2)
    )
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img_ids


def _set_module_tensor_by_name(model: torch.nn.Module, name: str, tensor: torch.Tensor) -> bool:
    """Assign one tensor into a module without building a full state_dict.

    This is used for large safetensors checkpoints in small demo jobs where
    loading a 70GB+ state_dict all at once can kill the worker by CPU OOM.
    """
    module: torch.nn.Module = model
    parts = name.split(".")
    for part in parts[:-1]:
        try:
            module = getattr(module, part)
        except AttributeError:
            return False
    leaf = parts[-1]
    if leaf in module._parameters:
        old = module._parameters[leaf]
        requires_grad = bool(getattr(old, "requires_grad", False))
        module._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=requires_grad)
        return True
    if leaf in module._buffers:
        module._buffers[leaf] = tensor
        return True
    if hasattr(module, leaf):
        setattr(module, leaf, tensor)
        return True
    return False


def _stream_load_safetensors(model, ckpt_path, target_dtype=torch.bfloat16, target_device="cpu", strict=False):
    model_keys = set(model.state_dict().keys())
    loaded: set[str] = set()
    unexpected: list[str] = []
    target_device = torch.device(target_device)
    if target_device.type == "cuda":
        torch.cuda.set_device(target_device.index if target_device.index is not None else 0)
    print(
        f"Streaming safetensors from {ckpt_path} target_dtype={target_dtype} target_device={target_device}...",
        flush=True,
    )
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        for i, key in enumerate(keys, 1):
            if key not in model_keys:
                unexpected.append(key)
                continue
            tensor = f.get_tensor(key)
            if tensor.is_floating_point() and target_dtype is not None:
                tensor = tensor.to(device=target_device, dtype=target_dtype)
            else:
                tensor = tensor.to(device=target_device)
            ok = _set_module_tensor_by_name(model, key, tensor)
            if ok:
                loaded.add(key)
            else:
                unexpected.append(key)
            if i % 500 == 0:
                print(f"  streamed {i}/{len(keys)} tensors", flush=True)
            del tensor
    missing = sorted(model_keys - loaded)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Error(s) in streaming state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


def load_state_dict(model, ckpt_path, device="cpu", strict=False, assign=True):
    print(f"Loading state dict from {ckpt_path}... for {type(model)}")
    use_stream = (
        Path(ckpt_path).suffix == ".safetensors"
        and os.environ.get("VGO_STREAM_LOAD_SAFETENSORS", "1") != "0"
    )
    if use_stream:
        dtype_name = os.environ.get("VGO_STREAM_LOAD_DTYPE", "bfloat16").lower()
        target_dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
            "none": None,
            "keep": None,
        }.get(dtype_name, torch.bfloat16)
        target_device = os.environ.get("VGO_STREAM_LOAD_DEVICE", str(device or "cpu"))
        missing, unexpected = _stream_load_safetensors(
            model,
            ckpt_path,
            target_dtype=target_dtype,
            target_device=target_device,
            strict=strict,
        )
    else:
        if Path(ckpt_path).suffix == ".safetensors":
            state_dict = load_file(ckpt_path, "cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=assign)

    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(list(missing)[:50]))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(list(unexpected)[:50]))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys:\n\t" + "\n\t".join(list(missing)[:50]))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys:\n\t" + "\n\t".join(list(unexpected)[:50]))
    return model

def replace_norm_layers(module):
    for name, child in module.named_children():
        if "llm_encoder" in name:
            continue

        if isinstance(child, torch.nn.LayerNorm):
            new_layer = LayerNormAutoCast(
                normalized_shape=child.normalized_shape,  # type: ignore
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
            )
            if child.elementwise_affine:
                new_layer.weight.data.copy_(child.weight.data)
                new_layer.bias.data.copy_(child.bias.data)
            setattr(module, name, new_layer)
        else:
            replace_norm_layers(child)


def load_models(
    dit_path="/mnt/sirui/model_zoo/Qwen-Image-Edit-2509.safetensors",
    ae_path: str | None = None,
    qwenvl_model_path: str | None = None,
    config_path="",
    device="cuda",
    max_length=2048,
    dtype=torch.bfloat16,
):
    if ae_path is None:
        ae_path = "/mnt/sirui/model_zoo/Qwen-Image-Edit-2509/vae/"
    if qwenvl_model_path is None:
        qwenvl_model_path = "/data/midjourney/model_zoo/ckpts/Qwen2.5-VL-7B-Instruct"

    if config_path:
        from omegaconf import OmegaConf

        default_config = OmegaConf.structured(DiTParams)
        config = OmegaConf.load(config_path)
        pipe_config = config.engine_config.pipe
        if (not qwenvl_model_path) and pipe_config.llm_model_path is not None:
            qwenvl_model_path = pipe_config.llm_model_path
        if (not ae_path) and pipe_config.get("ae_path", None) is not None:
            ae_path = pipe_config.ae_path
        llm_output_layer_index = pipe_config.get("llm_output_layer_index", -1)
        dit_config = pipe_config.dit
        merged_config = OmegaConf.merge(default_config, dit_config)
        dit_params: DiTParams = OmegaConf.to_object(merged_config)
        llm_image_min_token = pipe_config.get("llm_image_min_token", 188)
        llm_image_max_token = pipe_config.get("llm_image_max_token", 188)
        if pipe_config.llm_encoder_type == "naive":
            llm_encoder_dtype = Qwen25VL7B_Embedder
        elif pipe_config.llm_encoder_type == "naive_qwen3vl8b":
            llm_encoder_dtype = Qwen3VL8B_Embedder
        else:
            raise ValueError(f"Unrecognized {pipe_config.llm_encoder_type=}")
    else:
        dit_params = DiTParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=None,
            context_in_dim=3572,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=60,
            depth_single_blocks=0,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        )
        llm_output_layer_index = -1
        llm_image_min_token = 188
        llm_image_max_token = 188
        llm_encoder_dtype = Qwen25VL7B_Embedder

    if not ae_path:
        raise ValueError("Autoencoder path is empty")
    if not qwenvl_model_path:
        raise ValueError("Qwen VL model path is empty")

    ae = AutoencoderKLQwenImage.from_pretrained(ae_path)

    with torch.device("meta"):
        dit = VarLenDiT(dit_params, build_llm_encoder=None)
        replace_norm_layers(dit)

    llm_encoder = llm_encoder_dtype(
        qwenvl_model_path,
        max_length,
        dtype,
        device,
        llm_image_min_token=llm_image_min_token,
        llm_image_max_token=llm_image_max_token,
        out_embedding_layer_index=llm_output_layer_index,
    )
    dit.llm_encoder = None

    dit = load_state_dict(dit, dit_path, device=device)
    dit = convert_precision(dit, dtype=dtype)
    dit = dit.to(device=device)
    llm_encoder = llm_encoder.to(device=device, dtype=torch.bfloat16)  # type: ignore
    ae = ae.to(device=device, dtype=torch.float32)  # type: ignore

    return ae, dit, llm_encoder


class ImageGenerator:
    def __init__(
        self,
        dit_path=None,
        ae_path=None,
        qwenvl_model_path=None,
        lora_path=None,
        world_mesh: DeviceMesh | None = None,
        device="cuda",
        max_length=2048,
        dtype=torch.bfloat16,
        config_path="",
    ) -> None:
        self.device = torch.device(device) if world_mesh is None else torch.cuda.current_device()
        self.ae, self.dit, self.llm_encoder = load_models(
            dit_path=dit_path,
            ae_path=ae_path,
            qwenvl_model_path=qwenvl_model_path,
            max_length=max_length,
            dtype=dtype,
            device=self.device,
            config_path=config_path,
        )

        if world_mesh is not None:
            if world_mesh["tp_w_sp"].size() > 1:
                print("Detected world size > 1, use tensor parallel.")
                self.dit.parallelize_module(world_mesh["tp_w_sp"], use_async_tp=True)  # type: ignore

        self.dit.apply_compile(inference_mode=True)  # type: ignore
        self.world_mesh = world_mesh

    @torch.no_grad()
    def prepare_txt_for_dit(self, txt, txt_lens, max_img_ids):
        _max_img_ids = max_img_ids[0]

        txt_ids = [
            torch.arange(x, device=txt.device, dtype=torch.float32)[:, None].repeat(1, 3) + _max_img_ids + 1
            for i, x in enumerate(txt_lens)
        ]

        txt_ids = torch.cat(txt_ids, dim=0)
        return txt_ids

    def prepare(
        self,
        prompt,
        initial_noise: torch.Tensor,
        ref_image_latents: list[torch.Tensor],
        task_type,
    ):
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = initial_noise.shape[0]
        img_ids = generate_image_position_qwen_image_ids(
            initial_noise.shape[-2] // 2, initial_noise.shape[-1] // 2, batch_size, 0, initial_noise.device
        ).flatten(0, 1)
        initial_noise = rearrange(initial_noise, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_lens = [initial_noise.shape[1]] * initial_noise.shape[0]
        img = initial_noise.flatten(0, 1)

        ref_img = None
        ref_img_ids = None
        ref_img_lens = [0]
        # For sref/RoPE-FA inference. Stores, per sample, the range of the
        # final reference image (style image) inside the concatenated ref tokens.
        self._last_sref_ref_token_ranges = None

        if task_type == "t2i":
            texts: list[str] = prompt
            ref_img = torch.Tensor([]).reshape(-1, 64).to(img)  # type: ignore
            ref_img_ids = torch.Tensor([]).reshape(-1, 3).to(img.device, torch.float32)  # type: ignore
            ref_img_lens = [0]
        elif task_type == "edit":
            texts: list[str] = [prompt[0], " "] if len(prompt) == 2 else prompt
            ref_img_lens = []
            ref_img_ids = []
            for img_idx, ref_image_latents_i in enumerate(ref_image_latents):
                ref_img_ids.append(
                    generate_image_position_qwen_image_ids(
                        ref_image_latents_i.shape[-2] // 2,
                        ref_image_latents_i.shape[-1] // 2,
                        batch_size,
                        img_idx + 1,
                        ref_image_latents_i.device,
                    ).flatten(0, 1)
                )
                ref_image_latents_i = rearrange(
                    ref_image_latents_i, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2
                )
                ref_img_lens.append(ref_image_latents_i.shape[1])
                ref_image_latents[img_idx] = ref_image_latents_i.flatten(0, 1)
            # Keep individual reference token lengths before collapsing to total.
            # In sref tasks ref images are [content_ref, style_ref]; RoPE-FA needs
            # the key range of the style ref (the final ref image).
            _ref_img_lens_each = [int(x) for x in ref_img_lens]
            _ref_total_len = int(sum(_ref_img_lens_each))
            _sref_len = int(_ref_img_lens_each[-1]) if _ref_img_lens_each else 0
            self._last_sref_ref_token_ranges = [(_ref_total_len - _sref_len, _ref_total_len)] * batch_size

            ref_img = torch.cat(ref_image_latents)
            ref_img_ids = torch.cat(ref_img_ids)
            ref_img = ref_img[None].repeat(batch_size, 1, 1).flatten(0, 1)
            ref_img_lens = [_ref_total_len] * batch_size
        else:
            raise ValueError(f"Unrecognized task type: {task_type}")

        max_img_ids = [
            max(x.max().item(), y.max().item()) if y.numel() > 0 else x.max().item()
            for x, y in zip(img_ids.split(img_lens), ref_img_ids.split(ref_img_lens))
        ]

        return img, img_ids, img_lens, ref_img, ref_img_ids, ref_img_lens, max_img_ids, texts

    @torch.no_grad()
    def encode_text(self, prompt: list[str], ref_images: list[torch.Tensor], task_type: str):
        ref_images = len(prompt) * [ref_images]
        task_types = len(prompt) * [task_type]
        return self.llm_encoder(prompt, ref_images, task_types)

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        img_lens: list[int],
        ref_img: torch.Tensor,
        ref_img_ids: torch.Tensor,
        ref_img_lens: list[int],
        txt,
        txt_ids,
        txt_lens,
        timesteps: list[float],
        sigmas: torch.Tensor,
        cfg_guidance: float = 4.5,
    ):
        latents = img.clone()

        from tqdm import tqdm

        if len(img_lens) > 1:
            txt = torch.cat([txt[:, None]] * len(img_lens), dim=1).flatten(0, 1)
            txt_ids = torch.cat([txt_ids[:, None]] * len(img_lens), dim=1).flatten(0, 1)
            txt_lens = combine_list([[x] * len(img_lens) for x in txt_lens])

        if cfg_guidance > 1:
            img_lens = img_lens * 2
            ref_img_lens = ref_img_lens * 2

            img_ids = torch.cat([img_ids, img_ids], dim=0).contiguous()
            ref_img = torch.cat([ref_img, ref_img], dim=0).contiguous()
            ref_img_ids = torch.cat([ref_img_ids, ref_img_ids], dim=0).contiguous()

        all_img_ids = None
        all_img_lens = [x + y for x, y in zip(img_lens, ref_img_lens)]  # type: ignore
        for _idx, (t_curr, t_prev) in tqdm(enumerate(itertools.pairwise(timesteps)), total=len(timesteps) - 1):
            if cfg_guidance > 1:
                latents = torch.cat([latents, latents], dim=0)
            t_vec = torch.full((len(img_lens),), t_curr, dtype=torch.float32, device=latents.device)

            device = img.device  # type: ignore
            bs = len(img_lens)  # type: ignore

            # VAE token should be put in img transformer
            img_varlen_config = VarLenConfig.from_seq_lens(img_lens, device)
            ref_img_varlen_config = VarLenConfig.from_seq_lens(ref_img_lens, device)
            all_img = cat_seq([latents, ref_img], [img_varlen_config.split_index, ref_img_varlen_config.split_index])
            if all_img_ids is None:
                all_img_ids = cat_seq(
                    [img_ids, ref_img_ids],  # type: ignore
                    [img_varlen_config.split_index, ref_img_varlen_config.split_index],
                )

            # DiT forward call
            # Guidance scale passed here
            guidance_value = torch.full((bs,), 3.5, device=device, dtype=torch.float32)

            # Qwen 2511 的设置，暂时不清楚是否有增益
            zero_t_seq_lens = [(x, 0) for x in ref_img_lens] if self.dit.enable_zero_t_embed else None  # type: ignore

            sref_key_ranges = None
            if bool(getattr(self.dit, "use_frequency_aware_rope", False)):
                sref_ref_token_ranges = getattr(self, "_last_sref_ref_token_ranges", None)
                if sref_ref_token_ranges is not None:
                    if cfg_guidance > 1:
                        # img/ref/txt lens lists were duplicated for CFG above.
                        sref_ref_token_ranges = sref_ref_token_ranges * 2
                    sref_key_ranges = []
                    for img_len, txt_len, ref_img_len, (ref_start, ref_end) in zip(
                        img_lens, txt_lens, ref_img_lens, sref_ref_token_ranges
                    ):
                        img_len = int(img_len)
                        txt_len = int(txt_len)
                        ref_img_len = int(ref_img_len)
                        ref_start = int(ref_start)
                        ref_end = int(ref_end)
                        if ref_start < 0 or ref_end < ref_start or ref_end > ref_img_len:
                            raise ValueError(
                                f"Invalid sref ref-token range {(ref_start, ref_end)} with total ref length {ref_img_len}"
                            )
                        if self.dit.enable_zero_t_embed:
                            k_start = img_len + txt_len + ref_start
                            k_end = img_len + txt_len + ref_end
                        else:
                            k_start = img_len + ref_start
                            k_end = img_len + ref_end
                        sref_key_ranges.append((k_start, k_end))

            if self.world_mesh is not None:
                if self.world_mesh["tp_w_sp"].size() > 1:
                    pred: torch.Tensor = self.dit(  # Use the potentially wrapped model
                        img=all_img,
                        img_ids=all_img_ids,
                        txt=txt,
                        txt_ids=txt_ids,
                        y=None,
                        timesteps=t_vec,
                        img_seq_lens=all_img_lens,
                        txt_seq_lens=txt_lens,
                        guidance=guidance_value,  # Check if DiT uses this directly
                        zero_t_seq_lens=zero_t_seq_lens,
                        sref_key_ranges=sref_key_ranges,
                    )
            else:
                with torch.inference_mode():
                    pred: torch.Tensor = self.dit(  # Use the potentially wrapped model
                        img=all_img,
                        img_ids=all_img_ids,
                        txt=txt,
                        txt_ids=txt_ids,
                        y=None,
                        timesteps=t_vec,
                        img_seq_lens=all_img_lens,
                        txt_seq_lens=txt_lens,
                        guidance=guidance_value,  # Check if DiT uses this directly
                        zero_t_seq_lens=zero_t_seq_lens,
                        sref_key_ranges=sref_key_ranges,
                    )

            pred = pred.view_as(all_img)  # type: ignore

            pred, _ = split_seq_by_len_list(pred, [img_lens, ref_img_lens])

            pred = pred.view_as(latents).type_as(latents)

            if cfg_guidance > 1:
                cond, uncond = (
                    pred[0 : pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )

                latents = latents[: latents.shape[0] // 2, :]
                comb_pred = uncond + cfg_guidance * (cond - uncond)
                cond_norm = torch.norm(cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                pred = comb_pred * (cond_norm / noise_norm)

            _latents = latents + (t_prev - t_curr) * pred
            # FIXME: 我知道上面的算子应该更合理，但是下面的是 Diffusers 的实现，二者有区别
            latents = latents + (sigmas[_idx + 1] - sigmas[_idx]) * pred
        import gc

        gc.collect(2)
        torch.cuda.empty_cache()

        return latents

    @staticmethod
    def unpack(x: torch.Tensor, height: int, width: int, batch_size: int) -> torch.Tensor:
        return rearrange(
            x,
            "(b h w) (c ph pw) -> b c (h ph) (w pw)",
            b=batch_size,
            h=math.ceil(height / 16),
            w=math.ceil(width / 16),
            ph=2,
            pw=2,
        )

    @staticmethod
    def generate_suitable_shape(width, height, base_size: int, step_size=16, range_scale=2 / 5):
        size_min = np.floor(np.sqrt(base_size * base_size * range_scale) / step_size).astype(np.int64) * step_size
        size_all = list(range(size_min, base_size, step_size))
        area = base_size * base_size
        aspect_size = []
        for size in size_all:
            if area % (size * step_size) == 0:
                aspect_size.append(
                    (
                        size,
                        np.ceil(area / size / step_size).astype(np.int64) * step_size,
                    )
                )
            else:
                aspect_size.append(
                    (
                        size,
                        np.ceil(area / size / step_size).astype(np.int64) * step_size,
                    )
                )
                aspect_size.append(
                    (
                        size,
                        np.floor(area / size / step_size).astype(np.int64) * step_size,
                    )
                )

        aspect_size = [*aspect_size, (base_size, base_size)]
        for h, w in aspect_size[::-1]:
            if h == w:
                continue
            aspect_size.append((w, h))

        suitable_shapes = np.array(aspect_size).tolist()
        t_h, t_w = suitable_shapes[0]
        min_aspect_ratio_error = abs(t_w / t_h - width / height)

        for h, w in suitable_shapes[1:]:
            error = abs(width / height - w / h)
            if error < min_aspect_ratio_error:
                min_aspect_ratio_error = error
                t_w, t_h = w, h
        return t_w, t_h

    @staticmethod
    def load_image(image):
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, Image.Image):
            image = F.to_tensor(image.convert("RGB"))
            image = image.unsqueeze(0)
            return image
        elif isinstance(image, torch.Tensor):
            return image
        elif isinstance(image, str):
            image = F.to_tensor(Image.open(image).convert("RGB"))
            image = image.unsqueeze(0)
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def encode_images(self, ref_image_pixels: list[torch.Tensor]) -> list[torch.Tensor]:
        """图像转 VAE Latent，需要将图像转为 latent
        Args:
            ref_image_pixels (list[torch.Tensor]): 3xHxW 的 torch.Tensor 构成的 list
        """
        ref_image_latents: list[torch.Tensor] = []
        for ref_image_i in ref_image_pixels:
            ref_image_latents_i = self.ae.encode(ref_image_i[None, :, None]).latent_dist.mean[:, :, 0]
            latents_mean = torch.tensor(self.ae.config.latents_mean).view(1, -1, 1, 1)
            latents_mean = latents_mean.to(ref_image_latents_i.device, ref_image_latents_i.dtype)

            latents_std = torch.tensor(self.ae.config.latents_std).view(1, -1, 1, 1)
            latents_std = latents_std.to(ref_image_latents_i.device, ref_image_latents_i.dtype)

            ref_image_latents_i = (ref_image_latents_i - latents_mean) / latents_std
            ref_image_latents.append(ref_image_latents_i)
        return ref_image_latents

    def process_ref_images(
        self,
        ref_image: list[PIL.Image.Image] | list[str] | str | PIL.Image.Image | None,
        base_size,
    ):
        target_height = None
        target_width = None
        if ref_image is not None:
            if not isinstance(ref_image, list):
                _ref_images: list[PIL.Image.Image] | list[str] = [ref_image]
            else:
                _ref_images = ref_image

            _ref_image: list[torch.Tensor] = []
            for ref_image_i in _ref_images:  # type: ignore
                if isinstance(ref_image_i, str):
                    with megfile.smart_open(ref_image_i, "rb") as image_file:
                        ref_image_i = PIL.Image.open(image_file).copy()

                ref_width, ref_height = ref_image_i.size  # type: ignore
                width_, height_ = self.generate_suitable_shape(
                    ref_width,
                    ref_height,
                    base_size,
                )

                if len(_ref_images) == 1:  # type: ignore
                    target_height = height_
                    target_width = width_

                ref_image_i = default_pil_resize(ref_image_i, height=height_, width=width_)

                ref_image_i = self.load_image(ref_image_i)
                ref_image_i = ref_image_i.to(self.device)

                ref_image_i = ref_image_i * 2 - 1

                ref_image_i = ref_image_i[0]
                _ref_image.append(ref_image_i)
            ref_image_pixels = _ref_image
        else:
            ref_image_pixels: list[torch.Tensor] = []
        ref_image_latents = self.encode_images(ref_image_pixels)

        return ref_image_pixels, ref_image_latents, target_height, target_width

    def generate_noise(self, seed: int | list[int], height: int, width: int):
        if isinstance(seed, int):
            seed = int(seed)
            seed = torch.Generator(device="cpu").seed() if seed < 0 else seed
            x = torch.randn(
                1,
                16,
                height // 8,
                width // 8,
                device=self.device,
                dtype=torch.float32,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
        elif isinstance(seed, list):
            assert all(isinstance(x, int) for x in seed)
            x = torch.cat(
                [
                    torch.randn(
                        1,
                        16,
                        height // 8,
                        width // 8,
                        device=self.device,
                        dtype=torch.float32,
                        generator=torch.Generator(device=self.device).manual_seed(seed_i),
                    )
                    for seed_i in seed  # type: ignore
                ],
                dim=0,
            )
        else:
            raise ValueError(f"Unrecognize seed type: {seed=}")

        return x

    def get_timesteps(self, num_steps: int, latent_size: int):
        # FIXME: this is different from Qwen Image, change max_shift to 0.6935483870967742
        timesteps = sampling.get_schedule(
            num_steps,
            latent_size,  # initial_noise.shape[-1] * initial_noise.shape[-2] // 4,
            shift=True,
            max_shift=0.6935483870967742,
            align_to_diffusers=True,
        )

        # DEBUG: terminal
        timesteps = 1 - (1 - timesteps) / ((1 - timesteps)[-2] / (1 - 0.02))  # type: ignore

        # 下面的 code 没什么道理，只是为了和 Diffusers 对齐
        # 注意这里需要先转到 GPU 上，然后除以1000，才能够和 Diffusers 对齐； CPU 的结果和 GPU 有差异
        timesteps = torch.from_numpy(timesteps).to(dtype=torch.float32, device=torch.cuda.current_device())
        sigmas = timesteps.clone()
        sigmas[-1] = 0
        timesteps = timesteps * 1000
        # 注意， GPU 上 /1000 和 / torch.tensor(1000).cuda() 不同，和 / torch.tensor(1000) 相同
        timesteps = timesteps / 1000
        timesteps = [*timesteps.tolist()[:-1], 0.0]
        # timesteps = [*timesteps[:-1], 0.0]
        # FIXME: check whether this will affect precision later
        timesteps[0] = 1.0

        return timesteps, sigmas

    @torch.no_grad()
    def decode_image(self, x: torch.Tensor):
        with torch.autocast(device_type=device_type, dtype=self.ae.dtype):
            latents_mean = torch.tensor(self.ae.config.latents_mean).view(1, -1, 1, 1)
            latents_mean = latents_mean.to(torch.cuda.current_device(), torch.float32)

            latents_std = torch.tensor(self.ae.config.latents_std).view(1, -1, 1, 1)
            latents_std = latents_std.to(torch.cuda.current_device(), torch.float32)

            x = x * latents_std + latents_mean
            x = self.ae.decode(x[:, :, None]).sample[:, :, 0]

            x = x.clamp(-1, 1)
            x = x.mul(0.5).add(0.5)
            return x

    @torch.no_grad()
    def generate_image(
        self,
        prompt,
        negative_prompt,
        width,
        height,
        num_steps,
        cfg_guidance,
        seed,
        ref_image: list[PIL.Image.Image] | list[str] | PIL.Image.Image | str | None = None,
        task_type="t2i",
    ):
        ref_image = ref_image
        assert task_type in ["t2i", "edit"]

        assert not all([task_type == "t2i", ref_image is not None])

        # allow for packing
        height = 16 * math.ceil(height / 16)
        width = 16 * math.ceil(width / 16)
        base_size = int(round(((height * width) ** 0.5) / 256) * 256)

        t0 = time.perf_counter()

        ref_image_pixels, ref_image_latents, target_height, target_width = self.process_ref_images(
            ref_image, base_size=base_size
        )
        if target_height is None or target_width is None:
            target_height = height
            target_width = width

        initial_noise = self.generate_noise(seed, height=target_height, width=target_width)
        timesteps, sigmas = self.get_timesteps(
            num_steps,
            initial_noise.shape[-1] * initial_noise.shape[-2] // 4,
        )

        img, img_ids, img_lens, ref_img, ref_img_ids, ref_img_lens, max_img_ids, texts = self.prepare(
            [prompt, negative_prompt] if cfg_guidance > 1.0 else [prompt],
            initial_noise,
            ref_image_latents,
            task_type,
        )
        txt, txt_lens = self.encode_text(texts, ref_images=ref_image_pixels, task_type=task_type)  # type: ignore
        txt_ids = self.prepare_txt_for_dit(txt, txt_lens, max_img_ids)

        x = self.denoise(
            img,
            img_ids,
            img_lens,
            ref_img,
            ref_img_ids,
            ref_img_lens,
            txt,
            txt_ids,
            txt_lens,
            cfg_guidance=cfg_guidance,
            timesteps=timesteps,
            sigmas=sigmas,
        )

        x = self.unpack(x.float(), target_height, target_width, initial_noise.shape[0])
        torch.cuda.empty_cache()
        x = self.decode_image(x)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        return x


def init_distributed_engine():
    # we only support One Node inference
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return None
    device = torch.device(f"{device_type}:{local_rank}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)

    # init distributed
    parallel_dims = ParallelDims(
        dp=1,
        tp_w_sp=world_size,
        world_size=world_size,
    )

    init_distributed(
        init_timeout_seconds=300,
        dump_folder=Path(Path("tmp") / "dump"),
        trace_buf_size=1000,
        enable_cpu_offload=False,
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    return world_mesh


def grid_images(images, w, h):
    from PIL import Image

    grid_image = Image.new("RGB", (2 * w, 2 * h))

    for i, image in enumerate(images):
        image = image.resize((w, h))
        grid_image.paste(image, ((i % 2) * w, (i // 2) * h))

    return grid_image


def main(
    config_path="",
    text_prompt="生成一个好看的图片",
    ref_image_file="",
    dit_path: str = "",
    save_path: str = "",
    seed: int = 1234,
):
    if ref_image_file == "":
        ref_image_file = [
            "/data/sirui/scripts/Kris_bench/multi_element_synthesis/source/1-1.jpg",
            "/data/sirui/scripts/Kris_bench/multi_element_synthesis/source/1-2.jpg",
        ]

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    world_mesh = init_distributed_engine()

    setting: dict = dict(
        num_steps=50,
        cfg_guidance=4,
    )

    image_edit = ImageGenerator(world_mesh=world_mesh, dit_path=dit_path, config_path=config_path)

    task_type = "edit"

    h, w = 1024, 1024

    images = image_edit.generate_image(
        text_prompt,
        negative_prompt="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting",  # noqa: E501
        width=w,
        height=h,
        num_steps=setting["num_steps"],
        cfg_guidance=setting.get("cfg_guidance", 6),
        seed=seed if isinstance(seed, list) else [seed],
        ref_image=ref_image_file,
        task_type=task_type,
    )

    if world_mesh is None or world_mesh.get_rank() == 0:
        images_list = [F.to_pil_image(img) for img in images.float()]
        with megfile.smart_open(save_path, "wb") as image_file:
            images_list[0].save(image_file, lossless=True)


if __name__ == "__main__":
    fire.Fire(main)
