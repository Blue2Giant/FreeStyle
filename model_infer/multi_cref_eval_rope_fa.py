"""multi_cref_eval_rope_fa.py

批量推理（RoPE Frequency-Aware 版）。接口与 multi_cref_eval.py 完全兼容，
使用带 rope_fa 配置的训练 config 时会自动计算 sref_key_ranges 并传入 DiT。
"""
import itertools
import json
import math
import os
import time
from typing import List, Literal, Optional

import fire
import imageio
import torch
from torchvision.transforms import functional as F
from tqdm import tqdm

from vgo.inference import ImageGenerator, init_distributed_engine
from vgo.models.modules.varlen_ops import VarLenConfig, cat_seq, split_seq_by_len_list
from vgo.utils.common_utils import combine_list


class ImageGeneratorRopeFA(ImageGenerator):
    """ImageGenerator that passes sref_key_ranges to the DiT when rope_fa is enabled."""

    def denoise(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        img_lens: list,
        ref_img: torch.Tensor,
        ref_img_ids: torch.Tensor,
        ref_img_lens: list,
        txt,
        txt_ids,
        txt_lens,
        timesteps: list,
        sigmas: torch.Tensor,
        cfg_guidance: float = 4.5,
        per_sample_ref_img_lens: Optional[List[List[int]]] = None,
        sref_ref_index: int = 1,
    ):
        latents = img.clone()

        # Save pre-CFG img_lens for sref_key_ranges computation
        original_img_lens = list(img_lens)

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

        # Build sref_key_ranges when DiT has rope_fa enabled.
        # Joint sequence per sample: [target_tokens][cref_tokens][sref_tokens][txt_tokens]
        sref_key_ranges = None
        if getattr(self.dit, "use_frequency_aware_rope", False) and per_sample_ref_img_lens is not None:
            base_ranges = []
            for tgt_len, ref_lens_i in zip(original_img_lens, per_sample_ref_img_lens):
                offset = tgt_len + sum(ref_lens_i[:sref_ref_index])
                sref_len = ref_lens_i[sref_ref_index] if sref_ref_index < len(ref_lens_i) else 0
                base_ranges.append((offset, offset + sref_len))
            sref_key_ranges = base_ranges * 2 if cfg_guidance > 1 else list(base_ranges)

        all_img_ids = None
        all_img_lens = [x + y for x, y in zip(img_lens, ref_img_lens)]

        for _idx, (t_curr, t_prev) in tqdm(
            enumerate(itertools.pairwise(timesteps)), total=len(timesteps) - 1
        ):
            if cfg_guidance > 1:
                latents = torch.cat([latents, latents], dim=0)
            t_vec = torch.full((len(img_lens),), t_curr, dtype=torch.float32, device=latents.device)

            device = img.device
            bs = len(img_lens)

            img_varlen_config = VarLenConfig.from_seq_lens(img_lens, device)
            ref_img_varlen_config = VarLenConfig.from_seq_lens(ref_img_lens, device)
            all_img = cat_seq(
                [latents, ref_img],
                [img_varlen_config.split_index, ref_img_varlen_config.split_index],
            )
            if all_img_ids is None:
                all_img_ids = cat_seq(
                    [img_ids, ref_img_ids],
                    [img_varlen_config.split_index, ref_img_varlen_config.split_index],
                )

            guidance_value = torch.full((bs,), 3.5, device=device, dtype=torch.float32)
            zero_t_seq_lens = (
                [(x, 0) for x in ref_img_lens] if self.dit.enable_zero_t_embed else None
            )

            dit_kwargs = dict(
                img=all_img,
                img_ids=all_img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=None,
                timesteps=t_vec,
                img_seq_lens=all_img_lens,
                txt_seq_lens=txt_lens,
                guidance=guidance_value,
                zero_t_seq_lens=zero_t_seq_lens,
            )
            if sref_key_ranges is not None:
                dit_kwargs["sref_key_ranges"] = sref_key_ranges

            if self.world_mesh is not None:
                if self.world_mesh["tp_w_sp"].size() > 1:
                    pred: torch.Tensor = self.dit(**dit_kwargs)
            else:
                with torch.inference_mode():
                    pred: torch.Tensor = self.dit(**dit_kwargs)

            pred = pred.view_as(all_img)
            pred, _ = split_seq_by_len_list(pred, [img_lens, ref_img_lens])
            pred = pred.view_as(latents).type_as(latents)

            if cfg_guidance > 1:
                cond, uncond = (
                    pred[: pred.shape[0] // 2, :],
                    pred[pred.shape[0] // 2 :, :],
                )
                latents = latents[: latents.shape[0] // 2, :]
                comb_pred = uncond + cfg_guidance * (cond - uncond)
                cond_norm = torch.norm(cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                pred = comb_pred * (cond_norm / noise_norm)

            latents = latents + (sigmas[_idx + 1] - sigmas[_idx]) * pred

        import gc

        gc.collect(2)
        torch.cuda.empty_cache()
        return latents

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
        ref_image=None,
        task_type="edit",
        sref_ref_index: int = 1,
    ):
        assert task_type in ["t2i", "edit"]
        assert not all([task_type == "t2i", ref_image is not None])

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

        # Compute per-ref-image token lengths.
        # VAE latent shape: (1, C, H//8, W//8); after pack(ph=2,pw=2): tokens = H//16 * W//16
        per_sample_ref_img_lens = None
        if getattr(self.dit, "use_frequency_aware_rope", False) and ref_image_latents:
            per_sample_ref_img_lens = [
                [(lat.shape[-2] // 2) * (lat.shape[-1] // 2) for lat in ref_image_latents]
            ]

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
        txt, txt_lens = self.encode_text(texts, ref_images=ref_image_pixels, task_type=task_type)
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
            per_sample_ref_img_lens=per_sample_ref_img_lens,
            sref_ref_index=sref_ref_index,
        )

        x = self.unpack(x.float(), target_height, target_width, initial_noise.shape[0])
        torch.cuda.empty_cache()
        x = self.decode_image(x)

        t1 = time.perf_counter()
        print(f"Done in {t1 - t0:.1f}s.")
        return x


def main(  # noqa: C901
    name="default",
    bmk: Literal[
        "multi", "single", "omnicontext", "gref", "sref",
        "ljh_sref", "ljh_sref_recaption", "ljh_sref_recaption_sref",
    ] = "ljh_sref_recaption",
    config_path="",
    dit_path: str = "",
    save_path: str = "",
    cfg=3,
    dp_rank=0,
    dp_size=1,
    width=1024,
    height=1024,
    key_txt="",
    ae_path="",
    qwenvl_model_path="",
    sref_ref_index: int = 1,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    world_mesh = init_distributed_engine()

    pre_defined_settings = {
        "fast": dict(num_steps=35),
        "default": dict(num_steps=50, cfg_guidance=6),
        "few": dict(num_steps=16),
        "16-few32-cfg=2": dict(num_steps=32, cfg_guidance=2.0),
        "t2i": dict(num_steps=50, cfg_guidance=6),
        "edit": dict(num_steps=28, cfg_guidance=8),
    }

    setting: dict = pre_defined_settings[name]

    t2i = ImageGeneratorRopeFA(
        world_mesh=world_mesh,
        dit_path=dit_path,
        config_path=config_path,
        ae_path=ae_path or None,
        qwenvl_model_path=qwenvl_model_path or None,
    )

    if bmk == "multi":
        test_data_folder = "/mnt/chengwei/multi_cref_bmk"
        with open(f"{test_data_folder}/prompts2.json") as f:
            prompts = json.load(f)
    elif bmk == "single":
        test_data_folder = "/mnt/chengwei/single_cref_bmk"
        with open(f"{test_data_folder}/tiny_bmk_v2.json") as f:
            prompts = json.load(f)
    elif bmk == "omnicontext":
        test_data_folder = "/mnt/chengwei/omnicontext_cref_bmk"
        with open(f"{test_data_folder}/prompt.json") as f:
            prompts = json.load(f)
    elif bmk == "gref":
        test_data_folder = "/mnt/chengwei/gref_cref_bmk"
        with open(f"{test_data_folder}/prompts_en.json") as f:
            prompts = json.load(f)
    elif bmk == "sref":
        test_data_folder = "/mnt/chengwei/sref_cref_bmk"
        with open(f"{test_data_folder}/prompt_output.json") as f:
            prompts = json.load(f)
    elif bmk == "ljh_sref":
        test_data_folder = "/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
        with open(f"{test_data_folder}/prompts.json") as f:
            prompts = json.load(f)
    elif bmk == "ljh_sref_recaption":
        test_data_folder = "/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content"
        with open(f"{test_data_folder}/prompt_output.json") as f:
            prompts = json.load(f)
    elif bmk == "ljh_sref_recaption_sref":
        test_data_folder = "/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
        with open(f"{test_data_folder}/prompts_recaption.json") as f:
            prompts = json.load(f)
    else:
        raise ValueError(f"Unknown bmk: {bmk}")

    if key_txt:
        with open(key_txt) as f:
            ordered_keys = [line.strip() for line in f if line.strip()]
        prompts = {k: prompts[k] for k in ordered_keys if k in prompts}

    task_type = "edit"
    setting = pre_defined_settings[task_type]
    os.makedirs(save_path, exist_ok=True)

    h, w = int(height), int(width)

    for i, (filename, text_prompt) in tqdm(
        enumerate(prompts.items()), total=len(prompts), desc=save_path
    ):
        if i % dp_size != dp_rank:
            continue
        out_path = f"{save_path}/{filename}.png"
        if os.path.exists(out_path):
            tqdm.write(f"[skip] {filename}")
            continue
        tqdm.write(f'Run Image "{text_prompt}"')

        if bmk == "multi":
            ref_image_file = []
            for j in range(4):
                file = f"{test_data_folder}/ref{j + 1}/results/{filename}.png"
                img_arr = imageio.imread(file)
                if img_arr.shape[0] == 512 and img_arr.shape[1] == 512 and img_arr.mean() == 255:
                    continue
                ref_image_file.append(file)
        elif bmk == "single":
            ref_image_file = [f"{test_data_folder}/reference/{filename}.jpg"]
        elif bmk == "omnicontext":
            ref_image_file = []
            for j in range(3):
                file = f"{test_data_folder}/ref{j}/{filename}.png"
                img_arr = imageio.imread(file)
                if img_arr.shape[0] == 512 and img_arr.shape[1] == 512 and img_arr.mean() == 255:
                    continue
                ref_image_file.append(file)
        elif bmk == "gref":
            ref_image_file = []
            for j in range(6):
                file = f"{test_data_folder}/ref_img_{j}/{filename}.jpg"
                img_arr = imageio.imread(file)
                if img_arr.shape[0] == 512 and img_arr.shape[1] == 512 and img_arr.mean() == 255:
                    continue
                ref_image_file.append(file)
        elif bmk == "sref":
            ref_image_file = [f"{test_data_folder}/ref_{j}/{filename}.webp" for j in range(2)]
        elif bmk in ("ljh_sref", "ljh_sref_recaption", "ljh_sref_recaption_sref"):
            ref_image_file = [
                f"{test_data_folder}/cref/{filename}.png",
                f"{test_data_folder}/sref/{filename}.png",
            ]

        images = t2i.generate_image(
            text_prompt,
            negative_prompt="worst quality, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting",  # noqa: E501
            width=w,
            height=h,
            num_steps=setting["num_steps"],
            cfg_guidance=cfg,
            seed=[42],
            ref_image=ref_image_file,
            task_type=task_type,
            sref_ref_index=sref_ref_index,
        )
        if world_mesh is None or world_mesh.get_rank() == 0:
            images_list = [F.to_pil_image(img) for img in images.float()]

        images_list = [F.to_pil_image(img) for img in images.float()]
        image = images_list[0]
        image.save(f"{save_path}/{filename}.png")


if __name__ == "__main__":
    fire.Fire(main)
