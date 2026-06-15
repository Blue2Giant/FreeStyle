#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import multiprocessing as mp
import torch
from PIL import Image
from tqdm import tqdm


PREFERRED_KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


BASE_MODEL_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/sdxl"
IMAGE_ENCODER_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/ip_adaptor/sdxl_models/image_encoder"
CSGO_CKPT = "/mnt/task_runtime/shiyl_workspace/work/CSGO/csgo_4_32.bin"
VAE_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/VAEsdxl"
CONTROLNET_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/ttpcontrolnet"
WEIGHT_DTYPE = torch.float16


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CSGO (SDXL+ControlNet+IP-Adapter) on Sref/Cref prompts.json (multi-GPU, resize content to preferred ratio)"
    )
    p.add_argument(
        "--prompts_json",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark/data/sref/prompts.json",
        help="Path to prompts.json (id->prompt).",
    )
    p.add_argument(
        "--cref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark/data/sref/cref",
        help="Directory containing content reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--sref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark/data/sref/sref",
        help="Directory containing style reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./csgo_outputs_resize",
        help="Output directory to save generated images.",
    )

    p.add_argument("--steps", type=int, default=50, help="num_inference_steps")
    p.add_argument("--guidance_scale", type=float, default=10.0, help="guidance_scale")
    p.add_argument("--content_scale", type=float, default=1.0, help="content_scale")
    p.add_argument("--style_scale", type=float, default=1.0, help="style_scale")
    p.add_argument("--controlnet_conditioning_scale", type=float, default=0.4, help="controlnet_conditioning_scale")

    p.add_argument(
        "--negative_prompt",
        type=str,
        default="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
        help="negative_prompt",
    )

    p.add_argument("--seed", type=int, default=42, help="Base seed.")
    p.add_argument(
        "--seed_strategy",
        type=str,
        choices=["fixed", "per_id"],
        default="per_id",
        help='fixed: all samples use same seed; per_id: seed = base_seed + int(id).',
    )

    p.add_argument("--limit", type=int, default=0, help="Run first N items only. 0 = all.")
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help='Comma-separated ids to run, e.g. "000015,000010". Empty = all.',
    )

    p.add_argument(
        "--no_resize_cref",
        action="store_true",
        help="Disable resizing content(cref) to closest PREFERRED_KONTEXT_RESOLUTIONS aspect ratio.",
    )

    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help='GPU ids to use, e.g. "0,1,2,3". If CUDA not available, runs single-process on CPU.',
    )

    p.add_argument("--skip_existing", action="store_true", help="Skip ids whose output file already exists.")
    p.add_argument("--save_jsonl", action="store_true", help="Save metadata jsonl per-rank.")

    return p.parse_args()


def load_prompts(prompts_json: str) -> Dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def safe_open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def resize_cref(cref: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:

    cref_w, cref_h = cref.size
    aspect_ratio = cref_w / float(cref_h)

    _, target_w, target_h = min(
        (abs(aspect_ratio - (w / float(h))), w, h) for (w, h) in PREFERRED_KONTEXT_RESOLUTIONS
    )

    if (cref_w, cref_h) == (target_w, target_h):
        return cref, (target_w, target_h)

    resized = cref.resize((target_w, target_h), resample=_lanczos())
    return resized, (target_w, target_h)


def compute_seed(base_seed: int, seed_strategy: str, k: str) -> int:
    if seed_strategy == "per_id":
        try:
            return base_seed + int(k)
        except ValueError:
            return base_seed
    return base_seed


def build_csgo(device: torch.device):

    from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
    from ip_adapter.utils import BLOCKS as BLOCKS
    from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
    from ip_adapter import CSGO

    vae = AutoencoderKL.from_pretrained(VAE_PATH, torch_dtype=WEIGHT_DTYPE)
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_PATH,
        torch_dtype=WEIGHT_DTYPE,
        use_safetensors=True,
    )
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        BASE_MODEL_PATH,
        controlnet=controlnet,
        torch_dtype=WEIGHT_DTYPE,
        add_watermarker=False,
        vae=vae,
    )

    pipe.enable_vae_tiling()

    target_content_blocks = BLOCKS["content"]
    target_style_blocks = BLOCKS["style"]
    controlnet_target_content_blocks = controlnet_BLOCKS["content"]
    controlnet_target_style_blocks = controlnet_BLOCKS["style"]

    csgo = CSGO(
        pipe,
        IMAGE_ENCODER_PATH,
        CSGO_CKPT,
        device,
        num_content_tokens=4,
        num_style_tokens=32,
        target_content_blocks=target_content_blocks,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=True,
        controlnet_target_content_blocks=controlnet_target_content_blocks,
        controlnet_target_style_blocks=controlnet_target_style_blocks,
        content_model_resampler=True,
        style_model_resampler=True,
    )

    return csgo


def worker(rank: int, gpu_id: int, keys: List[str], prompts: Dict[str, str], args):
    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)

    meta_f = None
    if args.save_jsonl:
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")

    csgo = build_csgo(device)

    missing = 0
    skipped = 0

    autocast_ctx = torch.autocast(device_type="cuda", dtype=WEIGHT_DTYPE) if device.type == "cuda" else nullcontext()

    with torch.inference_mode():
        for k in tqdm(keys, desc=f"rank{rank} gpu{gpu_id}", position=rank, leave=True):
            out_path = out_dir / f"{k}.png"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue

            prompt = prompts[k]

            cref_path = cref_dir / f"{k}.png"
            sref_path = sref_dir / f"{k}.png"
            if not cref_path.exists() or not sref_path.exists():
                missing += 1
                print(f"[WARN][rank{rank}] missing id={k}: cref={cref_path.exists()} sref={sref_path.exists()}")
                continue

            content_img = safe_open_rgb(cref_path)
            style_img = safe_open_rgb(sref_path)

            # resize content to preferred ratio
            if not args.no_resize_cref:
                content_img, content_size = resize_cref(content_img)  # (W,H)
            else:
                content_size = content_img.size

            sample_seed = compute_seed(args.seed, args.seed_strategy, k)

            with autocast_ctx:
                try:
                    images = csgo.generate(
                        pil_content_image=content_img,
                        pil_style_image=style_img,  # 优先传 PIL
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        content_scale=args.content_scale,
                        style_scale=args.style_scale,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=1,
                        num_samples=1,
                        num_inference_steps=args.steps,
                        seed=sample_seed,
                        image=content_img.convert("RGB"),  # controlnet condition
                        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    )
                except Exception as e:
                    images = csgo.generate(
                        pil_content_image=content_img,
                        pil_style_image=str(sref_path),
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        content_scale=args.content_scale,
                        style_scale=args.style_scale,
                        guidance_scale=args.guidance_scale,
                        num_images_per_prompt=1,
                        num_samples=1,
                        num_inference_steps=args.steps,
                        seed=sample_seed,
                        image=content_img.convert("RGB"),
                        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    )

            out_img = images[0] if isinstance(images, (list, tuple)) else images

            if out_img.size != content_size:
                print(f"[WARN][rank{rank}] id={k} out.size={out_img.size} != content_size={content_size}")

            out_img.save(out_path)

            if meta_f is not None:
                record = {
                    "id": k,
                    "prompt": prompt,
                    "cref_path": str(cref_path),
                    "sref_path": str(sref_path),
                    "out_path": str(out_path),
                    "seed": sample_seed,
                    "steps": args.steps,
                    "guidance_scale": args.guidance_scale,
                    "content_scale": args.content_scale,
                    "style_scale": args.style_scale,
                    "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
                    "negative_prompt": args.negative_prompt,
                    "rank": rank,
                    "gpu_id": gpu_id,
                    "content_size_wh": [content_size[0], content_size[1]],
                    "out_size_wh": [out_img.size[0], out_img.size[1]],
                    "base_model_path": BASE_MODEL_PATH,
                    "controlnet_path": CONTROLNET_PATH,
                    "vae_path": VAE_PATH,
                    "image_encoder_path": IMAGE_ENCODER_PATH,
                    "csgo_ckpt": CSGO_CKPT,
                    "weight_dtype": str(WEIGHT_DTYPE),
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()

    if meta_f is not None:
        meta_f.close()

    print(f"[DONE][rank{rank} gpu{gpu_id}] processed={len(keys)} missing={missing} skipped={skipped} out_dir={args.out_dir}")


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_json)

    if args.ids.strip():
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        prompts = {k: prompts[k] for k in wanted if k in prompts}

    keys = sorted(prompts.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    # CPU fallback
    if not torch.cuda.is_available():
        worker(rank=0, gpu_id=-1, keys=keys, prompts=prompts, args=args)
        return

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    if len(gpus) == 0:
        raise ValueError("No GPUs parsed from --gpus")

    world = len(gpus)
    shards = [keys[r::world] for r in range(world)]

    ctx = mp.get_context("spawn")
    procs = []
    for r, gpu_id in enumerate(gpus):
        p = ctx.Process(target=worker, args=(r, gpu_id, shards[r], prompts, args))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker process exited with code {p.exitcode}")


class nullcontext:
    def __enter__(self): return self
    def __exit__(self, *args): return False


if __name__ == "__main__":
    main()