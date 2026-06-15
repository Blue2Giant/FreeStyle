#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/sref_cref/qwen_infer.py \
    --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
    --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref \
    --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref \
    --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit \
    --model_name /mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/ \
    --gpus 0
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import multiprocessing as mp
import torch
from PIL import Image
from tqdm import tqdm


# IMPORTANT: 按你截图写法，这里每个 tuple 视为 (w, h)
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


def parse_args():
    p = argparse.ArgumentParser(
        description="Run Qwen-Image-Edit-Plus on Sref/Cref benchmark prompts.json (multi-GPU, resize cref to preferred ratio)"
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
        default="./qwen_editplus_outputs_resize",
        help="Output directory to save generated images.",
    )

    # Model / inference
    p.add_argument(
        "--model_name",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/qwen-edit2511",
        help="Model path/name for QwenImageEditPlusPipeline.",
    )
    p.add_argument("--steps", type=int, default=28, help="num_inference_steps")
    p.add_argument("--true_cfg_scale", type=float, default=4.0, help="true_cfg_scale")
    p.add_argument("--negative_prompt", type=str, default=" ", help="negative_prompt (optional)")

    # Sampling / determinism
    p.add_argument("--seed", type=int, default=42, help="Base seed.")
    p.add_argument(
        "--seed_strategy",
        type=str,
        choices=["fixed", "per_id"],
        default="per_id",
        help='fixed: all samples use same seed; per_id: seed = base_seed + int(id).',
    )

    # Optional filtering
    p.add_argument("--limit", type=int, default=0, help="Run first N items only. 0 = all.")
    p.add_argument(
        "--ids",
        type=str,
        default="",
        help='Comma-separated ids to run, e.g. "000015,000010". Empty = all.',
    )

    # Size control
    p.add_argument(
        "--no_resize_cref",
        action="store_true",
        help="Disable resizing reference images to closest PREFERRED_KONTEXT_RESOLUTIONS aspect ratio.",
    )
    p.add_argument(
        "--output_resolution",
        type=str,
        default="",
        help='Set generated/noise resolution as "WIDTHxHEIGHT", e.g. "1024x1024". Reference images still use bucket resize.',
    )
    p.add_argument(
        "--input_resolution",
        type=str,
        default="",
        help='Deprecated alias for --output_resolution. It no longer forces reference images to this size.',
    )

    # Multi-GPU / multiprocessing
    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help='GPU ids to use, e.g. "0,1,2,3". If CUDA not available, runs single-process on CPU.',
    )

    # Save some metadata
    p.add_argument(
        "--save_jsonl",
        action="store_true",
        help="Save metadata jsonl. Multi-proc will write per-rank files: metadata.rank{r}.jsonl",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite outputs. Default: skip existing outputs.",
    )

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
    # Pillow>=10 uses Image.Resampling
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def parse_resolution(spec: str) -> Tuple[int, int]:
    text = str(spec).strip().lower()
    if "x" not in text:
        raise ValueError(f"invalid resolution {spec!r}; expected WIDTHxHEIGHT")
    width_text, height_text = text.split("x", 1)
    width = int(width_text)
    height = int(height_text)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resolution {spec!r}; width/height must be positive")
    return width, height


def resize_cref_like_screenshot(cref: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    完全按你截图逻辑：
      cref_h, cref_w from image shape; aspect_ratio = cref_w / cref_h
      _, w, h = min((abs(aspect_ratio - w/h), w, h) for w,h in PREFERRED...)
    这里 PIL 是 (W,H)，等价：
      aspect_ratio = cref_w / cref_h
    返回 resized_cref 以及 resized 后的 (W,H)。
    """
    cref_w, cref_h = cref.size  # PIL: (W,H)
    aspect_ratio = cref_w / float(cref_h)

    _, target_w, target_h = min(
        (abs(aspect_ratio - (w / float(h))), w, h) for (w, h) in PREFERRED_KONTEXT_RESOLUTIONS
    )

    if (cref_w, cref_h) == (target_w, target_h):
        return cref, (target_w, target_h)

    resized = cref.resize((target_w, target_h), resample=_lanczos())
    return resized, (target_w, target_h)


def resolve_output_size(args) -> Tuple[int, int] | None:
    """Return requested generated/noise size. --input_resolution is kept as a deprecated alias."""
    output_resolution = getattr(args, "output_resolution", "") or ""
    input_resolution = getattr(args, "input_resolution", "") or ""
    if output_resolution:
        return parse_resolution(output_resolution)
    if input_resolution:
        return parse_resolution(input_resolution)
    return None


def compute_seed(base_seed: int, seed_strategy: str, k: str) -> int:
    if seed_strategy == "per_id":
        try:
            return base_seed + int(k)
        except ValueError:
            return base_seed
    return base_seed


def worker(rank: int, gpu_id: int, keys: List[str], prompts: Dict[str, str], args):
    # device / dtype
    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch_dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    # Import in worker (important for spawn)
    from diffusers import QwenImageEditPlusPipeline

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)

    meta_f = None
    if args.save_jsonl:
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")

    requested_output_size = resolve_output_size(args)
    if getattr(args, "input_resolution", "") and not getattr(args, "output_resolution", ""):
        print(
            f"[WARN][rank{rank}] --input_resolution is deprecated; treating it as generated/noise "
            f"--output_resolution={args.input_resolution}. Reference images are bucket-resized.",
            flush=True,
        )

    missing = 0
    skipped = 0
    done = 0
    total = len(keys)
    with torch.inference_mode():
        for idx, k in enumerate(tqdm(keys, desc=f"rank{rank} gpu{gpu_id}", position=rank, leave=True), start=1):
            out_path = out_dir / f"{k}.png"
            if (not args.overwrite) and out_path.exists():
                skipped += 1
                if idx % 10 == 0 or idx == total:
                    print(
                        f"[Qwen][rank{rank} gpu{gpu_id}] scanned={idx}/{total} done={done} skipped={skipped} missing={missing}",
                        flush=True,
                    )
                continue

            prompt = prompts[k]

            cref_path = cref_dir / f"{k}.png"
            sref_path = sref_dir / f"{k}.png"
            if not cref_path.exists() or not sref_path.exists():
                missing += 1
                print(f"[WARN][rank{rank}] missing id={k}: cref={cref_path.exists()} sref={sref_path.exists()}")
                if idx % 10 == 0 or idx == total:
                    print(
                        f"[Qwen][rank{rank} gpu{gpu_id}] scanned={idx}/{total} done={done} skipped={skipped} missing={missing}",
                        flush=True,
                    )
                continue

            cref = safe_open_rgb(cref_path)
            sref = safe_open_rgb(sref_path)

            # 1) resize reference images with the Kontext bucket logic, independently from output/noise size.
            #    --output_resolution controls only the generated canvas/noise dimensions.
            cref_input_size: Tuple[int, int]
            sref_input_size: Tuple[int, int]
            if not args.no_resize_cref:
                cref, cref_input_size = resize_cref_like_screenshot(cref)  # (W,H)
                sref, sref_input_size = resize_cref_like_screenshot(sref)  # (W,H)
            else:
                cref_input_size = cref.size  # (W,H)
                sref_input_size = sref.size  # (W,H)

            content_size = requested_output_size or cref_input_size

            # 2) deterministic seed
            sample_seed = compute_seed(args.seed, args.seed_strategy, k)
            gen = torch.Generator(device=device).manual_seed(sample_seed)

            # run pipeline
            out = pipe(
                image=[cref, sref],
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                width=content_size[0],
                height=content_size[1],
                num_inference_steps=args.steps,
                true_cfg_scale=args.true_cfg_scale,
                generator=gen,
            ).images[0]

            if out.size != content_size:
                out = out.resize(content_size, resample=_lanczos())

            out.save(out_path)
            done += 1

            if meta_f is not None:
                record = {
                    "id": k,
                    "prompt": prompt,
                    "cref_path": str(cref_path),
                    "sref_path": str(sref_path),
                    "out_path": str(out_path),
                    "seed": sample_seed,
                    "steps": args.steps,
                    "true_cfg_scale": args.true_cfg_scale,
                    "negative_prompt": args.negative_prompt,
                    "model_name": args.model_name,
                    "rank": rank,
                    "gpu_id": gpu_id,
                    "requested_output_size_wh": [content_size[0], content_size[1]],
                    "content_size_wh": [content_size[0], content_size[1]],
                    "cref_input_size_wh": list(cref_input_size),
                    "sref_input_size_wh": list(sref_input_size),
                    "out_size_wh": [out.size[0], out.size[1]],
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()

            if idx % 10 == 0 or idx == total:
                print(
                    f"[Qwen][rank{rank} gpu{gpu_id}] scanned={idx}/{total} done={done} skipped={skipped} missing={missing}",
                    flush=True,
                )

    if meta_f is not None:
        meta_f.close()

    print(
        f"[DONE][rank{rank} gpu{gpu_id}] scanned={total} done={done} skipped={skipped} missing={missing} out_dir={args.out_dir}"
    )


def main():
    args = parse_args()

    prompts = load_prompts(args.prompts_json)

    if args.ids.strip():
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        prompts = {k: prompts[k] for k in wanted if k in prompts}

    keys = sorted(prompts.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    # CPU fallback: single-process
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


if __name__ == "__main__":
    main()
