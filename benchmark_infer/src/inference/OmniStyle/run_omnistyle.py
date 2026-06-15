#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

FLUX_DEV_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/flux_dev/flux1-dev.safetensors"
AE_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/flux_dev/ae.safetensors"
T5_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/xflux_text_encoders"
CLIP_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/clip-vit-p14-large"
LORA_PATH = "/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/omnistyle/dit_lora.safetensors"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run OmniStyle DSTPipeline on Sref/Cref ids (multi-GPU, resize cref to preferred ratio)."
    )
    p.add_argument(
        "--prompts_json",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/prompts.json",
        help="Path to prompts.json (id->prompt). Used only for id list/filtering.",
    )
    p.add_argument(
        "--cref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/cref",
        help="Directory containing content reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--sref_dir",
        type=str,
        default="/mnt/task_runtime/shiyl_workspace/work/gencompress/hf/Sref_Cref_Benchmark_new800/bench/sref/sref",
        help="Directory containing style/reference images, e.g., 000015.png",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./omnistyle_outputs_resize",
        help="Output directory to save generated images.",
    )

    p.add_argument(
        "--model_type",
        type=str,
        default="flux-dev",
        choices=["flux-dev", "flux-dev-fp8", "flux-schnell"],
        help='DSTPipeline model_type, e.g. "flux-dev".',
    )
    p.add_argument("--num_steps", type=int, default=25, help="DSTPipeline num_steps")
    p.add_argument("--guidance", type=float, default=4.0, help="DSTPipeline guidance")
    p.add_argument("--only_lora", default=True)
    p.add_argument("--lora_rank", type=int, default=512, help="DSTPipeline lora_rank")
    p.add_argument("--pe", type=str, default="d", choices=["d", "h", "w", "o"], help="DSTPipeline pe")
    p.add_argument("--flux_dev_path", type=str, default="")
    p.add_argument("--ae_path", type=str, default="")
    p.add_argument("--t5_path", type=str, default="")
    p.add_argument("--clip_path", type=str, default="")
    p.add_argument("--lora_path", type=str, default="")

    p.add_argument(
        "--concat_refs",
        action="store_true",
        help="If set, save a 3-panel image: [content | style | generated]. Otherwise save generated only.",
    )

    p.add_argument(
        "--no_images",
        action="store_true",
        help="(Not recommended) If set, will skip because DSTPipeline requires ref_imgs.",
    )
    p.add_argument(
        "--use_only_cref",
        action="store_true",
        help="If set, pass only content ref as ref_imgs=[cnt]. Default: ref_imgs=[sty,cnt].",
    )

    p.add_argument("--seed", type=int, default=42, help="Base seed.")
    p.add_argument(
        "--seed_strategy",
        type=str,
        choices=["fixed", "per_id"],
        default="fixed",
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

    # Resize control
    p.add_argument(
        "--no_resize_cref",
        action="store_true",
        help="Disable resizing cref to closest PREFERRED_KONTEXT_RESOLUTIONS aspect ratio.",
    )

    # Multi-GPU / multiprocessing
    p.add_argument(
        "--gpus",
        type=str,
        default="0,1,2,3",
        help='GPU ids to use, e.g. "0,1,2,3". If CUDA not available, runs single-process on CPU.',
    )

    # Resume / skipping
    p.add_argument(
        "--skip_existing",
        action="store_true",
        help="If set, skip ids whose output file already exists.",
    )

    # Save metadata
    p.add_argument(
        "--save_jsonl",
        action="store_true",
        help="Save metadata jsonl. Multi-proc writes per-rank files: metadata.rank{r}.jsonl",
    )

    return p.parse_args()


def load_prompts(prompts_json: str) -> Dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def safe_open_rgb(path: Path) -> Image.Image:
    # 用 copy() 断开底层文件句柄，避免长跑 fd 累积
    with Image.open(path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img.copy()


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def resize_cref(cref: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:

    cref_w, cref_h = cref.size  # PIL: (W,H)
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


def _set_omnistyle_env(args):
    os.environ["FLUX_DEV"] = args.flux_dev_path or os.environ.get("FLUX_DEV", FLUX_DEV_PATH)
    os.environ["AE"] = args.ae_path or os.environ.get("AE", AE_PATH)
    os.environ["T5"] = args.t5_path or os.environ.get("T5", T5_PATH)
    os.environ["CLIP"] = args.clip_path or os.environ.get("CLIP", CLIP_PATH)
    os.environ["LORA"] = args.lora_path or os.environ.get("LORA", LORA_PATH)


def worker(rank: int, gpu_id: int, keys: List[str], prompts: Dict[str, str], args):
    # device
    if torch.cuda.is_available() and gpu_id >= 0:
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    # IMPORTANT: set env before importing/initializing pipeline (some code reads env at import/init)
    _set_omnistyle_env(args)

    # Import in worker (important for spawn)
    from omnistyle.flux.pipeline import DSTPipeline

    # deepspeed flag: 你的原脚本是多进程多卡，不走 accelerate/deepspeed，这里传 False
    pipe = DSTPipeline(
        args.model_type,
        device,
        False,
        only_lora=bool(args.only_lora),
        lora_rank=int(args.lora_rank),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)

    meta_f = None
    if args.save_jsonl:
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")

    missing = 0
    skipped = 0
    noimg_skipped = 0

    with torch.inference_mode():
        for k in tqdm(keys, desc=f"rank{rank} gpu{gpu_id}", position=rank, leave=True):
            # output name
            out_path = out_dir / f"{k}.png"
            if args.skip_existing and out_path.exists():
                skipped += 1
                continue



            cref_path = cref_dir / f"{k}.png"
            sref_path = sref_dir / f"{k}.png"
            if not cref_path.exists() or (not args.use_only_cref and not sref_path.exists()):
                missing += 1
                print(f"[WARN][rank{rank}] missing id={k}: cref={cref_path.exists()} sref={sref_path.exists()}")
                continue

            cnt = safe_open_rgb(cref_path)

            if not args.no_resize_cref:
                cnt, content_size = resize_cref(cnt)  # (W,H)
            else:
                content_size = cnt.size  # (W,H)

            # style
            if args.use_only_cref:
                sty = None
            else:
                sty = safe_open_rgb(sref_path)


            # seed
            sample_seed = compute_seed(args.seed, args.seed_strategy, k)

            # build ref_imgs (按你给的示例：ref_imgs = [sty, cnt])
            if sty is None:
                ref_imgs = [cnt]
            else:
                ref_imgs = [sty, cnt]

            image_gen = pipe(
                prompt="",
                width=content_size[0],
                height=content_size[1],
                guidance=float(args.guidance),
                num_steps=int(args.num_steps),
                seed=int(sample_seed),
                ref_imgs=ref_imgs,
                pe=str(args.pe),
            )

            if args.concat_refs and sty is not None:
                new_blank = Image.new("RGB", (content_size[0] * 3, content_size[1]))
                new_blank.paste(cnt, (0, 0))
                new_blank.paste(sty, (content_size[0], 0))
                new_blank.paste(image_gen, (content_size[0] * 2, 0))
                new_blank.save(out_path)
                out_size = new_blank.size
            else:
                image_gen.save(out_path)
                out_size = image_gen.size

            if meta_f is not None:
                record = {
                    "id": k,
                    "model_type": args.model_type,
                    "rank": rank,
                    "gpu_id": gpu_id,
                    "num_steps": args.num_steps,
                    "guidance": args.guidance,
                    "seed": sample_seed,
                    "only_lora": bool(args.only_lora),
                    "lora_rank": int(args.lora_rank),
                    "pe": str(args.pe),
                    "concat_refs": bool(args.concat_refs),
                    "no_resize_cref": bool(args.no_resize_cref),
                    "use_only_cref": bool(args.use_only_cref),
                    "content_size_wh": [content_size[0], content_size[1]],
                    "out_size_wh": [out_size[0], out_size[1]],
                    "paths": {
                        "FLUX_DEV": os.environ.get("FLUX_DEV", ""),
                        "AE": os.environ.get("AE", ""),
                        "T5": os.environ.get("T5", ""),
                        "CLIP": os.environ.get("CLIP", ""),
                        "LORA": os.environ.get("LORA", ""),
                    },
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()

    if meta_f is not None:
        meta_f.close()

    print(
        f"[DONE][rank{rank} gpu{gpu_id}] processed={len(keys)} missing={missing} "
        f"skipped={skipped} noimg_skipped={noimg_skipped} out_dir={args.out_dir}"
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
