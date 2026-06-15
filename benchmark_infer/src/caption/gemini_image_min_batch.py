#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/caption_pipe/gemini_image_min_batch.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/gemini-edit \
  --model_id gemini-2.5-flash-image-native \
  --num_procs 8 \
  --num_generate 200
"""
import argparse
import json
import os
import multiprocessing as mp
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

from google import genai
from google.genai import types
from PIL import Image
from tqdm import tqdm


banana_aspect_ratio = ["1:1","1:4","1:8","2:3","3:2","3:4","4:1","4:3","4:5","5:4","8:1","9:16","16:9","21:9"]
banana_resolution = ["512px", "1K", "2K", "4K"]


def image_to_part(path: str) -> types.Part:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


def load_prompts(prompts_json: str) -> Dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("prompts_json must be a dict")
    return {str(k): str(v) for k, v in data.items()}


def build_client(base_url: str, api_version: str):
    return genai.Client(
        http_options={"api_version": api_version, "base_url": base_url},
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )


def _parse_ratio(ratio: str) -> float:
    left, right = ratio.split(":", 1)
    return float(left) / float(right)


def _select_aspect_ratio(width: int, height: int) -> str:
    target = width / float(height)
    best = None
    best_diff = None
    for r in banana_aspect_ratio:
        value = _parse_ratio(r)
        diff = abs(value - target)
        if best is None or diff < best_diff:
            best = r
            best_diff = diff
    return best



def save_first_image(resp, out_path: Path) -> bool:
    for part in resp.parts:
        if part.inline_data is not None:
            img = part.as_image()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_path)
            print(f"[SAVE] {out_path} size={img.size}")
            return True
    return False


def _worker(rank: int, tasks: List[Tuple[str, str]], args, result_queue: mp.Queue):
    client = build_client(args.base_url, args.api_version)
    out_dir = Path(args.out_dir)
    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)
    meta_f = None
    if args.save_jsonl:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")
    for k, prompt in tasks:
        try:
            out_path = out_dir / f"{k}.png"
            if (not args.overwrite) and out_path.exists():
                result_queue.put(("skip", k, None))
                continue
            cref_path = cref_dir / f"{k}.png"
            sref_path = sref_dir / f"{k}.png"
            if not cref_path.exists() or not sref_path.exists():
                result_queue.put(("missing", k, None))
                continue
            content_image = Image.open(cref_path).convert("RGB")
            if args.aspect_ratio:
                if args.aspect_ratio not in banana_aspect_ratio:
                    raise ValueError(f"aspect_ratio {args.aspect_ratio} not in {banana_aspect_ratio}")
                aspect_ratio = args.aspect_ratio
            else:
                aspect_ratio = _select_aspect_ratio(content_image.width, content_image.height)
            if args.resolution not in banana_resolution:
                raise ValueError(f"resolution {args.resolution} not in {banana_resolution}")
            image_size = args.resolution
            assert args.resolution in banana_resolution, f"resolution {args.resolution} not in {banana_resolution}"
            image_size = args.resolution
            content_part = image_to_part(str(cref_path))
            style_part = image_to_part(str(sref_path))
            text_part = types.Part.from_text(text=prompt)
            resp = client.models.generate_content(
                model=args.model_id,
                contents=[content_part, style_part, text_part],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=types.ImageConfig(
                        aspect_ratio=aspect_ratio,
                        image_size=image_size,
                    ),
                ),
            )
            if not save_first_image(resp, out_path):
                result_queue.put(("error", k, "no_image"))
                continue
            if meta_f is not None:
                record = {
                    "id": k,
                    "prompt": prompt,
                    "cref_path": str(cref_path),
                    "sref_path": str(sref_path),
                    "out_path": str(out_path),
                    "model_id": args.model_id,
                    "rank": rank,
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()
            result_queue.put(("done", k, None))
        except Exception as e:
            result_queue.put(("error", k, str(e)))
    if meta_f is not None:
        meta_f.close()


def parse_args():
    p = argparse.ArgumentParser(description="Gemini image edit batch (content + style)")
    p.add_argument("--prompts_json", required=True, help="Path to prompts.json (id->prompt).")
    p.add_argument("--cref_dir", required=True, help="Directory containing content reference images.")
    p.add_argument("--sref_dir", required=True, help="Directory containing style reference images.")
    p.add_argument("--out_dir", required=True, help="Output directory to save generated images.")
    p.add_argument("--model_id", default="gemini-3-pro-native")
    p.add_argument("--base_url", default="https://YOUR_OPENAI_COMPAT_ENDPOINT/gemini")
    p.add_argument("--api_version", default="v1alpha")
    p.add_argument("--num_procs", type=int, default=4)
    p.add_argument("--resolution", default="1K", help="Image resolution.")
    p.add_argument("--aspect_ratio", default="", help="Image aspect ratio.")
    p.add_argument("--num_generate", type=int, default=0, help="Generate first N items only. 0 = all.")
    p.add_argument("--limit", type=int, default=0, help="Run first N items only. 0 = all.")
    p.add_argument("--ids", type=str, default="", help='Comma-separated ids to run, e.g. "000015,000010".')
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save_jsonl", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts_json)
    if args.ids.strip():
        wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
        prompts = {k: prompts[k] for k in wanted if k in prompts}
    keys = list(prompts.keys())
    if args.num_generate and args.num_generate > 0:
        keys = keys[: args.num_generate]
    elif args.limit and args.limit > 0:
        keys = keys[: args.limit]
    tasks_all = [(k, prompts[k]) for k in keys]
    if not tasks_all:
        return
    num_procs = max(1, int(args.num_procs))
    if num_procs == 1:
        q = mp.Queue()
        _worker(0, tasks_all, args, q)
        return
    chunk_size = (len(tasks_all) + num_procs - 1) // num_procs
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = []
    for r in range(num_procs):
        sub = tasks_all[r * chunk_size : (r + 1) * chunk_size]
        if not sub:
            continue
        p = ctx.Process(target=_worker, args=(r, sub, args, q))
        p.start()
        procs.append(p)
    total = len(tasks_all)
    done = 0
    pbar = tqdm(total=total, unit="img")
    while done < total:
        try:
            status, _, _ = q.get(timeout=5)
            done += 1
            pbar.update(1)
        except Exception:
            if not any(p.is_alive() for p in procs) and q.empty():
                break
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"A worker process exited with code {p.exitcode}")
    pbar.close()


if __name__ == "__main__":
    main()
