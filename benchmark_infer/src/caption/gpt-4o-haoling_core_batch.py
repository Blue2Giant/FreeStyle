#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/benchmark_metrics/caption_pipe/gpt-4o-haoling_core_batch.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json \
  --cref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref \
  --sref_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/gpt4o-edit \
  --model gpt-4o-all \
  --base_url https://YOUR_OPENAI_COMPAT_ENDPOINT/v1 \
  --api_key YOUR_KEY \
  --num_procs 8
"""
import argparse
import base64
import json
import os
import re
import multiprocessing as mp
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import requests
from PIL import Image
from tqdm import tqdm


def image_to_data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def extract_image_urls(text: str) -> List[str]:
    urls = []
    urls.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
    urls.extend(re.findall(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", text))
    urls.extend(re.findall(r"https?://\S+\.(?:png|jpg|jpeg|webp)", text))
    seen = set()
    result = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        result.append(u)
    return result


def download_image(url: str) -> Image.Image:
    if url.startswith("data:image"):
        data = base64.b64decode(url.split(",", 1)[1])
        return Image.open(BytesIO(data)).convert("RGB")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def load_prompts(prompts_json: str) -> Dict[str, str]:
    with open(prompts_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("prompts_json must be a dict")
    return {str(k): str(v) for k, v in data.items()}


def build_payload(model: str, content_url: str, style_url: str, prompt: str) -> Dict[str, object]:
    guidance = (
        "Use the FIRST image as content reference and the SECOND image as style reference. "
        "Preserve the subject and layout from the FIRST image while applying the style, palette, "
        "texture, and lighting from the SECOND image."
    )
    return {
        "model": model,
        "stream": False,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{guidance}\n{prompt}"},
                    {"type": "image_url", "image_url": {"url": content_url}},
                    {"type": "image_url", "image_url": {"url": style_url}},
                ],
            }
        ],
    }


def _worker(rank: int, tasks: List[Tuple[str, str]], args, result_queue: mp.Queue):
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)
    meta_f = None
    if args.save_jsonl:
        meta_f = open(out_dir / f"metadata.rank{rank}.jsonl", "a", encoding="utf-8")
    missing = 0
    for k, prompt in tasks:
        try:
            out_path = out_dir / f"{k}.png"
            if (not args.overwrite) and out_path.exists():
                result_queue.put(("skip", k, None))
                continue
            cref_path = cref_dir / f"{k}.png"
            sref_path = sref_dir / f"{k}.png"
            if not cref_path.exists() or not sref_path.exists():
                missing += 1
                result_queue.put(("missing", k, None))
                continue
            content_url = image_to_data_url(str(cref_path))
            style_url = image_to_data_url(str(sref_path))
            payload = build_payload(args.model, content_url, style_url, prompt)
            resp = requests.post(
                f"{args.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=args.timeout,
            )
            resp.raise_for_status()
            try:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[{k}] response parse failed: {e}")
                result_queue.put(("error", k, f"parse_failed:{e}"))
                continue
            urls = extract_image_urls(content)
            if not urls:
                print(f"[{k}] no image url in response")
                result_queue.put(("error", k, "no_image"))
                continue
            image = download_image(urls[0])
            image.save(out_path)
            if meta_f is not None:
                record = {
                    "id": k,
                    "prompt": prompt,
                    "cref_path": str(cref_path),
                    "sref_path": str(sref_path),
                    "out_path": str(out_path),
                    "model": args.model,
                    "rank": rank,
                }
                meta_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_f.flush()
            result_queue.put(("done", k, None))
        except Exception as e:
            print(f"[{k}] request failed: {e}")
            result_queue.put(("error", k, str(e)))
    if meta_f is not None:
        meta_f.close()


def parse_args():
    p = argparse.ArgumentParser(description="GPT-4o core batch image generation (content + style)")
    p.add_argument("--prompts_json", required=True, help="Path to prompts.json (id->prompt).")
    p.add_argument("--cref_dir", required=True, help="Directory containing content reference images.")
    p.add_argument("--sref_dir", required=True, help="Directory containing style reference images.")
    p.add_argument("--out_dir", required=True, help="Output directory to save generated images.")
    p.add_argument("--model", default="gpt-4o-all")
    p.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://YOUR_OPENAI_COMPAT_ENDPOINT/v1"))
    p.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--num_procs", type=int, default=4)
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
    if args.limit and args.limit > 0:
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
