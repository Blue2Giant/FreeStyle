#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
import base64
import mimetypes
import multiprocessing as mp
from typing import Optional, Dict, Any, Tuple, List

from tqdm import tqdm
from openai import OpenAI
from megfile.smart import (
    smart_open as mopen,
    smart_exists,
    smart_listdir,
    smart_makedirs,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

CONTENT_SIMILARITY_PROMPT = r"""
Rate from 0 to 10:
Evaluate how consistent Image B is with Image A in terms of SUBJECT CONTENT / THEME, regardless of style.

Important Notes:
* Scoring should be strict — avoid giving high scores unless the subject is clearly and accurately consistent.
* Ignore style differences: rendering style, brushwork, lighting mood, color grading, resolution, noise, aesthetics.
* Focus on content: who/what is present, key attributes, layout, scene category, actions, and meaningful text/logos.
* If identity/subject cannot be verified due to blur/occlusion, keep the score conservative (prefer lower).

Detailed aspects to consider:
1) Human identity & attributes:
   - Facial identity (shape, features), hairstyle/color/length, body shape/build.
   - Clothing categories, colors, patterns, logos, accessories (glasses, hats, jewelry).
2) Objects & attributes:
   - Object categories, colors, materials, sizes, and COUNT for prominent items.
   - Presence/absence of key props; text/logo content where legible.
3) Spatial layout & composition:
   - Relative positions of major elements; subject scale and viewpoint within reasonable tolerance.
4) Background / scene category:
   - Indoor/outdoor; place type; major structures/furniture/landmarks.
5) Actions / poses / interactions:
   - Human/animal/body poses; gesture semantics.
6) Text / logos / symbols:
   - Words/numbers/symbols important to scene meaning.

Score rubric (0–10):
* 0: Completely unrelated.
* 1–3: Mostly inconsistent; minimal overlap.
* 4–6: Partially consistent with significant mismatches.
* 7–9: Mostly consistent with minor issues.
* 10: Fully consistent; all major content aligns.

Output rules (very important):
* Output ONLY one line in the format: score@reason
* score must be an integer from 0 to 10.
* reason must be 1-2 short sentences, specific and observable.
""".strip()


def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with mopen(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return s
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip()
    return s


def _clamp_score_0_10(score: int) -> int:
    if score < 0:
        return 0
    if score > 10:
        return 10
    return score


def parse_score_reason(raw_text: str) -> Dict[str, Any]:
    clean = strip_code_fences(raw_text).strip()
    if not clean:
        raise ValueError("模型输出为空")
    first_line = clean.splitlines()[0].strip()
    m = re.match(r"^\s*(\d{1,2})\s*@\s*(.+?)\s*$", first_line)
    if not m:
        raise ValueError(f"输出不符合 score@reason: {first_line!r}")
    score = _clamp_score_0_10(int(m.group(1)))
    reason = m.group(2).strip()
    if not reason:
        raise ValueError("reason 为空")
    return {"score": score, "reason": reason}


def build_messages(img_a: str, img_b: str):
    content = []
    content.append({"type": "text", "text": "Image A:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_a)}})
    content.append({"type": "text", "text": "Image B:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_b)}})
    content.append({"type": "text", "text": CONTENT_SIMILARITY_PROMPT})
    return [{"role": "user", "content": content}]


def run_content_score(
    client: OpenAI,
    model: str,
    img_a: str,
    img_b: str,
    max_tokens: int = 128,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    messages = build_messages(img_a, img_b)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parsed = parse_score_reason(raw)
    except Exception as e:
        return raw, None, {"error": str(e)}
    return raw, parsed, None


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def _worker_process(model: str, base_url: str, api_key: str, timeout: int, max_tokens: int, tasks: List[Tuple[str, str, str]], result_queue: mp.Queue):
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    for base, content_path, output_path in tasks:
        try:
            _, parsed, err = run_content_score(
                client=client,
                model=model,
                img_a=content_path,
                img_b=output_path,
                max_tokens=max_tokens,
            )
            if parsed is None:
                result_queue.put((base, None, None))
            else:
                result_queue.put((base, parsed["score"], parsed["reason"]))
        except Exception:
            result_queue.put((base, None, None))


def smart_write_json(path: str, obj: Any):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    dir_path = os.path.dirname(path) or "."
    if path.startswith(("s3://", "oss://")):
        smart_makedirs(dir_path, exist_ok=True)
        with mopen(path, "wb") as f:
            f.write(data)
    else:
        os.makedirs(dir_path, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description="内容相似度：双目录批量评分")
    parser.add_argument("--content_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_reason_json", required=True)
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--model", default="Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--num_procs", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    content_files = set(smart_listdir(args.content_dir))
    output_files = set(smart_listdir(args.output_dir))
    common_files = list(content_files & output_files)
    common_files = [f for f in common_files if is_image_name(f)]

    if args.num_samples > 0 and len(common_files) > args.num_samples:
        import random
        random.seed(args.seed)
        common_files = random.sample(common_files, args.num_samples)

    results = {}
    reason_results = {}
    if (not args.overwrite) and smart_exists(args.out_json) and smart_exists(args.out_reason_json):
        try:
            existing = None
            existing_reason = None
            if args.out_json.startswith(("s3://", "oss://")):
                with mopen(args.out_json, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            else:
                with open(args.out_json, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            if args.out_reason_json.startswith(("s3://", "oss://")):
                with mopen(args.out_reason_json, "r", encoding="utf-8") as f:
                    existing_reason = json.load(f)
            else:
                with open(args.out_reason_json, "r", encoding="utf-8") as f:
                    existing_reason = json.load(f)
            if isinstance(existing, dict) and isinstance(existing_reason, dict):
                results = existing
                reason_results = existing_reason
                processed_keys = set(results.keys()) & set(reason_results.keys())
                common_files = [f for f in common_files if os.path.splitext(f)[0] not in processed_keys]
        except Exception:
            pass

    if not common_files:
        smart_write_json(args.out_json, results)
        smart_write_json(args.out_reason_json, reason_results)
        return

    tasks = []
    for name in common_files:
        base = os.path.splitext(name)[0]
        content_path = args.content_dir.rstrip("/") + "/" + name
        output_path = args.output_dir.rstrip("/") + "/" + name
        tasks.append((base, content_path, output_path))

    num_procs = max(1, int(args.num_procs))
    chunk_size = (len(tasks) + num_procs - 1) // num_procs
    result_queue = mp.Queue()
    workers = []

    for i in range(num_procs):
        sub_tasks = tasks[i * chunk_size : (i + 1) * chunk_size]
        if not sub_tasks:
            continue
        p = mp.Process(
            target=_worker_process,
            args=(
                args.model,
                args.base_url,
                args.api_key,
                args.timeout,
                args.max_tokens,
                sub_tasks,
                result_queue,
            ),
        )
        p.start()
        workers.append(p)

    total_done = 0
    total_tasks = len(tasks)
    with tqdm(total=total_tasks, desc="Processing") as pbar:
        while total_done < total_tasks:
            try:
                base, score, reason = result_queue.get(timeout=5)
                results[base] = score
                reason_results[base] = reason
                total_done += 1
                pbar.update(1)
                if total_done % 50 == 0:
                    smart_write_json(args.out_json, results)
                    smart_write_json(args.out_reason_json, reason_results)
            except Exception:
                if not any(p.is_alive() for p in workers) and result_queue.empty():
                    break

    for p in workers:
        p.join()

    smart_write_json(args.out_json, results)
    smart_write_json(args.out_reason_json, reason_results)


if __name__ == "__main__":
    main()
