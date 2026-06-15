#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from codecs import raw_unicode_escape_decode
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

INSTRUCTION_FOLLOW_PROMPT = r"""
You will be given:
1) An image (the final edited result).
2) An editing instruction (prompt).

Task:
Score how well the final image follows the instruction, from 0 to 10.
Give PARTIAL CREDIT whenever the image implements some parts correctly.
Do NOT assume anything that is not clearly visible in the image.

Important evaluation rule:
- Ignore style/aesthetic differences (rendering style, brushwork, resolution, noise, overall beauty)
  ONLY IF the instruction does NOT explicitly ask for them.
- If the instruction explicitly mentions color, lighting, mood, style, lens, texture, etc.,
  then you MUST evaluate those aspects (do NOT ignore them).

How to score (internal steps; do not output these steps):
1) Break the instruction into atomic, checkable requirements:
   - main subject(s) / objects
   - key attributes (color, material, count, size, identity if specified)
   - actions / interactions
   - spatial/layout constraints (left/right, background, position)
   - any required text/logos (exact wording matters)
   Mark each requirement as MUST-HAVE or NICE-TO-HAVE.
2) For each requirement, judge: satisfied / partially satisfied / not satisfied / contradicted.
3) Combine into a final score using this guidance:
   - Start from 10 and subtract penalties:
     * Missing a MUST-HAVE requirement: -2 to -4 (bigger miss = bigger penalty)
     * Partially meeting a MUST-HAVE: -1 to -2
     * Minor detail wrong (NICE-TO-HAVE): -0.5 to -1
     * Direct contradiction (e.g., asked to add X but image removes X): -3 to -5
     * Wrong or missing required text/logo: -3 to -6
   Clamp the result to [0, 10].

Score anchors (use these to avoid extreme scores):
- 9–10: All MUST-HAVEs satisfied; at most tiny minor issues. Use 10 ONLY if nothing important is missing or wrong.
- 7–8.5: Most MUST-HAVEs satisfied; 1–2 notable misses or several small issues.
- 4–6.5: Some key parts are implemented, but major omissions/misapplied edits exist.
- 1–3.5: Slight relation, but most MUST-HAVEs are missing/wrong.
- 0–0.5: Completely unrelated or clearly opposite to the instruction. Use 0 ONLY when it is truly a total failure.

Output rules (very important):
* Output ONLY one line in the format: score@reason
* score must be a number from 0 to 10, allowing .5 steps (e.g., 7.5). Do NOT always use 0 or 10.
* reason must be 1-2 short sentences, specific and directly observable (mention 1-2 key satisfied/missed requirements).
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
    if "@" not in first_line:
        raise ValueError(f"输出不符合 score@reason: {first_line!r}")
    score_text, reason = first_line.split("@", 1)
    score_text = score_text.strip()
    reason = reason.strip()
    try:
        score = float(score_text)
    except Exception as e:
        raise ValueError(f"score 不是数字: {score_text!r}") from e
    if score < 0.0:
        score = 0.0
    if score > 10.0:
        score = 10.0
    if not reason:
        raise ValueError("reason 为空")
    return {"score": score, "reason": reason}


def build_messages(img_path: str, instruction: str):
    content = []
    content.append({"type": "text", "text": "Final image:"})
    content.append({"type": "image_url", "image_url": {"url": path_to_data_url(img_path)}})
    content.append({"type": "text", "text": f"Editing instruction:\n{instruction}"})
    content.append({"type": "text", "text": INSTRUCTION_FOLLOW_PROMPT})
    return [{"role": "user", "content": content}]


def select_instruction_text(text: str, mode: str) -> str:
    mode = (mode or "full").lower()
    s = text or ""
    if mode == "first_sentence":
        parts = s.split(".", 1)
        return parts[0].strip()
    if mode == "full":
        return s
    raise ValueError(f"Unsupported instruction_text_mode: {mode}")


def run_follow_score(
    client: OpenAI,
    model: str,
    img_path: str,
    instruction: str,
    max_tokens: int = 128,
) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    messages = build_messages(img_path, instruction)
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
        print(f"解析模型输出失败: {raw!r},path={img_path}")
        return raw, None, {"error": str(e)}
    return raw, parsed, None


def is_image_name(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def _worker_process(
    model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    max_tokens: int,
    tasks: List[Tuple[str, str, str]],
    result_queue: mp.Queue,
):
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    for base, image_path, instruction in tasks:
        try:
            _, parsed, err = run_follow_score(
                client=client,
                model=model,
                img_path=image_path,
                instruction=instruction,
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
    parser = argparse.ArgumentParser(description="指令遵循度评测：json + 图片目录")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--prompt_json", required=True, help="key=basename, value=instruction")
    parser.add_argument("--out_score_json", required=True)
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
    parser.add_argument("--instruction_text_mode", choices=["full", "first_sentence"], default="full")
    args = parser.parse_args()

    if not smart_exists(args.prompt_json):
        raise FileNotFoundError(f"prompt_json not found: {args.prompt_json}")
    if args.prompt_json.startswith(("s3://", "oss://")):
        with mopen(args.prompt_json, "r", encoding="utf-8") as f:
            prompt_map = json.load(f)
    else:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            prompt_map = json.load(f)
    if not isinstance(prompt_map, dict):
        raise ValueError("prompt_json must be a dict")

    image_files = set(smart_listdir(args.image_dir))
    image_files = [f for f in image_files if is_image_name(f)]
    image_files = list(image_files)

    tasks = []
    for name in image_files:
        base = os.path.splitext(name)[0]
        instruction = prompt_map.get(base)
        if not isinstance(instruction, str) or not instruction.strip():
            continue
        instruction_text = select_instruction_text(instruction.strip(), args.instruction_text_mode)
        if not instruction_text:
            continue
        image_path = args.image_dir.rstrip("/") + "/" + name
        tasks.append((base, image_path, instruction_text))

    if args.num_samples > 0 and len(tasks) > args.num_samples:
        import random
        random.seed(args.seed)
        tasks = random.sample(tasks, args.num_samples)

    score_results = {}
    reason_results = {}
    if (not args.overwrite) and smart_exists(args.out_score_json) and smart_exists(args.out_reason_json):
        try:
            if args.out_score_json.startswith(("s3://", "oss://")):
                with mopen(args.out_score_json, "r", encoding="utf-8") as f:
                    score_results = json.load(f)
            else:
                with open(args.out_score_json, "r", encoding="utf-8") as f:
                    score_results = json.load(f)
            if args.out_reason_json.startswith(("s3://", "oss://")):
                with mopen(args.out_reason_json, "r", encoding="utf-8") as f:
                    reason_results = json.load(f)
            else:
                with open(args.out_reason_json, "r", encoding="utf-8") as f:
                    reason_results = json.load(f)
            if isinstance(score_results, dict) and isinstance(reason_results, dict):
                processed_keys = set(score_results.keys()) & set(reason_results.keys())
                tasks = [t for t in tasks if t[0] not in processed_keys]
        except Exception:
            score_results = {}
            reason_results = {}

    if not tasks:
        smart_write_json(args.out_score_json, score_results)
        smart_write_json(args.out_reason_json, reason_results)
        return

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
                score_results[base] = score
                reason_results[base] = reason
                total_done += 1
                pbar.update(1)
                if total_done % 50 == 0:
                    smart_write_json(args.out_score_json, score_results)
                    smart_write_json(args.out_reason_json, reason_results)
            except Exception:
                if not any(p.is_alive() for p in workers) and result_queue.empty():
                    break

    for p in workers:
        p.join()

    smart_write_json(args.out_score_json, score_results)
    smart_write_json(args.out_reason_json, reason_results)


if __name__ == "__main__":
    main()
