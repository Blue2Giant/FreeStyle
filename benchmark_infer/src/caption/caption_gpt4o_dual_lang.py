#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 /data/benchmark_metrics/caption_gpt4o_dual_lang.py \
    --root /mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_new \
    --workers 4 \
    --limit 10 \
    --overwrite \
    --out prompts_dual.json
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import time
from multiprocessing import get_context
from io import BytesIO
from PIL import Image
import requests

# ==== 使用 caption_gpt4o_demo.py 的模型配置 ====
BASE_URL = "https://YOUR_OPENAI_COMPAT_ENDPOINT/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
MODEL = "gpt-4o"
TIMEOUT = 360

# ==== 图片处理（与 caption_gpt4o_demo 同风格） ====
RESIZE_MAX_SIDE = 384
JPEG_QUALITY = 85

SYSTEM_PROMPT = """
You are a professional image-synthesis prompt generator.
Your task is to analyze the reference image and create one JSON object containing **exactly 10 distinct and diverse** image-generation scenarios based on the image content.
For each scenario, provide a detailed caption in both Chinese and English.

If it is a scene, you can add some characters or objects to interact within the scene, or change the perspective of the scene.
If there are multiple subjects in the picture, you can have them interact with each other or just focus on one of them for your imagination.
If there is a clear subject in the picture, then take this subject as the theme and imagine various actions, behaviors, positions, and environments in which it is located.

Important rules:
- **Diversity is key**: The 10 scenarios must be significantly different from each other (e.g., different actions, environments, lighting, styles, or perspectives).
- The synthesis must not look like pasted or cut-out elements. The result must appear as a newly rendered, seamless picture.
- The subject from the reference image must interact naturally with the new environment or objects.
- The final image must associate with the reference image but offer a fresh perspective or story.

Output format:
Return only valid JSON, no extra explanation. The JSON must have this structure:
{
  "scenarios": [
    {
      "id": 1,
      "CN": "Chinese caption for scenario 1",
      "EN": "English caption for scenario 1"
    },
    ...
    {
      "id": 10,
      "CN": "Chinese caption for scenario 10",
      "EN": "English caption for scenario 10"
    }
  ]
}
You MUST give me exactly 10 diverse scenarios!
"""

USER_INSTRUCTION = """
You will be provided with a reference image.
Task:
Analyze the image and return exactly ONE JSON object with the key "scenarios". 
Its value must be an array of exactly 10 objects, where each object represents a unique imagination/scenario.
Each object must contain:
1. "id": integer ID (1-10)
2. "CN": Chinese instruction
3. "EN": English instruction

Ensure the 10 scenarios are highly diverse and creative.
"""

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SESSION = None
HEADERS = None


def _guess_mime(path: str) -> str:
    m, _ = mimetypes.guess_type(path)
    return m or "image/png"


def _load_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _resize_keep_long_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    side = max(w, h)
    if side <= max_side:
        return img
    scale = max_side / float(side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _img_to_data_uri_jpeg(img: Image.Image, quality: int = JPEG_QUALITY) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def to_data_uri_resized_local(path: str) -> str:
    img = _load_image(path)
    img = _resize_keep_long_side(img, RESIZE_MAX_SIDE)
    return _img_to_data_uri_jpeg(img, JPEG_QUALITY)


def build_payload(data_uri: str) -> dict:
    return {
        "model": MODEL,
        "stream": False,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_INSTRUCTION},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
        "response_format": {"type": "json_object"}
    }


def list_images(folder: str) -> dict:
    items = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        base, ext = os.path.splitext(name)
        if ext.lower() not in IMAGE_EXTS:
            continue
        items[base] = path
    return items


def load_existing(out_path: str) -> dict:
    if not os.path.isfile(out_path):
        return {}
    try:
        with open(out_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def sort_key(key):
    # Try to extract number from basename for sorting
    # e.g. "123" -> 123, "img_123" -> 123
    # If no number found, use string sorting
    nums = re.findall(r'\d+', key)
    if nums:
        return int(nums[0])
    return key

def save_json(out_path: str, data: dict) -> None:
    # Sort data by key (using numeric sort if possible) before saving
    sorted_items = sorted(data.items(), key=lambda x: sort_key(x[0]))
    sorted_data = dict(sorted_items)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)


def init_session():
    global SESSION, HEADERS
    SESSION = requests.Session()
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def worker(task):
    base, cref_path = task
    t0 = time.time()
    try:
        data_uri = to_data_uri_resized_local(cref_path)
        payload = build_payload(data_uri)
        url = BASE_URL.rstrip("/") + "/chat/completions"
        resp = SESSION.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        
        # Parse JSON
        try:
            # Clean markdown code blocks if present
            raw_clean = raw
            if "```json" in raw:
                raw_clean = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw_clean = raw.split("```")[1].split("```")[0].strip()

            res = json.loads(raw_clean)
            scenarios = res.get("scenarios", [])
            
            prompts_cn = []
            prompts_en = []
            
            for item in scenarios:
                prompts_cn.append(item.get("CN", ""))
                prompts_en.append(item.get("EN", ""))
            
            if not prompts_cn and not prompts_en:
                # Fallback if structure is unexpected
                prompts_en = [raw]
                prompts_cn = [raw]

        except Exception as e:
            # Fallback
            prompts_en = [raw]
            prompts_cn = [raw]
            
        dt = time.time() - t0
        return base, (prompts_en, prompts_cn), True, dt, ""
    except Exception as e:
        dt = time.time() - t0
        return base, ([], []), False, dt, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--cref", default="cref")
    parser.add_argument("--sref", default="sref")
    parser.add_argument("--out", default="prompts_dual.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--exists_action", choices=["skip", "overwrite", "abort"], default="skip")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    root = args.root
    cref_dir = os.path.join(root, args.cref)
    sref_dir = os.path.join(root, args.sref)
    
    # Define output paths
    out_base, out_ext = os.path.splitext(args.out)
    out_path_en = os.path.join(root, f"{out_base}_en{out_ext}")
    out_path_zh = os.path.join(root, f"{out_base}_zh{out_ext}")

    if not os.path.isdir(cref_dir):
        raise FileNotFoundError(f"cref dir not found: {cref_dir}")
    # sref_dir check is kept for compatibility but not strictly used for logic pairing anymore 
    # unless we want to filter common basenames.
    if not os.path.isdir(sref_dir):
        raise FileNotFoundError(f"sref dir not found: {sref_dir}")

    cref_map = list_images(cref_dir)
    sref_map = list_images(sref_dir)
    common = sorted(set(cref_map.keys()) & set(sref_map.keys()), key=lambda x: sort_key(x))
    
    if args.limit and args.limit > 0:
        common = common[: args.limit]
    if not common:
        raise ValueError("No matching basenames found between cref and sref.")

    if args.dry_run:
        print(f"Matched {len(common)} image pairs.")
        return

    existing_en = load_existing(out_path_en)
    existing_zh = load_existing(out_path_zh)
    exists_action = "overwrite" if args.overwrite else args.exists_action
    
    if existing_en or existing_zh:
        print(f"[INFO] Found existing prompts: EN={len(existing_en)}, ZH={len(existing_zh)}")
        if exists_action == "abort":
            print("[INFO] Output already has content, abort due to exists_action=abort")
            return
        if exists_action == "skip":
            common = [b for b in common if b not in existing_en or b not in existing_zh]
    
    if not common:
        print("No new items to process.")
        return

    # No idx needed since we don't use STYLE_SENTENCES
    tasks = [(base, cref_map[base]) for base in common]
    ctx = get_context("spawn")
    errors = {}

    with ctx.Pool(processes=max(1, args.workers), initializer=init_session) as pool:
        for base, prompts, ok, dt, err in pool.imap_unordered(worker, tasks):
            if ok:
                prompt_en, prompt_zh = prompts
                existing_en[base] = prompt_en
                existing_zh[base] = prompt_zh
                save_json(out_path_en, existing_en)
                save_json(out_path_zh, existing_zh)
                print(f"[OK] {base} {dt:.2f}s")
            else:
                errors[base] = err
                print(f"[ERR] {base} {dt:.2f}s {err}")

    print(f"Saved {len(existing_en)} EN prompts to {out_path_en}")
    print(f"Saved {len(existing_zh)} ZH prompts to {out_path_zh}")
    if errors:
        print(f"Failed {len(errors)} items")


if __name__ == "__main__":
    main()
