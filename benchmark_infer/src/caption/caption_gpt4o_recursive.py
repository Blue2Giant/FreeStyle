#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursively scan a directory for images, generate 10 diverse prompts (bilingual) using GPT-4o,
and save them to a JSON file with the same basename as the image.

Usage:
python3 /data/benchmark_metrics/caption_gpt4o_recursive.py \
    --root /mnt/jfs/bench-bucket/sref_bench/bench_1106_content_prompt_new/ --workers 16 
python3 /data/benchmark_metrics/caption_pipe/caption_gpt4o_recursive.py \
    --root /mnt/jfs/bench-bucket/sref_bench/bench_0228_content_prompt/  --workers 16 --overwrite
"""

import argparse
import base64
import json
import mimetypes
import os
import time
import requests
from io import BytesIO
from multiprocessing import get_context
from PIL import Image

# ==== Configuration ====
BASE_URL = "https://YOUR_OPENAI_COMPAT_ENDPOINT/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = "gpt-4o"
TIMEOUT = 360

RESIZE_MAX_SIDE = 384
JPEG_QUALITY = 85

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

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
- **Ignore style**: Just consider the content in the picture. If there is a style, don't take it into account.
- **Avoid overly exaggerated imagination**: The captions written should be as reasonable as possible and depict scenarios that could exist in daily life.

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

# Global session for workers
SESSION = None
HEADERS = None

def init_session():
    global SESSION, HEADERS
    SESSION = requests.Session()
    HEADERS = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

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
        "max_tokens": 2000, # Increased slightly to ensure full JSON
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

def worker(task):
    image_path, overwrite = task
    # Determine output JSON path
    base, _ = os.path.splitext(image_path)
    json_path = f"{base}.json"
    
    # Check if exists
    if os.path.exists(json_path) and not overwrite:
        return image_path, None, True, 0, "Skipped (exists)"

    t0 = time.time()
    try:
        data_uri = to_data_uri_resized_local(image_path)
        payload = build_payload(data_uri)
        url = BASE_URL.rstrip("/") + "/chat/completions"
        
        resp = SESSION.post(url, headers=HEADERS, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        raw = data["choices"][0]["message"]["content"]
        
        # Parse JSON
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
            # Fallback
            prompts_en = [raw]
            prompts_cn = [raw]

        output_data = {
            "caption_en": prompts_en,
            "caption_zh": prompts_cn
        }

        # Save individual JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        dt = time.time() - t0
        return image_path, output_data, True, dt, ""

    except Exception as e:
        dt = time.time() - t0
        return image_path, None, False, dt, str(e)

def find_images(root_dir):
    image_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTS:
                image_files.append(os.path.join(root, file))
    return image_files

def main():
    parser = argparse.ArgumentParser(description="Recursively caption images with GPT-4o (bilingual, 10 variations).")
    parser.add_argument("--root", required=True, help="Root directory to scan for images.")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing JSON files.")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"Error: Root directory '{args.root}' not found.")
        return

    print(f"Scanning {args.root} for images...")
    image_files = find_images(args.root)
    print(f"Found {len(image_files)} images.")

    if not image_files:
        return

    ctx = get_context("spawn")
    
    # Check for existing work to report progress accurately
    todo_files = []
    skipped_count = 0
    for img_path in image_files:
        base, _ = os.path.splitext(img_path)
        json_path = f"{base}.json"
        if os.path.exists(json_path) and not args.overwrite:
            skipped_count += 1
        else:
            todo_files.append((img_path, args.overwrite))

    print(f"Skipping {skipped_count} already processed images.")
    print(f"Processing {len(todo_files)} images with {args.workers} workers...")

    if not todo_files:
        print("All done.")
        return

    with ctx.Pool(processes=max(1, args.workers), initializer=init_session) as pool:
        for img_path, _, ok, dt, err in pool.imap_unordered(worker, todo_files):
            name = os.path.basename(img_path)
            if ok:
                if err == "Skipped (exists)":
                     # Should be handled by pre-filtering, but just in case
                     pass
                else:
                    print(f"[OK] {name} {dt:.2f}s")
            else:
                print(f"[ERR] {name} {dt:.2f}s {err}")

if __name__ == "__main__":
    main()
