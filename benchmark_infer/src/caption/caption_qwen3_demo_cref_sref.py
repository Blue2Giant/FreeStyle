#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 /data/benchmark_metrics/caption_qwen3_demo_cref_sref.py   --root /data/benchmark_metrics/sample_1500_bench_cref_sref   --workers 64 --overwrite
"""
import argparse
import base64
import json
import mimetypes
import os
import re
import time
from multiprocessing import get_context
from openai import OpenAI
BASE_URL = "http://YOUR_QWEN_VLM_ENDPOINT/Qwen3-VL-235B-A22B-W8A8/v1"
API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL_NAME = "qwen3vlw8a8"

BASE_URL = "http://stepcast-router.shai-core:9200/v1"
MODEL_NAME = "v1p3"
TIMEOUT = 600

STYLE_TERMS = [
    r"二次元", r"动漫", r"漫画风", r"漫画", r"插画", r"插图", r"水彩", r"油画", r"丙烯", r"素描", r"速写",
    r"线稿", r"线描", r"国画", r"泼墨", r"像素风", r"赛博朋克", r"蒸汽波", r"低多边形", r"渲染",
    r"CG", r"3D", r"3D渲染", r"次世代", r"卡通", r"写实风", r"电影感", r"胶片感", r"赛璐璐", r"上色",
    r"赛璐璐上色", r"喷枪", r"版画", r"水墨", r"扁平风", r"抽象风", r"风格化", r"风格", r"画风",
    r"anime", r"manga", r"cartoon", r"illustration", r"watercolor", r"oil\s*paint(?:ing)?",
    r"sketch", r"line\s*art", r"ink", r"charcoal", r"pastel", r"pixel\s*art",
    r"low\s*poly", r"render", r"3d", r"cgi", r"stylized", r"digital\s*art",
    r"concept\s*art", r"matte\s*painting", r"cinematic", r"film\s*grain", r"bokeh",
]

# SYSTEM_PROMPT = (
#     "You are an image prompt writer. Output must be in ENGLISH ONLY.\n"
#     "Goal: propose a concise, actionable generation prompt for a NEW image inspired by the provided content reference.\n"
#     "Rules:\n"
#     "1) Focus on main subjects, salient attributes, actions, scene/background, composition, and spatial cues.\n"
#     "2) Use an imperative or descriptive text-to-image prompt style suitable for image generation.\n"
#     "3) Do NOT mention art style, medium, rendering, filters, or camera terms (e.g., lens, shot, bokeh).\n"
#     "4) Do NOT mention 'in the image/photo/picture' or refer to a reference; just describe the target image to generate.\n"
#     "5) One sentence, concise and specific; no second person; no speculation beyond visible content.\n"
#     "Respond in clear, concise English only."
# )

# USER_INSTRUCTION = (
#     "Write a single-sentence English prompt to generate a new image inspired by the provided content reference. "
#     "Describe the main subject(s), key attributes and actions, the scene/background, and useful composition cues. "
#     "Avoid style/media/camera terms and do not mention a reference."
# )
"""
根据画面内容想象一段文字，描述新的场景。这个场景里要有一个主体和内容图里的主体相关。请你用英文描述出来。
"""
SYSTEM_PROMPT = (
    "You are a CREATIVE text-to-image prompt writer. Output must be ENGLISH ONLY.\n"
    "Goal: Based on the content of the picture, imagine a piece of text to describe a new scene. In this scene, there should be a subject related to the subject in the content picture. Please describe it in English.\n"
    "Hard rules:\n"
    "1) Do NOT describe the exact visible details from the reference; do NOT copy entities verbatim; avoid 'same/exact/identical/replicate'.\n"
    "2) Do NOT mention the reference, or 'in the image/photo/picture'.\n"
    "3) Do NOT mention art style, medium, rendering, filters, or camera terms (lens, shot, bokeh, cinematic, film grain, etc.).\n"
    "4) Output exactly ONE sentence, no lists, no extra text."
)

"""
描述的画面内容应该和内容图的差异大一点，不要和内容图完全一样。要更多样并且更加丰富。
"""
USER_INSTRUCTION = (
    "Write a single-sentence English text-to-image prompt for a new scene reimagined from the content reference. "
    "The described scene should have more differences from the content picture, rather than being exactly the same as it. It should be more diverse and richer."
    "avoid style/media/camera terms."
)
# USER_INSTRUCTION = (
#     "Write a single-sentence English text-to-image prompt for a new scene reimagined from the content reference. "
#     "Keep 1-2 abstract anchors (subject type, mood, or composition), but change at least 5 concrete aspects "
#     "(location/time/weather/action/props/background elements). "
#     "Do not mention the reference and do not copy exact details; avoid style/media/camera terms."
# )

STYLE_SENTENCES = [
    "Transfer the style into the style reference picture.",
    "Transfer the style into the second picture.",
    "Transfer the style into the style reference image.",
    "Transfer the style to the style reference picture.",
    "Transfer the style to the second picture.",
    "Apply the style to the style reference picture.",
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

CLIENT = None

def init_client():
    global CLIENT
    CLIENT = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=TIMEOUT)

def path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def sanitize_caption(text: str) -> str:
    if not text:
        return ""
    clean = text.strip()
    clean = re.sub(r"^[\"'`‘’“”\s]+|[\"'`‘’“”\s]+$", "", clean)
    for term in STYLE_TERMS:
        pattern = rf"(?:{term})(?:\s*(?:风格|风))?"
        clean = re.sub(pattern, "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean)
    clean = re.sub(r"\s*([,.;:!?])\s*", r"\1 ", clean)
    clean = re.sub(r"\s+$", "", clean)
    if len(clean) > 240:
        clean = clean[:240].rstrip(",.;:!? ")
    return clean or text.strip()

def build_messages(image_path: str) -> list:
    data_url = path_to_data_url(image_path)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": USER_INSTRUCTION},
            ],
        },
    ]

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

def build_prompt(caption: str, idx: int) -> str:
    sentence = STYLE_SENTENCES[idx % len(STYLE_SENTENCES)]
    if caption:
        return f"{caption} {sentence}"
    return f"{sentence}"

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

def save_json(out_path: str, data: dict) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def worker(task):
    base, cref_path, idx = task
    t0 = time.time()
    try:
        messages = build_messages(cref_path)
        resp = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=32768,
            temperature=0.4,
        )
        raw = resp.choices[0].message.content if resp and resp.choices else ""
        clean = sanitize_caption(raw)
        prompt = build_prompt(clean, idx)
        dt = time.time() - t0
        return base, prompt, True, dt, ""
    except Exception as e:
        dt = time.time() - t0
        return base, "", False, dt, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/mnt/jfs/bench-bucket/sref_bench/sample_1500_bench_cref_sref")
    parser.add_argument("--cref", default="cref")
    parser.add_argument("--sref", default="sref")
    parser.add_argument("--out", default="prompts.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.root
    cref_dir = os.path.join(root, args.cref)
    sref_dir = os.path.join(root, args.sref)
    out_path = os.path.join(root, args.out)

    if not os.path.isdir(cref_dir):
        raise FileNotFoundError(f"cref dir not found: {cref_dir}")
    if not os.path.isdir(sref_dir):
        raise FileNotFoundError(f"sref dir not found: {sref_dir}")

    cref_map = list_images(cref_dir)
    sref_map = list_images(sref_dir)
    common = sorted(set(cref_map.keys()) & set(sref_map.keys()))
    if args.limit and args.limit > 0:
        common = common[: args.limit]

    if not common:
        raise ValueError("No matching basenames found between cref and sref.")

    if args.dry_run:
        print(f"Matched {len(common)} image pairs.")
        return

    existing = load_existing(out_path)
    if existing and not args.overwrite:
        common = [base for base in common if base not in existing]

    if not common:
        print("No new items to process.")
        return

    tasks = [(base, cref_map[base], idx) for idx, base in enumerate(common)]
    ctx = get_context("spawn")
    errors = {}

    with ctx.Pool(processes=max(1, args.workers), initializer=init_client) as pool:
        for base, prompt, ok, dt, err in pool.imap_unordered(worker, tasks):
            if ok:
                existing[base] = prompt
                save_json(out_path, existing)
                print(f"[OK] {base} {dt:.2f}s")
            else:
                errors[base] = err
                print(f"[ERR] {base} {dt:.2f}s {err}")

    print(f"Saved {len(existing)} prompts to {out_path}")
    if errors:
        print(f"Failed {len(errors)} items")

if __name__ == "__main__":
    main()
