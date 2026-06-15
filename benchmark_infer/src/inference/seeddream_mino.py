#!/usr/bin/env python3
"""
https://www.volcengine.com/docs/82379/1541523?lang=zh 
other model reference to above
"""
import argparse
import base64
import json
import os
from io import BytesIO

import requests
from PIL import Image


ASPECT_RATIO_1K=[
    "1024x1024",
    "864x1152",
    "1152x864",
    "1280x720",
    "720x1280",
    "832x1248",
    "1248x832",
    "1512x648"
]


def image_to_base64(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_image_bytes(path: str, fmt: str) -> tuple[bytes, str]:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    if fmt == "jpeg":
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    if fmt == "png":
        img.save(buf, format="PNG")
        return buf.getvalue(), "image/png"
    raise ValueError(f"unsupported image fmt: {fmt}")


def _build_image_value(path: str, fmt: str, mode: str):
    raw, mime = _encode_image_bytes(path, fmt)
    b64 = base64.b64encode(raw).decode("utf-8")
    if mode == "b64":
        return b64
    if mode == "data_url":
        return f"data:{mime};base64,{b64}"
    if mode == "obj_b64_json":
        return {"b64_json": b64}
    if mode == "obj_image":
        return {"image": b64}
    if mode == "obj_data_mime":
        return {"data": b64, "mime_type": mime}
    raise ValueError(f"unsupported image mode: {mode}")


def _parse_size(s: str):
    w, h = s.lower().split("x", 1)
    return int(w), int(h)


def _select_size(width: int, height: int, candidates: list[str], min_area: int = 0) -> str:
    target = width / float(height)
    best = None
    best_diff = None
    best_area = None
    for s in candidates:
        w, h = _parse_size(s)
        area = w * h
        if min_area and area < min_area:
            continue
        diff = abs((w / float(h)) - target)
        if best is None or diff < best_diff or (diff == best_diff and area < best_area):
            best = s
            best_diff = diff
            best_area = area
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cref", required=True)
    ap.add_argument("--sref", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="doubao-seedream-4.5")
    ap.add_argument("--base_url", default="https://YOUR_OPENAI_COMPAT_ENDPOINT/v1/images/generations")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only_cref", action="store_true", help="Only send cref as images[0].")
    ap.add_argument(
        "--image_format",
        type=str,
        default="auto",
        choices=["auto", "jpeg", "png"],
        help="Image re-encode format for request. auto tries jpeg then png.",
    )
    ap.add_argument("--min_area", type=int, default=0, help="Minimum size area (width*height). 0 = no limit.")
    ap.add_argument(
        "--image_mode",
        type=str,
        default="auto",
        choices=["auto", "data_url", "b64"],
        help="Images field format. auto tries url-like formats first.",
    )
    args = ap.parse_args()

    api_key = os.getenv("SEEDREAM_API_KEY", "")
    if not api_key:
        raise SystemExit("SEEDREAM_API_KEY is required")

    content_image = Image.open(args.cref).convert("RGB")
    size = _select_size(
        content_image.width,
        content_image.height,
        candidates=ASPECT_RATIO_1K,
        min_area=max(0, int(args.min_area)),
    )
    if not size:
        size = _select_size(
            content_image.width,
            content_image.height,
            candidates=ASPECT_RATIO,
            min_area=max(0, int(args.min_area)),
        )
    if not size:
        raise SystemExit("no valid size found")

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    image_paths = [args.cref] if args.only_cref else [args.cref, args.sref]

    fmts = ["jpeg", "png"] if args.image_format == "auto" else [args.image_format]
    modes = ["data_url"] if args.image_mode == "auto" else [args.image_mode]

    last_error = None
    data = None
    for fmt in fmts:
        for mode in modes:
            images_value = [_build_image_value(p, fmt=fmt, mode=mode) for p in image_paths]
            payload = {
                "model": args.model,
                "prompt": args.prompt,
                "images": images_value,
                "size": size,
                "response_format": "url",
                "sequential_image_generation": "auto",
                "sequential_image_generation_options": {"max_images": 1},
            }
            resp = requests.post(args.base_url, headers=headers, json=payload, timeout=600)
            if resp.status_code == 200:
                data = resp.json()
                if args.debug:
                    print(f"[OK] image_format={fmt} image_mode={mode} size={size} images={len(image_paths)}")
                break
            last_error = (fmt, mode, resp.status_code, resp.text[:2000])
            if args.debug:
                print(f"[FAIL] image_format={fmt} image_mode={mode} status={resp.status_code}")
                try:
                    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
                except Exception:
                    print(resp.text[:2000])
        if data is not None:
            break

    if data is None and last_error is not None:
        _, _, status, body = last_error
        if status == 400 and "image size must be at least" in body:
            digits = ""
            for ch in body:
                if ch.isdigit():
                    digits += ch
                elif digits:
                    break
            try:
                min_area = int(digits)
            except Exception:
                min_area = 3686400
            size2 = _select_size(
                content_image.width,
                content_image.height,
                candidates=ASPECT_RATIO,
                min_area=min_area,
            )
            if size2:
                size = size2
                fmt = "jpeg"
                mode = "data_url"
                images_value = [_build_image_value(p, fmt=fmt, mode=mode) for p in image_paths]
                payload = {
                    "model": args.model,
                    "prompt": args.prompt,
                    "images": images_value,
                    "size": size,
                    "response_format": "url",
                    "sequential_image_generation": "auto",
                    "sequential_image_generation_options": {"max_images": 1},
                }
                resp = requests.post(args.base_url, headers=headers, json=payload, timeout=600)
                if resp.status_code == 200:
                    data = resp.json()
                    if args.debug:
                        print(f"[OK] retry_min_area={min_area} size={size} image_mode={mode} image_format={fmt}")
                else:
                    last_error = (fmt, mode, resp.status_code, resp.text[:2000])

    if data is None:
        if last_error is None:
            raise SystemExit("request failed")
        fmt, mode, status, body = last_error
        raise SystemExit(f"request failed: status={status} last_try=image_format={fmt} image_mode={mode} body={body}")

    print(json.dumps(data, indent=2, ensure_ascii=False))
    url = data["data"][0]["url"]
    image_resp = requests.get(url, timeout=300)
    image_resp.raise_for_status()
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    Image.open(BytesIO(image_resp.content)).convert("RGB").save(args.out)


if __name__ == "__main__":
    main()
