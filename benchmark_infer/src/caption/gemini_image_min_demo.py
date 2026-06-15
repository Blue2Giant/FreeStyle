"""
export GEMINI_API_KEY=YOUR_API_KEY

python /data/benchmark_metrics/caption_pipe/gemini_image_min_demo.py \
  --content /data/benchmark_metrics/assets/jiegeng.png \
  --style /data/benchmark_metrics/assets/style.webp \
  --prompt "Transfer the style while keeping the content." \
  --out /data/benchmark_metrics/logs/result.png \
  --model_id "gemini-2.5-flash-image-native"
"""
import argparse
import os
from io import BytesIO

from google import genai
from google.genai import types
from PIL import Image
banana_aspect_ratio = ["1:1","1:4","1:8","2:3","3:2","3:4","4:1","4:3","4:5","5:4","8:1","9:16","16:9","21:9"]
banana_resolution = ["512px", "1K", "2K", "4K"]

def image_to_part(path: str) -> types.Part:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model_id", default="gemini-3-pro-native")
    ap.add_argument("--base_url", default="https://YOUR_OPENAI_COMPAT_ENDPOINT/gemini")
    ap.add_argument("--resolution", default="1K")
    ap.add_argument("--aspect_ratio", default="")
    ap.add_argument("--api_version", default="v1alpha")
    args = ap.parse_args()

    client = genai.Client(
        http_options={"api_version": args.api_version, "base_url": args.base_url},
        api_key=os.getenv("GEMINI_API_KEY", ""),
    )

    content_image = Image.open(args.content).convert("RGB")
    content_part = image_to_part(args.content)
    style_part = image_to_part(args.style)
    text_part = types.Part.from_text(text=args.prompt)
    if args.aspect_ratio:
        if args.aspect_ratio not in banana_aspect_ratio:
            raise ValueError(f"aspect_ratio {args.aspect_ratio} not in {banana_aspect_ratio}")
        aspect_ratio = args.aspect_ratio
    else:
        aspect_ratio = _select_aspect_ratio(content_image.width, content_image.height)
    if args.resolution not in banana_resolution:
        raise ValueError(f"resolution {args.resolution} not in {banana_resolution}")
    image_size = args.resolution

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

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    saved = False
    for part in resp.parts:
        if part.inline_data is not None:
            img = part.as_image()
            print(f"save image to {args.out}")
            img.save(args.out)
            saved = True
            break
    if not saved:
        raise SystemExit("No image returned")


if __name__ == "__main__":
    main()
