import argparse
import base64
import os
import re
from io import BytesIO

import requests
from PIL import Image


def image_to_data_url(path: str) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def extract_image_urls(text: str):
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


def main():
    ap = argparse.ArgumentParser("GPT-4o core image generation demo (content + style)")
    ap.add_argument("--content", required=True)
    ap.add_argument("--style", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gpt-4o-all")
    ap.add_argument("--base_url", default=os.getenv("OPENAI_BASE_URL", "https://YOUR_OPENAI_COMPAT_ENDPOINT/v1"))
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    args = ap.parse_args()

    content_url = image_to_data_url(args.content)
    style_url = image_to_data_url(args.style)
    guidance = (
        "Use the FIRST image as content reference and the SECOND image as style reference. "
        "Preserve the subject and layout from the FIRST image while applying the style, palette, "
        "texture, and lighting from the SECOND image."
    )
    payload = {
        "model": args.model,
        "stream": False,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{guidance}\n{args.prompt}"},
                    {"type": "image_url", "image_url": {"url": content_url}},
                    {"type": "image_url", "image_url": {"url": style_url}},
                ],
            }
        ],
    }
    headers = {"Authorization": f"Bearer {args.api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{args.base_url}/chat/completions", json=payload, headers=headers, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    urls = extract_image_urls(content)
    if not urls:
        raise SystemExit("No image URL or base64 found in response")
    image = download_image(urls[0])
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    image.save(args.out)


if __name__ == "__main__":
    main()
