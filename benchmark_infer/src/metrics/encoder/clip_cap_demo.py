"""
python3 /data/benchmark_metrics/benchmark_metrics/clip_caption_similarity_demo.py \
  --image /data/benchmark_metrics/assets/content.webp \
  --caption "an elf standing" \
  --model_path /mnt/jfs/model_zoo/clip-vit-large-patch14 \
  --device cuda
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def clip_image_text_similarity(
    image: Image.Image,
    caption: str,
    processor: CLIPProcessor,
    model: CLIPModel,
    device: str = "cpu",
) -> float:
    inputs = processor(
        text=[caption],
        images=image,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    sim = (image_embeds * text_embeds).sum(dim=-1)
    return float(sim.item())


def main():
    parser = argparse.ArgumentParser(description="Minimal CLIP image-caption similarity demo")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--caption", required=True, help="待评估的 caption 文本")
    parser.add_argument(
        "--model_path",
        default="/mnt/jfs/model_zoo/clip-vit-large-patch14",
        help="本地 CLIP 权重路径或 HF 名称",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu 或 cuda",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    device = torch.device(args.device)
    processor = CLIPProcessor.from_pretrained(args.model_path)
    model = CLIPModel.from_pretrained(args.model_path).to(device)
    model.eval()

    img = load_image(args.image)
    score = clip_image_text_similarity(
        image=img,
        caption=args.caption,
        processor=processor,
        model=model,
        device=args.device,
    )
    print(f"CLIP image-text cosine similarity: {score:.6f}")


if __name__ == "__main__":
    main()

