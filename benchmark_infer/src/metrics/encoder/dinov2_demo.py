#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dino adain 的设计
python /data/benchmark_metrics/benchmark_metrics/dinov2_demo.py --content /data/benchmark_metrics/assets/content.webp --targets /data/benchmark_metrics/assets/style.webp --model /mnt/jfs/model_zoo/dinov2-with-registers-large  --device cuda --size 518
"""
import argparse, os, glob
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# ---- AdaIN on token features -------------------------------------------------
def adain_pool(tokens: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    tokens: (B, N, D)  # N 为 patch tokens 数, D 为 hidden size
    返回: (B, D)       # 对 AdaIN 后的 tokens 做均值池化
    """
    mu  = tokens.mean(dim=1, keepdim=True)               # (B,1,D)
    std = tokens.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    norm = (tokens - mu) / std                           # Ada(F)
    return norm.mean(dim=1)                              # 全局向量

# ---- 特征提取 -----------------------------------------------------------------
@torch.no_grad()
def phi_dinov2(img: Image.Image, processor, model, device="cpu", size=518):
    inputs = processor(
        images=img, return_tensors="pt",
        do_resize=True, size={"height": size, "width": size},
        do_center_crop=False
    )
    pixel_values = inputs["pixel_values"].to(device)
    out = model(pixel_values=pixel_values)
    # 取 patch tokens（去掉 CLS），形状 (B, N, D)
    tokens = out.last_hidden_state[:, 1:, :]
    return tokens


@torch.no_grad()
def token_content_similarity(
    img_a: Image.Image,
    img_b: Image.Image,
    processor,
    model,
    device: str = "cpu",
    size: int = 518,
) -> float:
    tokens_a = phi_dinov2(img_a, processor, model, device=device, size=size)
    tokens_b = phi_dinov2(img_b, processor, model, device=device, size=size)
    vec_a = tokens_a.mean(dim=1)
    vec_b = tokens_b.mean(dim=1)
    vec_a = vec_a / (vec_a.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    vec_b = vec_b / (vec_b.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    sim = (vec_a * vec_b).sum(dim=-1)
    return float(sim.item())

def load_model(model_path_or_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_path_or_id)
    model = AutoModel.from_pretrained(model_path_or_id)
    model.eval().to(device)
    return processor, model

def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def main():
    ap = argparse.ArgumentParser(description="CAS demo with DINOv2 + AdaIN")
    ap.add_argument("--content", required=True, help="内容图 C 的路径")
    ap.add_argument("--targets", required=True, help="生成图 T 的若干路径或含图片的目录")
    ap.add_argument("--model", default="facebook/dinov2-base",
                    help="DINOv2 模型（HF 名称或本地路径），如 /mnt/jfs/model_zoo/dinov2-with-registers-large")
    ap.add_argument("--size", type=int, default=518, help="DINOv2 推荐 518")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    processor, model = load_model(args.model, args.device)

    # 内容图特征（AdaIN + pool）
    # C = read_image(args.content)
    # C_tokens = phi_dinov2(C, processor, model, args.device, size=args.size)
    # C_vec = adain_pool(C_tokens)                         # (1, D)
    # print(C_vec.shape)
    img1 = read_image(args.content)
    img2 = read_image(args.targets)
    sim = token_content_similarity(img1, img2, processor, model, device=args.device, size=args.size)
    print("token content similarity:", sim)
if __name__ == "__main__":
    main()
