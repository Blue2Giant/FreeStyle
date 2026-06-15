#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Optional, List

# =========================
# 硬编码输入与超参
# =========================
# 输入图：第一个是 subject/content ref（作为 ID 参考），第二个是 style 参考

# 把环境变量全部写进去
os.environ["FLUX_DEV"] = "/data/USO/weights/FLUX.1-dev/flux1-dev.safetensors"
os.environ["AE"] = "/data/USO/weights/FLUX.1-dev/ae.safetensors"
os.environ["LORA"] = "/data/USO/weights/USO/uso_flux_v1.0/dit_lora.safetensors"
os.environ["PROJECTION_MODEL"] = "/data/USO/weights/USO/uso_flux_v1.0/projector.safetensors"
os.environ["SIGLIP_PATH"] = "/data/USO/weights/siglip"
os.environ["T5"] = "/mnt/jfs/model_zoo/t5-v1_1-xxl/"
os.environ["CLIP"] = "/mnt/jfs/model_zoo/clip-vit-large-patch14/"

import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from uso.flux.pipeline import USOPipeline, preprocess_ref

SUBJECT_PATH = "/data/benchmark_metrics/assets/content.webp"
STYLE_PATH   = "/data/benchmark_metrics/assets/style.webp"

# 提示词（留空）
PROMPT = ""

# 输出路径
OUTPUT_PATH = "/data/benchmark_metrics/assets/stylized.png"
SAVE_ATTN = False
SAVE_ATTN_PATH = "/data/benchmark_metrics/assets/attn/"

# 生成尺寸；若 INSTRUCT_EDIT=True 则自动改为 subject 图的尺寸
WIDTH, HEIGHT = 1024, 1024
INSTRUCT_EDIT = True

# 采样与引导
NUM_STEPS = 25
GUIDANCE  = 4.0
SEED      = 3407

# 参考图预处理：ID 参考（第一个）会被缩到 content_ref 的较短边
CONTENT_REF_SIZE = 512
PE = "d"  # 位置编码：["d","h","w","o"]

# Pipeline 配置
MODEL_TYPE   = "flux-dev"  # ["flux-dev","flux-dev-fp8","flux-schnell"]
OFFLOAD      = False
ONLY_LORA    = True
LORA_RANK    = 128
HF_DOWNLOAD  = False

# SigLIP
USE_SIGLIP   = True
SIGLIP_PATH  = os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384")

# =========================


def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 初始化 USOPipeline（内部会从环境变量读取各权重路径）
    pipe = USOPipeline(
        MODEL_TYPE,
        device,
        OFFLOAD,
        only_lora=ONLY_LORA,
        lora_rank=LORA_RANK,
        hf_download=HF_DOWNLOAD,
        save_attn=SAVE_ATTN,
    )

    # 2) SigLIP（style/其他参考图走视觉编码）
    siglip_processor = None
    if USE_SIGLIP:
        siglip_processor = SiglipImageProcessor.from_pretrained(SIGLIP_PATH)
        siglip_model = SiglipVisionModel.from_pretrained(SIGLIP_PATH).to(device).eval()
        pipe.model.vision_encoder = siglip_model

    # 3) 加载两张参考图
    id_img    = load_image(SUBJECT_PATH)  # 第1张作为 ID/content 参考
    style_img = load_image(STYLE_PATH)    # 第2张作为 style 参考

    # 4) 预处理：ID 参考图进入 preprocess_ref；style 进入 SigLIP
    ref_imgs_pil: List[Image.Image] = [preprocess_ref(id_img, CONTENT_REF_SIZE)]
    siglip_inputs = []
    if USE_SIGLIP and siglip_processor is not None and isinstance(style_img, Image.Image):
        with torch.no_grad():
            siglip_inputs.append(siglip_processor(style_img, return_tensors="pt").to(pipe.device))

    # 5) instruct_edit：根据 ID 图设置最终生成分辨率
    w, h = WIDTH, HEIGHT
    if INSTRUCT_EDIT and len(ref_imgs_pil) > 0:
        w, h = ref_imgs_pil[0].size
        print(f"[info] instruct_edit=True → use subject size: {w}x{h}")

    # 6) 生成
    ensure_dir(OUTPUT_PATH)
    image = pipe(
        prompt=PROMPT,
        width=w,
        height=h,
        guidance=GUIDANCE,
        num_steps=NUM_STEPS,
        seed=SEED,
        ref_imgs=ref_imgs_pil,       # 只放 ID 参考
        pe=PE,
        siglip_inputs=siglip_inputs, # style 参考走 SigLIP
        save_attn_path=SAVE_ATTN_PATH if SAVE_ATTN else None,
    )

    image.save(OUTPUT_PATH)
    print(f"[done] saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
