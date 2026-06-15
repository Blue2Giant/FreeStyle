#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python /data/Sref_Cref/CSD/single_demo.py   --image_path /mnt/jfs/xhs_style_dir/54ef53b14fac633bec15d131/1000g0082jvfpeqmja00g401jrb9r3k9hmqcbepo.jpg   --output_dir /data/Sref_Cref/CSD/output   --pt_style csd   --arch vit_base   --model_path /data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar  
"""
import argparse
import json
import os
import shutil
from pathlib import Path
#倒入通文件夹的模块
from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict
from CSD.loss_utils import transforms_branch0
import torch
from PIL import Image
def ensure_dir(p: str) -> str:
    out = os.path.abspath(os.path.expanduser(p))
    os.makedirs(out, exist_ok=True)
    return out

def load_image(image_path: str):
    img = Image.open(image_path).convert("RGB")
    return img

def build_csd_model(arch: str, model_path: str):
    """
    构建 CSD 模型与其预处理 transform（transforms_branch0）
    需要你的仓库里存在: CSD.model / CSD.utils / CSD.loss_utils
    """


    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建模型
    content_proj_head = "default"
    model = CSD_CLIP(arch, content_proj_head)
    if has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 加载 checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = convert_state_dict(ckpt["state_dict"])
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"[CSD] load_state_dict: {msg}")

    model.eval().to(device)
    preprocess = transforms_branch0  # 这是一个 torchvision.transforms.Compose
    return model, preprocess, device

def build_clip_model(arch: str):
    """
    构建（OpenAI）CLIP 模型与预处理
    你之前使用过: from models import clip; model, preprocess = clip.load(...)
    """
    try:
        from models import clip  # 你的本地 'models' 包（与项目一致）
    except Exception:
        import clip  # 兜底: 若系统装了 openai/clip 的 pip 包

    mapping = {
        "vit_large": "ViT-L/14",
        "vit_base": "ViT-B/16",
    }
    if arch not in mapping:
        raise ValueError(f"Unsupported arch for CLIP: {arch} (choose vit_base|vit_large)")
    model, preprocess = clip.load(mapping[arch], device="cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    return model, preprocess, device

@torch.inference_mode()
def extract_feature_csd(model, preprocess, device, pil_img, eval_embed: str = "head"):
    """
    简化版：走 CSD 模型的前向，得到单张图片的特征。
    若你更偏好使用 CSD.utils.extract_features（基于 DataLoader），也可以改成那种方式。
    这里直接用 transforms_branch0 + 模型 encode。
    """
    # 大多数 CLIP-like 模型习惯是返回 image embedding；CSD_CLIP 里通常有类似接口。
    # 如果你的 CSD_CLIP 暴露 encode_image，可直接用；否则走 model.forward_image 或统一的 forward。
    x = preprocess(pil_img).unsqueeze(0).to(device)

    # 尝试常见接口
    if hasattr(model, "encode_image"):
        print("Use model.encode_image() to extract feature")
        feats = model.encode_image(x)
    elif hasattr(model, "forward_image"):
        feats = model.forward_image(x)
    else:
        # 最兜底：直接前向（根据你模型定义，如果 forward 接受 image）
        feats = model(x)

    # 统一成 2D
    feats = feats[-1].clone()
    if feats.ndim > 2:
        feats = feats.flatten(1)

    # L2 normalize（常见检索特征做法）
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats  # shape: [1, D]

@torch.inference_mode()
def extract_feature_clip(model, preprocess, device, pil_img):
    x = preprocess(pil_img).unsqueeze(0).to(device)
    # OpenAI CLIP 接口
    if hasattr(model, "encode_image"):
        feats = model.encode_image(x)
    else:
        # 某些封装里是 model.visual(x) -> [B, L, D]；此时取 CLS 或平均
        out = model.visual(x)
        feats = out if out.ndim == 2 else out[:, 0, :]
    feats = torch.nn.functional.normalize(feats, dim=1)
    return feats  # [1, D]

def main():
    ap = argparse.ArgumentParser("Single-image feature demo: copy image + save feature tensor")
    ap.add_argument("--image_path", required=True, type=str, help="输入图片路径")
    ap.add_argument("--output_dir", required=True, type=str, help="输出目录（会自动创建）")
    ap.add_argument("--pt_style", default="clip", choices=["csd", "clip"], help="后端类型：csd 或 clip")
    ap.add_argument("--arch", default="vit_base", type=str, help="模型结构：csd/clip 都支持 vit_base | clip 还支持 vit_large")
    ap.add_argument("--model_path", default=None, type=str, help="当 pt_style=csd 时必须提供 checkpoint 路径")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"], help="保存特征张量 dtype")
    ap.add_argument("--eval_embed", default="head", choices=["head", "backbone"], help="CSD 的抽特征分支（若有）")
    args = ap.parse_args()

    img_path = os.path.abspath(os.path.expanduser(args.image_path))
    if not os.path.isfile(img_path):
        raise SystemExit(f"[Error] image not found: {img_path}")

    out_dir = ensure_dir(args.output_dir)
    stem = Path(img_path).stem

    # --------- 构建模型 + 预处理 ----------
    if args.pt_style == "csd":
        if not args.model_path:
            raise SystemExit("[Error] --model_path 必须提供（CSD 权重路径）")
        model, preprocess, device = build_csd_model(args.arch, args.model_path)
    else:
        model, preprocess, device = build_clip_model(args.arch)

    # --------- 读取图片，抽特征 ----------
    pil_img = load_image(img_path)
    if args.pt_style == "csd":
        feats = extract_feature_csd(model, preprocess, device, pil_img, eval_embed=args.eval_embed)
    else:
        feats = extract_feature_clip(model, preprocess, device, pil_img)

    # 转 dtype & 存盘
    if args.dtype == "float16":
        feats = feats.to(torch.float16)
    else:
        feats = feats.to(torch.float32)

    feat_path = os.path.join(out_dir, f"{stem}_feat.pt")
    torch.save(feats.cpu(), feat_path)

    # 复制原图
    img_dst = os.path.join(out_dir, f"{stem}{Path(img_path).suffix}")
    shutil.copy2(img_path, img_dst)

    # 写 meta
    meta = {
        "pt_style": args.pt_style,
        "arch": args.arch,
        "model_path": os.path.abspath(os.path.expanduser(args.model_path)) if args.model_path else None,
        "image_src": img_path,
        "image_copied_to": img_dst,
        "feature_saved_to": feat_path,
        "feature_shape": list(feats.shape),
        "feature_dtype": str(feats.dtype).replace("torch.", ""),
        "device_used": device,
    }
    with open(os.path.join(out_dir, f"{stem}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Copied image -> {img_dst}")
    print(f"[OK] Saved feature tensor -> {feat_path}")
    print(f"[OK] Meta -> {os.path.join(out_dir, f'{stem}_meta.json')}")
    print(f"[INFO] Feature shape: {tuple(feats.shape)}; dtype: {feats.dtype}")

if __name__ == "__main__":
    main()
