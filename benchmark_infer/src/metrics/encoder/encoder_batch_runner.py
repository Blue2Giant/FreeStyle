#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import multiprocessing as mp
import os
import re
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    CLIPProcessor,
    CLIPModel,
    AutoModelForCausalLM,
)

from CSD.model import CSD_CLIP
from CSD.utils import has_batchnorms, convert_state_dict
from CSD.loss_utils import transforms_branch0
from csd_utils import CSDStyleEmbedding, SEStyleEmbedding


IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def read_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def sort_key(name: str):
    base = os.path.splitext(os.path.basename(name))[0]
    nums = re.findall(r"\d+", base)
    if nums:
        return int(nums[0])
    return base


def list_images(folder: str) -> Dict[str, str]:
    items = {}
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext not in IMG_EXTS:
            continue
        items[os.path.splitext(name)[0]] = path
    return items


def save_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_overwrite(val) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid overwrite value: {val}")


def normalize_cosine_score(value: float) -> float:
    v = (float(value) + 1.0) / 2.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def compute_similarity(vec_a: torch.Tensor, vec_b: torch.Tensor, metric: str) -> float:
    metric = (metric or "cosine01").lower()
    if metric in {"cosine", "cosine01", "l2"}:
        vec_a = vec_a / vec_a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        vec_b = vec_b / vec_b.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    if metric == "cosine":
        return float((vec_a * vec_b).sum(dim=-1).item())
    if metric == "cosine01":
        return normalize_cosine_score((vec_a * vec_b).sum(dim=-1).item())
    if metric == "l2":
        dist = torch.norm(vec_a - vec_b, dim=-1).item()
        return 1.0 / (1.0 + float(dist))
    if metric == "dot":
        return float((vec_a * vec_b).sum(dim=-1).item())
    raise ValueError(f"Unsupported sim_metric: {metric}")


@torch.no_grad()
def dinov2_similarity(img_a: Image.Image, img_b: Image.Image, processor, model, device: str, size: int, sim_metric: str) -> float:
    inputs_a = processor(
        images=img_a, return_tensors="pt",
        do_resize=True, size={"height": size, "width": size},
        do_center_crop=False
    )
    inputs_b = processor(
        images=img_b, return_tensors="pt",
        do_resize=True, size={"height": size, "width": size},
        do_center_crop=False
    )
    tokens_a = model(pixel_values=inputs_a["pixel_values"].to(device)).last_hidden_state[:, 1:, :]
    tokens_b = model(pixel_values=inputs_b["pixel_values"].to(device)).last_hidden_state[:, 1:, :]
    vec_a = tokens_a.mean(dim=1)
    vec_b = tokens_b.mean(dim=1)
    return compute_similarity(vec_a, vec_b, sim_metric)


def cas_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std


@torch.no_grad()
def cas_similarity(img_a: Image.Image, img_b: Image.Image, processor, model, device: str) -> float:
    if img_a.size != (512, 512):
        img_a = img_a.resize((512, 512))
    if img_b.size != (512, 512):
        img_b = img_b.resize((512, 512))
    inputs1 = processor(images=img_a, return_tensors="pt").to(device)
    outputs1 = model(**inputs1)
    feat1 = outputs1.last_hidden_state
    mean1, std1 = cas_mean_std(feat1.transpose(-1, -2))
    size1 = feat1.transpose(-1, -2).size()
    norm1 = (feat1.transpose(-1, -2) - mean1.expand(size1)) / std1.expand(size1)

    inputs2 = processor(images=img_b, return_tensors="pt").to(device)
    outputs2 = model(**inputs2)
    feat2 = outputs2.last_hidden_state
    mean2, std2 = cas_mean_std(feat2.transpose(-1, -2))
    size2 = feat2.transpose(-1, -2).size()
    norm2 = (feat2.transpose(-1, -2) - mean2.expand(size2)) / std2.expand(size2)
    return float(torch.mean((norm2 - norm1) ** 2).item())


def build_oneig_encoder(csd_model_path: str, se_model_path: str, device: str, clip_model_path: str):
    clip_model_path = clip_model_path.strip() if clip_model_path else ""
    clip_model_path = clip_model_path or None
    csd_encoder = CSDStyleEmbedding(model_path=csd_model_path, device=device, clip_model_path=clip_model_path)
    se_encoder = SEStyleEmbedding(pretrained_path=se_model_path, device=device)
    return csd_encoder, se_encoder


@torch.no_grad()
def oneig_similarity(img_a: Image.Image, img_b: Image.Image, encoder, device: str, sim_metric: str, size: int) -> float:
    if size:
        img_a = img_a.resize((size, size))
        img_b = img_b.resize((size, size))
    csd_encoder, se_encoder = encoder
    csd_embed_a = csd_encoder.get_style_embedding(img_a)
    csd_embed_b = csd_encoder.get_style_embedding(img_b)
    se_embed_a = se_encoder.get_style_embedding(img_a)
    se_embed_b = se_encoder.get_style_embedding(img_b)
    csd_vec_a = torch.tensor(csd_embed_a, device=device).unsqueeze(0)
    csd_vec_b = torch.tensor(csd_embed_b, device=device).unsqueeze(0)
    se_vec_a = torch.tensor(se_embed_a, device=device).unsqueeze(0)
    se_vec_b = torch.tensor(se_embed_b, device=device).unsqueeze(0)
    csd_score = compute_similarity(csd_vec_a, csd_vec_b, "cosine")
    se_score = compute_similarity(se_vec_a, se_vec_b, "cosine")
    return (csd_score + se_score) / 2.0


def _load_checkpoint(model_path: str):
    try:
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def build_csd_model(arch: str, model_path: str, device: str):
    content_proj_head = "default"
    model = CSD_CLIP(arch, content_proj_head)
    if has_batchnorms(model):
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ckpt = _load_checkpoint(model_path)
    state_dict = convert_state_dict(ckpt["state_dict"])
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    preprocess = transforms_branch0
    return model, preprocess


@torch.no_grad()
def csd_similarity(img_a: Image.Image, img_b: Image.Image, model, preprocess, device: str) -> float:
    x_a = preprocess(img_a).unsqueeze(0).to(device)
    x_b = preprocess(img_b).unsqueeze(0).to(device)
    if hasattr(model, "encode_image"):
        fa = model.encode_image(x_a)
        fb = model.encode_image(x_b)
    elif hasattr(model, "forward_image"):
        fa = model.forward_image(x_a)
        fb = model.forward_image(x_b)
    else:
        fa = model(x_a)
        fb = model(x_b)
    fa = fa[-1].clone() if isinstance(fa, (list, tuple)) else fa
    fb = fb[-1].clone() if isinstance(fb, (list, tuple)) else fb
    if fa.ndim > 2:
        fa = fa.flatten(1)
    if fb.ndim > 2:
        fb = fb.flatten(1)
    fa = torch.nn.functional.normalize(fa, dim=1)
    fb = torch.nn.functional.normalize(fb, dim=1)
    return compute_similarity(fa, fb, "cosine")


@torch.no_grad()
def clip_image_text_similarity(image: Image.Image, caption: str, processor: CLIPProcessor, model: CLIPModel, device: str, sim_metric: str) -> float:
    inputs = processor(text=[caption], images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds
    return compute_similarity(image_embeds, text_embeds, sim_metric)


@torch.no_grad()
def clip_t_similarity(image: Image.Image, text: str, processor, tokenizer, model, device: str, sim_metric: str) -> float:
    image_inputs = processor(images=image, return_tensors="pt")
    text_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)
    return compute_similarity(image_features, text_features, sim_metric)


def select_clipcap_text(caption: str, mode: str) -> str:
    mode = (mode or "full").lower()
    text = caption or ""
    if mode == "first_sentence":
        parts = text.split(".", 1)
        return parts[0].strip()
    if mode == "full":
        return text
    raise ValueError(f"Unsupported clipcap_text_mode: {mode}")


def build_tasks_pair(dir_a: str, dir_b: str) -> List[Tuple[str, str, str]]:
    map_a = list_images(dir_a)
    map_b = list_images(dir_b)
    keys = list(set(map_a.keys()) & set(map_b.keys()))
    return [(k, map_a[k], map_b[k]) for k in keys]


def build_tasks_clipcap(image_dir: str, prompt_json: str) -> List[Tuple[str, str, str]]:
    with open(prompt_json, "r", encoding="utf-8") as f:
        prompt_map = json.load(f)
    if not isinstance(prompt_map, dict):
        raise ValueError("prompt_json must be a dict")
    image_map = list_images(image_dir)
    keys = list(set(image_map.keys()) & set(prompt_map.keys()))
    return [(k, image_map[k], str(prompt_map[k])) for k in keys]


def build_tasks_clip_t(image_dir: str, prompt_json: str) -> List[Tuple[str, str, str]]:
    return build_tasks_clipcap(image_dir, prompt_json)


def build_tasks_onealign(image_dir: str) -> List[Tuple[str, str]]:
    image_map = list_images(image_dir)
    keys = list(image_map.keys())
    return [(k, image_map[k]) for k in keys]


def build_tasks_aesthetic(image_dir: str) -> List[Tuple[str, str]]:
    return build_tasks_onealign(image_dir)


def _setup_cuda_env(args_dict: dict):
    cuda_visible = args_dict.get("cuda_visible_devices")
    if cuda_visible is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible


def worker_pair(encoder: str, args_dict: dict, tasks: List[Tuple[str, str, str]], out_queue: mp.Queue):
    _setup_cuda_env(args_dict)
    device = args_dict["device"]
    if encoder == "dinov2":
        processor = AutoImageProcessor.from_pretrained(args_dict["model"])
        model = AutoModel.from_pretrained(args_dict["model"]).to(device)
        model.eval()
        size = args_dict["size"]
        sim_metric = args_dict["sim_metric"]
        for k, p1, p2 in tasks:
            img_a = read_image(p1)
            img_b = read_image(p2)
            score = dinov2_similarity(img_a, img_b, processor, model, device, size, sim_metric)
            out_queue.put((k, score))
        return
    if encoder == "cas":
        config = AutoConfig.from_pretrained(args_dict["model"])
        config.output_hidden_states = True
        processor = AutoImageProcessor.from_pretrained(args_dict["model"])
        model = AutoModel.from_pretrained(args_dict["model"], config=config).to(device)
        model.eval()
        for k, p1, p2 in tasks:
            img_a = read_image(p1)
            img_b = read_image(p2)
            score = cas_similarity(img_a, img_b, processor, model, device)
            out_queue.put((k, score))
        return
    if encoder == "oneig":
        encoder = build_oneig_encoder(
            args_dict["model_path"],
            args_dict["se_model_path"],
            device,
            args_dict["clip_model_path"],
        )
        sim_metric = args_dict["sim_metric"]
        size = args_dict.get("oneig_size", 512)
        for k, p1, p2 in tasks:
            img_a = read_image(p1)
            img_b = read_image(p2)
            score = oneig_similarity(img_a, img_b, encoder, device, sim_metric, size)
            out_queue.put((k, score))
        return
    if encoder == "csd":
        model, preprocess = build_csd_model(args_dict["arch"], args_dict["model_path"], device)
        for k, p1, p2 in tasks:
            img_a = read_image(p1)
            img_b = read_image(p2)
            score = csd_similarity(img_a, img_b, model, preprocess, device)
            out_queue.put((k, score))
        return
    raise ValueError(f"Unsupported encoder: {encoder}")


def worker_clipcap(args_dict: dict, tasks: List[Tuple[str, str, str]], out_queue: mp.Queue):
    _setup_cuda_env(args_dict)
    device = args_dict["device"]
    processor = CLIPProcessor.from_pretrained(args_dict["model"])
    model = CLIPModel.from_pretrained(args_dict["model"]).to(device)
    model.eval()
    sim_metric = args_dict["sim_metric"]
    text_mode = args_dict["clipcap_text_mode"]
    for k, img_path, caption in tasks:
        img = read_image(img_path)
        text = select_clipcap_text(caption, text_mode)
        score = clip_image_text_similarity(img, text, processor, model, device, sim_metric)
        out_queue.put((k, score))


def worker_clip_t(args_dict: dict, tasks: List[Tuple[str, str, str]], out_queue: mp.Queue):
    _setup_cuda_env(args_dict)
    device = args_dict["device"]
    model = AutoModel.from_pretrained(args_dict["model"]).to(device)
    processor = AutoProcessor.from_pretrained(args_dict["model"])
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"])
    model.eval()
    sim_metric = args_dict["sim_metric"]
    text_mode = args_dict["clipcap_text_mode"]
    for k, img_path, text in tasks:
        img = read_image(img_path)
        text = select_clipcap_text(text, text_mode)
        score = clip_t_similarity(img, text, processor, tokenizer, model, device, sim_metric)
        out_queue.put((k, score))


def worker_onealign(args_dict: dict, tasks: List[Tuple[str, str]], out_queue: mp.Queue):
    _setup_cuda_env(args_dict)
    model = AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=args_dict["dtype"],
        device_map="auto",
    )
    for k, img_path in tasks:
        img = read_image(img_path)
        score = model.score([img], task_=args_dict["task"], input_="image")
        if isinstance(score, (list, tuple)):
            score = score[0]
        if hasattr(score, "item"):
            score = score.item()
        out_queue.put((k, float(score)))


def _laion_head_dim(clip_model: str) -> int:
    name = clip_model.lower().replace("_", "-")
    if name in ["vit-l-14", "vit-l-14-336"]:
        return 768
    if name in ["vit-b-32", "vit-b-16"]:
        return 512
    raise ValueError(f"Unsupported clip_model for laion: {clip_model}")


def load_laion_aesthetic(clip_model: str, clip_ckpt: str, pretrained_tag: str, linear_path: str, device: str):
    import torch.nn as nn
    from urllib.request import urlretrieve
    import open_clip

    linear_path = os.path.expanduser(linear_path)
    if not os.path.isfile(linear_path):
        os.makedirs(os.path.dirname(linear_path), exist_ok=True)
        url_key = clip_model.lower().replace("-", "_")
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"
            + url_key
            + "_linear.pth?raw=true"
        )
        urlretrieve(url_model, linear_path)
    head_dim = _laion_head_dim(clip_model)
    linear = nn.Linear(head_dim, 1)
    state = torch.load(linear_path, map_location="cpu")
    linear.load_state_dict(state)
    linear.eval().to(device)
    pretrained = clip_ckpt if clip_ckpt and os.path.isfile(clip_ckpt) else pretrained_tag
    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model,
        pretrained=pretrained,
        device=device,
    )
    model.eval()
    return model, preprocess, linear


def load_aesthetic_v25(encoder_model_name: str, device: str, dtype):
    from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

    model, preprocessor = convert_v2_5_from_siglip(
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        encoder_model_name=encoder_model_name,
    )
    model = model.to(dtype).to(device)
    return model, preprocessor


def worker_aesthetic(args_dict: dict, tasks: List[Tuple[str, str]], out_queue: mp.Queue):
    _setup_cuda_env(args_dict)
    device = args_dict["device"]
    backend = args_dict["backend"]
    if backend == "laion":
        model, preprocess, linear = load_laion_aesthetic(
            args_dict["laion_clip_model"],
            args_dict["laion_clip_ckpt"],
            args_dict["laion_pretrained_tag"],
            args_dict["laion_linear_path"],
            device,
        )
        for k, img_path in tasks:
            img = read_image(img_path)
            x = preprocess(img).unsqueeze(0).to(device)
            feats = model.encode_image(x)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            score = linear(feats).squeeze()
            if hasattr(score, "item"):
                score = score.item()
            out_queue.put((k, float(score)))
        return
    if backend == "v25":
        model, preprocess = load_aesthetic_v25(
            args_dict["v25_encoder_model_name"],
            device,
            args_dict["dtype"],
        )
        model.eval()
        for k, img_path in tasks:
            img = read_image(img_path)
            inputs = preprocess(images=img, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(device, dtype=args_dict["dtype"])
            with torch.inference_mode():
                outputs = model(pixel_values)
                score = outputs.logits.squeeze().float()
            out_queue.put((k, float(score.item())))
        return
    raise ValueError(f"Unsupported backend: {backend}")


def run_batch(tasks: List[Tuple], out_path: str, overwrite: bool, worker_fn, args_dict: dict):
    results = load_json(out_path)
    if overwrite:
        results = {}
    remaining = []
    for item in tasks:
        key = item[0]
        if key in results:
            continue
        remaining.append(item)
    if not remaining:
        save_json(out_path, results)
        return
    gpus = str(args_dict.get("gpus") or "").strip()
    gpus = [x for x in gpus.split(",") if x != ""]
    num_procs = max(1, int(args_dict.get("num_procs") or 1))
    if gpus:
        num_procs = min(num_procs, len(gpus))
    num_procs = min(num_procs, len(remaining))
    chunk = int(math.ceil(len(remaining) / float(num_procs)))
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    workers = []
    for i in range(num_procs):
        sub = remaining[i * chunk:(i + 1) * chunk]
        if not sub:
            continue
        sub_args = dict(args_dict)
        if gpus:
            sub_args["cuda_visible_devices"] = str(gpus[i % len(gpus)])
            sub_args["device"] = "cuda:0"
        if sub_args.get("encoder"):
            p = ctx.Process(target=worker_fn, args=(sub_args.get("encoder"), sub_args, sub, q))
        else:
            p = ctx.Process(target=worker_fn, args=(sub_args, sub, q))
        p.start()
        workers.append(p)
    total = len(remaining)
    done = 0
    pbar = tqdm(total=total, unit="img")
    while done < total:
        try:
            k, score = q.get(timeout=5)
            results[k] = score
            done += 1
            pbar.update(1)
        except Exception:
            if not any(p.is_alive() for p in workers) and q.empty():
                break
    for p in workers:
        p.join()
    pbar.close()
    save_json(out_path, results)


def main():
    ap = argparse.ArgumentParser(description="Batch runner for image encoders")
    sub = ap.add_subparsers(dest="mode", required=True)

    ap_pair = sub.add_parser("pair")
    ap_pair.add_argument("--encoder", choices=["dinov2", "cas", "oneig", "csd"], required=True)
    ap_pair.add_argument("--dir_a", required=True)
    ap_pair.add_argument("--dir_b", required=True)
    ap_pair.add_argument("--out_json", required=True)
    ap_pair.add_argument("--model", required=True)
    ap_pair.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_pair.add_argument("--size", type=int, default=518)
    ap_pair.add_argument("--dtype", default="bfloat16")
    ap_pair.add_argument("--num_procs", type=int, default=4)
    ap_pair.add_argument("--gpus", default="")
    ap_pair.add_argument("--overwrite", type=parse_overwrite, default=False)
    ap_pair.add_argument("--sim_metric", choices=["cosine", "cosine01", "l2", "dot"], default="cosine01")
    ap_pair.add_argument("--oneig_arch", default="vit_base")
    ap_pair.add_argument("--oneig_model_path", default="")
    ap_pair.add_argument("--oneig_se_model_path", default="")
    ap_pair.add_argument("--oneig_clip_model_path", default="")
    ap_pair.add_argument("--oneig_size", type=int, default=512)
    ap_pair.add_argument("--csd_arch", default="vit_base")
    ap_pair.add_argument("--csd_model_path", default="")

    ap_clip = sub.add_parser("clip_cap")
    ap_clip.add_argument("--image_dir", required=True)
    ap_clip.add_argument("--prompt_json", required=True)
    ap_clip.add_argument("--out_json", required=True)
    ap_clip.add_argument("--model", default="/mnt/jfs/model_zoo/clip-vit-large-patch14")
    ap_clip.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_clip.add_argument("--num_procs", type=int, default=4)
    ap_clip.add_argument("--gpus", default="")
    ap_clip.add_argument("--overwrite", type=parse_overwrite, default=False)
    ap_clip.add_argument("--sim_metric", choices=["cosine", "cosine01", "l2", "dot"], default="cosine01")
    ap_clip.add_argument("--clipcap_text_mode", choices=["full", "first_sentence"], default="full")

    ap_clip_t = sub.add_parser("clip_t")
    ap_clip_t.add_argument("--image_dir", required=True)
    ap_clip_t.add_argument("--prompt_json", required=True)
    ap_clip_t.add_argument("--out_json", required=True)
    ap_clip_t.add_argument("--model", default="openai/clip-vit-base-patch32")
    ap_clip_t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_clip_t.add_argument("--num_procs", type=int, default=4)
    ap_clip_t.add_argument("--gpus", default="")
    ap_clip_t.add_argument("--overwrite", type=parse_overwrite, default=False)
    ap_clip_t.add_argument("--sim_metric", choices=["cosine", "cosine01", "l2", "dot"], default="cosine01")
    ap_clip_t.add_argument("--clipcap_text_mode", choices=["full", "first_sentence"], default="full")

    ap_one = sub.add_parser("onealign")
    ap_one.add_argument("--image_dir", required=True)
    ap_one.add_argument("--out_json", required=True)
    ap_one.add_argument("--model", default="/mnt/jfs/model_zoo/one-align")
    ap_one.add_argument("--task", default="aesthetics")
    ap_one.add_argument("--dtype", default="float16")
    ap_one.add_argument("--num_procs", type=int, default=4)
    ap_one.add_argument("--gpus", default="")
    ap_one.add_argument("--overwrite", type=parse_overwrite, default=False)

    ap_aes = sub.add_parser("aesthetic")
    ap_aes.add_argument("--backend", choices=["laion", "v25"], required=True)
    ap_aes.add_argument("--image_dir", required=True)
    ap_aes.add_argument("--out_json", required=True)
    ap_aes.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap_aes.add_argument("--num_procs", type=int, default=4)
    ap_aes.add_argument("--gpus", default="")
    ap_aes.add_argument("--overwrite", type=parse_overwrite, default=False)
    ap_aes.add_argument("--laion_clip_model", default="ViT-L-14")
    ap_aes.add_argument("--laion_clip_ckpt", default="/mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors")
    ap_aes.add_argument("--laion_pretrained_tag", default="openai")
    ap_aes.add_argument("--laion_linear_path", default="~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth")
    ap_aes.add_argument("--v25_encoder_model_name", default="/mnt/jfs/model_zoo/siglip-so400m-patch14-384/")
    ap_aes.add_argument("--dtype", default="bfloat16")

    args = ap.parse_args()

    if args.mode == "pair":
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        tasks = build_tasks_pair(args.dir_a, args.dir_b)
        args_dict = {
            "encoder": args.encoder,
            "model": args.model,
            "device": args.device,
            "size": args.size,
            "dtype": dtype,
            "num_procs": args.num_procs,
            "gpus": args.gpus,
            "sim_metric": args.sim_metric,
            "arch": args.oneig_arch,
            "model_path": args.oneig_model_path,
            "clip_model_path": args.oneig_clip_model_path,
            "se_model_path": args.oneig_se_model_path,
            "oneig_size": args.oneig_size,
        }
        if args.encoder == "oneig" and not args.oneig_model_path:
            raise SystemExit("--oneig_model_path is required when encoder=oneig")
        if args.encoder == "oneig" and not args.oneig_se_model_path:
            raise SystemExit("--oneig_se_model_path is required when encoder=oneig")
        if args.encoder == "csd":
            args_dict["arch"] = args.csd_arch
            args_dict["model_path"] = args.csd_model_path
            if not args.csd_model_path:
                raise SystemExit("--csd_model_path is required when encoder=csd")
        run_batch(tasks, args.out_json, args.overwrite, worker_pair, args_dict)
        return

    if args.mode == "clip_cap":
        tasks = build_tasks_clipcap(args.image_dir, args.prompt_json)
        args_dict = {
            "model": args.model,
            "device": args.device,
            "num_procs": args.num_procs,
            "gpus": args.gpus,
            "sim_metric": args.sim_metric,
            "clipcap_text_mode": args.clipcap_text_mode,
        }
        run_batch(tasks, args.out_json, args.overwrite, worker_clipcap, args_dict)
        return

    if args.mode == "clip_t":
        tasks = build_tasks_clip_t(args.image_dir, args.prompt_json)
        args_dict = {
            "model": args.model,
            "device": args.device,
            "num_procs": args.num_procs,
            "gpus": args.gpus,
            "sim_metric": args.sim_metric,
            "clipcap_text_mode": args.clipcap_text_mode,
        }
        run_batch(tasks, args.out_json, args.overwrite, worker_clip_t, args_dict)
        return

    if args.mode == "onealign":
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        tasks = build_tasks_onealign(args.image_dir)
        args_dict = {
            "model": args.model,
            "task": args.task,
            "dtype": dtype,
            "num_procs": args.num_procs,
            "gpus": args.gpus,
        }
        run_batch(tasks, args.out_json, args.overwrite, worker_onealign, args_dict)
        return

    if args.mode == "aesthetic":
        dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        tasks = build_tasks_aesthetic(args.image_dir)
        args_dict = {
            "backend": args.backend,
            "device": args.device,
            "num_procs": args.num_procs,
            "gpus": args.gpus,
            "laion_clip_model": args.laion_clip_model,
            "laion_clip_ckpt": args.laion_clip_ckpt,
            "laion_pretrained_tag": args.laion_pretrained_tag,
            "laion_linear_path": args.laion_linear_path,
            "v25_encoder_model_name": args.v25_encoder_model_name,
            "dtype": dtype,
        }
        run_batch(tasks, args.out_json, args.overwrite, worker_aesthetic, args_dict)
        return


if __name__ == "__main__":
    main()
