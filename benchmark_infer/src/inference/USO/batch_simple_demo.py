import os
os.environ["FLUX_DEV"] = "/data/USO/weights/FLUX.1-dev/flux1-dev.safetensors"
os.environ["AE"] = "/data/USO/weights/FLUX.1-dev/ae.safetensors"
os.environ["LORA"] = "/data/USO/weights/USO/uso_flux_v1.0/dit_lora.safetensors"
os.environ["PROJECTION_MODEL"] = "/data/USO/weights/USO/uso_flux_v1.0/projector.safetensors"
os.environ["SIGLIP_PATH"] = "/data/USO/weights/siglip"
os.environ["T5"] = "/data/USO/weights/t5-xxl"
os.environ["CLIP"] = "/mnt/jfs/model_zoo/clip-vit-large-patch14"

import argparse
import json
from typing import Dict, List

import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor
from tqdm import tqdm
from uso.flux.pipeline import USOPipeline, preprocess_ref

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def list_images_map(dir_path: str, exts: List[str]) -> Dict[str, str]:
    if not os.path.isdir(dir_path):
        return {}
    out: Dict[str, str] = {}
    for root, _, files in os.walk(dir_path):
        for fn in files:
            low = fn.lower()
            if any(low.endswith(e) for e in exts):
                base = os.path.splitext(fn)[0]
                if base not in out:
                    out[base] = os.path.join(root, fn)
    return out


def load_image(path: str) -> Image.Image:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--prompts-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--exts", default=".png,.jpg,.jpeg,.webp,.bmp,.tiff,.avif,.heic")
    ap.add_argument("--num-steps", type=int, default=25)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--instruct-edit", action="store_true")
    ap.add_argument("--content-ref-size", type=int, default=512)
    ap.add_argument("--pe", default="d")
    ap.add_argument("--model-type", default="flux-dev")
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--lora-rank", type=int, default=128)
    ap.add_argument("--hf-download", action="store_true")
    ap.add_argument("--save-attn", action="store_true")
    ap.add_argument("--save-attn-path", default="")
    ap.add_argument("--use-siglip", action="store_true")
    ap.add_argument("--sref-only", action="store_true")
    ap.add_argument("--no-preprocess-ref", action="store_true",
                    help="Do not resize/crop the content reference before VAE encoding; pass the input image at its original pixel size.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Total number of parallel shards.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="This worker's 0-based shard index.")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exts = [x.strip().lower() for x in args.exts.split(",") if x.strip()]

    cref_dir = os.path.join(args.input_dir, "cref")
    sref_dir = os.path.join(args.input_dir, "sref")
    cref_map = list_images_map(cref_dir, exts)
    sref_map = list_images_map(sref_dir, exts)

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        prompts_obj = json.load(f)
    if not isinstance(prompts_obj, dict):
        raise RuntimeError("prompts-json 不是字典")

    keys = sorted(set(prompts_obj.keys()) & set(cref_map.keys()) & set(sref_map.keys()))
    missing = sorted(set(prompts_obj.keys()) - set(keys))
    if missing:
        print(f"[warn] 缺少配对的key数量: {len(missing)}")

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    all_key_count = len(keys)
    if args.num_shards > 1:
        keys = [k for i, k in enumerate(keys) if i % args.num_shards == args.shard_index]
        print(f"[info] shard {args.shard_index}/{args.num_shards}: {len(keys)}/{all_key_count} samples")

    if not keys:
        raise RuntimeError("没有可处理的样本")

    os.makedirs(args.out_dir, exist_ok=True)
    print("initializing pipeline......")
    pipe = USOPipeline(
        args.model_type,
        device,
        args.offload,
        only_lora=True,
        lora_rank=args.lora_rank,
        hf_download=args.hf_download,
        save_attn=args.save_attn,
    )

    siglip_processor = None
    if args.use_siglip:
        siglip_path = os.getenv("SIGLIP_PATH", "google/siglip-so400m-patch14-384")
        siglip_processor = SiglipImageProcessor.from_pretrained(siglip_path)
        siglip_model = SiglipVisionModel.from_pretrained(siglip_path).to(device).eval()
        pipe.model.vision_encoder = siglip_model

    processed = 0
    skipped = 0
    failed = 0
    total = len(keys)
    it = tqdm(keys, total=total, unit="img") if tqdm else keys
    for b in it:
        try:
            out_path = os.path.join(args.out_dir, b + ".png")
            if (not args.overwrite) and os.path.exists(out_path):
                skipped += 1
                continue
            id_img = load_image(cref_map[b])
            style_img = load_image(sref_map[b])
            if args.no_preprocess_ref:
                if id_img.size != (16, 16) and (id_img.size[0] % 16 != 0 or id_img.size[1] % 16 != 0):
                    raise ValueError(f"--no-preprocess-ref requires content ref size to be divisible by 16, got {id_img.size}")
                ref_imgs_pil = [id_img]
            else:
                ref_imgs_pil = [preprocess_ref(id_img, args.content_ref_size)]
            siglip_inputs = []
            if args.use_siglip and siglip_processor is not None:
                with torch.no_grad():
                    siglip_inputs.append(siglip_processor(style_img, return_tensors="pt").to(pipe.device))
            w, h = args.width, args.height
            if args.instruct_edit and len(ref_imgs_pil) > 0:
                w, h = ref_imgs_pil[0].size
            if args.sref_only:
                prompt = ""
            else:
                prompt = str(prompts_obj[b])
            img = pipe(
                prompt=prompt,
                width=w,
                height=h,
                guidance=args.guidance,
                num_steps=args.num_steps,
                seed=args.seed,
                ref_imgs=ref_imgs_pil,
                pe=args.pe,
                siglip_inputs=siglip_inputs,
                save_attn_path=args.save_attn_path if args.save_attn else None,
            )
            img.save(out_path)
            processed += 1
            if (not tqdm) and (processed % 10 == 0 or processed == total):
                print(f"[progress] {processed}/{total}")
        except Exception as e:
            failed += 1
            print(f"[warn] skip {b}: {e}")

    if tqdm and hasattr(it, "close"):
        it.close()
    print(f"[done] processed={processed} skipped={skipped} failed={failed} total={total} -> {args.out_dir}")


if __name__ == "__main__":
    main()
