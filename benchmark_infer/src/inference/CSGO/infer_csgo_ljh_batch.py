import argparse
import os
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from ip_adapter.utils import resize_content
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from diffusers import AutoencoderKL, ControlNetModel, StableDiffusionXLControlNetPipeline
from ip_adapter import CSGO
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cref_dir", required=True)
    p.add_argument("--sref_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--ids", type=str, default="")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=10.0)
    p.add_argument("--content_scale", type=float, default=0.5)
    p.add_argument("--style_scale", type=float, default=1.0)
    p.add_argument("--controlnet_conditioning_scale", type=float, default=0.6)
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def list_images(dir_path: Path):
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    result = {}
    for p in dir_path.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            result[p.stem] = p
    return result


def build_csgo(device: torch.device):
    base_model_path = "/mnt/jfs/model_zoo/stable-diffusion-xl-base-1.0"
    image_encoder_path = "/mnt/jfs/model_zoo/IP-Adapter/sdxl_models/image_encoder"
    csgo_ckpt = "/mnt/jfs/model_zoo/CSGO/csgo_4_32.bin"
    pretrained_vae_name_or_path = "/mnt/jfs/model_zoo/sdxl-vae-fp16-fix"
    controlnet_path = "/mnt/jfs/model_zoo/TTPLanet_SDXL_Controlnet_Tile_Realistic"
    weight_dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path, torch_dtype=weight_dtype)
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=weight_dtype, use_safetensors=True)
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=weight_dtype,
        add_watermarker=False,
        vae=vae,
    )
    pipe.enable_vae_tiling()

    target_content_blocks = BLOCKS["content"]
    target_style_blocks = BLOCKS["style"]
    controlnet_target_content_blocks = controlnet_BLOCKS["content"]
    controlnet_target_style_blocks = controlnet_BLOCKS["style"]

    csgo = CSGO(
        pipe,
        image_encoder_path,
        csgo_ckpt,
        device,
        num_content_tokens=4,
        num_style_tokens=32,
        target_content_blocks=target_content_blocks,
        target_style_blocks=target_style_blocks,
        controlnet_adapter=True,
        controlnet_target_content_blocks=controlnet_target_content_blocks,
        controlnet_target_style_blocks=controlnet_target_style_blocks,
        content_model_resampler=True,
        style_model_resampler=True,
    )
    return csgo


def main():
    args = parse_args()
    cref_dir = Path(args.cref_dir)
    sref_dir = Path(args.sref_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cref_map = list_images(cref_dir)
    sref_map = list_images(sref_dir)
    common = sorted(set(cref_map.keys()) & set(sref_map.keys()))
    if args.ids.strip():
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        common = [k for k in common if k in wanted]
    if args.limit and args.limit > 0:
        common = common[: args.limit]
    if not common:
        raise SystemExit("no matched pairs found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_processor = BlipProcessor.from_pretrained("/mnt/jfs/model_zoo/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("/mnt/jfs/model_zoo/blip-image-captioning-large").to(device)
    csgo = build_csgo(device)

    with torch.inference_mode():
        for k in tqdm(common, unit="img"):
            out_path = out_dir / f"{k}.png"
            if args.skip_existing and out_path.exists():
                continue
            style_image = Image.open(sref_map[k]).convert("RGB")
            content_image = Image.open(cref_map[k]).convert("RGB")

            inputs = blip_processor(content_image, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)

            width, height, content_image = resize_content(content_image)
            images = csgo.generate(
                pil_content_image=content_image,
                pil_style_image=style_image,
                prompt=caption,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                content_scale=args.content_scale,
                style_scale=args.style_scale,
                guidance_scale=args.guidance_scale,
                num_images_per_prompt=1,
                num_samples=1,
                num_inference_steps=args.steps,
                seed=args.seed,
                image=content_image.convert("RGB"),
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            )
            images[0].save(out_path)


if __name__ == "__main__":
    main()
