import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import torch
from ip_adapter.utils import BLOCKS as BLOCKS
from ip_adapter.utils import controlnet_BLOCKS as controlnet_BLOCKS
from ip_adapter.utils import resize_content
import cv2
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,

)
from ip_adapter import CSGO
from transformers import BlipProcessor, BlipForConditionalGeneration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_root = "/mnt/jfs/model_zoo/CSGO"
base_model_path =  "/mnt/jfs/model_zoo/stable-diffusion-xl-base-1.0"
image_encoder_path = "/mnt/jfs/model_zoo/IP-Adapter/sdxl_models/image_encoder"
csgo_ckpt = "/mnt/jfs/model_zoo/CSGO/csgo_4_32.bin"
pretrained_vae_name_or_path = "/mnt/jfs/model_zoo/sdxl-vae-fp16-fix"
controlnet_path = "/mnt/jfs/model_zoo/TTPLanet_SDXL_Controlnet_Tile_Realistic"
weight_dtype = torch.float16

blip_processor = BlipProcessor.from_pretrained("/mnt/jfs/model_zoo/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("/mnt/jfs/model_zoo/blip-image-captioning-large",
                                                          ).to(device)


vae = AutoencoderKL.from_pretrained(pretrained_vae_name_or_path,torch_dtype=torch.float16)
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16,use_safetensors=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
    vae=vae
)
pipe.enable_vae_tiling()


target_content_blocks = BLOCKS['content']
target_style_blocks = BLOCKS['style']
controlnet_target_content_blocks = controlnet_BLOCKS['content']
controlnet_target_style_blocks = controlnet_BLOCKS['style']

csgo = CSGO(pipe, image_encoder_path, csgo_ckpt, device, num_content_tokens=4,num_style_tokens=32,
                          target_content_blocks=target_content_blocks, target_style_blocks=target_style_blocks,controlnet_adapter=True,
                              controlnet_target_content_blocks=controlnet_target_content_blocks,
                              controlnet_target_style_blocks=controlnet_target_style_blocks,
                              content_model_resampler=True,
                              style_model_resampler=True,
                              )

style_name = '/data/benchmark_metrics/assets/style.webp'
content_name = '/data/benchmark_metrics/assets/content.webp'
style_image = Image.open(style_name).convert('RGB')
content_image = Image.open(content_name).convert('RGB')


with torch.no_grad():
    inputs = blip_processor(content_image, return_tensors="pt").to(device)
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

num_sample=1

width,height,content_image  = resize_content(content_image)
images = csgo.generate(pil_content_image= content_image, pil_style_image=style_image,
                           prompt=caption,
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           height=height,
                           width=width,
                           content_scale=0.5,
                           style_scale=1.0,
                           guidance_scale=10,
                           num_images_per_prompt=num_sample,
                           num_samples=1,
                           num_inference_steps=50,
                           seed=42,
                           image=content_image.convert('RGB'),
                           controlnet_conditioning_scale=0.6,
                          )
images[0].save("/data/Depth-Anything/CSGO/assets/content_img_0_style_imag_1.png")