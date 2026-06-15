import torch
import os
from typing import List, Tuple
from PIL import Image
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from huggingface_hub import hf_hub_download




PREFERRED_KONTEXT_RESOLUTIONS: List[Tuple[int, int]] = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def _lanczos():
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def resize_like_kontext_bucket(img: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
    w, h = img.size
    aspect_ratio = w / float(h)
    _, target_w, target_h = min(
        (abs(aspect_ratio - (rw / float(rh))), rw, rh)
        for (rw, rh) in PREFERRED_KONTEXT_RESOLUTIONS
    )
    if (w, h) == (target_w, target_h):
        return img, (target_w, target_h)
    return img.resize((target_w, target_h), resample=_lanczos()), (target_w, target_h)


def parse_resolution(spec: str) -> Tuple[int, int]:
    text = str(spec).strip().lower()
    if "x" not in text:
        raise ValueError(f"invalid resolution: {spec}")
    w_str, h_str = text.split("x", 1)
    w = int(w_str)
    h = int(h_str)
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid resolution: {spec}")
    return w, h


def output_size_from_resolution(output_resolution: str) -> Tuple[int, int] | None:
    if not output_resolution:
        return None
    w, h = parse_resolution(output_resolution)
    w = w - w % 16
    h = h - h % 16
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid output resolution after /16 alignment: {output_resolution}")
    return w, h

class ImageStyleInference:
   
    def __init__(self,):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_models()
    
    def _load_models(self):
        os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/jfs/model_zoo")
        os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
        os.environ.setdefault("DIFFSYNTH_DOWNLOAD_SOURCE", "huggingface")

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
                download_source='huggingface',
                origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
                download_source='huggingface',origin_file_pattern="text_encoder/model*.safetensors"),
                ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
                download_source='huggingface',origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
            ],
            tokenizer_config=None,
            processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit-2509", 
            download_source='huggingface',origin_file_pattern="processor/"),
        )



        telestyle_base = os.getenv("TELESTYLE_DIR", "/mnt/jfs/model_zoo/Tele-AI/TeleStyle")
        telestyle_image = os.path.join(
            telestyle_base, "weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors"
        )
        speedup = os.path.join(
            telestyle_base, "weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors"
        )
        if not os.path.isfile(telestyle_image):
            telestyle_image = hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-telestyle.safetensors",
                local_files_only=True,
            )
        if not os.path.isfile(speedup):
            speedup = hf_hub_download(
                repo_id="Tele-AI/TeleStyle",
                filename="weights/diffsynth_Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors",
                local_files_only=True,
            )
        #https://huggingface.co/lightx2v/Qwen-Image-Lightning converted to diffsynth format

        self.pipe.load_lora(self.pipe.dit, telestyle_image)
        self.pipe.load_lora(self.pipe.dit, speedup)

    def inference(self,
        prompt,
        content_ref,
        style_ref,
        seed=123,
        num_inference_steps=4,
        minedge=1024,
        output_resolution="",
        ):
        # Input/reference images use the same bucket logic as flux_klein_9B.py.
        # output_resolution controls only the sampled noise/output canvas.
        with Image.open(content_ref) as img:
            content_img = img.convert("RGB").copy()
        with Image.open(style_ref) as img:
            style_img = img.convert("RGB").copy()

        content_img, content_size = resize_like_kontext_bucket(content_img)
        style_img, _style_size = resize_like_kontext_bucket(style_img)

        output_size = output_size_from_resolution(output_resolution)
        if output_size is None:
            output_size = content_size
        w, h = output_size

        image = self.pipe(
            prompt,
            edit_image=[content_img, style_img],
            seed=seed,
            num_inference_steps=num_inference_steps,
            height=h,
            width=w,
            edit_image_auto_resize=False,
            cfg_scale=1.0
        )  # lightning

        if image.size != (w, h):
            image = image.resize((w, h), resample=_lanczos())
        return image




if __name__ == "__main__":
    inference_engine = ImageStyleInference()

    prompt = 'Style Transfer the style of Figure 2 to Figure 1, and keep the content and characteristics of Figure 1.'
        
    content_ref='/data/benchmark_metrics/assets/content.webp' #content reference image
    style_ref='/data/benchmark_metrics/assets/style.webp'#style reference image
    
    with torch.no_grad():
        generated_image = inference_engine.inference(prompt, content_ref, style_ref, seed=123, num_inference_steps=4, minedge=1024)

    save_dir=f'./qwen_style_output/'

    os.makedirs(save_dir,exist_ok=True)
    prefix=style_ref.split('/')[-1].split('.')[0]


    generated_image.save(os.path.join(save_dir, f'{prefix}_result.png'))


    print(f"saved to {os.path.join(save_dir, f'{prefix}_result.png')}")
            
