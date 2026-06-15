# omnistyle_single.py
import argparse, dataclasses, torch
from PIL import Image
from omnistyle.flux.pipeline import DSTPipeline   # 确保项目路径在 PYTHONPATH
@dataclasses.dataclass
class Cfg:
    model_type: str = "flux-dev"
    width: int = 1024
    height: int = 1024
    steps: int = 25
    guidance: float = 4
    seed: int = 0
    pe: str = "d"

def find_closest_size(width, height):
    #找到最接近的能被16整除的尺寸
    target_width = round(width / 16) * 16
    target_height = round(height / 16) * 16
    return target_width, target_height

def run(content: str, style: str, out: str):
    cfg = Cfg()
    pipe = DSTPipeline(cfg.model_type, torch.device("cuda"), False, only_lora=True, lora_rank=512)
    c = Image.open(content).convert("RGB")
    width, height = c.size
    s = Image.open(style).convert("RGB")
    c = c.resize((cfg.width, cfg.height))
    s = s.resize((cfg.width, cfg.height))
    width, height = find_closest_size(1024, 1024)
    gen = pipe(prompt="", width=width, height=height,
               guidance=cfg.guidance, num_steps=cfg.steps,
               seed=cfg.seed, ref_imgs=[s, c], pe=cfg.pe)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    gen.save(out)
    print("saved →", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("content")
    ap.add_argument("style")
    ap.add_argument("out")
    run(**vars(ap.parse_args()))