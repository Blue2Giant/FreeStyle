# You can use the following code to call our trained style encoder. Hope it helps.
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, AutoConfig,
                            CLIPImageProcessor, CLIPVisionModelWithProjection)
import torch
from PIL import Image
class SEStyleEmbedding:
    def __init__(self, pretrained_path: str = "xingpng/OneIG-StyleEncoder", device: str = "cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        self.image_encoder.to(self.device, dtype=self.dtype)
        self.image_encoder.eval()
        self.processor = CLIPImageProcessor()

    def _l2_normalize(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def get_style_embedding(self, image_path: str):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.image_encoder(inputs)
            image_embeds = outputs.image_embeds
            image_embeds_norm = self._l2_normalize(image_embeds)
        return image_embeds_norm

if __name__ == "__main__":
    style_encoder = SEStyleEmbedding(pretrained_path="/mnt/jfs/model_zoo/OneIG-StyleEncoder", device="cuda", dtype=torch.bfloat16)
    style_embedding = style_encoder.get_style_embedding("/data/benchmark_metrics/assets/style.webp")# shape=(1,1280)
    print(style_embedding.shape)