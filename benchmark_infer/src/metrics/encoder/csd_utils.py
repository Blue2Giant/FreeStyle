import os
from typing import Optional
from PIL import Image
from typing import Optional
import torch
torch.cuda.empty_cache()
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.functional as F
from torchvision import transforms
from transformers import (AutoModel, AutoProcessor, AutoTokenizer, AutoConfig,
                            CLIPImageProcessor, CLIPVisionModelWithProjection)
# from qwen_vl_utils import process_vision_info

torch.manual_seed(42) 
torch.cuda.manual_seed_all(42)

import torch
import torch.nn as nn
import clip
import copy
from torch.autograd import Function

from collections import OrderedDict

def convert_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k.replace("module.", "")
        new_state_dict[k] = v
    return new_state_dict

def convert_weights_float(model: nn.Module):
    """Convert applicable model parameters to fp32"""

    def _convert_weights_to_fp32(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp32)


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim,
        dropout=0
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


def init_weights(m): # TODO: do we need init for layernorm?
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.normal_(m.bias, std=1e-6)


class CSD_CLIP(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='vit_large',content_proj_head='default', model_path=None):
        super(CSD_CLIP, self).__init__()
        self.content_proj_head = content_proj_head
        if name == 'vit_large':
            if model_path is None:
                try:
                    clipmodel, _ = clip.load("/mnt/jfs/model_zoo/open_clip/ViT-L-14-openai.pt")
                except Exception:
                    clipmodel, _ = clip.load("ViT-L/14")
            else:
                try:
                    import open_clip
                    clipmodel, _, _ = open_clip.create_model_and_transforms(
                        "ViT-L-14",
                        pretrained=model_path,
                        device="cpu",
                    )
                except Exception:
                    try:
                        clipmodel, _ = clip.load(model_path)
                    except Exception:
                        clipmodel, _ = clip.load("ViT-L/14")
            self.backbone = clipmodel.visual
            self.embedding_dim = getattr(self.backbone, "output_dim", 1024)
        else:
            raise Exception('This model is not implemented')

        convert_weights_float(self.backbone)
        self.last_layer_style = copy.deepcopy(self.backbone.proj)
        if content_proj_head == 'custom':
            self.last_layer_content = ProjectionHead(self.embedding_dim,self.feat_dim)
            self.last_layer_content.apply(init_weights)
            
        else:
            self.last_layer_content = copy.deepcopy(self.backbone.proj)

        self.backbone.proj = None

    @property
    def dtype(self):
        return self.backbone.conv1.weight.dtype

    def forward(self, input_data, alpha=None):
        
        feature = self.backbone(input_data)

        if alpha is not None:
            reverse_feature = ReverseLayerF.apply(feature, alpha)
        else:
            reverse_feature = feature

        style_output = feature @ self.last_layer_style
        style_output = nn.functional.normalize(style_output, dim=1, p=2)

        # if alpha is not None:
        if self.content_proj_head == 'custom':
            content_output =  self.last_layer_content(reverse_feature)
        else:
            content_output = reverse_feature @ self.last_layer_content
        content_output = nn.functional.normalize(content_output, dim=1, p=2)
        return feature, content_output, style_output

class CSDStyleEmbedding:
    def __init__(self, model_path: str = "scripts/style/models/checkpoint.pth", device: str = "cuda", clip_model_path: Optional[str] = None):
        self.device = torch.device(device)
        self.model = self._load_model(model_path, clip_model_path).to(self.device)
        self.preprocess = transforms.Compose([
            transforms.Resize(size=224, interpolation=F.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def _load_model(self, model_path: str, clip_model_path: Optional[str] = None):
        model = CSD_CLIP("vit_large", "default", model_path=clip_model_path)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = convert_state_dict(checkpoint['model_state_dict'])
        model.load_state_dict(state_dict, strict=False)
        return model

    def get_style_embedding(self, image: Image.Image):
        # image = Image.open(image_path).convert('RGB')
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, style_output = self.model(image_tensor)
        return style_output.squeeze(0).cpu().numpy().tolist()


class SEStyleEmbedding:
    def __init__(self, pretrained_path: str = "xingpng/OneIG-StyleEncoder", device: str = "cuda", dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(pretrained_path)
        self.image_encoder.to(self.device, dtype=self.dtype)
        self.image_encoder.eval()
        self.processor = CLIPImageProcessor()

    def _l2_normalize(self, x):
        return torch.nn.functional.normalize(x, p=2, dim=-1)

    def get_style_embedding(self, image: Image.Image):
        # image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device, dtype=self.dtype)

        with torch.no_grad():
            outputs = self.image_encoder(inputs)
            image_embeds = outputs.image_embeds
            image_embeds_norm = self._l2_normalize(image_embeds)
        return image_embeds_norm.squeeze(0).cpu().numpy().tolist()


# class LLM2CLIP:
#     def __init__(self, processor_model="openai/clip-vit-large-patch14-336", 
#                  model_name="microsoft/LLM2CLIP-Openai-L-14-336", 
#                  llm_model_name="microsoft/LLM2CLIP-Llama-3-8B-Instruct-CC-Finetuned", 
#                  device='cuda'):
#         # Initialize processor and models
#         self.processor = CLIPImageProcessor.from_pretrained(processor_model)

#         self.model = AutoModel.from_pretrained(
#             model_name, 
#             torch_dtype=torch.bfloat16,
#             trust_remote_code=True
#         ).to(device).eval()

#         self.llm_model_name = llm_model_name
#         self.config = AutoConfig.from_pretrained(
#             self.llm_model_name, trust_remote_code=True
#         )
#         self.llm_model = AutoModel.from_pretrained(
#             self.llm_model_name, torch_dtype=torch.bfloat16, config=self.config, trust_remote_code=True
#         )
#         self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
        
#         self.llm_model.config._name_or_path = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Workaround for LLM2VEC
        
#         from scripts.utils.llm2clip.llm2vec import LLM2Vec
        
#         self.l2v = LLM2Vec(self.llm_model, self.tokenizer, pooling_mode="mean", max_length=512, doc_max_length=512)

#         self.device = device

#     def text_img_similarity_score(self, image_path_list, text_prompt):
#         try:
#             captions = [text_prompt]
#             images = [Image.open(image_path) for image_path in image_path_list]
            
#             # Process images and encode text
#             input_pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
#             text_features = self.l2v.encode(captions, convert_to_tensor=True).to(self.device)

#             # Get image and text features
#             with torch.no_grad(), torch.amp.autocast(self.device):
#                 image_features = self.model.get_image_features(input_pixels)
#                 text_features = self.model.get_text_features(text_features)

#                 # Normalize features
#                 image_features /= image_features.norm(dim=-1, keepdim=True)
#                 text_features /= text_features.norm(dim=-1, keepdim=True)

#                 # Compute similarity score (dot product)
#                 text_probs = image_features @ text_features.T
#                 text_probs = text_probs.cpu().tolist()

#             return [text_prob[0] for text_prob in text_probs]
#         except Exception as e:
#             print(f"Error: {e}")
#             return None
