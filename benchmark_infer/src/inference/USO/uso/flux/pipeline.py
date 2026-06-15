# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.
# Copyright (c) 2024 Black Forest Labs and The XLabs-AI Team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
import numpy as np
import os
import math
from typing import Literal, Optional
from torch import Tensor
import torch.nn as nn
import torch
from einops import rearrange
from PIL import ExifTags, Image
import torchvision.transforms.functional as TVF
from tqdm import tqdm
from uso.flux.modules.layers import (
    DoubleStreamBlockLoraProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    SingleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor_hotmap,
    DoubleStreamBlockLoraProcessor_hotmap,
    SingleStreamBlock,
    DoubleStreamBlock,
    LastLayer
)
import inspect
from uso.flux.sampling import denoise, get_noise, get_schedule, prepare_multi_ip, unpack
from uso.flux.util import (
    get_lora_rank,
    load_ae,
    load_checkpoint,
    load_clip,
    load_flow_model,
    load_flow_model_only_lora,
    load_t5,
)
from uso.flux.for_replace import  pipeline_forward
def replace_processor_module(root: nn.Module):
    """
    逐级替换root模块下所有的processor 
    """
    def make_new(old_module, new_module_class):
        #替换模块，但是因为模块的参数不变，只是forward函数变了，因此要保证参数的权重一样
        if isinstance(old_module,SingleStreamBlockLoraProcessor):
            new_module = new_module_class(dim=getattr(old_module.qkv_lora.down,'in_features', None), 
                                        rank=getattr(old_module.qkv_lora.down, 'out_features', None),
                                        network_alpha = getattr(old_module.qkv_lora, 'network_alpha', None),
                                        lora_weight = getattr(old_module, 'lora_weight', 1.0))
        else:
            new_module = new_module_class(dim=getattr(old_module.qkv_lora1.down,'in_features', None), 
                                        rank=getattr(old_module.qkv_lora1.down, 'out_features', None),
                                        network_alpha = getattr(old_module.qkv_lora1, 'network_alpha', None),
                                        lora_weight = getattr(old_module, 'lora_weight', 1.0))
        new_module.load_state_dict(old_module.state_dict())
        # Use the device of one of the parameters as reference
        param_device = next(old_module.parameters()).device
        new_module.to(param_device) #move to same device
        return new_module
        
    # 深度优先替换 children
    for name, child in list(root.named_children()):
        #如果children的名称中含有processor，则替换
        #得到layer的class类型 
        # print(f"Visiting module: {name}, class: {child.__class__.__name__}")
        if "processor" in name and isinstance(child,SingleStreamBlockLoraProcessor):
            setattr(root, name, make_new(child, SingleStreamBlockLoraProcessor_hotmap))  # 真正替换
        elif "processor" in name and isinstance(child,DoubleStreamBlockLoraProcessor):
            setattr(root, name, make_new(child, DoubleStreamBlockLoraProcessor_hotmap))  # 真正替换
        replace_processor_module(child)

def find_nearest_scale(image_h, image_w, predefined_scales):
    """
    根据图片的高度和宽度，找到最近的预定义尺度。

    :param image_h: 图片的高度
    :param image_w: 图片的宽度
    :param predefined_scales: 预定义尺度列表 [(h1, w1), (h2, w2), ...]
    :return: 最近的预定义尺度 (h, w)
    """
    # 计算输入图片的长宽比
    image_ratio = image_h / image_w

    # 初始化变量以存储最小差异和最近的尺度
    min_diff = float("inf")
    nearest_scale = None

    # 遍历所有预定义尺度，找到与输入图片长宽比最接近的尺度
    for scale_h, scale_w in predefined_scales:
        predefined_ratio = scale_h / scale_w
        diff = abs(predefined_ratio - image_ratio)

        if diff < min_diff:
            min_diff = diff
            nearest_scale = (scale_h, scale_w)

    return nearest_scale


def preprocess_ref(raw_image: Image.Image, long_size: int = 512, scale_ratio: int = 1):
    # 获取原始图像的宽度和高度
    image_w, image_h = raw_image.size
    if image_w == image_h and image_w == 16:
        return raw_image

    # 计算长边和短边
    if image_w >= image_h:
        new_w = long_size
        new_h = int((long_size / image_w) * image_h)
    else:
        new_h = long_size
        new_w = int((long_size / image_h) * image_w)

    # 按新的宽高进行等比例缩放
    raw_image = raw_image.resize((new_w, new_h), resample=Image.LANCZOS)

    # 为了能让canny img进行scale
    scale_ratio = int(scale_ratio)
    target_w = new_w // (16 * scale_ratio) * (16 * scale_ratio)
    target_h = new_h // (16 * scale_ratio) * (16 * scale_ratio)

    # 计算裁剪的起始坐标以实现中心裁剪
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    right = left + target_w
    bottom = top + target_h

    # 进行中心裁剪
    raw_image = raw_image.crop((left, top, right, bottom))

    # 转换为 RGB 模式
    raw_image = raw_image.convert("RGB")
    return raw_image

def get_arg_by_name(module, args, kwargs, name, default=None):
    # 1) if passed as keyword
    if name in kwargs:
        return kwargs[name]
    # 2) else map positional args to parameter names via the forward signature
    sig = inspect.signature(module.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    name_to_pos = {p.name: i for i, p in enumerate(params)}
    if name in name_to_pos and name_to_pos[name] < len(args):
        return args[name_to_pos[name]]
    return default

def pre_hook_get_vec(module, **kwargs):
    module._last_vec = kwargs.get("vec", None)
    print("pre_hook_get_vec",module._last_vec)

def resize_and_centercrop_image(image, target_height_ref1, target_width_ref1):
    target_height_ref1 = int(target_height_ref1 // 64 * 64)
    target_width_ref1 = int(target_width_ref1 // 64 * 64)
    h, w = image.shape[-2:]
    if h < target_height_ref1 or w < target_width_ref1:
        # 计算长宽比
        aspect_ratio = w / h
        if h < target_height_ref1:
            new_h = target_height_ref1
            new_w = new_h * aspect_ratio
            if new_w < target_width_ref1:
                new_w = target_width_ref1
                new_h = new_w / aspect_ratio
        else:
            new_w = target_width_ref1
            new_h = new_w / aspect_ratio
            if new_h < target_height_ref1:
                new_h = target_height_ref1
                new_w = new_h * aspect_ratio
    else:
        aspect_ratio = w / h
        tgt_aspect_ratio = target_width_ref1 / target_height_ref1
        if aspect_ratio > tgt_aspect_ratio:
            new_h = target_height_ref1
            new_w = new_h * aspect_ratio
        else:
            new_w = target_width_ref1
            new_h = new_w / aspect_ratio
    # 使用 TVF.resize 进行图像缩放
    image = TVF.resize(image, (math.ceil(new_h), math.ceil(new_w)))
    # 计算中心裁剪的参数
    top = (image.shape[-2] - target_height_ref1) // 2
    left = (image.shape[-1] - target_width_ref1) // 2
    # 使用 TVF.crop 进行中心裁剪
    image = TVF.crop(image, top, left, target_height_ref1, target_width_ref1)
    return image


class USOPipeline:
    def __init__(
        self,
        model_type: str,
        device: torch.device,
        offload: bool = False,
        only_lora: bool = False,#默认是true
        lora_rank: int = 16,
        hf_download: bool = True,
        save_attn: bool = False,
    ):
        self.device = device
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        self.use_fp8 = "fp8" in model_type
        if only_lora:
            #替换了processor，DoubleStreamBlockLoraProcessor，SingleStreamBlockProcessor
            self.model = load_flow_model_only_lora(
                model_type,
                device="cpu" if offload else self.device,
                lora_rank=lora_rank,
                use_fp8=self.use_fp8,
                hf_download=hf_download,
            )
        else:
            self.model = load_flow_model(
                model_type, device="cpu" if offload else self.device
            )
        #write model parameters names and shape into a file 
        with open("model_parameters.txt", "w") as f:
            for name, param in self.model.named_parameters():
                # Get the module class name for the parameter
                module_name = ".".join(name.split(".")[:-1])
                module = dict(self.model.named_modules()).get(module_name, None)
                class_name = module.__class__.__name__ if module is not None else param.__class__.__name__
                f.write(f"{name}: {param.shape},{class_name}\n")
    
        ## 设置hook进行attenion map的可视化
        self.handles = []#用来存储挂上的钩子，可以方便的删除
        self.model.hotmap_list = []
        self.model.index_dict = {}
        if save_attn == True:  #是否保存attn map
            ## 替换model中的processor模块，用来得到attnhotmap
            print("replace processor module to save attn hotmap...")
            replace_processor_module(self.model)
            self.set_hook_for_save_attn(self.model,self.model.hotmap_list)
            self.model.hidden_states_list = []
            print("set hook for save hidden states...")
            self.set_hook_for_save_hidden_states(self.model,self.model.hidden_states_list)
            print("replace pipeline forward function to save attn hotmap...")
            self.forward = pipeline_forward.__get__(self)
            print('get img token start and end index with hook')
            self.set_hook_for_get_index(self.model,self.model.index_dict)
    
    def set_hook_for_save_hidden_states(self, model: nn.Module, hidden_states_list: list):
        def decode_per_hidden_states(module, args, kwargs, output):
            # Only save for relevant block types
            if isinstance(module, DoubleStreamBlock):
                out = output[0]  # img
            elif isinstance(module, SingleStreamBlock):
                out = output
            else:
                return
            # #因为double block的img其实是拼接了ref_img经过siglip的输出因此，这里的img length太长了
            # vec = kwargs.get("vec", None)
            # img = kwargs.get("img", None)
            # txt = kwargs.get("txt", None)
            # x = kwargs.get("x", None)
            vec = kwargs.get("vec", None)
            hidden_states_list.append({
                "name": module.__class__.__name__,
                "output": out,
                'vec' : vec
            })
        # Register hook for each relevant block
        for name, submodule in model.named_modules():
            if isinstance(submodule, (DoubleStreamBlock, SingleStreamBlock)):
                submodule.register_forward_hook(decode_per_hidden_states,with_kwargs=True)

    def remove_hooks(self):
        """移除所有注册的钩子"""
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
    
    def set_hook_for_get_index(self, module, index_dict):
        """
        保存img token的长度
        """
        def get_img_token_length(module, args, kwargs, output):
            img = kwargs.get("img", None)
            txt = kwargs.get("txt", None)
            img = module.img_in(img)
            siglip_inputs = kwargs.get("siglip_inputs", None)
            txt = module.txt_in(txt)
            #txt hidden 拼接了siglip的feature
            #ref 图同时也经过了vae和 img_in,拼接到img上
            if module.feature_embedder is not None and siglip_inputs is not None and len(siglip_inputs) > 0 and module.vision_encoder is not None:
                siglip_embedding = [module.vision_encoder(**emb, output_hidden_states=True) for emb in siglip_inputs]
                siglip_embedding = torch.cat([module.feature_embedder(emb) for emb in siglip_embedding], dim=1)
                txt = torch.cat((siglip_embedding, txt), dim=1)
            index_dict['txt_len'] = txt.shape[1]
            index_dict['img_len'] = img.shape[1]
        module.register_forward_hook(get_img_token_length,with_kwargs=True)
    
    def set_hook_for_save_attn(self, model: nn.Module, hotmap_list: list):
        """
        注册钩子，保存每个 block.processor 的 hotmap 到 hotmap_list。
        """
        def save_hotmap(name, proc):
            hotmap = getattr(proc, "hotmap", None)
            vec = getattr(proc, "vec", None)
            if hotmap is not None:
                hotmap_list.append({
                    "name": name,
                    "map": hotmap,
                })

        for name, module in model.named_modules():
            if isinstance(module, (SingleStreamBlock, DoubleStreamBlock)):
                def hook(mod, inp, out, n=name):
                    proc = getattr(mod, "processor", None)
                    if proc is not None:
                        save_hotmap(n, proc)
                h = module.register_forward_hook(hook)
                self.handles.append(h)

    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            from safetensors.torch import load_file as load_sft

            print("Loading checkpoint to replace old keys")
            # load_sft doesn't support torch.device
            if ckpt_path.endswith("safetensors"):
                sd = load_sft(ckpt_path, device="cpu")
                missing, unexpected = self.model.load_state_dict(
                    sd, strict=False, assign=True
                )
            else:
                dit_state = torch.load(ckpt_path, map_location="cpu")
                sd = {}
                for k in dit_state.keys():
                    sd[k.replace("module.", "")] = dit_state[k]
                missing, unexpected = self.model.load_state_dict(
                    sd, strict=False, assign=True
                )
            self.model.to(str(self.device))
            print(f"missing keys: {missing}\n\n\n\n\nunexpected keys: {unexpected}")

    def set_lora(
        self,
        local_path: str = None,
        repo_id: str = None,
        name: str = None,
        lora_weight: int = 0.7,
    ):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(
        self, lora_type: str = "realism", lora_weight: int = 0.7
    ):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1 :]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(
                        dim=3072, rank=rank
                    )
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(
                        dim=3072, rank=rank
                    )
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def __call__(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        save_attn_path: Optional[str] = None,
        **kwargs,
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        device_type = self.device if isinstance(self.device, str) else self.device.type
        dtype = torch.bfloat16 if device_type != "mps" else torch.float16
        with torch.autocast(
            enabled=self.use_fp8, device_type=device_type, dtype=dtype
        ):
            return self.forward(
                prompt, width, height, guidance, num_steps, seed, save_attn_path=save_attn_path, **kwargs
            )

    @torch.inference_mode()
    def gradio_generate(
        self,
        prompt: str,
        image_prompt1: Image.Image,
        image_prompt2: Image.Image,
        image_prompt3: Image.Image,
        seed: int,
        width: int = 1024,
        height: int = 1024,
        guidance: float = 4,
        num_steps: int = 25,
        keep_size: bool = False,
        content_long_size: int = 512,
    ):
        ref_content_imgs = [image_prompt1]
        ref_content_imgs = [img for img in ref_content_imgs if isinstance(img, Image.Image)]
        ref_content_imgs = [preprocess_ref(img, content_long_size) for img in ref_content_imgs]

        ref_style_imgs = [image_prompt2, image_prompt3]
        ref_style_imgs = [img for img in ref_style_imgs if isinstance(img, Image.Image)]
        ref_style_imgs = [self.model.vision_encoder_processor(img, return_tensors="pt").to(self.device) for img in ref_style_imgs]

        seed = seed if seed != -1 else torch.randint(0, 10**8, (1,)).item()

        # whether keep input image size
        if keep_size and len(ref_content_imgs)>0:
            width, height = ref_content_imgs[0].size
            width, height = int(width * (1024 / content_long_size)), int(height * (1024 / content_long_size))
        img = self(
            prompt=prompt,
            width=width,
            height=height,
            guidance=guidance,
            num_steps=num_steps,
            seed=seed,
            ref_imgs=ref_content_imgs,
            siglip_inputs=ref_style_imgs,
        )

        filename = f"output/gradio/{seed}_{prompt[:20]}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "USO"
        exif_data[ExifTags.Base.Model] = self.model_type
        info = f"{prompt=}, {seed=}, {width=}, {height=}, {guidance=}, {num_steps=}"
        exif_data[ExifTags.Base.ImageDescription] = info
        img.save(filename, format="png", exif=exif_data)
        return img, filename

    @torch.inference_mode
    def forward(
        self,
        prompt: str,
        width: int,
        height: int,
        guidance: float,
        num_steps: int,
        seed: int,
        ref_imgs: list[Image.Image] | None = None,
        pe: Literal["d", "h", "w", "o"] = "d",
        siglip_inputs: list[Tensor] | None = None,
        save_attn_path: Optional[str] = None,
    ):
        x = get_noise(
            1, height, width, device=self.device, dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        if self.offload:
            self.ae.encoder = self.ae.encoder.to(self.device)
        x_1_refs = [
            self.ae.encode(
                (TVF.to_tensor(ref_img) * 2.0 - 1.0)
                .unsqueeze(0)
                .to(self.device, torch.float32)
            ).to(torch.bfloat16)
            for ref_img in ref_imgs
        ]

        if self.offload:
            self.offload_model_to_cpu(self.ae.encoder)
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp_cond = prepare_multi_ip(
            t5=self.t5,
            clip=self.clip,
            img=x,
            prompt=prompt,
            ref_imgs=x_1_refs,
            pe=pe,
        )

        if self.offload:
            self.offload_model_to_cpu(self.t5, self.clip)
            self.model = self.model.to(self.device)

        x = denoise(
            self.model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
            siglip_inputs=siglip_inputs,
        )

        if self.offload:
            self.offload_model_to_cpu(self.model)
            self.ae.decoder.to(x.device)
        x = unpack(x.float(), height, width)
        x = self.ae.decode(x)
        self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload:
            return
        for model in models:
            model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
