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
from .modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    SigLIPMultiFeatProjModel,
)
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
import torch

def denoise_save_hotmap(
    model,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    ref_img: Tensor = None,
    ref_img_ids: Tensor = None,
    siglip_inputs: list[Tensor] | None = None,
    save_attn_path: Optional[str] = None,
    time_save_step: Optional[int] = 5,
    block_save_step: Optional[int] = 10,
    target_size: Optional[int] = 256,
    ae : Optional[nn.Module] = None,
    height: Optional[int] = 1024,
    width: Optional[int] = 1024,
):
    """
    输入的img才是我们要的图像token，是噪声之路的token
    其余的txt,txt_ids,vec都是条件
    """
    i = 0
    guidance_vec = torch.full(
        (img.shape[0],), guidance, device=img.device, dtype=img.dtype
    )
    for t_curr, t_prev in tqdm(
        zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1
    ):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        # 清空hotmap和hidden_states列表，避免保存到上一次的内容
        if hasattr(model, "hotmap_list"):
            model.hotmap_list.clear()
        if hasattr(model, "hidden_states_list"):
            model.hidden_states_list.clear()
        pred = model(
            img=img,
            img_ids=img_ids,
            ref_img=ref_img,
            ref_img_ids=ref_img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            siglip_inputs=siglip_inputs,
        )
        img = img + (t_prev - t_curr) * pred
        i += 1

        txt_token_len = model.index_dict.get("txt_len", None)
        img_token_len = model.index_dict.get("img_len", None)
        #前向传播后得到图像token和文本token的长度等等
        print('image height,width in latent :',height//16,width//16, height * width // 256)
        assert len(model.hotmap_list) == len(model.hidden_states_list), "hotmap_list and hidden_states_list should have the same length."
        blocks_num = len(model.hotmap_list)
        if i % time_save_step == 0 and save_attn_path is not None and ae is not None:
            # 新建一个文件夹，按顺序存储hotmap
            for idx in range(blocks_num):
                if idx % block_save_step == 0:
                    print('saving attn map at block idx:',idx)
                    hotmap = model.hotmap_list[idx]
                    os.makedirs(f"{save_attn_path}/attnmap/{str(i)}", exist_ok=True)
                    if hotmap.get("map", None) is None or not torch.is_tensor(hotmap["map"]):
                        continue
                    map = hotmap["map"][0]  # (heads, seq_len, seq_len)
                    mean_map = map.mean(0)  # (seq_len, seq_len)
                    #只选取图像部分做注意力热力图保存
                    mean_map = mean_map[txt_token_len:txt_token_len+img_token_len, txt_token_len:txt_token_len+img_token_len]  # (img_len, txt_len)
                    mean_map = (mean_map - mean_map.min()) / (mean_map.max() - mean_map.min() + 1e-6)
                    mean_map = (mean_map.float() * 255).cpu().numpy()
                    if plt is None:
                        raise RuntimeError('matplotlib is required when save_attn_path is enabled')
                    colored_map = plt.get_cmap('hot')(mean_map)[:, :, :3]
                    img_mean = Image.fromarray((colored_map * 255).astype(np.uint8))
                    img_mean = img_mean.convert("RGB")
                    img_mean = img_mean.resize((target_size, target_size), resample=Image.LANCZOS)
                    img_mean.save(f"{save_attn_path}/attnmap/{str(i)}/{hotmap['name']}_mean_idx{idx}.png")
                    
                    #保存hidden_states_list
                    print('saving hidden states at block idx:',idx)
                    hidden_vae_decode = model.hidden_states_list[idx]
                    os.makedirs(f"{save_attn_path}/hidden_states/{str(i)}", exist_ok=True)
                    output = hidden_vae_decode.get('output', None)
                    # output = model.final_layer(output, vec)
                    if output is None:
                        print(f"Warning: hidden_vae_decode['output'] is None at idx {idx}")
                        continue
                    print("hidden states shape before decode", output.shape)
                    if 'double' in hidden_vae_decode['name'].lower():
                        print('double stream block, take the img part of the output')
                        output = output[:,:img_token_len,:]
                        print(output.shape)
                    else:
                        print('single stream block, take the img part of the output')
                        output = output[:,txt_token_len:txt_token_len+img_token_len,:]
                        print(output.shape)
                    output = model.final_layer(output, hidden_vae_decode['vec'])
                    output = unpack(output.float(), height, width)
                    output = ae.decode(output).clamp(-1, 1)
                    output = output[-1]
                    # print("hidden states shape", output.shape)
                    output = rearrange(output, "c h w -> h w c")
                    output = (127.5 * (output + 1.0)).cpu().byte().numpy()
                    # output = output[..., [2, 1, 0]]
                    img_output = Image.fromarray(output).convert("RGB")
                    img_output.save(f"{save_attn_path}/hidden_states/{str(i)}/{hidden_vae_decode['name']}_vae_decode_idx{idx}.png")
            #把时间步t的输出结果保存下来
            os.makedirs(f"{save_attn_path}/middle", exist_ok=True)
            print('saving middle image at step, img shape', img.shape)
            x1 = unpack(img.float(), height, width)
            x1 = ae.decode(x1)
            x1 = x1.clamp(-1, 1)
            x1 = rearrange(x1[-1], "c h w -> h w c")
            output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
            output_img.save(f"{save_attn_path}/middle/denoise_step_{str(i)}.png")
    return img


@torch.inference_mode
def pipeline_forward(
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

    #在这里进行hotmap的可视化
    if len(self.handles)>0:#说明我进行了hook的挂载
        x = denoise_save_hotmap(
            self.model,
            **inp_cond,
            timesteps=timesteps,
            guidance=guidance,
            siglip_inputs=siglip_inputs,
            save_attn_path = save_attn_path,
            ae = self.ae,
            height = height,
            width = width,
        )
    else:
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
