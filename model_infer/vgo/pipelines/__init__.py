import os
from dataclasses import dataclass, field
from functools import partial

import torch
from loguru import logger
from omegaconf import MISSING

from vgo.models.text_encoder.qwen25vl import Qwen25VL7B_Embedder
from vgo.models.transformers.model import DiTParams, VarLenDiT
from vgo.models.vae.qwen_autoencoder import AutoencoderKLQwenImage
from vgo.utils.common_utils import convert_precision, load_state_dict


@dataclass
class LoRAArgs:
    rank: int = 128
    init_lora_weights: str = "gaussian"


@dataclass
class PipelineArgs:
    type: str = ""
    pipeline_path: str = MISSING

    def build(self) -> dict:
        raise NotImplementedError("build method not implemented")


@dataclass
class NaivePipelineArgs(PipelineArgs):
    dit: DiTParams = field(
        default_factory=lambda: DiTParams(
            in_channels=64,
            out_channels=64,
            vec_in_dim=None,
            context_in_dim=3584,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=60,
            depth_single_blocks=0,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=False,
        )
    )

    llm_model_path: str | None = None
    llm_encoder_type: str = "common"
    llm_output_layer_index: int = -1
    llm_image_min_token: int = 256
    llm_image_max_token: int = 400
    dit_path: str | None = None
    ae_path: str | None = None
    max_length: int = 2048
    lora: LoRAArgs | None = field(default_factory=LoRAArgs)
    fuse_llm_dit: bool = False

    def __post_init__(self):
        if self.pipeline_path is not MISSING:
            self.llm_model_path = self.llm_model_path or os.path.join(self.pipeline_path, "Qwen2.5-VL-7B-Instruct")
            self.ae_path = self.ae_path or os.path.join(self.pipeline_path, "vae.safetensors")

    def build_ae(self, device):
        ae = AutoencoderKLQwenImage.from_pretrained(self.ae_path)
        ae = ae.to(device=device, dtype=torch.float32).eval()  # type: ignore
        ae.requires_grad_(False)
        logger.debug("load autoencoder done")
        return ae

    def build_llm_encoder(self, device, dtype, non_skip=False):
        if not non_skip:
            if self.fuse_llm_dit:
                return None

        if self.llm_encoder_type in ["naive", "naive_qwen3vl8b"]:
            assert self.llm_output_layer_index in [-1, -2]
            if self.llm_output_layer_index == -2:
                logger.warning("当前正采用 Qwen2.5VL 的倒数第二层特征给 DiT。")
            if self.llm_encoder_type == "naive":
                qwenvl_encoder_type = Qwen25VL7B_Embedder
            else:
                from vgo.models.text_encoder.qwen3vl8b import Qwen3VL8B_Embedder

                qwenvl_encoder_type = Qwen3VL8B_Embedder
            logger.info(f"当前采用 {qwenvl_encoder_type.__name__} 作为 Text Encoder.")

            qwenvl_encoder = qwenvl_encoder_type(
                model_path=self.llm_model_path,
                device=device,
                max_length=self.max_length,
                dtype=dtype,
                llm_image_min_token=self.llm_image_min_token,
                llm_image_max_token=self.llm_image_max_token,
                out_embedding_layer_index=self.llm_output_layer_index,
            )
        else:
            raise NotImplementedError(f"Unrecognized {self.llm_encoder_type=}")

        logger.debug("load qwenvl_encoder done")
        qwenvl_encoder.eval()
        qwenvl_encoder.requires_grad_(False)
        return qwenvl_encoder

    def build_dit(self, device, dtype):
        # with torch.device("meta"):
        llm_build_func = partial(self.build_llm_encoder, device, dtype, True) if self.fuse_llm_dit else None
        dit = VarLenDiT(self.dit, build_llm_encoder=llm_build_func)  # type: ignore

        if self.dit_path:
            logger.info(f"Loading DiT from {self.dit_path}")
            dit = load_state_dict(dit, self.dit_path)
        else:
            logger.info("Building DiT from scratch")

        dit = convert_precision(dit, dtype=dtype)
        dit = dit.to(device=device).eval()
        dit.requires_grad_(False)
        logger.debug("load dit done")
        return dit
