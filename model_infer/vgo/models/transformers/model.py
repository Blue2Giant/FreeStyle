import os
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.distributed.tensor
import torch.utils.checkpoint
from loguru import logger
from torch import Tensor, nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from vgo.models.modules import RMSNorm
from vgo.models.modules.attention import VarlenAttentionConfig
from vgo.models.modules.distributed_ops import LocalReplicateToDTensorSP
from vgo.models.modules.varlen_ops import VarLenConfig, cat_seq, merge_varlen_seqs, split_seq
from vgo.models.transformers.layers import (
    EmbedND,
    FrequencyAwareRopeConfig,
    MLPEmbedder,
    VarLenDoubleStreamBlock,
    VarlenLastLayer,
    VarLenSingleStreamBlock,
    timestep_embedding,
)
from vgo.utils.common_utils import combine_list
from vgo.utils.dist_utils import (
    NoParallel,
    fully_shard_w_optimizer_dtype,
    get_modules_w_grad,
    set_requires_gradient_sync,
    set_reshard_after_backward,
)


@dataclass
class DiTParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int | None
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    enable_txt_norm: bool = False
    enable_zero_t_embed: bool = False
    adaln_dim: int | None = None
    timestep_mlp_ratio: int = 1
    rope_fa: FrequencyAwareRopeConfig = field(default_factory=FrequencyAwareRopeConfig)


class VarLenDiT(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: DiTParams, build_llm_encoder: Callable[[], nn.Module] | None = None):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.use_frequency_aware_rope = bool(params.rope_fa.enabled)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # 设置 Timestep MLP Ratio
        self.adaln_dim = self.hidden_size if params.adaln_dim is None else params.adaln_dim
        logger.info(f"AdaLN DIM = {self.adaln_dim}")
        logger.info(f"Timestep MLP Ratio = {params.timestep_mlp_ratio}")

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.adaln_dim, mlp_ratio=params.timestep_mlp_ratio)
        if params.vec_in_dim is not None:
            self.vector_in = MLPEmbedder(params.vec_in_dim, self.adaln_dim)
        else:
            self.vector_in = None
        if params.guidance_embed:
            self.guidance_in = MLPEmbedder(in_dim=256, hidden_dim=self.adaln_dim)
        else:
            self.guidance_in = None

        if params.enable_txt_norm:
            self.txt_norm = RMSNorm(params.context_in_dim)
        else:
            self.txt_norm = None
        logger.debug("VarLenDiT init: txt_norm ready")

        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)
        logger.debug("VarLenDiT init: txt_in ready")

        double_blocks: list[VarLenDoubleStreamBlock] = []
        for block_idx in range(params.depth):
            if block_idx == 0 or block_idx == params.depth - 1 or block_idx % 5 == 0:
                logger.debug(f"VarLenDiT init: building double_block[{block_idx}]")
            double_blocks.append(
                VarLenDoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    adaln_dim=self.adaln_dim,
                    qkv_bias=params.qkv_bias,
                    rope_fa_config=params.rope_fa,
                    rope_axes_dim=params.axes_dim,
                )
            )
        self.double_blocks = nn.ModuleList(double_blocks)
        logger.debug(f"VarLenDiT init: double_blocks ready ({params.depth})")

        single_blocks: list[VarLenSingleStreamBlock] = []
        for block_idx in range(params.depth_single_blocks):
            if block_idx == 0 or block_idx == params.depth_single_blocks - 1 or block_idx % 5 == 0:
                logger.debug(f"VarLenDiT init: building single_block[{block_idx}]")
            single_blocks.append(
                VarLenSingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    adaln_dim=self.adaln_dim,
                    mlp_ratio=params.mlp_ratio,
                )
            )
        self.single_blocks = nn.ModuleList(single_blocks)
        logger.debug(f"VarLenDiT init: single_blocks ready ({params.depth_single_blocks})")

        self.final_layer = VarlenLastLayer(self.hidden_size, 1, self.out_channels, self.adaln_dim)
        logger.debug("VarLenDiT init: final_layer ready")

        self.llm_encoder: nn.Module | None = build_llm_encoder() if build_llm_encoder is not None else None  # type: ignore
        logger.debug("VarLenDiT init: llm_encoder ready")

        self.enable_zero_t_embed = params.enable_zero_t_embed
        if self.enable_zero_t_embed:
            assert self.guidance_in is None, f"当前，{self.enable_zero_t_embed=} 下，self.guidance_in 需要为 None"
            assert self.vector_in is None, f"当前，{self.enable_zero_t_embed=} 下，self.guidance_in 需要为 None"
            logger.warning(
                f"当前，`{self.enable_zero_t_embed=}`，部分 token 的调制参数 t 的始终为 0 的设置"
                "这个设置在 Qwen 2511 中使用，暂时并不清楚这个设置是否会有增益。"
                "如果要启用，请务必确保每个 sequence 中，需要设置为 t=0 的 token 放在 img/txt token 的末尾。"
            )

        self.gradient_checkpointing = False
        self.is_shard = False
        self.use_async_tp = False
        self.compiled = False

    @torch.no_grad()
    def get_vae_downsample_affine_mat(self):
        if not hasattr(self, "vae_affine_mat"):
            self.vae_affine_mat = torch.Tensor(
                [
                    [1 / 16, 0, -11 / 16],
                    [0, 1 / 16, -11 / 16],
                    [0, 0, 1],
                ]
            ).to(device=self.img_in.weight.device, dtype=torch.float32)
        return self.vae_affine_mat

    @torch.no_grad()
    def set_vision_tokens_position_ids(
        self, txt_ids, vision_token_hw, affine_mat, vision_token_mask, device, debug=False
    ):
        # txt_ids: should be flattened txt_idx, shape: (L, 3)
        # affine_mat: should be the pixel value space to the vision token space, (3, 3)
        # vision_token_mask: should be the bool (L, 3)
        dtype = torch.float32

        v_h = vision_token_hw[0]
        v_w = vision_token_hw[1]
        vision_token_idx = torch.zeros((v_h, v_w, 3), dtype=dtype, device=device)
        vision_token_idx[..., 1] = vision_token_idx[..., 1] + torch.arange(v_h, dtype=dtype, device=device)[:, None]
        vision_token_idx[..., 2] = vision_token_idx[..., 2] + torch.arange(v_w, dtype=dtype, device=device)[None, :]

        vision_token_idx_xy_shape = vision_token_idx[..., 1:].shape
        # notice that the axis 1 is y, axis 2 is x
        vision_token_idx_xy = vision_token_idx[..., [2, 1]].reshape(-1, 2)
        affin_mat = self.get_vae_downsample_affine_mat() @ torch.linalg.inv(affine_mat).to(dtype)
        vision_token_idx_xy = vision_token_idx_xy @ affin_mat[:2, :2].T + affin_mat[:2, -1][None]

        if debug:
            from loguru import logger

            logger.info(f"Vision Token HxW: {vision_token_idx_xy[..., 1].max()}x{vision_token_idx_xy[..., 0].max()}")

        # notice that the axis 1 is y, axis 2 is x, set xy to [2, 1]
        vision_token_idx[..., [2, 1]] = vision_token_idx_xy.view(vision_token_idx_xy_shape)
        vision_token_idx = vision_token_idx.reshape(-1, 3)

        # do not change position of time
        txt_ids[..., 1:].masked_scatter_(vision_token_mask[:, 1:], vision_token_idx[:, 1:])

    def parallelize_module(self, device_mesh: DeviceMesh, use_async_tp=False, use_replicate_modulation_linear=False):
        self.device_mesh = device_mesh

        parallelize_plan = {
            "img_in": SequenceParallel(sequence_dim=0),
            "txt_in": SequenceParallel(sequence_dim=0),
            "final_layer.linear": SequenceParallel(sequence_dim=0),
            "final_layer.adaLN_modulation.1": NoParallel(
                input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False
            ),
            "time_in.in_layer": ColwiseParallel(),
            "time_in.out_layer": RowwiseParallel(),
        }

        if self.final_layer.norm_final.elementwise_affine:
            parallelize_plan["final_layer.norm_final"] = SequenceParallel(sequence_dim=0)

        if self.guidance_in is not None:
            parallelize_plan.update(
                {"guidance_in.in_layer": ColwiseParallel(), "guidance_in.out_layer": RowwiseParallel()}
            )
        if self.vector_in is not None:
            parallelize_plan.update(
                {"vector_in.in_layer": ColwiseParallel(), "vector_in.out_layer": RowwiseParallel()}
            )

        parallelize_module(
            module=self,
            device_mesh=device_mesh,
            parallelize_plan=parallelize_plan,
        )

        for double_block_i in self.double_blocks:
            double_block_i: VarLenDoubleStreamBlock = double_block_i
            double_block_i.parallelize_module(
                device_mesh=device_mesh, use_replicate_modulation_linear=use_replicate_modulation_linear
            )

        for single_block_i in self.single_blocks:
            single_block_i: VarLenSingleStreamBlock = single_block_i
            single_block_i.parallelize_module(
                device_mesh=device_mesh, use_replicate_modulation_linear=use_replicate_modulation_linear
            )

        self.is_shard = True
        self.use_replicate_modulation_linear = use_replicate_modulation_linear

        tp_params_name_list = []
        for name, params in self.named_parameters():
            if isinstance(params, torch.distributed.tensor.DTensor):
                tp_params_name_list.append(name)
        self.tp_params_name_list = set(tp_params_name_list)

        if use_async_tp:
            from torch.distributed._symmetric_memory import enable_symm_mem_for_group

            enable_symm_mem_for_group(device_mesh.get_group().group_name)

            torch._inductor.config._micro_pipeline_tp = True
            self.use_async_tp = True

        self.compiled = False

    def apply_ac(self, non_activation_checkpointing_every: int = -1, apply_to_llm=False):
        """Apply activation checkpointing to the model."""

        self.ac_applied = True

        # when non_activation_checkpointing_every == 1, activation checkpointing will be disabled
        if non_activation_checkpointing_every == 1:
            return

        for layer_id, block in self.double_blocks.named_children():
            if non_activation_checkpointing_every == -1 or int(layer_id) % non_activation_checkpointing_every != 0:
                block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
                self.double_blocks.register_module(layer_id, block)

        for layer_id, block in self.single_blocks.named_children():
            if non_activation_checkpointing_every == -1 or int(layer_id) % non_activation_checkpointing_every != 0:
                block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
                self.single_blocks.register_module(layer_id, block)

        if apply_to_llm:
            assert self.llm_encoder is not None
            for layer_id, block in self.llm_encoder.model.model.layers.named_children():
                if non_activation_checkpointing_every == -1 or int(layer_id) % non_activation_checkpointing_every != 0:
                    block = ptd_checkpoint_wrapper(block, preserve_rng_state=False)
                    self.llm_encoder.model.model.layers.register_module(layer_id, block)

    def apply_compile(self, apply_to_llm: bool = True, inference_mode=False):
        if self.compiled:
            return
        if os.environ.get("VGO_DISABLE_TORCH_COMPILE", "0") == "1":
            logger.info("Skip torch.compile because VGO_DISABLE_TORCH_COMPILE=1")
            return

        if self.use_async_tp or inference_mode or not self.is_shard:
            for _layer_id, double_block in self.double_blocks.named_children():
                double_block.apply_compile()  # type: ignore

            for _layer_id, single_block in self.single_blocks.named_children():
                single_block.apply_compile()  # type: ignore

        # FIXME: Currenctly llm decode layer is cannot be used in torch.compile.
        if apply_to_llm:
            if self.llm_encoder is not None:
                for _layer_id, llm_layer in self.llm_encoder.model.model.layers.named_children():
                    llm_layer.apply_compile()  # type: ignore
                    # llm_layer = torch.compile(self.llm_encoder.model.model.layers, dynamic=True)
                    # self.llm_encoder.model.model.layers.register_module(layer_id, llm_layer)
        self.compiled = True

    def apply_fsdp(  # noqa: C901
        self,
        world_mesh: DeviceMesh,
        param_dtype: torch.dtype,
        reduce_dtype: torch.dtype,
        cpu_offload: bool = False,
        use_hsdp: bool = True,
        apply_to_llm_encoder: bool = False,
        reshard_after_forward: bool = False,
        optimizer_dtype: torch.dtype | None = None,
        norm_param_dtype: torch.dtype | None = torch.float32,
    ):
        fully_shard = fully_shard_w_optimizer_dtype(None)

        # turn on per-TransformerBlock compile after AC wrapping and before FSDP
        if getattr(self, "use_async_tp", False):
            assert getattr(self, "ac_applied", False), (
                "`use_async_tp` can only be used when activation checkpointint wrapper is called, please read https://github.com/pytorch/torchtitan/blob/1923ce4/torchtitan/parallelisms/parallelize_llama.py#L66C5-L66C77."
            )
            self.apply_compile()

        mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
        # set normalization policy
        norm_mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32
        )
        if self.is_shard:
            # 如果开启了 TP+SP，设置 梯度反向传播 时的 reduce 方式为求和而非取均值
            _reduce_scatter_weight = 1.0 if use_hsdp else world_mesh["dp"].size()
            fsdp_config = {"mesh": world_mesh["dp"], "mp_policy": mp_policy}
            _world_fsdp_config = {"mesh": world_mesh if use_hsdp else world_mesh._flatten(), "mp_policy": mp_policy}
            fsdp_config_norm = {"mesh": world_mesh["dp"], "mp_policy": norm_mp_policy}
            world_fsdp_config_norm = {
                "mesh": world_mesh if use_hsdp else world_mesh._flatten(),
                "mp_policy": norm_mp_policy,
            }
            if cpu_offload:
                fsdp_config["offload_policy"] = CPUOffloadPolicy()
        else:
            fsdp_config = _world_fsdp_config = {
                "mesh": world_mesh if use_hsdp else world_mesh._flatten(),
                "mp_policy": mp_policy,
            }
            fsdp_config_norm = world_fsdp_config_norm = {
                "mesh": world_mesh if use_hsdp else world_mesh._flatten(),
                "mp_policy": norm_mp_policy,
            }

        if norm_param_dtype != torch.float32:
            assert norm_param_dtype is None
            logger.warning(
                "当前的设置下，建议必须在使用 ReFL 的训练，或者需要采用 DiT 多次 forward 后 backward 的训练方式。"
                "当前设置下，必须将 DiT 中的 LayerNorm 层修改为 FP32 推理的 LayerNormAutoCast 否则可能出现精度对不齐的问题"  # noqa: E501
            )

        for name, module in self.named_modules():
            # HARDCODE 暂时不训练 llm Encoder
            if "llm_encoder" in name:
                continue

            if isinstance(module, (nn.LayerNorm, RMSNorm)):
                module = module.to(torch.float32)

                logger.info(f"Use FLOAT32 in module: {name=}")
                if "double_blocks" in name or "single_blocks" in name or "final_layer" in name:
                    if norm_param_dtype == torch.float32:
                        fully_shard(
                            module,
                            **fsdp_config_norm,
                            reshard_after_forward=reshard_after_forward,
                        )
                elif name != "txt_norm":
                    fully_shard(
                        module,
                        **world_fsdp_config_norm,
                        reshard_after_forward=reshard_after_forward,
                    )
                else:
                    fully_shard(
                        module,
                        mesh=world_mesh if use_hsdp else world_mesh._flatten(),
                        mp_policy=MixedPrecisionPolicy(
                            param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=param_dtype
                        ),
                        reshard_after_forward=reshard_after_forward,
                    )

        # shard on DP
        # if world_mesh["dp"].size() > 1:
        linear_layers = [
            self.img_in,
            self.time_in,
            self.txt_in,
        ]
        if self.vector_in is not None:
            linear_layers.append(self.vector_in)

        for layer in linear_layers:
            fully_shard(layer, **fsdp_config, reshard_after_forward=reshard_after_forward)

        if self.guidance_in is not None:
            fully_shard(self.guidance_in, **fsdp_config, reshard_after_forward=reshard_after_forward)

        for block in self.double_blocks:
            double_block: VarLenDoubleStreamBlock = block

            fully_shard(
                double_block,
                mesh=fsdp_config["mesh"],
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    cast_forward_inputs=False,  # use false for FP32 RoPE
                ),
                reshard_after_forward=reshard_after_forward,
            )
            # double_block._get_fsdp_state()._fsdp_param_group.post_forward_mesh_info
            # double_block._get_fsdp_state()._fsdp_param_group._reshard_after_forward

        # shard on DP
        for block in self.single_blocks:
            single_block: VarLenSingleStreamBlock = block

            fully_shard(
                single_block,
                mesh=fsdp_config["mesh"],
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    cast_forward_inputs=False,  # use false for FP32 RoPE
                ),
                reshard_after_forward=reshard_after_forward,
            )

        # shard on all gpus
        # apply FSDP to last layer. Set reshard_after_forward=False for last layer to avoid gather right after reshard

        if self.final_layer.norm_final.elementwise_affine:
            fully_shard(self.final_layer.norm_final, **fsdp_config, reshard_after_forward=reshard_after_forward)
        fully_shard(self.final_layer.adaLN_modulation, **fsdp_config, reshard_after_forward=reshard_after_forward)
        fully_shard(self.final_layer.linear, **fsdp_config, reshard_after_forward=False)

        # FIXME: I have no idea why the following line will cause reduce_tensor in varlen config will be converted,
        # to local tensor.
        # fully_shard(self.final_layer, **fsdp_config, reshard_after_forward=reshard_after_forward)

        # 将可训练参数转为 optimizer_dtype 的设置值。
        # On NPU + DTensor hybrid sharding, casting a sharded module after FSDP can fail during
        # sharding propagation, so skip modules whose params are already DTensors.
        if optimizer_dtype is not None:
            _, modules_w_grad = get_modules_w_grad(self)
            for module in modules_w_grad:
                direct_params = list(module.parameters(recurse=False))
                if any(
                    isinstance(param, torch.distributed.tensor.DTensor)
                    or isinstance(getattr(param, "data", None), torch.distributed.tensor.DTensor)
                    for param in direct_params
                ):
                    logger.info(f"Skip optimizer_dtype cast on DTensor module: {module.__class__.__name__}")
                    continue
                module.to(optimizer_dtype)

        # HSDP ( reduce scatter (AVG) on Shard -> all reduce (AVG) on Replicate )
        # FSDP ( reduce scatter (AVG) on Shard )

    def set_reshard_after_backward(self, reshard_after_backward: bool):
        set_reshard_after_backward(self, reshard_after_backward)

    def set_requires_gradient_sync(self, requires_gradient_sync: bool):
        set_requires_gradient_sync(self, requires_gradient_sync)

    def prepare_for_transformer_blocks(
        self, img, img_ids, txt, txt_ids, vec, img_seq_lens, txt_seq_lens, zero_t_seq_lens
    ):
        img_seq_lens = img_seq_lens.copy()
        txt_seq_lens = txt_seq_lens.copy()
        total_img_len = sum(img_seq_lens)
        total_txt_len = sum(txt_seq_lens)
        img_pad_len = 0
        txt_pad_len = 0
        # 这里需要对数据长度进行 pad 到 TP SIZE 的倍数，因为 Async TP 要求要能够整除
        if self.is_shard:
            block_size = 128
            img_pad_len = (total_img_len + block_size - 1) // block_size * block_size - total_img_len
            txt_pad_len = (total_txt_len + block_size - 1) // block_size * block_size - total_txt_len
            if img_pad_len > 0 or txt_pad_len > 0:
                img = torch.cat([img, torch.zeros((img_pad_len, *img.shape[1:]), device=img.device, dtype=img.dtype)])
                txt = torch.cat([txt, torch.zeros((txt_pad_len, *txt.shape[1:]), device=txt.device, dtype=txt.dtype)])
                img_ids = torch.cat(
                    [
                        img_ids,
                        torch.zeros((img_pad_len, *img_ids.shape[1:]), device=img_ids.device, dtype=img_ids.dtype),
                    ],
                    dim=0,
                )
                txt_ids = torch.cat(
                    [
                        txt_ids,
                        torch.zeros((txt_pad_len, *txt_ids.shape[1:]), device=txt_ids.device, dtype=txt_ids.dtype),
                    ],
                    dim=0,
                )
                vec = torch.cat([vec, vec[:1]], dim=0)
                img_seq_lens = [*img_seq_lens, img_pad_len]
                txt_seq_lens = [*txt_seq_lens, txt_pad_len]

        if self.enable_zero_t_embed:
            assert zero_t_seq_lens is not None
            # 注意 pad 带来的问题
            if self.is_shard and (img_pad_len > 0 or txt_pad_len > 0):
                zero_t_seq_lens.append((0, 0))
                vec = torch.cat([vec, vec[:1]], dim=0)

            assert vec.shape[0] == 2 * len(img_seq_lens)
            assert len(zero_t_seq_lens) == len(img_seq_lens)
            assert all(x >= y[0] for x, y in zip(img_seq_lens, zero_t_seq_lens))
            assert all(x >= y[1] for x, y in zip(txt_seq_lens, zero_t_seq_lens))

            _img_seq_lens = combine_list([[x - y[0], y[0]] for x, y in zip(img_seq_lens, zero_t_seq_lens)])
            _txt_seq_lens = combine_list([[x - y[1], y[1]] for x, y in zip(txt_seq_lens, zero_t_seq_lens)])
        else:
            _img_seq_lens = img_seq_lens
            _txt_seq_lens = txt_seq_lens

        img_varlen_config = VarLenConfig.from_seq_lens(_img_seq_lens, device=img.device)
        txt_varlen_config = VarLenConfig.from_seq_lens(_txt_seq_lens, device=txt.device)
        img_txt_varlen_config = merge_varlen_seqs(img_varlen_config, txt_varlen_config)
        img_varlen_config.set_gather_index(vec.shape[-1])
        txt_varlen_config.set_gather_index(vec.shape[-1])
        img_txt_varlen_config.set_gather_index(vec.shape[-1])

        # prepare varlen
        if self.is_shard:
            img = LocalReplicateToDTensorSP(img, self.device_mesh)
            txt = LocalReplicateToDTensorSP(txt, self.device_mesh)
            img_varlen_config.set_device_mesh(self.device_mesh)
            txt_varlen_config.set_device_mesh(self.device_mesh)
            img_txt_varlen_config.set_device_mesh(self.device_mesh)

        varlen_attention_config = VarlenAttentionConfig.from_seq_lens(
            [x + y for x, y in zip(img_seq_lens, txt_seq_lens)], device=txt.device
        )

        if self.compiled:
            torch._dynamo.mark_dynamic(img, 0)
            torch._dynamo.mark_dynamic(img_ids, 0)
            torch._dynamo.mark_dynamic(txt, 0)
            torch._dynamo.mark_dynamic(txt_ids, 0)
            torch._dynamo.mark_dynamic(vec, 0)

            torch._dynamo.mark_dynamic(img_varlen_config.reduce_tensor, (0, 1))
            torch._dynamo.mark_dynamic(txt_varlen_config.reduce_tensor, (0, 1))
            torch._dynamo.mark_dynamic(img_txt_varlen_config.reduce_tensor, (0, 1))
            torch._dynamo.mark_dynamic(img_varlen_config.gather_index, 0)
            torch._dynamo.mark_dynamic(txt_varlen_config.gather_index, 0)
            torch._dynamo.mark_dynamic(img_txt_varlen_config.gather_index, 0)

            torch._dynamo.mark_dynamic(varlen_attention_config.cu_seqlens_q, 0)
            torch._dynamo.mark_dynamic(varlen_attention_config.cu_seqlens_kv, 0)

            torch._dynamo.mark_static(img, 1)
            torch._dynamo.mark_static(img_ids, 1)
            torch._dynamo.mark_static(txt, 1)
            torch._dynamo.mark_static(txt_ids, 1)
            torch._dynamo.mark_static(vec, 1)

        return (
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
            img_seq_lens,
            txt_seq_lens,
            img_varlen_config,
            txt_varlen_config,
            img_txt_varlen_config,
            varlen_attention_config,
            img_pad_len,
            txt_pad_len,
        )

    def forward(  # noqa: C901
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor | None,
        img_seq_lens: list[int],
        txt_seq_lens: list[int],
        guidance: Tensor | None = None,
        zero_t_seq_lens: list[tuple[int, int]] | None = None,  # 仅当 enable_zero_t_embed 开启时起作用
        return_sref_enrichment: bool = False,
        return_sref_entropy: bool = False,
        sref_key_ranges: list[tuple[int, int]] | None = None,
        sref_query_ranges: list[tuple[int, int]] | None = None,
        sref_enrichment_lower_bound: float = 0.08,
        sref_enrichment_upper_bound: float = 0.5,
        sref_enrichment_eps: float = 1e-6,
        sref_entropy_lower_bound: float = 0.06,
        sref_entropy_upper_bound: float = 0.14,
        sref_entropy_eps: float = 1e-6,
        sref_entropy_schedule_enabled: bool = False,
        sref_entropy_schedule_start_timestep: float = 0.75,
        sref_entropy_schedule_start_lower_bound: float = 0.06,
        sref_entropy_schedule_end_lower_bound: float = 0.10,
        sref_entropy_schedule_power: float = 2.0,
        sref_enrichment_timestep_weighting: bool = False,
        sref_enrichment_timestep_weight_power: float = 1.0,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if img.ndim != 2 or txt.ndim != 2:
            raise ValueError(f"Input img and txt tensors must have 2 dimensions, got {img.ndim} and {txt.ndim}")

        assert img.shape[0] == img_ids.shape[0] == sum(img_seq_lens), (
            f"{img.shape[0]=} {img_ids.shape[0]=} {sum(img_seq_lens)=}"
        )
        assert txt.shape[0] == txt_ids.shape[0] == sum(txt_seq_lens), (
            f"{txt.shape[0]=} {txt_ids.shape[0]=} {sum(txt_seq_lens)=}"
        )

        if self.txt_norm is not None:
            # Liger-Kernel 的 RMSNorm 算子并不支持 DTensor 作为输入
            txt = self.txt_norm(txt)

            # norm 层有时会被强制转到 FP32 进行计算，如果精度和 txt_in 不一致会报错
            if txt.dtype != self.txt_in.weight.dtype:
                txt = txt.to(self.txt_in.weight.dtype)

        return_sref_aux = return_sref_enrichment or return_sref_entropy
        sref_timestep_weights = None
        sref_entropy_target_lower_bounds = None
        rope_fa_progress = None
        if self.use_frequency_aware_rope:
            if timesteps.ndim != 1:
                raise ValueError(f"Expected timesteps to be 1D for frequency-aware RoPE, got shape {tuple(timesteps.shape)}")
            expected_samples = len(img_seq_lens)
            if timesteps.shape[0] != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} timesteps for frequency-aware RoPE, got {timesteps.shape[0]}"
                )
            # In training, t=1 corresponds to the high-noise start and t=0 to the late denoise stage.
            rope_fa_progress = (1.0 - timesteps.to(torch.float32)).clamp_(0.0, 1.0)
        if return_sref_aux and sref_enrichment_timestep_weighting:
            if timesteps.ndim != 1:
                raise ValueError(f"Expected timesteps to be 1D for enrichment weighting, got shape {tuple(timesteps.shape)}")
            expected_samples = len(img_seq_lens)
            if timesteps.shape[0] != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} timesteps for enrichment weighting, got {timesteps.shape[0]}"
                )
            sref_timestep_weights = (1.0 - timesteps.to(torch.float32)).clamp_(0.0, 1.0)
            if sref_enrichment_timestep_weight_power != 1.0:
                sref_timestep_weights = sref_timestep_weights.pow(sref_enrichment_timestep_weight_power)
        if return_sref_entropy and sref_entropy_schedule_enabled:
            if timesteps.ndim != 1:
                raise ValueError(
                    f"Expected timesteps to be 1D for entropy scheduling, got shape {tuple(timesteps.shape)}"
                )
            expected_samples = len(img_seq_lens)
            if timesteps.shape[0] != expected_samples:
                raise ValueError(
                    f"Expected {expected_samples} timesteps for entropy scheduling, got {timesteps.shape[0]}"
                )
            timesteps_f = timesteps.to(torch.float32)
            progress = ((sref_entropy_schedule_start_timestep - timesteps_f) / sref_entropy_schedule_start_timestep).clamp_(
                0.0, 1.0
            )
            if sref_entropy_schedule_power != 1.0:
                progress = progress.pow(sref_entropy_schedule_power)
            active = (timesteps_f <= sref_entropy_schedule_start_timestep).to(torch.float32)
            lower_bound_delta = sref_entropy_schedule_end_lower_bound - sref_entropy_schedule_start_lower_bound
            sref_entropy_target_lower_bounds = active * (
                sref_entropy_schedule_start_lower_bound + lower_bound_delta * progress
            )

        if self.enable_zero_t_embed:
            timesteps = (
                torch.cat([timesteps[:, None], torch.zeros_like(timesteps[:, None])], dim=1).flatten().contiguous()
            )

        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.guidance_in is not None:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        if self.vector_in is not None and y is not None:
            vec = vec + self.vector_in(y)

        # 这里对变长数据的格式进行整理
        (
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
            img_seq_lens,
            txt_seq_lens,
            img_varlen_config,
            txt_varlen_config,
            img_txt_varlen_config,
            varlen_attention_config,
            img_pad_len,
            txt_pad_len,
        ) = self.prepare_for_transformer_blocks(
            img, img_ids, txt, txt_ids, vec, img_seq_lens, txt_seq_lens, zero_t_seq_lens
        )

        img = self.img_in(img.type_as(self.img_in.weight))

        txt = self.txt_in(txt)

        ids = cat_seq([img_ids, txt_ids], [img_varlen_config.split_index, txt_varlen_config.split_index])
        pe = self.pe_embedder(ids)
        cos = pe[None, :, 0, :, 0, 0].contiguous()
        sin = pe[None, :, 0, :, 1, 0].contiguous()
        pe = (cos, sin)

        joint_seq_lens = None
        normalized_sref_key_ranges = None
        normalized_sref_query_ranges = None
        if return_sref_aux or self.use_frequency_aware_rope:
            if sref_key_ranges is None:
                raise ValueError("sref_key_ranges must be provided when sref auxiliary regularization or RoPE FA is enabled.")
            joint_seq_lens = [int(img_len) + int(txt_len) for img_len, txt_len in zip(img_seq_lens, txt_seq_lens)]
            expected_samples = len(joint_seq_lens)
            if rope_fa_progress is not None and rope_fa_progress.shape[0] < expected_samples:
                pad_len = expected_samples - rope_fa_progress.shape[0]
                pad = torch.zeros(
                    pad_len,
                    dtype=rope_fa_progress.dtype,
                    device=rope_fa_progress.device,
                )
                rope_fa_progress = torch.cat([rope_fa_progress, pad], dim=0)
            if sref_timestep_weights is not None and sref_timestep_weights.shape[0] < expected_samples:
                pad_len = expected_samples - sref_timestep_weights.shape[0]
                pad = torch.zeros(
                    pad_len,
                    dtype=sref_timestep_weights.dtype,
                    device=sref_timestep_weights.device,
                )
                sref_timestep_weights = torch.cat(
                    [sref_timestep_weights, pad], dim=0
                )
            if (
                sref_entropy_target_lower_bounds is not None
                and sref_entropy_target_lower_bounds.shape[0] < expected_samples
            ):
                pad_len = expected_samples - sref_entropy_target_lower_bounds.shape[0]
                pad = torch.zeros(
                    pad_len,
                    dtype=sref_entropy_target_lower_bounds.dtype,
                    device=sref_entropy_target_lower_bounds.device,
                )
                sref_entropy_target_lower_bounds = torch.cat(
                    [sref_entropy_target_lower_bounds, pad], dim=0
                )
            if len(sref_key_ranges) > expected_samples:
                raise ValueError(
                    f"Expected at most {expected_samples} sref_key_ranges entries, got {len(sref_key_ranges)}"
                )
            normalized_sref_key_ranges = list(sref_key_ranges)
            if len(normalized_sref_key_ranges) < expected_samples:
                normalized_sref_key_ranges.extend([(0, 0)] * (expected_samples - len(normalized_sref_key_ranges)))

            for sample_idx, (seq_len, (k_start, k_end)) in enumerate(zip(joint_seq_lens, normalized_sref_key_ranges)):
                if k_start < 0 or k_end < k_start or k_end > seq_len:
                    raise ValueError(
                        f"Invalid sref key range {(k_start, k_end)} for sample {sample_idx} with seq_len {seq_len}"
                    )

            if sref_query_ranges is not None:
                if len(sref_query_ranges) > expected_samples:
                    raise ValueError(
                        f"Expected at most {expected_samples} sref_query_ranges entries, got {len(sref_query_ranges)}"
                    )
                normalized_sref_query_ranges = list(sref_query_ranges)
                if len(normalized_sref_query_ranges) < expected_samples:
                    normalized_sref_query_ranges.extend([(0, 0)] * (expected_samples - len(normalized_sref_query_ranges)))

                for sample_idx, (seq_len, (q_start, q_end)) in enumerate(zip(joint_seq_lens, normalized_sref_query_ranges)):
                    if q_start < 0 or q_end < q_start or q_end > seq_len:
                        raise ValueError(
                            f"Invalid sref query range {(q_start, q_end)} for sample {sample_idx} with seq_len {seq_len}"
                        )

        block_sref_aux = None
        for block_idx, block in enumerate(self.double_blocks):
            if return_sref_aux and block_idx == 0:
                block_out = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    img_varlen_config=img_varlen_config,
                    txt_varlen_config=txt_varlen_config,
                    varlen_attention_config=varlen_attention_config,
                    return_sref_enrichment=return_sref_enrichment,
                    return_sref_entropy=return_sref_entropy,
                    joint_seq_lens=joint_seq_lens,
                    sref_key_ranges=normalized_sref_key_ranges,
                    sref_query_ranges=normalized_sref_query_ranges,
                    sref_enrichment_lower_bound=sref_enrichment_lower_bound,
                    sref_enrichment_upper_bound=sref_enrichment_upper_bound,
                    sref_enrichment_eps=sref_enrichment_eps,
                    sref_entropy_lower_bound=sref_entropy_lower_bound,
                    sref_entropy_upper_bound=sref_entropy_upper_bound,
                    sref_entropy_eps=sref_entropy_eps,
                    sref_entropy_target_lower_bounds=sref_entropy_target_lower_bounds,
                    sref_timestep_weights=sref_timestep_weights,
                    rope_fa_progress=rope_fa_progress,
                )
                img, txt, block_sref_aux = block_out
            else:
                img, txt = block(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    img_varlen_config=img_varlen_config,
                    txt_varlen_config=txt_varlen_config,
                    varlen_attention_config=varlen_attention_config,
                    joint_seq_lens=joint_seq_lens,
                    sref_key_ranges=normalized_sref_key_ranges,
                    rope_fa_progress=rope_fa_progress,
                )

        # As some models might not have any single blocks, we should avoid
        # the pre- or post- process for single layers.
        if len(self.single_blocks) > 0:
            if self.is_shard:
                img = img.full_tensor()
                txt = txt.full_tensor()

            x = cat_seq([img, txt], [img_varlen_config.split_index, txt_varlen_config.split_index])

            if self.is_shard:
                x = LocalReplicateToDTensorSP(x, self.device_mesh)

            for block in self.single_blocks:
                x = block(
                    x,
                    vec=vec,
                    pe=pe,
                    x_varlen_config=img_txt_varlen_config,
                    varlen_attention_config=varlen_attention_config,
                )

            if self.is_shard:
                x = x.full_tensor()

            img, txt = split_seq(x, [img_varlen_config.seq_lens, txt_varlen_config.seq_lens])

            if self.is_shard:
                img = LocalReplicateToDTensorSP(img, self.device_mesh)
        else:
            # just to make sure the backward works well with DTensor
            if self.training and self.is_shard:
                img = img.full_tensor()
                txt = txt.full_tensor()
                img = img + txt.mean() * 0
                img = LocalReplicateToDTensorSP(img, self.device_mesh)

        img = self.final_layer(img, vec, img_varlen_config)

        if self.is_shard:
            img = img.full_tensor()

        if img_pad_len > 0:
            img = img[:-img_pad_len]

        if return_sref_aux:
            zero = torch.zeros((), device=img.device, dtype=torch.float32)
            sref_aux: dict[str, Tensor] = {}
            if return_sref_enrichment:
                sref_aux["loss_sref_enrichment"] = zero
                sref_aux["sref_enrichment"] = zero
            if return_sref_entropy:
                sref_aux["loss_sref_entropy"] = zero
                sref_aux["sref_entropy"] = zero
            if block_sref_aux is not None:
                for key, value in block_sref_aux.items():
                    if hasattr(value, "full_tensor"):
                        value = value.full_tensor()
                    sref_aux[key] = value
            return img, sref_aux
        return img
