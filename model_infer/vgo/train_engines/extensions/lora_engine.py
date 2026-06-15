import gc
import itertools
import os
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import peft
import torch
import torch.distributed.checkpoint
import torch.distributed.checkpoint as dcp
import torch.distributed.fsdp
import torch.nn
import torch.nn as nn
import torch.nn.grad
import torch.optim as optim
import torch.torch_version
from einops import repeat
from loguru import logger
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict, set_state_dict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.tensor.experimental import implicit_replication
from torch.optim.lr_scheduler import LambdaLR

from vgo.data.processor.naive_collect import PackData
from vgo.models.modules.distributed_ops import (
    DataRecorder,
    broadcast_tensors,
    gather_combine_object_from_tensor_parallel_group,
    get_local_start_end_for_tensor_split,
)
from vgo.models.modules.varlen_ops import VarLenConfig, cat_seq, split_seq_by_len_list
from vgo.models.transformers.model import VarLenDiT
from vgo.pipelines import NaivePipelineArgs
from vgo.train import TrainState
from vgo.train_engines import StepInfo
from vgo.train_engines.naive_policy import (
    DiTInputOutput,
    NoiseOffsetArgs,
    PackedVarlenFlowMatchingPolicy,
    TimeShiftArgs,
)
from vgo.utils.common_utils import GarbageCollection, combine_list
from vgo.utils.dist_utils import average_sync_dict, clip_grad_norm_
from vgo.utils.memory_utils import MemoryLeakDetector, MemoryMonitorArgs, TensorCleaner
from vgo.utils.timer import Timer

TIMER_DATA_LOAD_BATCH = "data_load_batch_total"
TIMER_DATA_RECORD_BATCH = "data_record_batch_total"
TIMER_DATA_ENCODE_BATCH = "data_encod_batch"
TIMER_ENCODE_IMAGE = "encode_image"
TIMER_ENCODE_TEXT = "encode_text"
TIMER_COMPUTE_FORWARD = "compute_forward"
TIMER_COMPUTE_LOSS = "compute_loss"
TIMER_COMPUTE_BACKWARD = "compute_backward"
TIMER_COMPUTE_OPTIMIZE = "compute_optimize"
TIMER_SYNC_LOG = "sync_log"
TIMER_MEMORY_CLEANUP = "memory_cleanup"


def _supports_async_dcp_save() -> bool:
    if not torch.distributed.is_initialized():
        return False
    try:
        pg = torch.distributed.distributed_c10d._get_default_group()
    except Exception:
        return False
    device_types = getattr(pg, "_device_types", ())
    try:
        return torch.device("cpu") in device_types
    except TypeError:
        return "cpu" in device_types


@dataclass
class OptimArgs:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    epsilon: float = 1e-8
    beta1: float = 0.9
    beta2: float = 0.999
    grad_norm_clip: float = 1.0
    scheduler: str = "constant_with_warmup"  # Currently only constant_with_warmup implemented outside deepspeed
    warmup: int = 1000  # Number of *global* steps for warmup
    use_grad_scaler: bool = False


@dataclass
class PrecisionArgs:
    param_dtype: str = "bf16"
    reduce_dtype: str = "fp32"
    optimizer_dtype: str = "fp32"


PRECISION_TYPE: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class PolicyArgs:
    weighting_scheme: str = "truncated_logit_normal"
    shift_timesteps: bool = True
    recon_loss_weight: float = 0.0
    style_loss_weight: float = 0.0
    sref_enrichment_loss_weight: float = 0.0
    sref_enrichment_lower_bound: float = 0.08
    sref_enrichment_upper_bound: float = 0.5
    sref_enrichment_eps: float = 1e-6
    sref_entropy_loss_weight: float = 0.0
    sref_entropy_lower_bound: float = 0.06
    sref_entropy_upper_bound: float = 0.14
    sref_entropy_eps: float = 1e-6
    sref_entropy_schedule_enabled: bool = False
    sref_entropy_schedule_start_timestep: float = 0.75
    sref_entropy_schedule_start_lower_bound: float = 0.06
    sref_entropy_schedule_end_lower_bound: float = 0.10
    sref_entropy_schedule_power: float = 2.0
    sref_enrichment_noise_query_only: bool = False
    sref_enrichment_timestep_weighting: bool = False
    sref_enrichment_timestep_weight_power: float = 1.0

    noise_offset: NoiseOffsetArgs = field(default_factory=NoiseOffsetArgs)
    time_shift: TimeShiftArgs = field(default_factory=TimeShiftArgs)


@dataclass
class LoRAArgs:
    rank: int = 16
    alpha: int = 8
    init_lora_weights: str = "gaussian"


def generate_image_position_qwen_image_ids(h_feat, w_feat, bs, t_offset, device):
    img_ids = torch.zeros(h_feat, w_feat, 3, device=device, dtype=torch.float32)
    img_ids[..., 0] = t_offset
    img_ids[..., 1] = img_ids[..., 1] + (
        torch.arange(h_feat, device=device, dtype=torch.float32)[:, None] - (h_feat - h_feat // 2)
    )
    img_ids[..., 2] = img_ids[..., 2] + (
        torch.arange(w_feat, device=device, dtype=torch.float32)[None, :] - (w_feat - w_feat // 2)
    )
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img_ids


@dataclass
class EngineArgs:
    data_config: str
    model_precision: str = "bf16"  # "fp32", "fp16", "bf16"
    ce_loss_weight: float = 1.0
    enable_tensor_parallel: bool = True
    shard_text_encoder: bool = False

    pipe: NaivePipelineArgs = field(default_factory=NaivePipelineArgs)
    optim: OptimArgs = field(default_factory=OptimArgs)
    policy: PolicyArgs = field(default_factory=PolicyArgs)
    precision: PrecisionArgs = field(default_factory=PrecisionArgs)
    memory_monitor: MemoryMonitorArgs = field(default_factory=MemoryMonitorArgs)
    lora_config: LoRAArgs = field(default_factory=LoRAArgs)


def build_models(pipe_args, lora_args: LoRAArgs, device: torch.device, dtype: torch.dtype):
    def all_linear(module: torch.nn.Module, prefix=""):
        for name, child in module.named_children():
            current_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, torch.nn.Linear):
                yield current_name
            else:
                yield from all_linear(child, prefix=current_name)

    ema_updater = None

    if isinstance(pipe_args, NaivePipelineArgs):
        pipe = pipe_args

        ae = pipe.build_ae(device)

        dit: torch.nn.Module = pipe.build_dit(device, dtype)

        dit.train()

        llm_encoder = pipe.build_llm_encoder(device, dtype) if not pipe.fuse_llm_dit else dit.llm_encoder

        target_modules = [x for x in list(all_linear(dit)) if "double_blocks" in x or "single_blocks" in x]
        dit_lora_config = peft.LoraConfig(
            init_lora_weights=lora_args.init_lora_weights,  # type: ignore
            r=lora_args.rank,
            lora_alpha=lora_args.alpha,
            target_modules=target_modules,
        )

        peft_dit: peft.peft_model.PeftModel = peft.get_peft_model(dit, dit_lora_config)

        logger.success("build components done.")

        return (
            dict(
                llm_encoder=llm_encoder,
                ae=ae,
                dit=peft_dit,
            ),
            ema_updater,
        )

    raise NotImplementedError(f"{type(pipe_args)=} not implemented yet.")


def build_optimizer_and_scheduler(optim_args: OptimArgs, model: nn.Module):  # noqa: C901
    """Builds the optimizer and learning rate scheduler."""

    # Determine optimizer class (respect DeepSpeed config)
    optimizer_cls = optim.AdamW

    params_to_optimize = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_optimize.append(param)
            logger.debug(f"{name=} {param.shape=} {param.dtype=}")

    optimizer = optimizer_cls(
        params_to_optimize,
        lr=optim_args.learning_rate,
        betas=(optim_args.beta1, optim_args.beta2),
        weight_decay=optim_args.weight_decay,  # type: ignore
        eps=optim_args.epsilon,
    )

    # Determine scheduler class
    scheduler_cls = LambdaLR

    num_warmup_steps = optim_args.warmup
    if scheduler_cls is LambdaLR:
        # Simple linear warmup scheduler
        if optim_args.scheduler == "constant_with_warmup":

            def lr_lambda(step):
                return min(1.0, step / max(1, num_warmup_steps))  # Avoid division by zero
        elif optim_args.scheduler == "constant_with_decline":

            def lr_lambda(step):
                if step < num_warmup_steps:
                    rate = 10.0
                else:
                    rate = max(1.0, 10 * (num_warmup_steps - (step - num_warmup_steps)) / max(1, num_warmup_steps))
                return rate
        elif optim_args.scheduler == "constant_with_raise_decline":

            def lr_lambda(step):
                if step < num_warmup_steps:
                    rate = min(1.0, step / max(1, num_warmup_steps)) * 10.0
                elif step >= num_warmup_steps and step < num_warmup_steps * 2:
                    rate = 10.0
                elif step >= num_warmup_steps * 2 and step < num_warmup_steps * 3:
                    step -= num_warmup_steps * 2
                    rate = min(1.0, (num_warmup_steps - step) / max(1, num_warmup_steps)) * 9.0 + 1.0
                else:
                    rate = 1.0
                return rate

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)  # type: ignore
        logger.info(f"Using LambdaLR scheduler with {num_warmup_steps} warmup steps.")
    else:
        # Add other schedulers here if needed
        raise NotImplementedError(f"Scheduler type not implemented for {scheduler_cls}")

    scaler = ShardedGradScaler() if optim_args.use_grad_scaler else None

    return optimizer, lr_scheduler, scaler


def build_dataloader(data_config_path, micro_batch_size, llm_processor, seed, world_mesh: DeviceMesh = None):
    if world_mesh["tp_w_sp"].get_local_rank() != 0:
        import itertools

        return itertools.repeat(None)

    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from vgo.data.sequence_processor import NaiveSequenceCollateFn
    from vgo.data.vaultloader import DataConfigArgs, LoadBalancedVaultDataset

    base_config = OmegaConf.structured(DataConfigArgs)
    merged = OmegaConf.merge(base_config, OmegaConf.load(data_config_path))
    data_config: DataConfigArgs = OmegaConf.to_object(merged)

    pipeline = NaiveSequenceCollateFn(
        base_size=data_config.base_size,
        base_size_weights=data_config.base_size_weights,
        enable_random_degrade=data_config.enable_random_degrade,
    )

    dataset = LoadBalancedVaultDataset(
        data_config.train_database,
        source_weights=data_config.source_weights,
        base_size=data_config.base_size,
        base_size_weights=data_config.base_size_weights,
        max_length=4096 * micro_batch_size,
        load_balance_buffer_factor=max(4, world_mesh["dp"].size()),
        dp_rank=world_mesh["dp"].get_local_rank(),
        dp_size=world_mesh["dp"].size(),
        text_dropout_rate=data_config.text_dropout_rate,
    )
    dataloader_num_workers = max(0, int(os.environ.get("VGO_DATALOADER_NUM_WORKERS", "16")))

    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        num_workers=dataloader_num_workers,
        collate_fn=pipeline,
        pin_memory=True,
        in_order=True,
    )

    logger.info(f"图像随机质量退化: {data_config.enable_random_degrade}")
    logger.info(f"Dataloader workers: {dataloader_num_workers}")

    return train_dataloader


def build_policy(policy: PolicyArgs):
    return PackedVarlenFlowMatchingPolicy(
        weighting_scheme=policy.weighting_scheme,
        shift_timesteps=policy.shift_timesteps,
        seed=42 + torch.distributed.get_rank(),  # use different seed for each rank
        noise_offset_policy=policy.noise_offset,
        time_shift_policy=policy.time_shift,
        recon_loss_weight=policy.recon_loss_weight,
        style_loss_weight=policy.style_loss_weight,
        sref_enrichment_loss_weight=policy.sref_enrichment_loss_weight,
        sref_enrichment_lower_bound=policy.sref_enrichment_lower_bound,
        sref_enrichment_upper_bound=policy.sref_enrichment_upper_bound,
        sref_enrichment_eps=policy.sref_enrichment_eps,
        sref_entropy_loss_weight=policy.sref_entropy_loss_weight,
        sref_entropy_lower_bound=policy.sref_entropy_lower_bound,
        sref_entropy_upper_bound=policy.sref_entropy_upper_bound,
        sref_entropy_eps=policy.sref_entropy_eps,
        sref_entropy_schedule_enabled=policy.sref_entropy_schedule_enabled,
        sref_entropy_schedule_start_timestep=policy.sref_entropy_schedule_start_timestep,
        sref_entropy_schedule_start_lower_bound=policy.sref_entropy_schedule_start_lower_bound,
        sref_entropy_schedule_end_lower_bound=policy.sref_entropy_schedule_end_lower_bound,
        sref_entropy_schedule_power=policy.sref_entropy_schedule_power,
        sref_enrichment_noise_query_only=policy.sref_enrichment_noise_query_only,
        sref_enrichment_timestep_weighting=policy.sref_enrichment_timestep_weighting,
        sref_enrichment_timestep_weight_power=policy.sref_enrichment_timestep_weight_power,
    )


@torch.no_grad()
def _encode_image(
    ae: torch.nn.Module,
    images: list[torch.Tensor],
    policy: PackedVarlenFlowMatchingPolicy,
    device,
    device_mesh: DeviceMesh,
    generator: torch.Generator,
) -> tuple:
    """Encodes images into latents using the VAE."""
    ae.eval()  # Ensure AE is in eval mode
    vae_dtype = torch.float32  # Use higher precision for VAE stability
    latents_list: list[torch.Tensor] = []

    # Input to AE should be in float32 and range [-1, 1]
    start_idx, end_idx = get_local_start_end_for_tensor_split(
        len(images), device_mesh.get_local_rank(), device_mesh.size()
    )

    if start_idx == end_idx:
        return (
            torch.Tensor().to(device, vae_dtype).reshape(-1, 64),
            torch.Tensor().to(device, vae_dtype).reshape(-1, 64),
            torch.Tensor().to(device, vae_dtype).reshape(-1, 64),
            torch.Tensor().to(device, vae_dtype).reshape(-1, 64),
            torch.Tensor().to(device, vae_dtype),
            torch.Tensor().to(torch.int32),
            True,
        )

    for image_idx in range(start_idx, end_idx):  # type: ignore
        #  (assuming standard VAE)
        micro_batch = images[image_idx]
        micro_batch_float = micro_batch.to(device=device, dtype=vae_dtype)
        latents = ae.encode(micro_batch_float[None, :, None]).latent_dist.sample(generator)[0, :, 0]  # type: ignore
        latents_list.append(latents.flatten(-2, -1))

    latents = torch.cat(latents_list, dim=-1).T  # , latents_seq_len  # (B (H x W)) x C

    latents_mean = torch.tensor(ae.config.latents_mean).view(1, -1)  # type: ignore
    latents_mean = latents_mean.to(latents.device, latents.dtype)

    latents_std = torch.tensor(ae.config.latents_std).view(1, -1)  # type: ignore
    latents_std = latents_std.to(latents.device, latents.dtype)

    latents = (latents - latents_mean) / latents_std

    images_size = torch.tensor([x.shape[-2:] for x in images][start_idx:end_idx])
    x0, x1, xt, vt, t, token_lens = policy.compute_noisy_latent(latents=latents, images_size=images_size)

    return x0, x1, xt, vt, t, token_lens, False


@torch.no_grad()
def prepare_txt_for_dit(txt, txt_lens, max_img_ids):
    txt_ids = [
        torch.arange(x, device=x.device, dtype=torch.float32)[:, None].repeat(1, 3) + max_img_ids[i] + 1
        for i, x in enumerate(txt_lens)
    ]

    txt_ids = torch.cat(txt_ids, dim=0)
    return txt, txt_lens.tolist(), txt_ids


@torch.no_grad()
def prepare_img_for_dit(vae_latents, noisy_vae_latents, v_target, timesteps, vae_token_size, images_count_in_sequence):
    device = torch.cuda.current_device()
    # 将 vae latent 拆成每张图的
    vae_token_lens = vae_token_size[:, 0] * vae_token_size[:, 1]

    vae_latents_for_each_image = vae_latents.split(vae_token_lens.tolist())
    noisy_vae_latents_for_each_image = noisy_vae_latents.split(vae_token_lens.tolist())
    v_target_for_each_image = v_target.split(vae_token_lens.tolist())

    image_start_end_index_for_each_seq = np.cumsum([0, *images_count_in_sequence])

    ref_vae_latents = [
        vae_latents_for_each_image[x : y - 1] for x, y in itertools.pairwise(image_start_end_index_for_each_seq)
    ]
    noisy_vae_latents = [
        noisy_vae_latents_for_each_image[y - 1] for _, y in itertools.pairwise(image_start_end_index_for_each_seq)
    ]
    v_target = [v_target_for_each_image[y - 1] for _, y in itertools.pairwise(image_start_end_index_for_each_seq)]
    timesteps = torch.cat([timesteps[y - 1 : y] for _, y in itertools.pairwise(image_start_end_index_for_each_seq)])

    # compute pe
    vae_token_size_cpu = vae_token_size.tolist()

    position_ids_for_each_image = [
        generate_image_position_qwen_image_ids(h, w, 1, 0, device)[0] for h, w in vae_token_size_cpu
    ]

    ref_img_lens = [sum(y.shape[0] for y in x) for x in ref_vae_latents]
    ref_vae_latents = combine_list(ref_vae_latents)
    ref_img = (
        torch.cat(ref_vae_latents)
        if len(ref_vae_latents) > 0
        else torch.zeros((0, vae_latents.shape[1])).to(vae_latents)
    )
    ref_img_ids = combine_list(
        [position_ids_for_each_image[x : y - 1] for x, y in itertools.pairwise(image_start_end_index_for_each_seq)]
    )
    ref_img_ids = (
        torch.cat(ref_img_ids).clone() if len(ref_img_ids) > 0 else torch.zeros((0, 3)).to(device, torch.float32)
    )
    # 参考图 T=1，可以自行修改
    ref_img_ids[:, 0] += 1

    img_lens = [x.shape[0] for x in noisy_vae_latents]
    img = torch.cat(noisy_vae_latents)
    img_ids = torch.cat(
        [position_ids_for_each_image[y - 1] for _, y in itertools.pairwise(image_start_end_index_for_each_seq)]
    )
    v_target = torch.cat(v_target)

    return img, img_ids, img_lens, ref_img, ref_img_ids, ref_img_lens, v_target, timesteps


@torch.no_grad()
def _encode_text(
    models,
    pack_data: PackData,
    device_mesh: DeviceMesh,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encodes text prompts using QwenVL and CLIP encoders with dropout."""
    llm_encoder = models["llm_encoder"]

    # Encoders should handle device and dtype internally based on their init
    start_idx, end_idx = get_local_start_end_for_tensor_split(
        len(pack_data.text), device_mesh.get_local_rank(), device_mesh.size()
    )
    device = torch.cuda.current_device()

    llm_encoder.unshard_module()

    if start_idx < end_idx:
        texts = pack_data.text[start_idx:end_idx]  # type: ignore
        ref_images = pack_data.ref_images[start_idx:end_idx]
        task_types = [x.task_type for x in pack_data.sequences[start_idx:end_idx]]

        # 只用管下面这个
        txt, txt_lens = llm_encoder(texts, ref_images, task_types)
        txt_lens_pt = torch.tensor(txt_lens).to(device, torch.int32)
    else:
        txt = torch.Tensor([]).reshape(-1, llm_encoder.hidden_size()).to(device, torch.bfloat16)
        txt_lens_pt = torch.Tensor([]).reshape(-1, 2).to(device, torch.int32)

    llm_encoder.reshard_module()

    return txt, txt_lens_pt


def generate_image_position_ids(h_feat, w_feat, bs, device):
    img_ids = torch.zeros(h_feat, w_feat, 3, device=device, dtype=torch.float32)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h_feat, device=device, dtype=torch.float32)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w_feat, device=device, dtype=torch.float32)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)
    return img_ids


def generate_text_position_ids(bs, seq_len, device, dtype):
    text_ids = torch.zeros(bs, seq_len, 3, device=device, dtype=dtype)
    return text_ids


class Engine:
    def __init__(
        self,
        config: EngineArgs,
        models,
        ema_updater,
        optimizer,
        lr_scheduler,
        scaler,
        dataloader,
        policy,
        data_recorder,
        device_mesh,
        gradient_accumulation_steps,
        non_activation_checkpointing_every,
    ) -> None:
        self.config = config

        self.timers = {
            n: Timer(name=n)
            for n in [
                TIMER_DATA_ENCODE_BATCH,
                TIMER_DATA_LOAD_BATCH,
                TIMER_DATA_RECORD_BATCH,
                TIMER_ENCODE_IMAGE,
                TIMER_ENCODE_TEXT,
                TIMER_COMPUTE_FORWARD,
                TIMER_COMPUTE_BACKWARD,
                TIMER_COMPUTE_OPTIMIZE,
                TIMER_COMPUTE_LOSS,
                TIMER_SYNC_LOG,
                TIMER_MEMORY_CLEANUP,
            ]
        }

        self.device = torch.cuda.current_device()
        self.dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[config.model_precision]

        self.models = models
        self.ema_updater = ema_updater
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.dataloader = dataloader
        self.data_recorder: DataRecorder = data_recorder
        self.policy: PackedVarlenFlowMatchingPolicy = policy

        self.dit: VarLenDiT = self.models["dit"]
        self.device_mesh: DeviceMesh = device_mesh

        self.dit.train()

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.non_activation_checkpointing_every = non_activation_checkpointing_every
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}.")
        logger.info(f"Disable activation checkpointing every: {non_activation_checkpointing_every} layer(s).")

        self.gc_handler = GarbageCollection(gc_freq=50)
        self.ce_loss_weight = self.config.ce_loss_weight
        logger.info(f"Use Ce Loss Weight: {self.ce_loss_weight}")

        # Initialize memory monitor
        self.memory_monitor = None
        if config.memory_monitor.enable:
            self.memory_monitor = MemoryLeakDetector(
                device=self.device,
                check_interval=config.memory_monitor.check_interval,
                snapshot_interval=config.memory_monitor.snapshot_interval,
                alert_threshold_mb=config.memory_monitor.alert_threshold_mb,
                enable_snapshot=config.memory_monitor.enable_snapshot,
                track_tensors=config.memory_monitor.track_tensors,
                track_references=config.memory_monitor.track_references,
            )
            logger.warning("显存泄漏检测器：开启 ✅")
        else:
            logger.info("显存泄漏检测器：禁用")

        self.aggressive_cleanup = config.memory_monitor.aggressive_cleanup
        self.current_step = 0

    def get_timer(self) -> dict[str, Timer]:
        return self.timers

    def set_logdir(self, logdir, exp_id):
        self.logdir = logdir

        if self.device_mesh["tp_w_sp"].get_local_rank() == 0:
            dp_rank = self.device_mesh["dp"].get_local_rank()
            logger.info(
                f"设置 Dataloader 的错误日志保存在 `{(Path(logdir) / 'dataloader' / f'DP-RANK-{exp_id}-{dp_rank}.log').as_posix()}` 。"  # noqa: E501
            )
            self.dataloader.dataset.set_logfile(
                (Path(logdir) / "dataloader" / f"DP-RANK-{exp_id}-{dp_rank}.log").as_posix()
            )

    @classmethod
    def build(
        cls,
        config: EngineArgs,
        device_mesh: DeviceMesh,
        dataloader_seed: int = 42 * 42,
        micro_batch_size: int = 8,
        use_data_recorder: bool = False,
        gradient_accumulation_steps: int = 1,
        non_activation_checkpointing_every: int = -1,
    ):
        assert not config.enable_tensor_parallel, "LoRA 训练下请采用无TP的模式"
        assert device_mesh["tp_w_sp"].size() == 1, (
            f"LoRA 训练下，TP SIZE 务必为 1 ，当前为 {device_mesh['tp_w_sp'].size()}"
        )

        device = torch.cuda.current_device()
        dtype = PRECISION_TYPE[config.model_precision]

        models, ema_updater = build_models(config.pipe, config.lora_config, device=device, dtype=dtype)

        dit: VarLenDiT = models["dit"]

        dit.apply_ac(non_activation_checkpointing_every, apply_to_llm=False)

        dit.apply_compile(apply_to_llm=False)

        # 务必在 apply fsdp 之前配置好可优化参数，否则 `optimizer_dtype` 不会改变优化器中的参数类型
        # start specify trainable parameters
        dit.requires_grad_(False)

        for name, params in dit.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                params.requires_grad_(True)

        if not config.enable_tensor_parallel:
            from vgo.utils.dist_utils import ParallelDims, device_type

            parallel_dims = ParallelDims(
                dp=device_mesh.size() // torch.cuda.device_count(),
                tp_w_sp=torch.cuda.device_count(),
                world_size=device_mesh.size(),
            )
            fsdp_device_mesh = parallel_dims.build_mesh(device_type=device_type)
        else:
            fsdp_device_mesh = device_mesh

        should_apply_fsdp = fsdp_device_mesh.size() > 1
        if should_apply_fsdp:
            dit.apply_fsdp(
                world_mesh=fsdp_device_mesh,
                param_dtype=PRECISION_TYPE[config.precision.param_dtype],
                reduce_dtype=PRECISION_TYPE[config.precision.reduce_dtype],
                optimizer_dtype=PRECISION_TYPE[config.precision.optimizer_dtype],
                reshard_after_forward=True,
                apply_to_llm_encoder=False,
            )
        else:
            logger.info("Skip DiT FSDP because world_size=1; keep the full model on the single device.")

        if config.shard_text_encoder and should_apply_fsdp:
            models["llm_encoder"].apply_fsdp(world_mesh=device_mesh, use_hsdp=True)  # type: ignore
        else:
            logger.info("Text Encoder 不会切分，每个 GPU 会存储完整的参数")

        optimizer, lr_scheduler, scaler = build_optimizer_and_scheduler(config.optim, models["dit"])  # type: ignore

        dataloader = build_dataloader(
            data_config_path=config.data_config,
            micro_batch_size=micro_batch_size,
            llm_processor=models["llm_encoder"].processor,  # type: ignore
            seed=dataloader_seed,
            world_mesh=device_mesh,
        )

        policy = build_policy(config.policy)

        # 设置 source list names，以记录每个 source 的 loss 权重
        if device_mesh["tp_w_sp"].get_local_rank() == 0:
            policy.all_source_names = dataloader.dataset.source_list  # type: ignore
        else:
            policy.all_source_names = None
        policy.all_source_names = broadcast_tensors(policy.all_source_names, device_mesh=device_mesh["tp_w_sp"])

        data_recorder = (
            DataRecorder(
                save_path=None,
                world_mesh=device_mesh,
            )
            if use_data_recorder
            else None
        )

        return cls(
            config,
            models,
            ema_updater,
            optimizer,
            lr_scheduler,
            scaler,
            dataloader,
            policy,
            data_recorder,
            device_mesh,
            gradient_accumulation_steps,
            non_activation_checkpointing_every,
        )

    def dit_forward(self, batch: tuple[PackData, DiTInputOutput]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the DiT model, handling different modes (train, ema, base).
        Expects latents `img` with shape (LxC).
        """

        _, dit_input_output = batch
        img = dit_input_output.img
        prev_img = dit_input_output.ref_img
        txt = dit_input_output.txt
        t = dit_input_output.timesteps
        target = dit_input_output.v_target
        img_lens = dit_input_output.img_lens
        prev_img_lens = dit_input_output.ref_img_lens
        txt_lens = dit_input_output.txt_lens
        dit_img_position_ids = dit_input_output.img_ids
        prev_dit_img_position_ids = dit_input_output.ref_img_ids
        dit_txt_position_ids = dit_input_output.txt_ids

        device = img.device

        dit_model_to_call = self.dit

        # VAE token should be put in img transformer
        img_varlen_config = VarLenConfig.from_seq_lens(img_lens, device)
        prev_img_varlen_config = VarLenConfig.from_seq_lens(prev_img_lens, device)
        all_img = cat_seq([img, prev_img], [img_varlen_config.split_index, prev_img_varlen_config.split_index])
        all_dit_img_position_ids = cat_seq(
            [dit_img_position_ids, prev_dit_img_position_ids],
            [img_varlen_config.split_index, prev_img_varlen_config.split_index],
        )
        all_img_lens = [x + y for x, y in zip(img_lens, prev_img_lens)]

        # DiT forward call
        # Guidance scale passed here

        # Qwen 2511 的设置，暂时不清楚是否有增益
        zero_t_seq_lens = [(x, 0) for x in prev_img_lens] if self.dit.enable_zero_t_embed else None  # type: ignore

        pred: torch.Tensor = dit_model_to_call(  # Use the potentially wrapped model
            img=all_img,
            img_ids=all_dit_img_position_ids,
            txt=txt,
            txt_ids=dit_txt_position_ids,
            y=None,
            timesteps=t,
            img_seq_lens=all_img_lens,
            txt_seq_lens=txt_lens,
            guidance=None,  # Check if DiT uses this directly
            zero_t_seq_lens=zero_t_seq_lens,
        )
        pred = pred.view_as(all_img)

        pred, _ = split_seq_by_len_list(pred, [img_lens, prev_img_lens])

        return pred, target, t.repeat_interleave(torch.IntTensor(img_lens).to(t.device, torch.int), dim=0)  # type: ignore

    def get_trainable_params(self):
        dit_params = [params for params in self.dit.parameters() if params.requires_grad]

        return dit_params

    def get_trainable_param_names(self):
        dit_params_name = [name for name, params in self.dit.named_parameters() if params.requires_grad]
        dit_params_name = [
            x.replace("._orig_mod", "").replace("._checkpoint_wrapped_module", "") for x in dit_params_name
        ]

        return dit_params_name

    def set_init_train_state(self, train_state: TrainState):
        if self.data_recorder is not None:
            self.data_recorder.current_iterations = train_state.global_step

        if self.device_mesh["tp_w_sp"].get_local_rank() == 0:
            self.dataloader.dataset.resume_from(train_state.global_step, self.device_mesh["dp"].size())  # type: ignore

        self.current_step = train_state.global_step

    def _aggressive_cleanup(self):
        """Aggressively clean up memory."""
        with self.timers[TIMER_MEMORY_CLEANUP]:
            # Empty CUDA cache
            torch.cuda.empty_cache()

            # Force Python garbage collection
            gc.collect()

    def save_data_record(self, save_folder):
        if self.data_recorder is None:
            logger.warning("Skip saving data record because data_recorder is disabled.")
            return
        self.data_recorder.save_path = save_folder
        self.data_recorder._save()
        logger.success(
            f"Data record is saved successfully to {save_folder} for step {self.data_recorder.current_iterations}."
        )

    def load_checkpoint(self, load_path):
        # Load the state

        # optimizer.step() will be called in `get_state_dict` function, use `implicit_replication`
        # to avoid `_foreach_*_` ops errors
        with implicit_replication():
            model_state_dict, optimizer_state_dict = get_state_dict(self.dit, self.optimizer)

        # filter out llm params
        if self.dit.llm_encoder is not None:
            to_be_remove_key = [
                k for k in model_state_dict if "llm_encoder" in k and "lora" not in k and "vae_fusion_mapping" not in k
            ]
            for k in to_be_remove_key:
                model_state_dict.pop(k)

        state_dict = {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }
        if torch.__version__ >= torch.torch_version.TorchVersion("2.9.1"):
            dcp.load(
                state_dict,
                checkpoint_id=load_path,
                planner=dcp.default_planner.DefaultLoadPlanner(allow_partial_load=True),
            )
        else:
            dcp.load(state_dict, checkpoint_id=load_path)
        missing_keys, unexpected_keys = set_state_dict(
            self.dit,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            options=StateDictOptions(strict=False),
        )
        if len(unexpected_keys) > 0:
            raise ValueError(f"loading checkpoint from `{load_path}`, got {unexpected_keys=}")
        for missing_key in missing_keys:
            # 只有 Qwen 的非可学习参数才会是 missing 的
            if not ("llm_encoder" in missing_key and "lora" not in missing_key):
                raise ValueError(f"Unexpected missing key in {missing_key=}")

        # save dataloader
        if hasattr(self.dataloader, "state_dict"):
            dataloader_state_dict = torch.load(
                (Path(load_path) / f"dataloader_{self.device_mesh.get_rank()}.pth").as_posix(),
            )
            self.dataloader.load_state_dict(dataloader_state_dict)
            logger.success(f"Rank {self.device_mesh.get_rank()}: load dataloader status.")

        logger.success(f"Successfully resumed state from {load_path}.")

    #     if hasattr(self, "_staging_future"):
    #         if self._staging_future is not None:
    #             self._staging_future.result()
    #             self._staging_future = None

    @torch.no_grad()
    def save_checkpoint(self, checkpoint_folder: str):
        if getattr(self, "_async_save_future", None) is not None:
            self._async_save_future.result()  # type: ignore
            self._async_save_future = None

        # Save the state
        model_state_dict, optimizer_state_dict = get_state_dict(self.dit, self.optimizer)

        # filter out llm params
        if self.dit.llm_encoder is not None:
            to_be_remove_key = [
                k for k in model_state_dict if "llm_encoder" in k and "lora" not in k and "vae_fusion_mapping" not in k
            ]
            for k in to_be_remove_key:
                model_state_dict.pop(k)

        checkpoint = {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

        # use PIN Memory will lead to OOM
        # if torch.__version__ >= torch.torch_version.TorchVersion("2.9.1"):
        #     from torch.distributed.checkpoint.staging import DefaultStager, StagingOptions
        #     from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType

        #     self.gc_handler.collect("GC collection invoked by save_checkpointe.")

        #     self.stager = DefaultStager(StagingOptions(True, True, True, False))
        #     result = dcp.async_save(
        #         checkpoint,
        #         checkpoint_id=checkpoint_folder,
        #         process_group=self.pg,
        #         async_checkpointer_type=AsyncCheckpointerType.PROCESS,
        #         async_stager=self.stager,
        #     )
        #     self._async_save_future = result.upload_completion
        #     self._staging_future = result.staging_completion
        #     self._staging = True
        # if torch.__version__ >= torch.torch_version.TorchVersion("2.9.1"):
        #     from torch.distributed.checkpoint import HuggingFaceStorageWriter
        #     from torch.distributed.checkpoint._consolidate_hf_safetensors import (
        #         consolidate_safetensors_files_on_every_rank,
        #     )

        #     self.gc_handler.collect("GC collection invoked by save_checkpointe.")
        #     storage_writer = HuggingFaceStorageWriter(
        #         path=checkpoint_folder,
        #         save_distributed=True,
        #         enable_consolidation=True,
        #     )
        #     self._async_save_future = dcp.async_save(
        #         checkpoint, storage_writer=storage_writer, checkpoint_id=checkpoint_folder, process_group=self.pg
        #     )
        #     self.gc_handler.collect("GC collection invoked by save_checkpointe.")

        # else:
        self.gc_handler.collect("GC collection invoked by save_checkpointe.")
        if torch.__version__ >= torch.torch_version.TorchVersion("2.9.1"):
            dcp.save(checkpoint, checkpoint_id=checkpoint_folder)
        elif _supports_async_dcp_save():
            self._async_save_future = dcp.async_save(checkpoint, checkpoint_id=checkpoint_folder)
        else:
            logger.warning(
                "Async checkpoint save requires a CPU backend in the process group; falling back to synchronous save."
            )
            dcp.save(checkpoint, checkpoint_id=checkpoint_folder)
        self.gc_handler.collect("GC collection invoked by save_checkpointe.")

        # save dataloader
        if hasattr(self.dataloader, "state_dict"):
            Path(checkpoint_folder).mkdir(exist_ok=True)
            torch.save(
                self.dataloader.state_dict(),
                (Path(checkpoint_folder) / f"dataloader_{self.device_mesh.get_rank()}.pth").as_posix(),
            )
            logger.success(f"Rank {self.device_mesh.get_rank()}: save dataloader status.")

        logger.success(f"Checkpoint saved successfully to {checkpoint_folder}")

    def batch_generator(self) -> Iterator:
        """Wraps the dataloader to handle online encoding and epoch counting."""

        random_generator = torch.Generator(device=torch.cuda.current_device())
        random_generator.manual_seed(1234 + self.device_mesh.get_rank())

        while True:
            for batch_idx, batch in enumerate(self.dataloader):
                batch: PackData | None = batch
                batch = PackData.sync_tp(batch, self.device_mesh["tp_w_sp"])

                try:
                    split_outputs_ = {}

                    with torch.no_grad():
                        # 仅为了和 macro sequence 用法做对齐
                        _image_list = combine_list([[*x, y] for x, y in zip(batch.ref_images, batch.target_images)])
                        # 1. Encode Images (VAE) -> Latents
                        self.timers[TIMER_ENCODE_IMAGE].start()
                        (
                            split_outputs_["vae_latents"],
                            _,
                            split_outputs_["noisy_vae_latents"],
                            split_outputs_["v_target"],
                            split_outputs_["timesteps"],
                            split_outputs_["vae_token_size"],
                            is_pad_split,
                        ) = _encode_image(
                            self.models["ae"],
                            _image_list,
                            self.policy,
                            self.device,
                            self.device_mesh["tp_w_sp"],
                            generator=random_generator,
                        )
                        self.timers[TIMER_ENCODE_IMAGE].stop()

                        self.timers[TIMER_ENCODE_TEXT].start()
                        split_outputs_["txt"], split_outputs_["txt_lens"] = _encode_text(
                            self.models, batch, self.device_mesh["tp_w_sp"]
                        )
                        self.timers[TIMER_ENCODE_TEXT].stop()

                        gather_outputs = gather_combine_object_from_tensor_parallel_group(
                            split_outputs_, self.device_mesh["tp_w_sp"]
                        )

                        # 以上代码拿到了每个 Sequence 的 VAE Token 以及 Qwen 的 Prompt，需要将其 Pack 为变长的 DiT 输入
                        # 以下繁琐的代码主要在处理这个部分

                        # 将 VAE token 拆分为参考图和目标图
                        vae_token_size = gather_outputs["vae_token_size"]
                        vae_latents = gather_outputs["vae_latents"]
                        noisy_vae_latents = gather_outputs["noisy_vae_latents"]
                        v_target = gather_outputs["v_target"]
                        timesteps = gather_outputs["timesteps"]
                        images_count_in_sequence = [len(x.image) for x in batch.sequences]

                        img, img_ids, img_lens, ref_img, ref_img_ids, ref_img_lens, v_target, timesteps = (
                            prepare_img_for_dit(
                                vae_latents,
                                noisy_vae_latents,
                                v_target,
                                timesteps,
                                vae_token_size,
                                images_count_in_sequence,
                            )
                        )

                        #
                        txt = gather_outputs["txt"]
                        txt_lens = gather_outputs["txt_lens"]
                        max_img_ids = [
                            max(x.max().item(), y.max().item()) if y.numel() > 0 else x.max().item()
                            for x, y in zip(img_ids.split(img_lens), ref_img_ids.split(ref_img_lens))
                        ]
                        txt, txt_lens, txt_ids = prepare_txt_for_dit(txt, txt_lens, max_img_ids)

                        dit_input = DiTInputOutput(
                            img=img,
                            img_ids=img_ids,
                            img_lens=img_lens,
                            ref_img=ref_img,
                            ref_img_ids=ref_img_ids,
                            ref_img_lens=ref_img_lens,
                            txt=txt,
                            txt_lens=txt_lens,
                            txt_ids=txt_ids,
                            timesteps=timesteps,
                            v_target=v_target,
                        )

                        # 需要生成 PE

                        batch.cuda()

                        del gather_outputs, split_outputs_

                    # MEMORY FIX: Periodic aggressive cleanup
                    if self.aggressive_cleanup and batch_idx % 10 == 0:
                        self._aggressive_cleanup()

                    yield batch, dit_input

                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    logger.exception(e)  # Log traceback
                    logger.warning("Skipping problematic batch.")

                    # MEMORY FIX: Clean up on error
                    if "split_outputs_" in locals():
                        del split_outputs_  # type: ignore
                    if "gather_outputs" in locals():
                        del gather_outputs  # type: ignore
                    if "batch" in locals():
                        del batch
                    self._aggressive_cleanup()
                    continue

    def compute_loss(self, batch: tuple[PackData, DiTInputOutput]):
        return self.policy.compute_loss(
            self.dit_forward, batch, self.device_mesh["tp_w_sp"], self.ce_loss_weight, self.models["ae"]
        )

    def train_one_step(self, loader) -> StepInfo:  # noqa: C901
        global_grad_norm = None
        each_grad_norm_info = None
        gradient_accumulation_steps = self.gradient_accumulation_steps

        log_info = defaultdict(float)

        # Track memory at step start
        if self.memory_monitor:
            self.memory_monitor.track_step(self.current_step)

        for acc_step in range(gradient_accumulation_steps):
            sync_gradients = acc_step == gradient_accumulation_steps - 1

            self.gc_handler.run(1)
            self.timers[TIMER_DATA_LOAD_BATCH].start()
            batch = next(loader)
            self.timers[TIMER_DATA_LOAD_BATCH].stop()

            self.timers[TIMER_COMPUTE_LOSS].start()
            loss, _log_info = self.compute_loss(batch)
            self.timers[TIMER_COMPUTE_LOSS].stop()

            # FIXME: record dataload
            self.timers[TIMER_DATA_RECORD_BATCH].start()
            if self.data_recorder is not None:
                self.data_recorder(batch[0])
            self.timers[TIMER_DATA_RECORD_BATCH].stop()

            self.timers[TIMER_COMPUTE_BACKWARD].start()
            if gradient_accumulation_steps > 1:
                self.dit.set_requires_gradient_sync(sync_gradients)
            loss = loss / gradient_accumulation_steps
            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
            self.timers[TIMER_COMPUTE_BACKWARD].stop()

            if sync_gradients:
                with self.timers[TIMER_COMPUTE_OPTIMIZE]:
                    if self.scaler is None:
                        with implicit_replication():
                            global_grad_norm, each_grad_norm_info = clip_grad_norm_(
                                self.get_trainable_params(),
                                max_norm=self.config.optim.grad_norm_clip,
                                foreach=True,
                            )
                            self.optimizer.step()
                    else:
                        self.scaler.unscale_(self.optimizer)
                        with implicit_replication():
                            global_grad_norm, each_grad_norm_info = clip_grad_norm_(
                                self.get_trainable_params(),
                                max_norm=self.config.optim.grad_norm_clip,
                                foreach=True,
                            )
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

            with self.timers[TIMER_SYNC_LOG]:
                # MEMORY FIX: Properly detach loss for logging
                _log_info = dict(loss=loss.detach().clone(), **_log_info)
                average_sync_dict(_log_info, self.device_mesh, device=torch.cuda.current_device())  # type: ignore

                for k, v in _log_info.items():
                    log_info[k] += v

            # MEMORY FIX: Delete batch after processing
            del batch
            if self.aggressive_cleanup:
                self._aggressive_cleanup()

        if self.ema_updater is not None:
            # FIXME: unwrap self.dit
            self.ema_updater.step(self.dit)

        for k, v in log_info.items():
            log_info[k] = v / gradient_accumulation_steps
            if isinstance(log_info[k], torch.Tensor):
                log_info[k] = TensorCleaner.extract_scalar(log_info[k])

        loss = log_info.pop("loss")

        # 记录所有参数的 grad norm 信息，供 debug 使用
        if each_grad_norm_info is not None:  # type: ignore
            for name, grad in each_grad_norm_info:
                grad_value = grad.detach().item() if isinstance(grad, torch.Tensor) else grad
                log_info["grad_norm/" + name] = grad_value

        # Check for memory leaks periodically
        if self.memory_monitor and self.current_step % 50 == 0:
            if self.memory_monitor.check_for_leaks():
                logger.warning(f"[MemoryLeak] Detected at step {self.current_step}")
                self._aggressive_cleanup()

        self.current_step += 1

        return StepInfo(
            lr=self.lr_scheduler.get_last_lr()[0],  # type: ignore
            loss=loss,
            global_grad_norm=global_grad_norm.detach().item() if global_grad_norm is not None else 0.0,
            scalar_metrics=log_info,
        )

    def cleanup(self):
        """Clean up resources when training is done."""
        if self.memory_monitor:
            self.memory_monitor.cleanup()
            logger.info("Memory monitor cleaned up")
