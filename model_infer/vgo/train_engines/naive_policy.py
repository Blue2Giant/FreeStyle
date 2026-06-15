import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch.distributed.device_mesh import DeviceMesh

from vgo.data.processor.naive_collect import PackData


def _current_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return torch.device("npu", torch.npu.current_device())  # type: ignore[attr-defined]
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


@dataclass
class NoiseOffsetArgs:
    global_weight: float = field(default=0.0)
    low_freq_weight: float = field(default=0.0)


@dataclass
class TimeShiftArgs:
    x1: float = 256
    y1: float = 0.5
    x2: float = 4096
    y2: float = 1.15
    reverse: bool = False
    sigma: float = 1.0


def generate_low_freq_noise(size=(1, 1, 256, 256), sigma=15, device="cuda", generator=None):
    # 1. 创建高斯核（低通滤波器）
    kernel_size = int(8 * sigma + 1)

    # 2. 生成高斯白噪声
    size = list(size)
    size[-1] += 8 * sigma
    size[-2] += 8 * sigma
    noise = torch.randn(size, device=device, generator=generator)

    x = torch.arange(kernel_size) - kernel_size // 2
    gauss_1d = torch.exp(-(x**2) / (2 * sigma**2))
    gauss_1d /= gauss_1d.sum()
    gauss_2d = gauss_1d[:, None] * gauss_1d[None, :]
    gauss_2d = gauss_2d.to(noise.device).unsqueeze(0).unsqueeze(0)
    gauss_2d /= gauss_2d.norm()

    # 3. 应用卷积（低通滤波）
    padding = 0
    low_freq_noise = F.conv2d(noise, gauss_2d, padding=padding)

    return low_freq_noise.squeeze()


def shift_timesteps(
    timesteps,
    image_shape,
    x1: float = 256,
    y1: float = 0.5,
    x2: float = 4096,
    y2: float = 1.15,
    reverse=False,
    sigma: float = 1.0,
    patch_size=2,
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    height, width = image_shape[-2:]
    mu = m * (height / patch_size * width / patch_size) + b

    t = timesteps.clone()
    if reverse:
        t = 1 - t
    # 使用指数运算处理张量
    exp_mu = math.exp(mu)
    transformed_t = exp_mu / (exp_mu + (1 / t - 1) ** sigma)
    if reverse:
        transformed_t = 1 - transformed_t
    return transformed_t


def create_timestep_sampler(weighting_scheme="logit_normal", **sampling_kwargs):
    generator = sampling_kwargs.get("generator")
    if weighting_scheme == "uniform":

        def uniform_sampling(batch_size, device, t_start=0, t_end=1):
            timesteps = torch.rand((batch_size,), device=device, generator=generator)
            timesteps = (t_end - t_start) * timesteps + t_start
            return timesteps

        return uniform_sampling
    elif weighting_scheme == "logit_normal":

        def logit_normal_sampling(batch_size, device, t_start=0, t_end=1):
            timesteps = torch.sigmoid(torch.randn((batch_size,), device=device, generator=generator))
            timesteps = (t_end - t_start) * timesteps + t_start
            return timesteps

        return logit_normal_sampling

    elif weighting_scheme in ["truncated_logit_normal", "truncated_logit_normal_one_enhance"]:
        mu = sampling_kwargs.get("mu", 0.0)
        sigma = sampling_kwargs.get("sigma", 1.0)
        lower_percentile = sampling_kwargs.get("lower_percentile", 0.005)
        upper_percentile = sampling_kwargs.get("upper_percentile", 0.995)
        device = torch.device(sampling_kwargs.get("device", _current_device()))

        # 定义 Normal 分布
        dist = torch.distributions.Normal(mu, sigma)
        # 计算分布在下界和上界对应的数值
        lower_bound = torch.sigmoid(dist.icdf(torch.tensor(lower_percentile, device=device)))
        upper_bound = torch.sigmoid(dist.icdf(torch.tensor(upper_percentile, device=device)))

        def truncated_logit_normal_sampling(batch_size, device, t_start=0, t_end=1):
            # 在截断的 CDF 范围内均匀采样
            # 注意: 原始 CDF 在下界为 lower_percentile,上界为 upper_percentile

            u = (1 - torch.rand(batch_size, device=device, generator=generator)) * (
                upper_percentile - lower_percentile
            ) + lower_percentile

            # 利用逆变换采样法,将均匀采样值映射为分布中的样本
            timesteps = dist.icdf(u)
            timesteps = torch.sigmoid(timesteps)
            timesteps = (timesteps - lower_bound) / (upper_bound - lower_bound)

            timesteps = (t_end - t_start) * timesteps + t_start

            if weighting_scheme == "truncated_logit_normal_one_enhance":
                timesteps = torch.where(
                    torch.rand_like(timesteps) < 1 / 8,
                    1 - torch.rand_like(timesteps).clamp(0, 7 / 8) * 8 / 7 * 1 / 256,
                    timesteps,
                )

            return timesteps

        return truncated_logit_normal_sampling

    raise NotImplementedError(f"timestep sampling_type {weighting_scheme} 未实现")


@dataclass
class DiTInputOutput:
    img: torch.Tensor
    img_ids: torch.Tensor
    img_lens: list[int]
    ref_img: torch.Tensor
    ref_img_ids: torch.Tensor
    ref_img_lens: list[int]

    txt: torch.Tensor
    txt_ids: torch.Tensor
    txt_lens: list[int]

    timesteps: torch.Tensor
    v_target: torch.Tensor | None = None
    sref_ref_token_ranges: list[tuple[int, int]] | None = None


class PackedVarlenFlowMatchingPolicy:
    name: str = "flow_matching"

    def __init__(
        self,
        weighting_scheme="truncated_logit_normal",
        shift_timesteps=True,
        noise_offset_policy: NoiseOffsetArgs | None = None,
        time_shift_policy: TimeShiftArgs | None = None,
        seed: int | None = None,
        recon_loss_weight: float = 0.0,
        style_loss_weight: float = 0.0,
        sref_enrichment_loss_weight: float = 0.0,
        sref_enrichment_lower_bound: float = 0.08,
        sref_enrichment_upper_bound: float = 0.5,
        sref_enrichment_eps: float = 1e-6,
        sref_entropy_loss_weight: float = 0.0,
        sref_entropy_lower_bound: float = 0.06,
        sref_entropy_upper_bound: float = 0.14,
        sref_entropy_eps: float = 1e-6,
        sref_entropy_schedule_enabled: bool = False,
        sref_entropy_schedule_start_timestep: float = 0.75,
        sref_entropy_schedule_start_lower_bound: float = 0.06,
        sref_entropy_schedule_end_lower_bound: float = 0.10,
        sref_entropy_schedule_power: float = 2.0,
        sref_enrichment_noise_query_only: bool = False,
        sref_enrichment_timestep_weighting: bool = False,
        sref_enrichment_timestep_weight_power: float = 1.0,
    ) -> None:
        self.gpu_random_generator = None
        self.random = random
        if seed is not None:
            self.gpu_random_generator = torch.Generator(device=_current_device())
            self.gpu_random_generator.manual_seed(seed)
            self.random = random.Random(seed)

        self.timesteps_sampler = create_timestep_sampler(
            weighting_scheme=weighting_scheme, generator=self.gpu_random_generator
        )
        self.shift_timesteps = shift_timesteps

        self.noise_offset_policy = NoiseOffsetArgs() if noise_offset_policy is None else noise_offset_policy
        self.time_shift_policy = TimeShiftArgs() if time_shift_policy is None else time_shift_policy
        self.recon_loss_weight = float(recon_loss_weight)
        self.style_loss_weight = float(style_loss_weight)
        self.sref_enrichment_loss_weight = float(sref_enrichment_loss_weight)
        self.sref_enrichment_lower_bound = float(sref_enrichment_lower_bound)
        self.sref_enrichment_upper_bound = float(sref_enrichment_upper_bound)
        self.sref_enrichment_eps = float(sref_enrichment_eps)
        self.sref_entropy_loss_weight = float(sref_entropy_loss_weight)
        self.sref_entropy_lower_bound = float(sref_entropy_lower_bound)
        self.sref_entropy_upper_bound = float(sref_entropy_upper_bound)
        self.sref_entropy_eps = float(sref_entropy_eps)
        self.sref_entropy_schedule_enabled = bool(sref_entropy_schedule_enabled)
        self.sref_entropy_schedule_start_timestep = float(sref_entropy_schedule_start_timestep)
        self.sref_entropy_schedule_start_lower_bound = float(sref_entropy_schedule_start_lower_bound)
        self.sref_entropy_schedule_end_lower_bound = float(sref_entropy_schedule_end_lower_bound)
        self.sref_entropy_schedule_power = float(sref_entropy_schedule_power)
        self.sref_enrichment_noise_query_only = bool(sref_enrichment_noise_query_only)
        self.sref_enrichment_timestep_weighting = bool(sref_enrichment_timestep_weighting)
        self.sref_enrichment_timestep_weight_power = float(sref_enrichment_timestep_weight_power)
        if self.sref_enrichment_eps <= 0.0:
            raise ValueError(f"sref_enrichment_eps must be positive, got {self.sref_enrichment_eps}")
        if self.sref_enrichment_lower_bound < 0.0:
            raise ValueError(
                f"sref_enrichment_lower_bound must be non-negative, got {self.sref_enrichment_lower_bound}"
            )
        if self.sref_enrichment_lower_bound > self.sref_enrichment_upper_bound:
            raise ValueError(
                "sref_enrichment_lower_bound must be <= sref_enrichment_upper_bound, got "
                f"{self.sref_enrichment_lower_bound} > {self.sref_enrichment_upper_bound}"
            )
        if self.sref_entropy_eps <= 0.0:
            raise ValueError(f"sref_entropy_eps must be positive, got {self.sref_entropy_eps}")
        if self.sref_entropy_lower_bound < 0.0:
            raise ValueError(f"sref_entropy_lower_bound must be non-negative, got {self.sref_entropy_lower_bound}")
        if self.sref_entropy_lower_bound > self.sref_entropy_upper_bound:
            raise ValueError(
                "sref_entropy_lower_bound must be <= sref_entropy_upper_bound, got "
                f"{self.sref_entropy_lower_bound} > {self.sref_entropy_upper_bound}"
            )
        if not 0.0 < self.sref_entropy_schedule_start_timestep <= 1.0:
            raise ValueError(
                "sref_entropy_schedule_start_timestep must be in (0, 1], got "
                f"{self.sref_entropy_schedule_start_timestep}"
            )
        if not 0.0 <= self.sref_entropy_schedule_start_lower_bound <= 1.0:
            raise ValueError(
                "sref_entropy_schedule_start_lower_bound must be in [0, 1], got "
                f"{self.sref_entropy_schedule_start_lower_bound}"
            )
        if not 0.0 <= self.sref_entropy_schedule_end_lower_bound <= 1.0:
            raise ValueError(
                "sref_entropy_schedule_end_lower_bound must be in [0, 1], got "
                f"{self.sref_entropy_schedule_end_lower_bound}"
            )
        if self.sref_entropy_schedule_end_lower_bound < self.sref_entropy_schedule_start_lower_bound:
            raise ValueError(
                "sref_entropy_schedule_end_lower_bound must be >= sref_entropy_schedule_start_lower_bound, got "
                f"{self.sref_entropy_schedule_end_lower_bound} < {self.sref_entropy_schedule_start_lower_bound}"
            )
        if self.sref_entropy_schedule_power <= 0.0:
            raise ValueError(
                "sref_entropy_schedule_power must be positive, got "
                f"{self.sref_entropy_schedule_power}"
            )
        if self.sref_enrichment_timestep_weight_power < 0.0:
            raise ValueError(
                "sref_enrichment_timestep_weight_power must be non-negative, got "
                f"{self.sref_enrichment_timestep_weight_power}"
            )

        self.all_source_names: list[str] | None = None

    @torch.no_grad()
    def gen_noise(self, latent, latent_size, latent_lens, timesteps):
        device = latent.device
        x1 = torch.randn(*latent.shape, device=device, dtype=torch.float32, generator=self.gpu_random_generator)  # type: ignore

        if self.noise_offset_policy.global_weight > 0:
            latent_lens_tensor = torch.LongTensor(latent_lens).to(device=device, dtype=torch.long)
            global_offset = (
                torch.randn(
                    (latent_lens_tensor.shape[0], latent.shape[1]),
                    device=device,
                    dtype=torch.float32,
                    generator=self.gpu_random_generator,
                )
                * torch.rand(
                    latent_lens_tensor.shape[0],
                    device=device,
                    dtype=torch.float32,
                    generator=self.gpu_random_generator,
                )[:, None]
            ).repeat_interleave(latent_lens_tensor, dim=0) * self.noise_offset_policy.global_weight
        else:
            global_offset = 0

        if self.noise_offset_policy.low_freq_weight > 0:
            weight = torch.clamp((1 - timesteps) / timesteps, min=0, max=1.0)

            low_freq_offset = []
            for latent_idx in range(latent_size.shape[0]):
                h, w = latent_size[latent_idx].tolist()
                low_freq_offset_i = generate_low_freq_noise(
                    size=(latent.shape[1], 1, h, w),
                    sigma=13,
                    device=device,
                    generator=self.gpu_random_generator,
                )
                low_freq_offset.append(self.random.random() * weight[latent_idx] * low_freq_offset_i.flatten(1, 2).T)
            low_freq_offset = self.noise_offset_policy.low_freq_weight * torch.cat(low_freq_offset)
        else:
            low_freq_offset = 0

        ph = 2
        pw = 2
        noise_token = x1 + global_offset + low_freq_offset
        token_size = latent_size // 2

        noise_token = torch.cat(
            [
                rearrange(x, "(h ph w pw) c -> (h w) (c ph pw)", h=h, w=w, ph=ph, pw=pw)
                for x, (h, w) in zip(noise_token.split(latent_lens), token_size.tolist())
            ],
            dim=0,
        ).contiguous()
        return noise_token

    def compute_noisy_latent(self, latents, images_size: torch.Tensor):
        device, dtype = latents.device, latents.dtype

        images_size = images_size.to(torch.int32)
        latent_size = images_size // 8
        token_size = images_size // 16

        ph, pw = 2, 2
        latent_len = (latent_size[:, 0] * latent_size[:, 1]).tolist()
        image_tokens = torch.cat(
            [
                rearrange(x, "(h ph w pw) c -> (h w) (c ph pw)", h=h, w=w, ph=ph, pw=pw)
                for x, (h, w) in zip(latents.split(latent_len), token_size.tolist())
            ],
            dim=0,
        ).contiguous()
        token_lens = token_size[:, 0] * token_size[:, 1]

        # images -> L x C
        x0 = image_tokens

        # timesteps should be torch.float32
        timesteps = self.timesteps_sampler(images_size.size(0), device)
        if self.shift_timesteps:
            for i, latent_shape in enumerate(latent_size.tolist()):
                timesteps[i] = shift_timesteps(
                    timesteps[i],
                    latent_shape,
                    x1=self.time_shift_policy.x1,
                    y1=self.time_shift_policy.y1,
                    x2=self.time_shift_policy.x2,
                    y2=self.time_shift_policy.y2,
                    reverse=self.time_shift_policy.reverse,
                    sigma=self.time_shift_policy.sigma,
                )  # 在 latent 空间计算 shift
        timesteps = timesteps.to(device)

        x1 = self.gen_noise(latents, latent_size=latent_size, latent_lens=latent_len, timesteps=timesteps)

        _timesteps = timesteps.repeat_interleave(token_lens.to(device=device), dim=0)

        xt = _timesteps[:, None].type(dtype) * x1 + (1 - _timesteps[:, None]).type(dtype) * x0

        v_target = x1 - x0  # noise - image

        return x0, x1, xt, v_target, timesteps, token_size

    @staticmethod
    @torch.no_grad
    def get_vision_tokens_position_ids(vision_token_hw, affine_mat, device):
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
        vision_token_idx_xy = vision_token_idx_xy @ affine_mat[:2, :2].T + affine_mat[:2, -1][None]

        # notice that the axis 1 is y, axis 2 is x, set xy to [2, 1]
        vision_token_idx[..., [2, 1]] = vision_token_idx_xy.view(vision_token_idx_xy_shape)
        vision_token_idx = vision_token_idx.reshape(-1, 3)
        return vision_token_idx

    @torch.no_grad()
    def compute_loss_for_each_source(
        self,
        pack_data: PackData,
        timesteps: torch.Tensor,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        tp_mesh: DeviceMesh,
    ):
        if self.all_source_names is None:
            logger.error("无法记录每个 Source 的权重")
            return None

        world_size = torch.distributed.get_world_size()
        tp_size = tp_mesh.size()
        dp_size = world_size / tp_size

        loss_for_source = dict.fromkeys(
            [
                *["loss_diffusion/" + x + "/_loss_sum_" for x in self.all_source_names],
                *["loss_diffusion/" + x + "/_token_count_" for x in self.all_source_names],
            ],
            0.0,
        )
        if tp_mesh.get_local_rank() == 0:
            loss_diffusion = torch.mean((v_pred.float() - v_target.float()).reshape(v_target.shape[0], -1) ** 2, dim=1)
            learnable_vae_token_size = torch.tensor(
                [[x.shape[1] // 16, x.shape[2] // 16] for x in pack_data.target_images]
            )
            learnable_vae_token_lens = (learnable_vae_token_size[:, 0] * learnable_vae_token_size[:, 1]).tolist()

            # loss_for_source
            for doc_idx, loss_seq in enumerate(loss_diffusion.split(learnable_vae_token_lens, dim=0)):
                # 这里用了两个 magic str ，来设定 loss
                loss_i = loss_seq.sum().item()
                token_count = float(loss_seq.shape[0])
                loss_for_source["loss_diffusion/" + pack_data.sequences[doc_idx].source + "/_loss_sum_"] += (
                    loss_i * dp_size
                )
                loss_for_source["loss_diffusion/" + pack_data.sequences[doc_idx].source + "/_token_count_"] += float(
                    loss_seq.shape[0] * dp_size
                )

                if getattr(pack_data, "data_track_info", None) is not None:
                    if pack_data.data_track_info[doc_idx]._loss is None:  # type: ignore
                        pack_data.data_track_info[doc_idx]._loss = [loss_i, token_count]  # type: ignore
                    else:
                        pack_data.data_track_info[doc_idx]._loss[0] += loss_i  # type: ignore
                        pack_data.data_track_info[doc_idx]._loss[1] += token_count  # type: ignore
        if getattr(pack_data, "data_track_info", None) is not None:
            timesteps = timesteps.tolist()
            for doc_idx in range(len(pack_data.data_track_info)):  # type: ignore
                pack_data.data_track_info[doc_idx]._loss = (  # type: ignore
                    pack_data.data_track_info[doc_idx]._loss[0] / pack_data.data_track_info[doc_idx]._loss[1]  # type: ignore
                )
                pack_data.data_track_info[doc_idx]._timestep = timesteps[doc_idx]  # type: ignore
        return loss_for_source

    # we should patchify latent to avoid to much rearrange
    def compute_loss(
        self,
        model_fn: Callable,
        data: tuple[PackData, DiTInputOutput],
        tp_mesh: DeviceMesh,
        ce_loss_weight: float = 1.0,
        ae: torch.nn.Module | None = None,
    ):
        model_fn_out = model_fn(data)
        v_pred, v_target = model_fn_out[:2]
        aux_info = model_fn_out[3] if len(model_fn_out) >= 4 and isinstance(model_fn_out[3], dict) else {}

        device = v_pred.device
        world_size = torch.distributed.get_world_size()
        tp_size = tp_mesh.size()
        dp_size = world_size / tp_size

        # # flow-matching loss
        total_dit_tokens = torch.tensor(v_pred.shape[0], device=device) / tp_mesh.size()
        torch.distributed.all_reduce(total_dit_tokens, op=torch.distributed.ReduceOp.SUM)
        # 预测的速度场一步得到x0 x0 = xt-t*v_pred, 计算x0与target的cas

        # 注意，这里除以 world size 可以让计算得到的 loss/grad norm 不会随着 world size 的改变而改变量级
        loss_diffusion = torch.mean(
            ((v_pred.float() - v_target.float()) ** 2).reshape(v_target.shape[0], -1), dim=1
        ).sum() / (total_dit_tokens / dp_size)
        
        loss = loss_diffusion

        info: dict = {"loss_diffusion": loss_diffusion}

        if self.sref_enrichment_loss_weight > 0.0:
            if "loss_sref_enrichment" not in aux_info:
                raise RuntimeError("Missing block-0 sref enrichment auxiliary loss while regularization is enabled.")

            loss_sref_enrichment = aux_info["loss_sref_enrichment"].to(device=device, dtype=loss_diffusion.dtype)
            loss = loss + self.sref_enrichment_loss_weight * loss_sref_enrichment
            info["loss_sref_enrichment"] = loss_sref_enrichment
            if "sref_enrichment" in aux_info:
                info["sref_enrichment"] = aux_info["sref_enrichment"]

        if self.sref_entropy_loss_weight > 0.0:
            if "loss_sref_entropy" not in aux_info:
                raise RuntimeError("Missing block-0 sref entropy auxiliary loss while regularization is enabled.")

            loss_sref_entropy = aux_info["loss_sref_entropy"].to(device=device, dtype=loss_diffusion.dtype)
            loss = loss + self.sref_entropy_loss_weight * loss_sref_entropy
            info["loss_sref_entropy"] = loss_sref_entropy
            if "sref_entropy" in aux_info:
                info["sref_entropy"] = aux_info["sref_entropy"]
            if "sref_entropy_target_lower_bound" in aux_info:
                info["sref_entropy_target_lower_bound"] = aux_info["sref_entropy_target_lower_bound"]

        info.update(
            {
                "dit_token_count": float(v_pred.shape[0]),
            }
        )

        # 统计 sample 的数量，这里需要考虑后续对所有卡求均值的影响，先主动乘上 DP size
        info.update(
            {
                "sample_count": float(len(data[0].sequences)) * torch.distributed.get_world_size() / tp_size,
            }
        )

        with torch.no_grad():
            if len(model_fn_out) >= 3:
                t = model_fn_out[2]
                info.update(
                    {
                        "loss_diffusion_x0": float(
                            torch.mean(
                                (
                                    ((v_pred.float() - v_target.float()).reshape(v_target.shape[0], -1) * t[:, None])
                                    ** 2
                                ),
                                dim=1,
                            ).sum()
                            / (total_dit_tokens / dp_size)
                        )
                    }
                )

        if (self.recon_loss_weight > 0.0 or self.style_loss_weight > 0.0) and ae is not None:
            pack_data, dit_io = data
            xt_tokens = dit_io.img
            img_lens = dit_io.img_lens
            t_per_token = model_fn_out[2] if len(model_fn_out) >= 3 else torch.zeros(
                (xt_tokens.shape[0],), device=xt_tokens.device, dtype=xt_tokens.dtype
            )
            x0_hat_tokens = xt_tokens.to(v_pred.dtype) - t_per_token[:, None].to(v_pred.dtype) * v_pred

            loss_recon = torch.tensor(0.0, device=device)
            latents_mean = torch.tensor(ae.config.latents_mean, device=device, dtype=torch.float32).view(-1, 1, 1)
            latents_std = torch.tensor(ae.config.latents_std, device=device, dtype=torch.float32).view(-1, 1, 1)
            start = 0
            for i, tok_len in enumerate(img_lens):
                end = start + tok_len
                tokens_i = x0_hat_tokens[start:end]
                target_i = pack_data.target_images[i].to(device=device, dtype=torch.float32)
                h, w = target_i.shape[-2], target_i.shape[-1]
                h_tok = h // 16
                w_tok = w // 16
                ph = 2
                pw = 2
                latents_norm_i = rearrange(
                    tokens_i, "(h w) (c ph pw) -> c (h ph) (w pw)", h=h_tok, w=w_tok, ph=ph, pw=pw
                ).contiguous()

                latents_dec_i = (latents_norm_i.to(torch.float32) * latents_std) + latents_mean
                img_dec_i = ae.decode(latents_dec_i.unsqueeze(0).unsqueeze(2)).sample[:, :, 0]
                img_dec_i = img_dec_i.clamp(-1, 1)[0]

                loss_recon = loss_recon + torch.mean(torch.abs(img_dec_i - target_i))
                start = end

            if len(img_lens) > 0:
                loss_recon = loss_recon / len(img_lens)
                loss = loss + self.recon_loss_weight * loss_recon
                info["loss_recon"] = loss_recon

            if self.style_loss_weight > 0.0:
                loss_style = torch.tensor(0.0, device=device)
                count_pairs = 0
                for i in range(len(pack_data.sequences)):
                    target_i = pack_data.target_images[i].to(device=device, dtype=torch.float32)
                    for ref_img in pack_data.ref_images[i]:
                        ref_i = ref_img.to(device=device, dtype=torch.float32)
                        loss_style = loss_style + torch.mean(torch.abs(ref_i - target_i))
                        count_pairs += 1
                if count_pairs > 0:
                    loss_style = loss_style / count_pairs
                    loss = loss + self.style_loss_weight * loss_style
                    info["loss_style"] = loss_style

        info = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in info.items()}

        loss_for_each_source = self.compute_loss_for_each_source(data[0], data[1].timesteps, v_pred, v_target, tp_mesh)
        if loss_for_each_source is not None:
            info.update(loss_for_each_source)

        return loss, info
