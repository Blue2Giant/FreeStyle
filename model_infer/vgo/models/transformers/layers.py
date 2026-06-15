import math
import os
from importlib.util import find_spec
from dataclasses import dataclass
from functools import cache

import torch
import torch.distributed.tensor
from einops import rearrange
from torch import Tensor, nn
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from vgo.models.modules import RMSNorm, apply_rope, layernorm_and_scale_shift, scale_add_residual, triton_apply_rope
from vgo.models.modules.attention import VarlenAttentionConfig, attention
from vgo.models.modules.varlen_ops import (
    VarLenConfig,
    cat_seq,
    split_seq,
    varlen_gate,
    varlen_scale_shift,
)
from vgo.utils.dist_utils import NoParallel


def _can_use_torch_compile() -> bool:
    if os.environ.get("VGO_DISABLE_TORCH_COMPILE", "0") == "1":
        return False
    if not hasattr(torch, "compile"):
        return False
    if hasattr(torch, "npu") and torch.npu.is_available():  # type: ignore[attr-defined]
        return False
    if not torch.cuda.is_available():
        return False
    return find_spec("triton") is not None


def _optional_torch_compile(*, dynamic: bool = True, mode: str = "max-autotune-no-cudagraphs"):
    def decorator(fn):
        if not _can_use_torch_compile():
            return fn
        return torch.compile(fn, dynamic=dynamic, mode=mode)

    return decorator


def _compile_if_available(module, *, dynamic: bool = True, mode: str = "max-autotune-no-cudagraphs"):
    if not _can_use_torch_compile():
        return module
    return torch.compile(module, dynamic=dynamic, mode=mode)


@cache
def _generate_freqs(dim: int, max_period: int = 10000) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
    return freqs


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10000, time_factor: float = 1000.0) -> Tensor:
    """
    生成正弦时间步嵌入(合并版),支持动态时间缩放和维度对齐

    数学公式:
    1. 频率计算:freqs_j = exp( -ln(max_period) * j/(half) ), j ∈ [0, half-1]
    2. 相位参数:args_ij = (t_i * time_factor) * freqs_j
    3. 嵌入计算:embedding = cat([cos(args), sin(args)], dim=-1)
    4. 维度对齐:当dim为奇数时末尾补零

    Shape:
        Input:  t - (N,)      # 批大小N的时间步序列
        Output:   - (N, dim)  # 时间步嵌入矩阵

    参数说明:
        t: 输入时间步序列,支持浮点型张量
        dim: 输出嵌入维度
        max_period: 控制最小频率(公式中的最大周期),默认10000
        time_factor: 时间步缩放因子,默认1000(适配不同数值范围的时间输入)
    """

    assert t.dtype == torch.float32, "We should set t.dtype to torch.float32"

    # 时间步缩放(原始实现差异合并)
    # shape保持 (N,),与输入t设备保持一致
    t_scaled = time_factor * t

    # 计算频率向量(数学公式第1步)
    # half = dim // 2
    # 生成指数衰减频率:shape (half,)
    freqs = _generate_freqs(dim=dim, max_period=max_period).to(t.device)

    # 计算相位参数(数学公式第2步)
    # t_scaled[:, None] 形状 (N, 1),freqs[None] 形状 (1, half)
    # 广播乘法后得到 (N, half)
    args = t_scaled[:, None].float() * freqs[None]

    # 生成基础嵌入(数学公式第3步)
    # torch.cos(args) 和 torch.sin(args) 各为 (N, half)
    # cat后得到 (N, 2*half) = (N, dim//2*2)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    # 处理奇数维度情况(数学公式第4步)
    if dim % 2:
        # 补零列保持形状对齐:zeros_like(embedding[:, :1]) 保持 (N,1)
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    # 类型传播
    if torch.is_floating_point(t):
        embedding = embedding.to(t.dtype)

    return embedding


def rope(pos, dim: int, theta: int):
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "... n d (i j) -> ... n d i j", i=2, j=2)
    return out.float()


class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        assert ids.dtype == torch.float32

        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, mlp_ratio: int = 1):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim * mlp_ratio, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim * mlp_ratio, hidden_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        """Initialize with truncated normal distribution."""
        nn.init.trunc_normal_(self.in_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.in_layer.bias)
        nn.init.trunc_normal_(self.out_layer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x.type(self.in_layer.weight.dtype)
        return self.out_layer(self.silu(self.in_layer(x)))


def attention_after_rope(q, k, v, pe, attention_mask=None):
    q, k = apply_rope(q, k, pe)
    x = attention(q, k, v, mode="flash", attn_mask=attention_mask)
    return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
        self._init_weights()

    def _init_weights(self):
        """Initialize attention weights with Xavier uniform."""
        # QKV projection - Xavier for better gradient flow
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        # Output projection
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention_after_rope(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


@dataclass
class FrequencyAwareRopeConfig:
    enabled: bool = False
    shf_min: float = 0.8
    slf_min: float = 1.0
    shf_max: float = 0.9
    slf_max: float = 1.1
    beta: float = 2.0
    spatial_axes_only: bool = True


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        input_dtype = q.dtype

        if input_dtype != self.query_norm.scale.dtype:
            q = q.to(self.query_norm.scale.dtype)  # type: ignore
            v = v.to(self.query_norm.scale.dtype)  # type: ignore

        q = self.query_norm(q)
        k = self.key_norm(k)

        return q, k


class Modulation(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(in_dim, self.multiplier * out_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        """
        Initialize with adaLN-Zero strategy.

        Zero-initialize the modulation layer so that:
        - shift = 0, scale = 0 (so 1 + scale = 1, identity transform)
        - gate = 0 (residual connection is identity)

        This ensures the model starts with identity transformations.
        """
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    @_optional_torch_compile(dynamic=True, mode="max-autotune-no-cudagraphs")
    def _forward(self, vec: Tensor) -> Tensor:
        return self.lin(nn.functional.silu(vec))

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        # FIXME:
        # 将 chunk 挪到后面，另外将 TP 模式下输出的 layout 修改为 Shard(1)，也许耗时会好一些
        out = self._forward(vec).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class MergedModulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 12 if double else 6
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        """Initialize with adaLN-Zero strategy (same as Modulation)."""
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    @_optional_torch_compile(dynamic=True, mode="max-autotune-no-cudagraphs")
    def _forward(self, vec: Tensor) -> Tensor:
        return self.lin(nn.functional.silu(vec))

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None, ModulationOut, ModulationOut | None]:
        # FIXME:
        # 将 chunk 挪到后面，另外将 TP 模式下输出的 layout 修改为 Shard(1)，也许耗时会好一些
        out = self._forward(vec).chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:6]) if self.is_double else None,
            ModulationOut(*out[6:9]),
            ModulationOut(*out[9:12]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP layers with truncated normal distribution."""
        for mlp in [self.img_mlp, self.txt_mlp]:
            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, attention_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        # img_modulated = self.img_norm1(img)
        # img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_modulated = layernorm_and_scale_shift(img, img_mod1.scale, img_mod1.shift)
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        # txt_modulated = self.txt_norm1(txt)
        # txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_modulated = layernorm_and_scale_shift(txt, txt_mod1.scale, txt_mod1.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)

        attn = attention_after_rope(q, k, v, pe=pe, attention_mask=attention_mask)
        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # calculate the img bloks
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img_mlp = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        # img = img + img_mod2.gate * img_mlp
        img = scale_add_residual(self.img_attn.proj(img_attn), img_mod1.gate, img)
        img_mlp = self.img_mlp(layernorm_and_scale_shift(img, img_mod2.scale, img_mod2.shift))
        img = scale_add_residual(img_mlp, img_mod2.gate, img)

        # calculate the txt bloks
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt_mlp = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # txt = txt + txt_mod2.gate * txt_mlp
        txt = scale_add_residual(self.txt_attn.proj(txt_attn), txt_mod1.gate, txt)
        txt_mlp = self.txt_mlp(layernorm_and_scale_shift(txt, txt_mod2.scale, txt_mod2.shift))
        txt = scale_add_residual(txt_mlp, txt_mod2.gate, txt)
        return img, txt


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, hidden_size, double=False)
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with truncated normal distribution."""
        nn.init.trunc_normal_(self.linear1.weight, mean=0.0, std=0.02)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.trunc_normal_(self.linear2.weight, mean=0.0, std=0.02)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor, attention_mask: Tensor | None) -> Tensor:
        mod, _ = self.modulation(vec)
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = layernorm_and_scale_shift(x, mod.scale, mod.shift)
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention_after_rope(q, k, v, pe=pe, attention_mask=attention_mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        # return x + mod.gate * output
        return scale_add_residual(output, mod.gate, x)


class VarLenDoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        adaln_dim: int,
        qkv_bias: bool = False,
        rope_fa_config: FrequencyAwareRopeConfig | None = None,
        rope_axes_dim: list[int] | None = None,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(adaln_dim, hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        # import copy

        # self.img_attn_qkv = copy.deepcopy(self.img_attn.qkv)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(adaln_dim, hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.rope_fa_config = rope_fa_config or FrequencyAwareRopeConfig()
        self.rope_axes_dim = tuple(int(x) for x in (rope_axes_dim or []))

        self.is_shard = False

        self.merged_mod = False
        self._init_weights()

    def _init_weights(self):
        """Initialize MLP layers with truncated normal distribution."""
        for mlp in [self.img_mlp, self.txt_mlp]:
            for module in mlp:
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

    def _build_rope_fa_axis_scale(
        self,
        axis_dim: int,
        shf: float,
        slf: float,
        modulate: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if axis_dim <= 0:
            return torch.empty((0,), device=device, dtype=dtype)
        if axis_dim % 2 != 0:
            raise ValueError(f"RoPE axis dim must be even, got {axis_dim}")
        pair_count = axis_dim // 2
        if (not modulate) or pair_count <= 1:
            return torch.full((axis_dim,), float(slf), device=device, dtype=dtype)

        idx = torch.arange(pair_count, device=device, dtype=torch.float32)
        norm = idx / float(max(pair_count - 1, 1))
        pair_scale = float(shf) + (float(slf) - float(shf)) * norm.pow(float(self.rope_fa_config.beta))
        return pair_scale.to(dtype=dtype).repeat_interleave(2)

    def _build_rope_fa_scale_vector(
        self, head_dim: int, shf: float, slf: float, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        pieces: list[Tensor] = []
        for axis_idx, axis_dim in enumerate(self.rope_axes_dim):
            modulate_axis = True
            if self.rope_fa_config.spatial_axes_only and axis_idx == 0:
                modulate_axis = False
            pieces.append(
                self._build_rope_fa_axis_scale(
                    axis_dim=int(axis_dim),
                    shf=shf,
                    slf=slf,
                    modulate=modulate_axis,
                    device=device,
                    dtype=dtype,
                )
            )

        scale = torch.cat(pieces, dim=0) if pieces else torch.ones((int(head_dim),), device=device, dtype=dtype)
        if int(scale.shape[0]) < int(head_dim):
            pad = torch.full((int(head_dim) - int(scale.shape[0]),), float(slf), device=device, dtype=dtype)
            scale = torch.cat([scale, pad], dim=0)
        elif int(scale.shape[0]) > int(head_dim):
            scale = scale[: int(head_dim)]
        return scale.view(1, 1, int(head_dim))

    def _apply_frequency_aware_rope(
        self,
        k: Tensor,
        joint_seq_lens: list[int],
        sref_key_ranges: list[tuple[int, int]],
        rope_fa_progress: Tensor | None,
    ) -> Tensor:
        if not self.rope_fa_config.enabled:
            return k
        if rope_fa_progress is None:
            raise ValueError("rope_fa_progress is required when frequency-aware RoPE is enabled.")
        if len(joint_seq_lens) != len(sref_key_ranges):
            raise ValueError(
                "joint_seq_lens and sref_key_ranges must have the same length, got "
                f"{len(joint_seq_lens)} and {len(sref_key_ranges)}"
            )
        if rope_fa_progress.shape[0] != len(joint_seq_lens):
            raise ValueError(
                "rope_fa_progress must have one entry per sample, got "
                f"{rope_fa_progress.shape[0]} and {len(joint_seq_lens)}"
            )

        k = k.clone()
        seq_offset = 0
        progress_tensor = rope_fa_progress.to(device=k.device, dtype=torch.float32)
        for seq_idx, (seq_len, (k_start, k_end)) in enumerate(zip(joint_seq_lens, sref_key_ranges)):
            seq_len = int(seq_len)
            if seq_len <= 0:
                continue

            k_start = min(max(int(k_start), 0), seq_len)
            k_end = min(max(int(k_end), k_start), seq_len)
            if k_end <= k_start:
                seq_offset += seq_len
                continue

            progress = float(progress_tensor[seq_idx].clamp(0.0, 1.0).item())
            shf = self.rope_fa_config.shf_min + (self.rope_fa_config.shf_max - self.rope_fa_config.shf_min) * progress
            slf = self.rope_fa_config.slf_min + (self.rope_fa_config.slf_max - self.rope_fa_config.slf_min) * progress
            scale = self._build_rope_fa_scale_vector(
                head_dim=int(k.shape[-1]),
                shf=float(shf),
                slf=float(slf),
                device=k.device,
                dtype=k.dtype,
            )
            k[seq_offset + k_start : seq_offset + k_end] = k[seq_offset + k_start : seq_offset + k_end] * scale
            seq_offset += seq_len

        if seq_offset != k.shape[0]:
            raise ValueError(
                f"Sequence slicing mismatch in frequency-aware RoPE modulation: used {seq_offset}, total {k.shape[0]}"
            )
        return k

    # 尝试将 img 和 txt 调制层合并，但是实际上速度并不会更快
    def merge_img_txt_mod(self):
        assert not hasattr(self, "img_txt_mod")
        assert not self.is_shard

        hidden_size = self.img_mod.lin.weight.shape[0] // 3
        self.img_txt_mod = MergedModulation(hidden_size, double=True)

        self.img_txt_mod.lin.weight.data = torch.cat(
            [self.img_mod.lin.weight.data, self.txt_mod.lin.weight.data], dim=0
        ).contiguous()
        self.img_txt_mod.lin.bias.data = torch.cat(
            [self.img_mod.lin.bias.data, self.txt_mod.lin.bias.data], dim=0
        ).contiguous()

        del self.img_mod
        del self.txt_mod
        self.merged_mod = True

    def parallelize_module(self, device_mesh: DeviceMesh, use_replicate_modulation_linear=False):
        # 试了一下单机训练，并不会更快，见
        # self.merge_img_txt_mod()

        modulation_linear_tp_style = NoParallel if use_replicate_modulation_linear else ColwiseParallel
        layer_tp_plan = {
            "img_norm1": SequenceParallel(sequence_dim=0, use_local_output=False),  # sp
            "txt_norm1": SequenceParallel(sequence_dim=0, use_local_output=False),  # sp
            "img_attn.qkv": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),  # all gather before qkv
            "img_attn.proj": RowwiseParallel(output_layouts=Shard(0), use_local_output=False),  # sp
            "txt_attn.qkv": ColwiseParallel(
                input_layouts=Replicate(), output_layouts=Shard(1)
            ),  # all gather before qkv
            "txt_attn.proj": RowwiseParallel(output_layouts=Shard(0), use_local_output=False),  # sp
            "img_norm2": SequenceParallel(sequence_dim=0, use_local_output=False),
            "txt_norm2": SequenceParallel(sequence_dim=0, use_local_output=False),
            "img_mlp.0": ColwiseParallel(input_layouts=Replicate()),
            "img_mlp.2": RowwiseParallel(output_layouts=Shard(0), use_local_output=False),
            "txt_mlp.0": ColwiseParallel(input_layouts=Replicate()),
            "txt_mlp.2": RowwiseParallel(output_layouts=Shard(0), use_local_output=False),
        }
        if self.merged_mod:
            layer_tp_plan.update(
                {
                    "img_txt_mod.lin": modulation_linear_tp_style(
                        input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False
                    ),
                }
            )
        else:
            layer_tp_plan.update(
                {
                    "img_mod.lin": modulation_linear_tp_style(
                        input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False
                    ),
                    "txt_mod.lin": modulation_linear_tp_style(
                        input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False
                    ),
                }
            )

        self.img_attn.qkv.weight.data = rearrange(
            self.img_attn.qkv.weight.data, "(K H D) I -> (H K D) I", K=3, H=self.num_heads
        )
        if self.img_attn.qkv.bias is not None:
            self.img_attn.qkv.bias.data = rearrange(
                self.img_attn.qkv.bias.data, "(K H D) -> (H K D)", K=3, H=self.num_heads
            )
        self.txt_attn.qkv.weight.data = rearrange(
            self.txt_attn.qkv.weight.data, "(K H D) I -> (H K D) I", K=3, H=self.num_heads
        )
        if self.txt_attn.qkv.bias is not None:
            self.txt_attn.qkv.bias.data = rearrange(
                self.txt_attn.qkv.bias.data, "(K H D) -> (H K D)", K=3, H=self.num_heads
            )

        # qk norm should be set to sequence parallel mode
        self.img_attn.norm.query_norm.set_device_mesh(device_mesh=device_mesh)
        self.img_attn.norm.key_norm.set_device_mesh(device_mesh=device_mesh)
        self.txt_attn.norm.query_norm.set_device_mesh(device_mesh=device_mesh)
        self.txt_attn.norm.key_norm.set_device_mesh(device_mesh=device_mesh)

        self.num_heads = self.num_heads // device_mesh.size()

        self.device_mesh = device_mesh
        self.is_shard = True

        # Custom parallelization plan for the model
        parallelize_module(module=self, device_mesh=device_mesh, parallelize_plan=layer_tp_plan)

    def set_fsdp_sequence_parallel(self, **fsdp_config):
        fully_shard(self.img_mod, **fsdp_config)
        fully_shard(self.txt_mod, **fsdp_config)
        fully_shard(self.img_attn.norm.query_norm, **fsdp_config)
        fully_shard(self.img_attn.norm.key_norm, **fsdp_config)
        fully_shard(self.txt_attn.norm.query_norm, **fsdp_config)
        fully_shard(self.txt_attn.norm.key_norm, **fsdp_config)

        self.img_mod.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.txt_mod.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.img_attn.norm.query_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.img_attn.norm.key_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.txt_attn.norm.query_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.txt_attn.norm.key_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore

    def get_unparallelized_params_grad(self):
        if not self.is_shard:
            return {k: v.grad.clone() if v.grad is not None else None for k, v in self.named_parameters()}
        else:
            state_dict = {
                k: v.grad.full_tensor().clone() if v.grad is not None else None for k, v in self.named_parameters()
            }
            num_heads = self.num_heads * self.device_mesh.size()
            if state_dict["img_attn.qkv.weight"] is not None:
                state_dict["img_attn.qkv.weight"] = rearrange(
                    state_dict["img_attn.qkv.weight"], "(H K D) I -> (K H D) I", K=3, H=num_heads
                )
            # if self.img_attn.qkv.bias is not None:
            #     self.img_attn.qkv.bias.data = rearrange(
            #         self.img_attn.qkv.bias.data, "(H K D) -> (K H D)", K=3, H=num_heads
            #     )
            if state_dict["txt_attn.qkv.weight"] is not None:
                state_dict["txt_attn.qkv.weight"] = rearrange(
                    state_dict["txt_attn.qkv.weight"], "(H K D) I -> (K H D) I", K=3, H=num_heads
                )
            # if self.txt_attn.qkv.bias is not None:
            #     self.txt_attn.qkv.bias.data = rearrange(
            #         self.txt_attn.qkv.bias.data, "(K H D) -> (H K D)", K=3, H=num_heads
            #     )
            return state_dict

    def apply_compile(self):
        for layer_name, layer in self.img_attn.named_children():
            if "qkv" in layer_name or "proj" in layer_name:
                layer = _compile_if_available(layer, dynamic=True, mode="max-autotune-no-cudagraphs")
                self.img_attn.register_module(layer_name, layer)

        for layer_name, layer in self.txt_attn.named_children():
            if "qkv" in layer_name or "proj" in layer_name:
                layer = _compile_if_available(layer, dynamic=True, mode="max-autotune-no-cudagraphs")
                self.txt_attn.register_module(layer_name, layer)

        for layer_name, layer in self.img_mlp.named_children():
            if isinstance(layer, nn.Linear):
                layer = _compile_if_available(layer, dynamic=True, mode="max-autotune-no-cudagraphs")
                self.img_mlp.register_module(layer_name, layer)

        for layer_name, layer in self.txt_mlp.named_children():
            if isinstance(layer, nn.Linear):
                layer = _compile_if_available(layer, dynamic=True, mode="max-autotune-no-cudagraphs")
                self.txt_mlp.register_module(layer_name, layer)

    def compute_sref_attention_aux(
        self,
        q: Tensor,
        k: Tensor,
        joint_seq_lens: list[int],
        sref_key_ranges: list[tuple[int, int]],
        sref_query_ranges: list[tuple[int, int]] | None,
        compute_enrichment: bool,
        enrichment_lower_bound: float,
        enrichment_upper_bound: float,
        enrichment_eps: float,
        compute_entropy: bool,
        entropy_lower_bound: float,
        entropy_upper_bound: float,
        entropy_eps: float,
        sref_entropy_target_lower_bounds: Tensor | None = None,
        sref_timestep_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if not compute_enrichment and not compute_entropy:
            return {}
        if compute_enrichment:
            if enrichment_lower_bound > enrichment_upper_bound:
                raise ValueError(
                    "Expected enrichment_lower_bound <= enrichment_upper_bound, got "
                    f"{enrichment_lower_bound} > {enrichment_upper_bound}"
                )
            if enrichment_eps <= 0.0:
                raise ValueError(f"Expected enrichment_eps > 0, got {enrichment_eps}")
        if compute_entropy:
            if entropy_lower_bound > entropy_upper_bound:
                raise ValueError(
                    f"Expected entropy_lower_bound <= entropy_upper_bound, got {entropy_lower_bound} > {entropy_upper_bound}"
                )
            if entropy_eps <= 0.0:
                raise ValueError(f"Expected entropy_eps > 0, got {entropy_eps}")
        if len(joint_seq_lens) != len(sref_key_ranges):
            raise ValueError(
                "joint_seq_lens and sref_key_ranges must have the same length, got "
                f"{len(joint_seq_lens)} and {len(sref_key_ranges)}"
            )
        if sref_query_ranges is not None and len(joint_seq_lens) != len(sref_query_ranges):
            raise ValueError(
                "joint_seq_lens and sref_query_ranges must have the same length, got "
                f"{len(joint_seq_lens)} and {len(sref_query_ranges)}"
            )
        if sref_timestep_weights is not None and sref_timestep_weights.shape[0] != len(joint_seq_lens):
            raise ValueError(
                "sref_timestep_weights must have one entry per sample, got "
                f"{sref_timestep_weights.shape[0]} and {len(joint_seq_lens)}"
            )
        if sref_entropy_target_lower_bounds is not None and sref_entropy_target_lower_bounds.shape[0] != len(joint_seq_lens):
            raise ValueError(
                "sref_entropy_target_lower_bounds must have one entry per sample, got "
                f"{sref_entropy_target_lower_bounds.shape[0]} and {len(joint_seq_lens)}"
            )

        q = q.to(torch.float32)
        k = k.to(torch.float32)
        if sref_timestep_weights is not None:
            sref_timestep_weights = sref_timestep_weights.to(device=q.device, dtype=torch.float32)
        if sref_entropy_target_lower_bounds is not None:
            sref_entropy_target_lower_bounds = sref_entropy_target_lower_bounds.to(device=q.device, dtype=torch.float32)

        enrichment_penalty_sum = None
        enrichment_value_sum = None
        enrichment_count = 0
        entropy_penalty_sum = None
        entropy_value_sum = None
        entropy_target_lower_bound_sum = None
        entropy_count = 0
        seq_offset = 0
        scale = 1.0 / math.sqrt(q.shape[-1])

        default_query_ranges = [(0, int(seq_len)) for seq_len in joint_seq_lens]
        iter_query_ranges = default_query_ranges if sref_query_ranges is None else sref_query_ranges

        for seq_idx, (seq_len, (k_start, k_end), (q_start, q_end)) in enumerate(
            zip(joint_seq_lens, sref_key_ranges, iter_query_ranges)
        ):
            seq_len = int(seq_len)
            if seq_len < 0:
                raise ValueError(f"Sequence length must be non-negative, got {seq_len} for sample {seq_idx}")
            if seq_len == 0:
                continue
            if seq_offset + seq_len > q.shape[0]:
                raise ValueError(
                    f"Sequence slicing overflow in block-0 sref attention auxiliary computation: "
                    f"offset {seq_offset}, seq_len {seq_len}, total {q.shape[0]}"
                )

            q_i = q[seq_offset : seq_offset + seq_len]
            k_i = k[seq_offset : seq_offset + seq_len]
            seq_offset += seq_len

            q_start = min(max(int(q_start), 0), seq_len)
            q_end = min(max(int(q_end), q_start), seq_len)
            if q_start == q_end:
                continue

            k_start = min(max(int(k_start), 0), seq_len)
            k_end = min(max(int(k_end), k_start), seq_len)
            group_len = k_end - k_start
            if group_len == 0:
                continue

            logits = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
            attn = logits.softmax(dim=-1)[:, q_start:q_end]
            if attn.shape[1] == 0:
                continue

            timestep_weight = None if sref_timestep_weights is None else sref_timestep_weights[seq_idx]
            entropy_target_lower_bound = (
                None if sref_entropy_target_lower_bounds is None else sref_entropy_target_lower_bounds[seq_idx]
            )

            if compute_enrichment:
                group_mass = attn[..., k_start:k_end].sum(dim=(-1, -2))
                total_mass = attn.sum(dim=(-1, -2)).clamp_min(enrichment_eps)
                ratio = group_mass / total_mass
                uniform_expectation = max(group_len / max(seq_len, 1), enrichment_eps)
                enrichment = ratio / uniform_expectation
                penalty = torch.relu(enrichment_lower_bound - enrichment).square() + torch.relu(
                    enrichment - enrichment_upper_bound
                ).square()
                penalty_value = penalty.sum()
                if timestep_weight is not None:
                    penalty_value = penalty_value * timestep_weight
                enrichment_value = enrichment.sum()
                enrichment_penalty_sum = (
                    penalty_value if enrichment_penalty_sum is None else enrichment_penalty_sum + penalty_value
                )
                enrichment_value_sum = (
                    enrichment_value if enrichment_value_sum is None else enrichment_value_sum + enrichment_value
                )
                enrichment_count += enrichment.numel()

            if compute_entropy and group_len > 1:
                sref_attn = attn[..., k_start:k_end]
                k_mass = sref_attn.sum(dim=-2)
                valid_mask = k_mass.sum(dim=-1) > entropy_eps
                if valid_mask.any():
                    k_probs = k_mass / k_mass.sum(dim=-1, keepdim=True).clamp_min(entropy_eps)
                    k_probs_safe = k_probs.clamp_min(entropy_eps)
                    entropy = -(k_probs * k_probs_safe.log()).sum(dim=-1) / math.log(float(group_len))
                    entropy = torch.where(valid_mask, entropy, torch.zeros_like(entropy))
                    if entropy_target_lower_bound is None:
                        penalty = torch.relu(entropy_lower_bound - entropy).square() + torch.relu(
                            entropy - entropy_upper_bound
                        ).square()
                    else:
                        penalty = torch.relu(entropy_target_lower_bound - entropy).square()
                    penalty = torch.where(valid_mask, penalty, torch.zeros_like(penalty))
                    penalty_value = penalty.sum()
                    if entropy_target_lower_bound is None and timestep_weight is not None:
                        penalty_value = penalty_value * timestep_weight
                    entropy_value = entropy.sum()
                    entropy_penalty_sum = (
                        penalty_value if entropy_penalty_sum is None else entropy_penalty_sum + penalty_value
                    )
                    entropy_value_sum = entropy_value if entropy_value_sum is None else entropy_value_sum + entropy_value
                    if entropy_target_lower_bound is not None:
                        target_lower_bound_value = entropy_target_lower_bound * valid_mask.to(torch.float32).sum()
                        entropy_target_lower_bound_sum = (
                            target_lower_bound_value
                            if entropy_target_lower_bound_sum is None
                            else entropy_target_lower_bound_sum + target_lower_bound_value
                        )
                    entropy_count += int(valid_mask.sum().item())

        if seq_offset != q.shape[0]:
            raise ValueError(
                f"Sequence slicing mismatch in block-0 sref attention auxiliary computation: used {seq_offset}, total {q.shape[0]}"
            )

        zero = q.sum() * 0.0
        aux: dict[str, Tensor] = {}
        if compute_enrichment:
            if enrichment_count == 0:
                aux["loss_sref_enrichment"] = zero
                aux["sref_enrichment"] = zero
            else:
                assert enrichment_penalty_sum is not None
                assert enrichment_value_sum is not None
                aux["loss_sref_enrichment"] = enrichment_penalty_sum / float(enrichment_count)
                aux["sref_enrichment"] = enrichment_value_sum / float(enrichment_count)
        if compute_entropy:
            if entropy_count == 0:
                aux["loss_sref_entropy"] = zero
                aux["sref_entropy"] = zero
            else:
                assert entropy_penalty_sum is not None
                assert entropy_value_sum is not None
                aux["loss_sref_entropy"] = entropy_penalty_sum / float(entropy_count)
                aux["sref_entropy"] = entropy_value_sum / float(entropy_count)
                if entropy_target_lower_bound_sum is not None:
                    aux["sref_entropy_target_lower_bound"] = entropy_target_lower_bound_sum / float(entropy_count)
        return aux

    def _forward_wo_dist(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        img_varlen_config: VarLenConfig,
        txt_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
        return_sref_enrichment: bool = False,
        return_sref_entropy: bool = False,
        joint_seq_lens: list[int] | None = None,
        sref_key_ranges: list[tuple[int, int]] | None = None,
        sref_query_ranges: list[tuple[int, int]] | None = None,
        sref_enrichment_lower_bound: float = 0.08,
        sref_enrichment_upper_bound: float = 0.5,
        sref_enrichment_eps: float = 1e-6,
        sref_entropy_lower_bound: float = 0.06,
        sref_entropy_upper_bound: float = 0.14,
        sref_entropy_eps: float = 1e-6,
        sref_entropy_target_lower_bounds: Tensor | None = None,
        sref_timestep_weights: Tensor | None = None,
        rope_fa_progress: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Double Stream Forward

        Args:
            img (Tensor): (BL) x D
            txt (Tensor): (BL) x D
            vec (Tensor): B x D
            pe (tuple[Tensor]): cos, sin: (BL) x D // 2
            img_csr_index (VarLenConfig)
            txt_csr_index (VarLenConfig)

        Returns:
            tuple[Tensor, Tensor]: img, txt
        """

        if self.merged_mod:
            img_mod1, img_mod2, txt_mod1, txt_mod2 = self.img_txt_mod(vec)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)

        img_modulated = varlen_scale_shift(
            img_modulated, img_mod1.scale, img_mod1.shift, img_varlen_config, out_dtype=self.img_attn.qkv.weight.dtype
        )

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "L (K H D) -> K L H D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = varlen_scale_shift(
            txt_modulated, txt_mod1.scale, txt_mod1.shift, txt_varlen_config, out_dtype=self.txt_attn.qkv.weight.dtype
        )

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (K H D) -> K L H D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = cat_seq([img_q, txt_q], [img_varlen_config.split_index, txt_varlen_config.split_index])
        k = cat_seq([img_k, txt_k], [img_varlen_config.split_index, txt_varlen_config.split_index])
        v = cat_seq([img_v, txt_v], [img_varlen_config.split_index, txt_varlen_config.split_index])

        q, k = triton_apply_rope(q, k, pe[0], pe[1], True)
        q, k = q.to(v), k.to(v)
        if self.rope_fa_config.enabled:
            if joint_seq_lens is None or sref_key_ranges is None:
                raise ValueError("joint_seq_lens and sref_key_ranges are required for frequency-aware RoPE.")
            k = self._apply_frequency_aware_rope(
                k=k,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                rope_fa_progress=rope_fa_progress,
            )

        sref_aux = None
        if return_sref_enrichment or return_sref_entropy:
            if joint_seq_lens is None or sref_key_ranges is None:
                raise ValueError("joint_seq_lens and sref_key_ranges are required for sref auxiliary computation.")
            sref_aux = self.compute_sref_attention_aux(
                q=q,
                k=k,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                sref_query_ranges=sref_query_ranges,
                compute_enrichment=return_sref_enrichment,
                enrichment_lower_bound=sref_enrichment_lower_bound,
                enrichment_upper_bound=sref_enrichment_upper_bound,
                enrichment_eps=sref_enrichment_eps,
                compute_entropy=return_sref_entropy,
                entropy_lower_bound=sref_entropy_lower_bound,
                entropy_upper_bound=sref_entropy_upper_bound,
                entropy_eps=sref_entropy_eps,
                sref_entropy_target_lower_bounds=sref_entropy_target_lower_bounds,
                sref_timestep_weights=sref_timestep_weights,
            )

        attn = attention(
            q,
            k,
            v,
            mode="flash_varlen",
            cu_seqlens_q=varlen_attention_config.cu_seqlens_q,
            cu_seqlens_kv=varlen_attention_config.cu_seqlens_kv,
            max_seqlen_q=varlen_attention_config.max_seqlen_q,
            max_seqlen_kv=varlen_attention_config.max_seqlen_kv,
        )

        img_attn, txt_attn = split_seq(attn, [img_varlen_config.seq_lens, txt_varlen_config.seq_lens])
        img_attn = img_attn.flatten(1, 2)
        txt_attn = txt_attn.flatten(1, 2)

        # calculate the img bloks
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img_mlp = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        # img = img + img_mod2.gate * img_mlp
        # img = scale_add_residual(self.img_attn.proj(img_attn), img_mod1.gate, img)
        img = img + varlen_gate(self.img_attn.proj(img_attn), img_mod1.gate, img_varlen_config)
        txt = txt + varlen_gate(self.txt_attn.proj(txt_attn), txt_mod1.gate, txt_varlen_config)

        img_mlp = self.img_mlp(
            varlen_scale_shift(
                self.img_norm2(img), img_mod2.scale, img_mod2.shift, img_varlen_config, out_dtype=img.dtype
            )
        )
        img = img + varlen_gate(img_mlp, img_mod2.gate, img_varlen_config)

        # calculate the txt bloks
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt_mlp = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # txt = txt + txt_mod2.gate * txt_mlp
        txt_mlp = self.txt_mlp(
            varlen_scale_shift(
                self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift, txt_varlen_config, out_dtype=txt.dtype
            )
        )
        txt = txt + varlen_gate(txt_mlp, txt_mod2.gate, txt_varlen_config)

        if return_sref_enrichment or return_sref_entropy:
            assert sref_aux is not None
            return img, txt, sref_aux
        return img, txt

    def _forward_w_dist(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        img_varlen_config: VarLenConfig,
        txt_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
        return_sref_enrichment: bool = False,
        return_sref_entropy: bool = False,
        joint_seq_lens: list[int] | None = None,
        sref_key_ranges: list[tuple[int, int]] | None = None,
        sref_query_ranges: list[tuple[int, int]] | None = None,
        sref_enrichment_lower_bound: float = 0.08,
        sref_enrichment_upper_bound: float = 0.5,
        sref_enrichment_eps: float = 1e-6,
        sref_entropy_lower_bound: float = 0.06,
        sref_entropy_upper_bound: float = 0.14,
        sref_entropy_eps: float = 1e-6,
        sref_entropy_target_lower_bounds: Tensor | None = None,
        sref_timestep_weights: Tensor | None = None,
        rope_fa_progress: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Double Stream Forward

        Args:
            img (Tensor): (BL) x D
            txt (Tensor): (BL) x D
            vec (Tensor): B x D
            pe (tuple[Tensor]): cos, sin: (BL) x D // 2
            img_csr_index (VarLenConfig)
            txt_csr_index (VarLenConfig)

        Returns:
            tuple[Tensor, Tensor]: img, txt
        """

        if self.merged_mod:
            img_mod1, img_mod2, txt_mod1, txt_mod2 = self.img_txt_mod(vec)
        else:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)

        img_modulated = varlen_scale_shift(
            img_modulated, img_mod1.scale, img_mod1.shift, img_varlen_config, out_dtype=self.img_attn.qkv.weight.dtype
        )

        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "L (H K D) -> K L H D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = varlen_scale_shift(
            txt_modulated, txt_mod1.scale, txt_mod1.shift, txt_varlen_config, out_dtype=self.txt_attn.qkv.weight.dtype
        )

        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "L (H K D) -> K L H D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = cat_seq([img_q, txt_q], [img_varlen_config.split_index, txt_varlen_config.split_index])
        k = cat_seq([img_k, txt_k], [img_varlen_config.split_index, txt_varlen_config.split_index])
        v = cat_seq([img_v, txt_v], [img_varlen_config.split_index, txt_varlen_config.split_index])

        q, k = triton_apply_rope(q, k, pe[0], pe[1], True)
        q, k = q.to(v), k.to(v)
        if self.rope_fa_config.enabled:
            if joint_seq_lens is None or sref_key_ranges is None:
                raise ValueError("joint_seq_lens and sref_key_ranges are required for frequency-aware RoPE.")
            k = self._apply_frequency_aware_rope(
                k=k,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                rope_fa_progress=rope_fa_progress,
            )

        sref_aux = None
        if return_sref_enrichment or return_sref_entropy:
            if joint_seq_lens is None or sref_key_ranges is None:
                raise ValueError("joint_seq_lens and sref_key_ranges are required for sref auxiliary computation.")
            sref_aux = self.compute_sref_attention_aux(
                q=q,
                k=k,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                sref_query_ranges=sref_query_ranges,
                compute_enrichment=return_sref_enrichment,
                enrichment_lower_bound=sref_enrichment_lower_bound,
                enrichment_upper_bound=sref_enrichment_upper_bound,
                enrichment_eps=sref_enrichment_eps,
                compute_entropy=return_sref_entropy,
                entropy_lower_bound=sref_entropy_lower_bound,
                entropy_upper_bound=sref_entropy_upper_bound,
                entropy_eps=sref_entropy_eps,
                sref_entropy_target_lower_bounds=sref_entropy_target_lower_bounds,
                sref_timestep_weights=sref_timestep_weights,
            )
        attn = attention(
            q,
            k,
            v,
            mode="flash_varlen",
            cu_seqlens_q=varlen_attention_config.cu_seqlens_q,
            cu_seqlens_kv=varlen_attention_config.cu_seqlens_kv,
            max_seqlen_q=varlen_attention_config.max_seqlen_q,
            max_seqlen_kv=varlen_attention_config.max_seqlen_kv,
        )

        img_attn, txt_attn = split_seq(attn, [img_varlen_config.seq_lens, txt_varlen_config.seq_lens])
        img_attn = img_attn.flatten(1, 2)
        txt_attn = txt_attn.flatten(1, 2)

        # calculate the img bloks
        # img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        # img_mlp = self.img_mlp((1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift)
        # img = img + img_mod2.gate * img_mlp
        # img = scale_add_residual(self.img_attn.proj(img_attn), img_mod1.gate, img)
        img = img + varlen_gate(self.img_attn.proj(img_attn), img_mod1.gate, img_varlen_config)
        txt = txt + varlen_gate(self.txt_attn.proj(txt_attn), txt_mod1.gate, txt_varlen_config)

        img_mlp = self.img_mlp(
            varlen_scale_shift(
                self.img_norm2(img), img_mod2.scale, img_mod2.shift, img_varlen_config, out_dtype=img.dtype
            )
        )
        img = img + varlen_gate(img_mlp, img_mod2.gate, img_varlen_config)

        # calculate the txt bloks
        # txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        # txt_mlp = self.txt_mlp((1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift)
        # txt = txt + txt_mod2.gate * txt_mlp
        txt_mlp = self.txt_mlp(
            varlen_scale_shift(
                self.txt_norm2(txt), txt_mod2.scale, txt_mod2.shift, txt_varlen_config, out_dtype=txt.dtype
            )
        )
        txt = txt + varlen_gate(txt_mlp, txt_mod2.gate, txt_varlen_config)

        if return_sref_enrichment or return_sref_entropy:
            assert sref_aux is not None
            return img, txt, sref_aux
        return img, txt

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        img_varlen_config: VarLenConfig,
        txt_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
        return_sref_enrichment: bool = False,
        return_sref_entropy: bool = False,
        joint_seq_lens: list[int] | None = None,
        sref_key_ranges: list[tuple[int, int]] | None = None,
        sref_query_ranges: list[tuple[int, int]] | None = None,
        sref_enrichment_lower_bound: float = 0.08,
        sref_enrichment_upper_bound: float = 0.5,
        sref_enrichment_eps: float = 1e-6,
        sref_entropy_lower_bound: float = 0.06,
        sref_entropy_upper_bound: float = 0.14,
        sref_entropy_eps: float = 1e-6,
        sref_entropy_target_lower_bounds: Tensor | None = None,
        sref_timestep_weights: Tensor | None = None,
        rope_fa_progress: Tensor | None = None,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.is_shard:
            return self._forward_w_dist(
                img,
                txt,
                vec,
                pe,
                img_varlen_config,
                txt_varlen_config,
                varlen_attention_config,
                return_sref_enrichment=return_sref_enrichment,
                return_sref_entropy=return_sref_entropy,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                sref_query_ranges=sref_query_ranges,
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
        else:
            return self._forward_wo_dist(
                img,
                txt,
                vec,
                pe,
                img_varlen_config,
                txt_varlen_config,
                varlen_attention_config,
                return_sref_enrichment=return_sref_enrichment,
                return_sref_entropy=return_sref_entropy,
                joint_seq_lens=joint_seq_lens,
                sref_key_ranges=sref_key_ranges,
                sref_query_ranges=sref_query_ranges,
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


class VarLenSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        adaln_dim: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(adaln_dim, hidden_size, double=False)

        self.is_shard = False
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with truncated normal distribution."""
        nn.init.trunc_normal_(self.linear1.weight, mean=0.0, std=0.02)
        if self.linear1.bias is not None:
            nn.init.zeros_(self.linear1.bias)
        nn.init.trunc_normal_(self.linear2.weight, mean=0.0, std=0.02)
        if self.linear2.bias is not None:
            nn.init.zeros_(self.linear2.bias)

    def apply_compile(self):
        for layer_name, layer in self.named_children():
            if "linear1" in layer_name or "linear2" in layer_name:
                layer = _compile_if_available(layer, dynamic=True, mode="max-autotune-no-cudagraphs")
                self.register_module(layer_name, layer)

    def parallelize_module(self, device_mesh: DeviceMesh, use_replicate_modulation_linear=False):
        modulation_linear_tp_style = NoParallel if use_replicate_modulation_linear else ColwiseParallel
        layer_tp_plan = {
            "modulation.lin": modulation_linear_tp_style(
                input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=False
            ),
            "pre_norm": SequenceParallel(sequence_dim=0, use_local_output=False),  # sp
            "linear1": ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(dim=1)),
            "linear2": RowwiseParallel(output_layouts=Shard(dim=0), use_local_output=False),
        }

        self.linear1.weight.data = rearrange(
            self.linear1.weight.data,
            "(K H D) I -> (H K D) I",
            K=self.mlp_hidden_dim // self.hidden_dim + 3,
            H=self.num_heads,
        )
        if self.linear1.bias is not None:
            self.linear1.bias.data = rearrange(
                self.linear1.bias.data,
                "(K H D) -> (H K D)",
                K=self.mlp_hidden_dim // self.hidden_dim + 3,
                H=self.num_heads,
            )
        self.linear2.weight.data = rearrange(
            self.linear2.weight.data,
            "I (M H D) -> I (H M D)",
            M=self.mlp_hidden_dim // self.hidden_dim + 1,
            H=self.num_heads,
        )

        self.norm.query_norm.set_device_mesh(device_mesh=device_mesh)
        self.norm.key_norm.set_device_mesh(device_mesh=device_mesh)

        self.num_heads = self.num_heads // device_mesh.size()
        self.hidden_size = self.hidden_size // device_mesh.size()
        self.mlp_hidden_dim = self.mlp_hidden_dim // device_mesh.size()
        self.device_mesh = device_mesh

        # Custom parallelization plan for the model
        parallelize_module(module=self, device_mesh=device_mesh, parallelize_plan=layer_tp_plan)

        self.is_shard = True

    def set_fsdp_sequence_parallel(self, **fsdp_config):
        fully_shard(self.modulation, **fsdp_config)
        fully_shard(self.norm.query_norm, **fsdp_config)
        fully_shard(self.norm.key_norm, **fsdp_config)

        self.modulation.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.norm.query_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore
        self.norm.key_norm.set_reduce_scatter_divide_factor(1.0)  # type: ignore

    def _forward_wo_dist(
        self,
        x: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        x_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = varlen_scale_shift(
            self.pre_norm(x), mod.scale, mod.shift, x_varlen_config, out_dtype=self.linear1.weight.dtype
        )

        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "L (K H D) -> K L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = triton_apply_rope(q, k, pe[0], pe[1], True)
        q, k = q.to(v), k.to(v)

        attn = attention(
            q,
            k,
            v,
            mode="flash_varlen",
            cu_seqlens_q=varlen_attention_config.cu_seqlens_q,
            cu_seqlens_kv=varlen_attention_config.cu_seqlens_kv,
            max_seqlen_q=varlen_attention_config.max_seqlen_q,
            max_seqlen_kv=varlen_attention_config.max_seqlen_kv,
        )
        attn = attn.flatten(1, 2)

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        # x + mod.gate * output
        return x + varlen_gate(output, mod.gate, x_varlen_config)

    def _forward_w_dist(
        self,
        x: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        x_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        # x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        x_mod = varlen_scale_shift(
            self.pre_norm(x), mod.scale, mod.shift, x_varlen_config, out_dtype=self.linear1.weight.dtype
        )

        # do not split
        # -> L (H//tp K D) --> L H//tp (3+4 D)
        qkv_mlp = rearrange(
            self.linear1(x_mod),
            "L (H D) -> L H D",
            H=self.num_heads,
        )

        # L H//tp (3+4 D) -> L H//tp 3D, L H//tp 4D
        qkv, mlp = torch.split(
            qkv_mlp, [3 * self.hidden_size // self.num_heads, self.mlp_hidden_dim // self.num_heads], dim=-1
        )

        q, k, v = rearrange(qkv, "L H (K D) -> K L H D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = triton_apply_rope(q, k, pe[0], pe[1], True)
        q, k = q.to(v), k.to(v)

        attn = attention(
            q,
            k,
            v,
            mode="flash_varlen",
            cu_seqlens_q=varlen_attention_config.cu_seqlens_q,
            cu_seqlens_kv=varlen_attention_config.cu_seqlens_kv,
            max_seqlen_q=varlen_attention_config.max_seqlen_q,
            max_seqlen_kv=varlen_attention_config.max_seqlen_kv,
        )

        # compute activation in mlp stream, cat again and run second linear layer
        # L H//tp D, L H//tp 4D -> L (H//tp 5D) -> L//tp (H D)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1).flatten(1, 2))
        # self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1).flatten(1, 2))
        # x + mod.gate * output
        return x + varlen_gate(output, mod.gate, x_varlen_config)

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: tuple[Tensor, Tensor],
        x_varlen_config: VarLenConfig,
        varlen_attention_config: VarlenAttentionConfig,
    ):
        if self.is_shard:
            return self._forward_w_dist(x, vec, pe, x_varlen_config, varlen_attention_config)
        else:
            return self._forward_wo_dist(x, vec, pe, x_varlen_config, varlen_attention_config)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self._init_weights()

    def _init_weights(self):
        """
        Initialize output layer to zero for diffusion models.
        This ensures the model outputs zeros initially.
        """
        # Zero-initialize the output projection
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Zero-initialize the adaLN modulation layer
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class VarlenLastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, adaln_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_dim, 2 * hidden_size, bias=True))
        self._init_weights()

    def _init_weights(self):
        """
        Initialize output layer to zero for diffusion models.
        This ensures the model outputs zeros initially.
        """
        # Zero-initialize the output projection
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Zero-initialize the adaLN modulation layer
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x: Tensor, vec: Tensor, varlen_config: VarLenConfig) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        norm_x = self.norm_final(x)
        x = varlen_scale_shift(norm_x, scale, shift, varlen_config, out_dtype=self.linear.weight.dtype)
        x = self.linear(x)
        return x


class LQTokenMapping(nn.Module):
    def __init__(self, input_dims=4096, intermediate_dims=4096, out_dims=64, mapping_type="LINEAR"):
        super().__init__()
        self.mapping_type = mapping_type
        if mapping_type == "LINEAR":
            self.layer_0 = nn.Linear(input_dims, out_dims)
            self.layer_0.weight.data.zero_()
            self.layer_0.bias.data.zero_()
        elif mapping_type == "LN+MLP" or mapping_type == "CAT+LN+MLP":
            self.norm = RMSNorm(input_dims)
            self.gate_proj = nn.Linear(input_dims, intermediate_dims, bias=False)
            self.up_proj = nn.Linear(input_dims, intermediate_dims, bias=False)
            self.act_fn = nn.SiLU()

            self.layer_0 = nn.Linear(intermediate_dims, out_dims)
            self.layer_0.weight.data.zero_()
            self.layer_0.bias.data.zero_()

            nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
            nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=0.02)
        else:
            raise NotImplementedError(f"Unregonized {self.mapping_type=}")

        self.compile_forward = _compile_if_available(
            self.compile_forward, dynamic=True, mode="max-autotune-no-cudagraphs"
        )

    def compile_forward(self, input_x, origin_x=None):
        if self.mapping_type == "LINEAR":
            return self.layer_0(input_x)
        elif self.mapping_type == "LN+MLP":
            norm_x = self.norm(input_x)
            return self.layer_0(self.act_fn(self.gate_proj(norm_x)) * self.up_proj(norm_x))
        elif self.mapping_type == "CAT+LN+MLP":
            assert origin_x is not None

            norm_x = self.norm(torch.cat([input_x, origin_x], dim=-1))
            return self.layer_0(self.act_fn(self.gate_proj(norm_x)) * self.up_proj(norm_x))
        else:
            raise NotImplementedError(f"Unregonized {self.mapping_type=}")

    def forward(self, input_x, origin_x=None):
        return self.compile_forward(input_x, origin_x=origin_x)
