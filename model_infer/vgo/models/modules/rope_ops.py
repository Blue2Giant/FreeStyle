from __future__ import annotations

"""
RoPE (Rotary Position Embedding) implementation for Ascend/NPU friendly runtime.

The original `minivgo` repo uses Triton kernels for RoPE. On Ascend environments,
Triton is typically unavailable and CUDA-only kernels cannot be imported.

This file provides a pure PyTorch implementation that:
- Works on both CUDA and NPU devices
- Supports autograd (forward/backward)
- Keeps the original public API: `apply_rope(q, k, cos, sin, inplace=False)`

Tensor shapes (same as the original implementation):
- q: (bsz, seq_len, n_q_heads, head_dim) or (seq_len, n_q_heads, head_dim)
- k: (bsz, seq_len, n_kv_heads, head_dim) or (seq_len, n_kv_heads, head_dim)
- cos/sin: (bsz or 1, seq_len, head_dim // 2), float32
"""

import os

import torch

try:
    import torch_npu  # type: ignore

    _HAS_TORCH_NPU = True
except ImportError:
    torch_npu = None
    _HAS_TORCH_NPU = False


_ROPE_FASTPATH_DISABLE_VALUES = {"0", "false", "off", "no"}


def _use_npu_rope_fastpath() -> bool:
    return os.getenv("VGO_USE_NPU_ROPE_FASTPATH", "1").strip().lower() not in _ROPE_FASTPATH_DISABLE_VALUES


def _compact_cos_sin(cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    expected_compact_dim = head_dim // 2
    expected_full_dim = head_dim
    if cos.shape[-1] == expected_compact_dim and sin.shape[-1] == expected_compact_dim:
        return cos, sin
    if cos.shape[-1] == expected_full_dim and sin.shape[-1] == expected_full_dim:
        return cos[..., 0::2], sin[..., 0::2]
    raise ValueError(
        "Expected cos/sin last dim to be either head_dim//2 or head_dim, "
        f"got {cos.shape[-1]=} {sin.shape[-1]=} for {head_dim=}"
    )


def _full_cos_sin(cos: torch.Tensor, sin: torch.Tensor, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.shape[-1] == head_dim and sin.shape[-1] == head_dim:
        return cos, sin
    compact_cos, compact_sin = _compact_cos_sin(cos, sin, head_dim)
    return compact_cos.repeat_interleave(2, dim=-1), compact_sin.repeat_interleave(2, dim=-1)


def _apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply RoPE to a single tensor (q or k).

    Args:
        x: (B, S, H, D)
        cos/sin: (B or 1, S, D//2), float32

    Returns:
        Tensor with same shape/dtype as `x`.
    """
    if x.ndim != 4:
        raise ValueError(f"Expected x.ndim == 4, got {x.ndim} with shape {tuple(x.shape)}")

    dtype = x.dtype
    compact_cos, compact_sin = _compact_cos_sin(cos, sin, x.shape[-1])
    x_float = x.float() if dtype in (torch.float16, torch.bfloat16) else x

    # (B, S, H, D) -> (B, S, H, D/2, 2)
    x_ = x_float.unflatten(-1, (x_float.shape[-1] // 2, 2))
    x_cos = x_[..., 0]
    x_sin = x_[..., 1]

    # Broadcast cos/sin to (B or 1, S, 1, D/2)
    cos_ = compact_cos.unsqueeze(-2)
    sin_ = compact_sin.unsqueeze(-2)

    out_cos = x_cos * cos_ - x_sin * sin_
    out_sin = x_sin * cos_ + x_cos * sin_

    out = torch.stack((out_cos, out_sin), dim=-1).flatten(-2, -1)
    return out.to(dtype=dtype)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inplace: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to q/k.

    Note:
      - `inplace` is kept for API compatibility. This implementation is functional
        (does not modify inputs in-place) to preserve autograd friendliness across devices.
    """
    if cos.dtype != torch.float32 or sin.dtype != torch.float32:
        raise ValueError(f"Expected cos/sin float32, got {cos.dtype=} {sin.dtype=}")

    n_dim = q.ndim
    if n_dim == 3:
        q_ = q.unsqueeze(0)
        k_ = k.unsqueeze(0)
    elif n_dim == 4:
        q_, k_ = q, k
    else:
        raise ValueError(f"Unsupported q/k dims: {q.ndim}")

    if not inplace:
        q_ = q_.clone()
        k_ = k_.clone()

    if q_.shape[-1] % 2 != 0 or k_.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE requires even head_dim, got {q_.shape[-1]=} {k_.shape[-1]=}")

    if _HAS_TORCH_NPU and _use_npu_rope_fastpath() and q_.device.type == "npu" and k_.device.type == "npu":
        cos_full, sin_full = _full_cos_sin(cos, sin, q_.shape[-1])
        q_out = torch_npu.npu_rotary_mul(q_, cos_full.unsqueeze(-2), sin_full.unsqueeze(-2), "interleave").to(q.dtype)
        k_out = torch_npu.npu_rotary_mul(k_, cos_full.unsqueeze(-2), sin_full.unsqueeze(-2), "interleave").to(k.dtype)
    else:
        q_out = _apply_rope_single(q_, cos, sin)
        k_out = _apply_rope_single(k_, cos, sin)

    if n_dim == 3:
        return q_out[0], k_out[0]
    return q_out, k_out


__all__ = ["apply_rope"]
