import math
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger

from vgo.utils.common_utils import DETERMINISTIC_MODE

is_flash_attn_3_available = False
try:
    from hopper.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

    is_flash_attn_3_available = True
    # https://github.com/huggingface/transformers/blob/2c56461194c89837d86fc806c507a7536f952db0/src/transformers/modeling_flash_attention_utils.py#L231
    torch._dynamo.config.capture_scalar_outputs = True
    # flash_attn_func = torch._dynamo.disable(flash_attn_func)
    # flash_attn_varlen_func = torch._dynamo.disable(flash_attn_varlen_func)
except Exception as e:
    logger.warning(f"Flash attn 3 not found: {e}. Try to use flash attn 2 instead.")
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
        from flash_attn.bert_padding import pad_input, unpad_input
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func

        torch._dynamo.config.capture_scalar_outputs = True
        # torch.compiler.disable(flash_attn_func)
        # torch.compiler.disable(flash_attn_varlen_func)
    except ImportError:
        logger.warning("Both Flash attn 2 and 3 not found.")
        flash_attn_varlen_func = None
        flash_attn_func = None
        flash_attn_varlen_qkvpacked_func = None
        pad_input = None
        unpad_input = None


# 定义不同注意力模式下的张量形状变换逻辑
MEMORY_LAYOUT = {
    # Flash Attention模式：
    # 预处理：将形状 (batch_size, seq_len, num_heads, head_dim) 转换为 (batch_size*seq_len, num_heads, head_dim)
    # 后处理：保持形状不变
    "flash_varlen": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]) if x.ndim == 4 else x,  # 展平batch和序列维度
        lambda x: x,  # 无变化
    ),
    "flash": (
        lambda x: x,  # 保持形状
        lambda x: x,  # 保持形状
    ),
    # PyTorch原生实现模式：
    # 预处理：交换序列维度和注意力头维度 → (batch_size, num_heads, seq_len, head_dim)
    # 后处理：恢复原始维度顺序
    "torch": (
        lambda x: x.transpose(1, 2),  # [b, s, h, d] → [b, h, s, d]
        lambda x: x.transpose(1, 2),  # [b, h, s, d] → [b, s, h, d]
    ),
    # 常规自注意力模式（同torch）
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "sageattn": (
        lambda x: x,
        lambda x: x,
    ),
}


def _maybe_to_local(x: torch.Tensor) -> torch.Tensor:
    dtensor_module = getattr(torch.distributed, "tensor", None)
    dtensor_cls = getattr(dtensor_module, "DTensor", None)
    if dtensor_cls is not None and isinstance(x, dtensor_cls):
        return x.to_local()
    return x


def _scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    drop_rate: float,
    causal: bool,
) -> torch.Tensor:
    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.to(q.dtype)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)


def _flash_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None,
    drop_rate: float,
    causal: bool,
) -> torch.Tensor:
    q = _maybe_to_local(q).transpose(1, 2)
    k = _maybe_to_local(k).transpose(1, 2)
    v = _maybe_to_local(v).transpose(1, 2)
    return _scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, drop_rate=drop_rate, causal=causal).transpose(
        1, 2
    )


def _flash_varlen_attention_fallback(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    drop_rate: float,
    causal: bool,
) -> torch.Tensor:
    q = _maybe_to_local(q)
    k = _maybe_to_local(k)
    v = _maybe_to_local(v)
    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    kv_offsets = cu_seqlens_kv.detach().cpu().tolist()
    outputs = []
    for q_start, q_end, kv_start, kv_end in zip(q_offsets[:-1], q_offsets[1:], kv_offsets[:-1], kv_offsets[1:]):
        q_chunk = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        k_chunk = k[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        v_chunk = v[kv_start:kv_end].transpose(0, 1).unsqueeze(0)
        out_chunk = _scaled_dot_product_attention(
            q_chunk,
            k_chunk,
            v_chunk,
            attn_mask=None,
            drop_rate=drop_rate,
            causal=causal,
        )
        outputs.append(out_chunk.squeeze(0).transpose(0, 1))
    if not outputs:
        return q.new_empty(q.shape)
    return torch.cat(outputs, dim=0)


def get_cu_seqlens(text_mask, img_len):
    """计算Flash Attention所需的累积序列长度(cu_seqlens)

    Args:
        text_mask (torch.Tensor): 文本掩码,形状 (batch_size, text_seq_len)
        img_len (int): 图像序列长度

    Returns:
        torch.Tensor: 累积序列长度,形状 (2*batch_size + 1, ),数据类型int32
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)  # 每个样本的文本实际长度,形状 (batch_size,)
    max_len = text_mask.shape[1] + img_len  # 每个样本总长度（文本最大长度 + 图像长度）

    # 初始化累积序列长度张量（包含起始的0和每样本两个分割点）
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        # 第i个样本的总长度：文本实际长度 + 图像长度
        s = text_len[i] + img_len
        # 分割点1：文本+图像的结束位置
        s1 = i * max_len + s
        # 分割点2：样本的结束位置（包含填充部分）
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens


@dataclass
class VarlenAttentionConfig(OrderedDict):
    cu_seqlens_q: torch.Tensor
    cu_seqlens_kv: torch.Tensor
    max_seqlen_q: int
    max_seqlen_kv: int

    @classmethod
    def from_seq_lens(cls, seq_lens: list, device):
        csr_index = torch.tensor([0, *seq_lens]).cumsum(dim=0).to(device, torch.int32)
        max_length = max(seq_lens)
        return cls(csr_index, csr_index, max_length, max_length)


def attention(  # noqa: C901
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """多模式注意力计算

    Args:
        q (torch.Tensor): 查询张量,形状 (batch_size, seq_len_q, num_heads, head_dim)
        k (torch.Tensor): 键张量,形状 (batch_size, seq_len_kv, num_heads, head_dim)
        v (torch.Tensor): 值张量,形状 (batch_size, seq_len_kv, num_heads, head_dim)
        mode (str): 注意力模式,可选 'flash'/'torch'/'vanilla'
        drop_rate (float): 注意力矩阵的dropout概率
        attn_mask (torch.Tensor, optional): 注意力掩码。当为交叉注意力时形状 (batch_size, seq_len_kv),
            常规模式时形状 (batch_size, num_heads, seq_len_q, seq_len_kv)
        causal (bool): 是否使用因果（单向）注意力
        cu_seqlens_q (torch.Tensor): 查询的累积序列长度,用于Flash模式,形状 (batch_size*2 + 1,)
        cu_seqlens_kv (torch.Tensor): 键值的累积序列长度,用于Flash模式
        max_seqlen_q (int): 查询的最大序列长度
        max_seqlen_kv (int): 键值的最大序列长度

    Returns:
        torch.Tensor: 注意力输出,形状 (batch_size, seq_len_q, num_heads*head_dim)
    """

    if DETERMINISTIC_MODE:
        drop_rate = 0

    # 根据模式选择形状变换函数
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    # 预处理：调整张量形状以适应不同注意力实现
    q, k, v = pre_attn_layout(q), pre_attn_layout(k), pre_attn_layout(v)

    if mode == "torch":
        # 使用PyTorch原生实现（需要形状为 [batch_size, num_heads, seq_len, head_dim]）
        x = _scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, drop_rate=drop_rate, causal=causal)
    elif mode == "flash":
        if flash_attn_func is None:
            x = _flash_attention_fallback(q, k, v, attn_mask=attn_mask, drop_rate=drop_rate, causal=causal)
        elif attn_mask is None:
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=drop_rate,
                causal=causal,
                softmax_scale=None,
                deterministic=DETERMINISTIC_MODE,
            )
        else:
            # qkv: (batch_size, seqlen, 3, nheads, head_dim)
            qkv = torch.stack([q, k, v], dim=2)
            x = flash_attn_no_pad(qkv, attn_mask, causal=causal, dropout_p=drop_rate, softmax_scale=None)  # type: ignore
    elif mode == "flash_varlen":
        if flash_attn_varlen_func is None:
            if cu_seqlens_q is None or cu_seqlens_kv is None:
                raise ValueError("cu_seqlens_q and cu_seqlens_kv are required for flash_varlen fallback.")
            x = _flash_varlen_attention_fallback(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                drop_rate=drop_rate,
                causal=causal,
            )
        else:
            # 使用Flash Attention变长序列实现
            attn_out = flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                deterministic=DETERMINISTIC_MODE,
            )  # type: ignore
            if isinstance(attn_out, tuple):
                x = attn_out[0]
            else:
                x = attn_out
    elif mode == "sageattn":
        from sageattention import sageattn

        x = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    elif mode == "vanilla":
        # 手动实现常规注意力
        scale_factor = 1 / math.sqrt(q.size(-1))  # 缩放因子 1/sqrt(d_k)
        b, num_heads, seq_len_q, _ = q.shape
        seq_len_kv = k.size(2)  # 键值序列长度

        # 初始化注意力偏置（用于因果掩码或自定义掩码）
        attn_bias = torch.zeros(b, num_heads, seq_len_q, seq_len_kv, dtype=q.dtype, device=q.device)

        # 构建因果掩码
        if causal:
            assert attn_mask is None, "因果掩码不能与其他掩码同时使用"
            causal_mask = torch.ones(b, num_heads, seq_len_q, seq_len_q, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(~causal_mask, float("-inf"))  # 下三角为0,其余-inf
            attn_bias = attn_bias.to(q.dtype)

        # 合并自定义掩码
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(~attn_mask, float("-inf"))
            else:
                attn_bias += attn_mask  # 支持加性掩码

        # 计算注意力权重
        attn = (q @ k.transpose(-2, -1)) * scale_factor  # [b, h, s_q, s_kv]
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v  # [b, h, s_q, d]

    else:
        raise NotImplementedError(f"不支持的注意力模式: {mode}")

    # 后处理：恢复原始形状
    x = post_attn_layout(x)
    # 展平最后两个维度： [batch_size, seq_len, num_heads, head_dim] → [batch_size, seq_len, num_heads*head_dim]
    return x.reshape(x.shape[0], x.shape[1], -1)


def flash_attn_no_pad(qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None):
    assert flash_attn_varlen_qkvpacked_func is not None, "flash_attn_varlen_qkvpacked_func未定义"
    assert key_padding_mask is not None and pad_input is not None and unpad_input is not None, (
        "请安装flash-attn以使用Flash Attention"
    )
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_input(x, key_padding_mask)

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        deterministic=DETERMINISTIC_MODE,
    )
    output = rearrange(
        pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output

