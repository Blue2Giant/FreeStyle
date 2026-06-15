# Copy from https://github.com/volcengine/verl/blob/main/verl/utils/flops_counter.py
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import time
from typing import TYPE_CHECKING

import torch
from loguru import logger
from transformers import PretrainedConfig

if TYPE_CHECKING:
    from vgo.data.processor.naive_collect import PackData
    from vgo.train_engines.naive_policy import DiTInputOutput


def is_torch_npu_available(check_device=True) -> bool:
    """Check if Ascend NPU is available for PyTorch operations.

    Attempts to detect NPU availability by checking for the torch.npu module
    and its is_available() function.

    Args:
        check_device : only check torch_npu package or strictly check if NPU device is available

    Returns:
        bool: True if NPU is available, False otherwise.
    """
    try:
        if not hasattr(torch, "npu"):
            return False

        if check_device:
            return torch.npu.is_available()
        else:
            return True
    except ImportError:
        return False


is_cuda_available = torch.cuda.is_available()
is_npu_available = is_torch_npu_available()


def get_device_name() -> str:
    """Get the device type string based on available accelerators.

    Detects the available accelerator and returns the corresponding PyTorch
    device type string. Currently supports CUDA, Ascend NPU, and CPU.

    Returns:
        str: Device type string ('cuda', 'npu', or 'cpu').
    """
    if is_cuda_available:
        device = "cuda"
    elif is_npu_available:
        device = "npu"
    else:
        device = "cpu"
    return device


def get_torch_device():
    """Get the PyTorch device module for the current accelerator.

    Returns the torch device namespace (e.g., torch.cuda, torch.npu) based on
    the detected accelerator type. Falls back to torch.cuda if the namespace
    is not found.

    Returns:
        module: The PyTorch device module (torch.cuda, torch.npu, etc.).
    """
    device_name = get_device_name()
    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load torch.cuda.")
        return torch.cuda


def get_current_torch_device() -> torch.device:
    """Get the current torch.device for the active accelerator."""
    device_name = get_device_name()
    if device_name == "cpu":
        return torch.device("cpu")

    device_module = get_torch_device()
    current_device = getattr(device_module, "current_device", None)
    if callable(current_device):
        try:
            return torch.device(device_name, current_device())
        except Exception as e:
            logger.debug(f"Failed to query current {device_name} device index: {e}")

    return torch.device(device_name)


def synchronize_torch_device(device_module=None) -> None:
    """Synchronize the current accelerator if the backend exposes a sync API."""
    if device_module is None:
        device_module = get_torch_device()

    synchronize = getattr(device_module, "synchronize", None)
    if callable(synchronize):
        try:
            synchronize()
        except Exception as e:
            logger.debug(f"Failed to synchronize device module {device_module}: {e}")


_DEVICE_FLOPS = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12,
}


def get_device_flops(unit="T", device_name=None):
    """Get the theoretical FLOPS (Floating Point Operations Per Second) capacity of the current device.

    Args:
        unit (str): The unit to return the FLOPS in. Supported values are:
            "B" - Billion (1e9)
            "K" - Thousand (1e3)
            "M" - Million (1e6)
            "G" - Giga (1e9)
            "T" - Tera (1e12, default)
            "P" - Peta (1e15)

    Returns:
        float: The theoretical FLOPS capacity of the current device in the specified unit.
        Returns float('inf') for unknown GPU types.
    """

    def unit_convert(number, level):
        units = ["B", "K", "M", "G", "T", "P"]
        if number <= 0:
            return number
        ptr = 0
        while ptr < len(units) and units[ptr] != level:
            number /= 1000
            ptr += 1
        return number

    # pass device_name is for testing purpose only
    if device_name is None:
        device = get_torch_device()
        device_name = "CPU" if device == torch.cpu else get_torch_device().get_device_name()  # type: ignore

    flops = float("inf")  # INF flops for unkown gpu type

    for key, value in sorted(_DEVICE_FLOPS.items(), reverse=True):
        if key in device_name:
            flops = value
            break
    flops_unit = unit_convert(flops, unit)
    return flops_unit


def _estimate_qwen2_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Qwen2/LLama use SwiGelu, gate, having up and down linear layer in mlp
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    intermediate_size = config.text_config.intermediate_size

    head_dim = hidden_size // num_attention_heads
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # qwen3_vl uses deepstack to merge visual embeds and text embeds, but it has no tensor operation.

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens")
    vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config) if images_seqlens is not None else 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vl_moe_flops(config, tokens_sum, batch_seqlens, delta_time, **kargs):
    # qwen3_vl uses text_config and vision_config to distinguish configs of different parts.
    hidden_size = config.text_config.hidden_size
    vocab_size = config.text_config.vocab_size
    num_hidden_layers = config.text_config.num_hidden_layers
    num_key_value_heads = config.text_config.num_key_value_heads
    num_attention_heads = config.text_config.num_attention_heads
    moe_intermediate_size = config.text_config.moe_intermediate_size
    moe_num_expert = config.text_config.num_experts
    moe_topk = config.text_config.num_experts_per_tok

    head_dim = getattr(
        config.text_config, "head_dim", config.text_config.hidden_size // config.text_config.num_attention_heads
    )
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has gate_proj, up_proj and down_proj using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk) * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers) + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # vit flops
    images_seqlens = kargs.get("images_seqlens")
    vit_flops = _estimate_qwen3_vit_flop(images_seqlens, config.vision_config) if images_seqlens is not None else 0

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops + vit_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_qwen3_vit_flop(images_seqlens, config):
    """
    Estimate the FLOPS of the vision encoder for Qwen3-VL
    """

    if config is None:
        return 0
    tokens_sum = sum(images_seqlens)

    num_heads = config.num_heads
    depth = config.depth

    dim = config.hidden_size
    mlp_hidden_dim = config.intermediate_size
    out_hidden_size = config.out_hidden_size

    spatial_merge_size = config.spatial_merge_size

    head_dim = dim // num_heads

    # every vision token's patch_embed comes from a conv of (C, T, H, W) -> (dim,)
    patch_embed_N = dim * config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    # Qwen3 VL vision mlp does not use GLU, thus 2.
    mlp_N = dim * mlp_hidden_dim * 2
    attn_linear_N = dim * (4 * dim)  # qkv and output proj
    merger_N = (out_hidden_size + (dim * (spatial_merge_size**2))) * (dim * (spatial_merge_size**2))

    # Qwen3 VL uses deep stack, one merger for every deepstack layer
    deepstack_merger_N = merger_N * len(getattr(config, "deepstack_visual_indexes", []))
    # non-attn all_layer parm
    dense_N = patch_embed_N + (mlp_N + attn_linear_N) * depth + deepstack_merger_N + merger_N

    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # In Qwen3 VL, full attention is used in all vision layers.
    full_attn_layer_num = depth

    # full attn layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in images_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 12 * seqlen_square_sum * head_dim * num_heads * full_attn_layer_num

    vit_flops = dense_N_flops + attn_qkv_flops

    return vit_flops


def _estimate_deepseek_v3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    moe_intermediate_size = config.moe_intermediate_size
    num_hidden_layers = config.num_hidden_layers
    first_k_dense_replace = config.first_k_dense_replace
    num_query_heads = config.num_attention_heads
    moe_num_expert = config.n_routed_experts

    moe_topk = config.num_experts_per_tok
    share_expert_num = config.n_shared_experts

    # non-attn per layer parm
    moe_gata_N = hidden_size * moe_num_expert
    # moe has fc1_1, fc1_2 and fc2 using SwiGLU in ExpertMlp layer & shared experts
    moe_expertmlp_N = hidden_size * moe_intermediate_size * (moe_topk + share_expert_num) * 3
    # MLA attn
    attn_linear_N = 0
    q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    if config.q_lora_rank is None:
        attn_linear_N += hidden_size * num_query_heads * q_head_dim
    else:
        attn_linear_N += hidden_size * config.q_lora_rank
        attn_linear_N += num_query_heads * q_head_dim * config.q_lora_rank

    attn_linear_N += hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
    attn_linear_N += num_query_heads * (q_head_dim - config.qk_rope_head_dim + config.v_head_dim) * config.kv_lora_rank
    attn_linear_N += num_query_heads * config.v_head_dim * hidden_size
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    moe_N = (
        (moe_gata_N + moe_expertmlp_N + attn_linear_N) * (num_hidden_layers - first_k_dense_replace)
        + (hidden_size * config.intermediate_size * 3 + attn_linear_N) * first_k_dense_replace
        + emd_and_lm_head_N
    )
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * moe_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen * num_hidden_layers

    # Core attention FLOPS for MLA with causal mask:
    # Q @ K^T: 3 * 2 * seq^2 * q_head_dim * num_heads / 2 (causal)
    # attn @ V: 3 * 2 * seq^2 * v_head_dim * num_heads / 2 (causal)
    attn_qkv_flops = 3 * seqlen_square_sum * (q_head_dim + config.v_head_dim) * num_query_heads
    # all_layer & all_token fwd & bwk flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12

    return flops_achieved


def _estimate_qwen2_moe_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    moe_intermediate_size = config.moe_intermediate_size
    moe_topk = config.num_experts_per_tok
    num_experts = config.num_experts

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # gate + moe export
    moe_mlp_N = hidden_size * moe_topk * moe_intermediate_size * 3 + hidden_size * num_experts
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gemma3_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # non-attn per layer parm
    # Gemma3 uses GeGLU (gelu_pytorch_tanh), having 3 matrices in MLP (inherited from Gemma2MLP)
    mlp_N = hidden_size * intermediate_size * 3
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer parm
    dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    # Gemma3 alternates between full and sliding window attention based on layer_types
    seqlen_square_sum = 0

    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 1024)  # default 1024
    # default pattern: every 6th layer is full
    sliding_window_pattern = getattr(config, "sliding_window_pattern", 6)

    # If layer_types is not provided, generate it based on sliding_window_pattern
    if layer_types is None and sliding_window is not None and sliding_window_pattern is not None:
        layer_types = [
            "sliding_attention" if bool((i + 1) % sliding_window_pattern) else "full_attention"
            for i in range(num_hidden_layers)
        ]

    if layer_types:
        # Calculate attention flops per layer based on attention type
        for layer_idx in range(num_hidden_layers):
            is_sliding = False
            if layer_types and layer_idx < len(layer_types):
                is_sliding = layer_types[layer_idx] == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # If no layer_types config, assume all layers use full attention
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_apertus_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size

    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # Apertus MLP with XIELU activation uses only 2 linear layers (up_proj, down_proj)
    # No gate_proj for XIELU, unlike SwiGLU which has 3 layers
    mlp_N = hidden_size * intermediate_size * 2
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)

    # ApertusConfig has qk_norm defaulting to True.
    # This adds params for q_norm (on H) and k_norm (on num_kv_heads * head_dim)
    qk_norm_params_per_layer = hidden_size + num_key_value_heads * head_dim  # q_norm + k_norm

    emd_and_lm_head_N = vocab_size * hidden_size * 2
    # non-attn all_layer params
    dense_N = (mlp_N + attn_linear_N + qk_norm_params_per_layer) * num_hidden_layers + emd_and_lm_head_N
    # non-attn all_layer & all_token fwd & bwd flops
    dense_N_flops = 6 * dense_N * tokens_sum

    # attn all_layer & all_token fwd & bwd flops
    seqlen_square_sum = 0
    for seqlen in batch_seqlens:
        seqlen_square_sum += seqlen * seqlen
    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads * num_hidden_layers

    # all_layer & all_token fwd & bwd flops
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_gpt_oss_flops(config, tokens_sum, batch_seqlens, delta_time):
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads

    # MoE params
    moe_intermediate_size = config.intermediate_size
    num_experts = config.num_local_experts
    num_experts_per_tok = config.num_experts_per_tok
    mlp_matrices = 3

    # Head dim
    head_dim = getattr(config, "head_dim", hidden_size // num_attention_heads)
    q_size = num_attention_heads * head_dim
    k_size = num_key_value_heads * head_dim
    v_size = num_key_value_heads * head_dim

    # 1. Attention Block (GQA)
    attn_linear_N = hidden_size * (q_size + k_size + v_size + num_attention_heads * head_dim)
    # 2. MLP / MoE Block
    # Gate network
    moe_gate_N = hidden_size * num_experts
    # Expert forward calculation, Active parameters: mlp_matrices * H * I * num_experts_per_tok
    moe_expert_N = hidden_size * moe_intermediate_size * mlp_matrices * num_experts_per_tok

    moe_mlp_N = moe_gate_N + moe_expert_N

    emd_and_lm_head_N = vocab_size * hidden_size * 2

    # Total non-attn params per layer * layers + embeddings
    # (moe_mlp_N + attn_linear_N) * layers
    dense_N = (moe_mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

    # FLOPs for dense part (fwd + bwd = 6 * N)
    dense_N_flops = 6 * dense_N * tokens_sum

    # 3. Attention Matrix FLOPs
    seqlen_square_sum = 0

    # Handle sliding window attention
    layer_types = getattr(config, "layer_types", None)
    sliding_window = getattr(config, "sliding_window", 128)

    if layer_types:
        for layer_type in layer_types:
            is_sliding = layer_type == "sliding_attention"

            for seqlen in batch_seqlens:
                if is_sliding and sliding_window:
                    # Sliding window limits each token to attend to at most window_size tokens
                    effective_seqlen = min(seqlen, sliding_window)
                    seqlen_square_sum += seqlen * effective_seqlen
                else:
                    # Full attention
                    seqlen_square_sum += seqlen * seqlen
    else:
        # Default to full attention for all layers
        for seqlen in batch_seqlens:
            seqlen_square_sum += seqlen * seqlen
        seqlen_square_sum *= num_hidden_layers

    attn_qkv_flops = 6 * seqlen_square_sum * head_dim * num_attention_heads

    # Total FLOPs
    flops_all_token = dense_N_flops + attn_qkv_flops
    flops_achieved = flops_all_token * (1.0 / delta_time) / 1e12
    return flops_achieved


def _estimate_unknown_flops(config, tokens_sum, batch_seqlens, delta_time):
    return 0


ESTIMATE_FUNC = {
    "qwen2": _estimate_qwen2_flops,
    "llama": _estimate_qwen2_flops,
    "qwen2_moe": _estimate_qwen2_moe_flops,
    "qwen2_vl": _estimate_qwen2_flops,
    "qwen2_5_vl": _estimate_qwen2_flops,
    "qwen3": _estimate_qwen2_flops,
    "qwen3_moe": _estimate_qwen2_moe_flops,
    "qwen3_vl": _estimate_qwen3_vl_flops,
    "qwen3_vl_moe": _estimate_qwen3_vl_moe_flops,
    "deepseek_v3": _estimate_deepseek_v3_flops,
    "minicpmv": _estimate_qwen2_flops,
    "minicpmo": _estimate_qwen2_flops,
    "mistral": _estimate_qwen2_flops,
    "gemma3_text": _estimate_gemma3_flops,
    "seed_oss": _estimate_qwen2_flops,
    "apertus": _estimate_apertus_flops,
    "glm4v": _estimate_qwen2_flops,
    "gpt_oss": _estimate_gpt_oss_flops,
    "mimo": _estimate_qwen2_flops,
}


class FlopsCounter:
    """
    Used to count mfu during training loop

    Example:
        flops_counter = FlopsCounter(config)
        flops_achieved, flops_promised = flops_counter.estimate_flops(tokens_list, delta_time)

    """

    def __init__(self, config: PretrainedConfig):
        VALID_CONFIG_TYPE = ESTIMATE_FUNC.keys()
        if config.model_type not in VALID_CONFIG_TYPE:
            print(
                f"Only support config type of {VALID_CONFIG_TYPE}, but got {config.model_type}. MFU will always be "
                f"zero."
            )

        self.config = config

    # TODO: actually we can make this a static method
    def estimate_flops(self, batch_seqlens, delta_time, **kargs):
        """
        Estimate the FLOPS based on the number of valid tokens in the current batch and the time taken.

        Args:
            batch_seqlens (List[int]): A list where each element represents the number of valid tokens in the
                current batch.
            delta_time (float): The time taken to process the batch, in seconds.

        Returns:
            estimated_flops (float): The estimated FLOPS based on the input tokens and time.
            promised_flops (float): The expected FLOPS of the current device.
        """
        tokens_sum = sum(batch_seqlens)
        func = ESTIMATE_FUNC.get(self.config.model_type, _estimate_unknown_flops)
        sig = inspect.signature(func)
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time, **kargs)
        else:
            estimated_flops = func(self.config, tokens_sum, batch_seqlens, delta_time)
        promised_flops = get_device_flops()
        return estimated_flops, promised_flops


# ==================== VAE and DiT FLOPs Estimation ====================


def estimate_vae_flops(  # noqa: C901
    image_sizes: list[tuple[int, int]],
    base_dim: int = 96,
    z_dim: int = 16,
    dim_mult: list[int] | None = None,
    num_res_blocks: int = 2,
    temporal_downsample: list[bool] | None = None,
    is_training: bool = True,
    encode_only: bool = True,
) -> float:
    """
    Estimate the FLOPs for QwenImageVAE encoder (and optionally decoder).

    Args:
        image_sizes: List of (height, width) tuples for each image in the batch.
        base_dim: Base channel dimension (default: 96).
        z_dim: Latent space dimension (default: 16).
        dim_mult: Channel multipliers for each stage (default: [1, 2, 4, 4]).
        num_res_blocks: Number of residual blocks per stage (default: 2).
        temporal_downsample: Temporal downsample flags (default: [False, True, True]).
        is_training: If True, multiply by 6 for fwd+bwd; else multiply by 2 for fwd only.
        encode_only: If True, only estimate encoder FLOPs; else include decoder.

    Returns:
        Total FLOPs as a float.
    """
    if dim_mult is None:
        dim_mult = [1, 2, 4, 4]
    if temporal_downsample is None:
        temporal_downsample = [False, True, True]

    flop_factor = 6 if is_training else 2

    total_flops = 0.0

    for height, width in image_sizes:
        # VAE processes images as 5D tensors: (B, C, T, H, W) with T=1 for images
        # Spatial compression ratio is 8 (3 stages of 2x downsample)
        # Each Conv3d with kernel (kt, kh, kw) has FLOPs = 2 * kt * kh * kw * C_in * C_out * T * H * W

        # ===== Encoder =====
        dims = [base_dim * m for m in [1] + dim_mult]  # [96, 96, 192, 384, 384]
        h, w = height, width
        t = 1  # single frame for images

        # conv_in: 3 -> dims[0], kernel 3x3x3
        flops_conv_in = flop_factor * 3 * 3 * 3 * 3 * dims[0] * t * h * w

        encoder_flops = flops_conv_in

        # down_blocks
        for i in range(len(dim_mult)):
            in_dim = dims[i]
            out_dim = dims[i + 1]

            # num_res_blocks ResidualBlocks
            for j in range(num_res_blocks):
                block_in = in_dim if j == 0 else out_dim
                # conv1: 3x3x3
                encoder_flops += flop_factor * 3 * 3 * 3 * block_in * out_dim * t * h * w
                # conv2: 3x3x3
                encoder_flops += flop_factor * 3 * 3 * 3 * out_dim * out_dim * t * h * w
                # shortcut if dims differ
                if block_in != out_dim:
                    encoder_flops += flop_factor * 1 * 1 * 1 * block_in * out_dim * t * h * w

            # Resample (downsample)
            if i != len(dim_mult) - 1:
                # Conv2d for spatial downsample: out_dim -> out_dim, kernel 3x3, stride 2
                encoder_flops += flop_factor * 3 * 3 * out_dim * out_dim * t * h * w
                h = (h + 1) // 2
                w = (w + 1) // 2
                # Temporal downsample (Conv3d 3x1x1) if enabled
                if temporal_downsample[i]:
                    encoder_flops += flop_factor * 3 * 1 * 1 * out_dim * out_dim * t * h * w
                    t = (t + 1) // 2

        # mid_block: 2 ResidualBlocks + 1 AttentionBlock
        mid_dim = dims[-1]  # 384
        for _ in range(2):
            encoder_flops += flop_factor * 3 * 3 * 3 * mid_dim * mid_dim * t * h * w
            encoder_flops += flop_factor * 3 * 3 * 3 * mid_dim * mid_dim * t * h * w

        # AttentionBlock: to_qkv (1x1) + proj (1x1) + attention
        encoder_flops += flop_factor * 1 * 1 * mid_dim * mid_dim * 3 * t * h * w  # to_qkv
        encoder_flops += flop_factor * 1 * 1 * mid_dim * mid_dim * t * h * w  # proj
        # Attention: 2 * seq^2 * dim (Q@K^T + attn@V)
        seq_len = h * w
        encoder_flops += flop_factor * 2 * seq_len * seq_len * mid_dim * t

        # conv_out: mid_dim -> z_dim*2, kernel 3x3x3
        encoder_flops += flop_factor * 3 * 3 * 3 * mid_dim * (z_dim * 2) * t * h * w

        # quant_conv: z_dim*2 -> z_dim*2, kernel 1x1x1
        encoder_flops += flop_factor * 1 * 1 * 1 * (z_dim * 2) * (z_dim * 2) * t * h * w

        total_flops += encoder_flops

        # ===== Decoder (if needed) =====
        if not encode_only:
            decoder_flops = 0.0
            # Latent size after encoding
            latent_h, latent_w = h, w
            latent_t = t

            # post_quant_conv: z_dim -> z_dim, kernel 1x1x1
            decoder_flops += flop_factor * 1 * 1 * 1 * z_dim * z_dim * latent_t * latent_h * latent_w

            # conv_in: z_dim -> dims[-1], kernel 3x3x3
            decoder_flops += flop_factor * 3 * 3 * 3 * z_dim * dims[-1] * latent_t * latent_h * latent_w

            # mid_block (same as encoder)
            for _ in range(2):
                decoder_flops += flop_factor * 3 * 3 * 3 * mid_dim * mid_dim * latent_t * latent_h * latent_w
                decoder_flops += flop_factor * 3 * 3 * 3 * mid_dim * mid_dim * latent_t * latent_h * latent_w
            # AttentionBlock
            decoder_flops += flop_factor * 1 * 1 * mid_dim * mid_dim * 3 * latent_t * latent_h * latent_w
            decoder_flops += flop_factor * 1 * 1 * mid_dim * mid_dim * latent_t * latent_h * latent_w
            seq_len = latent_h * latent_w
            decoder_flops += flop_factor * 2 * seq_len * seq_len * mid_dim * latent_t

            # up_blocks (reverse order)
            rev_dims = [dims[-1]] + dims[-1:0:-1]  # [384, 384, 192, 96]
            temporal_upsample = temporal_downsample[::-1]
            h, w, t = latent_h, latent_w, latent_t

            for i in range(len(dim_mult)):
                in_dim = rev_dims[i] // 2 if i > 0 else rev_dims[i]
                out_dim = rev_dims[i + 1] if i + 1 < len(rev_dims) else base_dim

                # num_res_blocks + 1 ResidualBlocks
                for j in range(num_res_blocks + 1):
                    block_in = in_dim if j == 0 else out_dim
                    decoder_flops += flop_factor * 3 * 3 * 3 * block_in * out_dim * t * h * w
                    decoder_flops += flop_factor * 3 * 3 * 3 * out_dim * out_dim * t * h * w
                    if block_in != out_dim:
                        decoder_flops += flop_factor * 1 * 1 * 1 * block_in * out_dim * t * h * w

                # Upsample
                if i != len(dim_mult) - 1:
                    # Conv2d: out_dim -> out_dim//2, kernel 3x3
                    decoder_flops += flop_factor * 3 * 3 * out_dim * (out_dim // 2) * t * h * w
                    h *= 2
                    w *= 2
                    if temporal_upsample[i]:
                        # time_conv for temporal upsample
                        decoder_flops += flop_factor * 3 * 1 * 1 * out_dim * out_dim * 2 * t * (h // 2) * (w // 2)
                        t *= 2

            # conv_out: base_dim -> 3, kernel 3x3x3
            decoder_flops += flop_factor * 3 * 3 * 3 * base_dim * 3 * t * h * w

            total_flops += decoder_flops

    return total_flops


def estimate_dit_flops(
    img_seq_lens: list[int],
    txt_seq_lens: list[int],
    hidden_size: int,
    num_heads: int,
    depth: int,
    depth_single_blocks: int,
    mlp_ratio: float = 4.0,
    in_channels: int = 64,
    out_channels: int = 64,
    context_in_dim: int = 4096,
    adaln_dim: int | None = None,
    timestep_mlp_ratio: int = 1,
    is_training: bool = True,
) -> float:
    """
    Estimate the FLOPs for VarLenDiT model.

    Args:
        img_seq_lens: List of image sequence lengths for each sample.
        txt_seq_lens: List of text sequence lengths for each sample.
        hidden_size: Hidden dimension of the transformer.
        num_heads: Number of attention heads.
        depth: Number of double stream blocks.
        depth_single_blocks: Number of single stream blocks.
        mlp_ratio: MLP expansion ratio (default: 4.0).
        in_channels: Input channels (default: 64).
        out_channels: Output channels (default: 64).
        context_in_dim: Text encoder output dimension (default: 4096).
        adaln_dim: AdaLN dimension (default: hidden_size).
        timestep_mlp_ratio: Timestep MLP ratio (default: 1).
        is_training: If True, multiply by 6 for fwd+bwd; else multiply by 2 for fwd only.

    Returns:
        Total FLOPs as a float.
    """
    if adaln_dim is None:
        adaln_dim = hidden_size

    flop_factor = 6 if is_training else 2
    mlp_hidden_dim = int(hidden_size * mlp_ratio)
    head_dim = hidden_size // num_heads

    total_img_tokens = sum(img_seq_lens)
    total_txt_tokens = sum(txt_seq_lens)
    total_tokens = total_img_tokens + total_txt_tokens
    batch_size = len(img_seq_lens)

    total_flops = 0.0

    # ===== Input Embeddings =====
    # img_in: Linear(in_channels, hidden_size)
    img_in_flops = flop_factor * in_channels * hidden_size * total_img_tokens

    # txt_in: Linear(context_in_dim, hidden_size)
    txt_in_flops = flop_factor * context_in_dim * hidden_size * total_txt_tokens

    # time_in: MLPEmbedder (256 -> adaln_dim * timestep_mlp_ratio -> adaln_dim)
    time_in_flops = (
        flop_factor * (256 * adaln_dim * timestep_mlp_ratio + adaln_dim * timestep_mlp_ratio * adaln_dim) * batch_size
    )

    total_flops += img_in_flops + txt_in_flops + time_in_flops

    # ===== Double Stream Blocks =====
    for _ in range(depth):
        # img_mod: Linear(adaln_dim, hidden_size * 6)
        img_mod_flops = flop_factor * adaln_dim * hidden_size * 6 * batch_size
        # txt_mod: Linear(adaln_dim, hidden_size * 6)
        txt_mod_flops = flop_factor * adaln_dim * hidden_size * 6 * batch_size

        # AdaLN scale_shift and gate element-wise operations
        # scale_shift: (1 + scale) * x + shift -> 3 ops per element
        # gate: gate * x -> 1 op per element
        # img: 2x scale_shift (before attn, before mlp) + 2x gate (after attn, after mlp)
        img_adaln_flops = flop_factor * (3 + 1 + 3 + 1) * hidden_size * total_img_tokens
        # txt: 2x scale_shift + 2x gate
        txt_adaln_flops = flop_factor * (3 + 1 + 3 + 1) * hidden_size * total_txt_tokens

        # img_attn.qkv: Linear(hidden_size, hidden_size * 3)
        img_qkv_flops = flop_factor * hidden_size * hidden_size * 3 * total_img_tokens
        # img_attn.proj: Linear(hidden_size, hidden_size)
        img_proj_flops = flop_factor * hidden_size * hidden_size * total_img_tokens

        # txt_attn.qkv: Linear(hidden_size, hidden_size * 3)
        txt_qkv_flops = flop_factor * hidden_size * hidden_size * 3 * total_txt_tokens
        # txt_attn.proj: Linear(hidden_size, hidden_size)
        txt_proj_flops = flop_factor * hidden_size * hidden_size * total_txt_tokens

        # Joint attention: Q @ K^T and attn @ V
        # Sequence length is img + txt for each sample
        attn_flops = 0.0
        for img_len, txt_len in zip(img_seq_lens, txt_seq_lens):
            seq_len = img_len + txt_len
            # Q @ K^T: 2 * seq^2 * head_dim * num_heads
            # attn @ V: 2 * seq^2 * head_dim * num_heads
            attn_flops += flop_factor * 2 * seq_len * seq_len * head_dim * num_heads

        # RoPE: q * cos + rotate_half(q) * sin -> 3 ops per element for Q and K
        # Applied to concatenated img+txt sequence
        rope_flops = flop_factor * 3 * 2 * hidden_size * total_tokens  # 3 ops * 2 (Q,K) * hidden_size * tokens

        # img_mlp: Linear(hidden_size, mlp_hidden_dim) + Linear(mlp_hidden_dim, hidden_size)
        img_mlp_flops = flop_factor * (hidden_size * mlp_hidden_dim + mlp_hidden_dim * hidden_size) * total_img_tokens

        # txt_mlp: Linear(hidden_size, mlp_hidden_dim) + Linear(mlp_hidden_dim, hidden_size)
        txt_mlp_flops = flop_factor * (hidden_size * mlp_hidden_dim + mlp_hidden_dim * hidden_size) * total_txt_tokens

        block_flops = (
            img_mod_flops
            + txt_mod_flops
            + img_adaln_flops
            + txt_adaln_flops
            + img_qkv_flops
            + img_proj_flops
            + txt_qkv_flops
            + txt_proj_flops
            + rope_flops
            + attn_flops
            + img_mlp_flops
            + txt_mlp_flops
        )
        total_flops += block_flops

    # ===== Single Stream Blocks =====
    for _ in range(depth_single_blocks):
        # modulation: Linear(adaln_dim, hidden_size * 3)
        mod_flops = flop_factor * adaln_dim * hidden_size * 3 * batch_size

        # AdaLN scale_shift and gate element-wise operations
        # 1x scale_shift (before linear1) + 1x gate (after linear2)
        adaln_flops = flop_factor * (3 + 1) * hidden_size * total_tokens

        # linear1: Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim)
        linear1_flops = flop_factor * hidden_size * (hidden_size * 3 + mlp_hidden_dim) * total_tokens

        # linear2: Linear(hidden_size + mlp_hidden_dim, hidden_size)
        linear2_flops = flop_factor * (hidden_size + mlp_hidden_dim) * hidden_size * total_tokens

        # Attention
        attn_flops = 0.0
        for img_len, txt_len in zip(img_seq_lens, txt_seq_lens):
            seq_len = img_len + txt_len
            attn_flops += flop_factor * 2 * seq_len * seq_len * head_dim * num_heads

        # RoPE: q * cos + rotate_half(q) * sin -> 3 ops per element for Q and K
        rope_flops = flop_factor * 3 * 2 * hidden_size * total_tokens

        block_flops = mod_flops + adaln_flops + linear1_flops + linear2_flops + rope_flops + attn_flops
        total_flops += block_flops

    # ===== Final Layer =====
    # adaLN_modulation: Linear(adaln_dim, hidden_size * 2)
    final_mod_flops = flop_factor * adaln_dim * hidden_size * 2 * batch_size
    # AdaLN scale_shift: (1 + scale) * x + shift -> 3 ops per element
    final_adaln_flops = flop_factor * 3 * hidden_size * total_img_tokens
    # linear: Linear(hidden_size, out_channels)
    final_linear_flops = flop_factor * hidden_size * out_channels * total_img_tokens

    total_flops += final_mod_flops + final_adaln_flops + final_linear_flops

    return total_flops


def estimate_vae_dit_flops(
    image_sizes: list[tuple[int, int]],
    txt_seq_lens: list[int],
    dit_params: dict,
    vae_params: dict | None = None,
    ref_image_sizes: list[tuple[int, int]] | None = None,
    vae_is_training: bool = False,
    dit_is_training: bool = True,
    include_vae: bool = True,
    vae_encode_only: bool = True,
) -> tuple[float, float, float]:
    """
    Estimate the combined FLOPs for VAE and DiT in a diffusion training step.

    Args:
        image_sizes: List of (height, width) tuples for each target image.
        txt_seq_lens: List of text sequence lengths.
        dit_params: Dictionary of DiT parameters (hidden_size, num_heads, depth, etc.).
        vae_params: Dictionary of VAE parameters (optional, uses defaults if None).
        ref_image_sizes: List of (height, width) tuples for each reference image (optional).
        vae_is_training: If True, use training FLOPs (6x) for VAE; else inference (2x).
        dit_is_training: If True, use training FLOPs (6x) for DiT; else inference (2x).
        include_vae: If True, include VAE FLOPs in the estimate.
        vae_encode_only: If True, only estimate VAE encoder FLOPs.

    Returns:
        Tuple of (vae_flops, dit_flops, total_flops).
    """
    ref_image_sizes = ref_image_sizes or []

    vae_flops = 0.0
    if include_vae:
        vae_cfg = vae_params or {}
        # VAE encodes both target images and reference images
        all_image_sizes = image_sizes + ref_image_sizes
        vae_flops = estimate_vae_flops(
            image_sizes=all_image_sizes,
            base_dim=vae_cfg.get("base_dim", 96),
            z_dim=vae_cfg.get("z_dim", 16),
            dim_mult=vae_cfg.get("dim_mult", [1, 2, 4, 4]),
            num_res_blocks=vae_cfg.get("num_res_blocks", 2),
            temporal_downsample=vae_cfg.get("temporal_downsample", [False, True, True]),
            is_training=vae_is_training,
            encode_only=vae_encode_only,
        )

    # Calculate image token sequence lengths from image sizes
    # Assuming spatial compression ratio of 8 (VAE) and patch size of 2 (DiT)
    # Token size = (H / 8 / 2) * (W / 8 / 2) = H * W / 256
    img_seq_lens = [(h // 16) * (w // 16) for h, w in image_sizes]

    dit_flops = estimate_dit_flops(
        img_seq_lens=img_seq_lens,
        txt_seq_lens=txt_seq_lens,
        hidden_size=dit_params["hidden_size"],
        num_heads=dit_params["num_heads"],
        depth=dit_params["depth"],
        depth_single_blocks=dit_params["depth_single_blocks"],
        mlp_ratio=dit_params.get("mlp_ratio", 4.0),
        in_channels=dit_params.get("in_channels", 64),
        out_channels=dit_params.get("out_channels", 64),
        context_in_dim=dit_params.get("context_in_dim", 4096),
        adaln_dim=dit_params.get("adaln_dim"),
        timestep_mlp_ratio=dit_params.get("timestep_mlp_ratio", 1),
        is_training=dit_is_training,
    )

    return vae_flops, dit_flops, vae_flops + dit_flops


# ==================== Text Encoder FLOPs using existing functions ====================

ESTIMATE_TEXT_ENCODER_FUNC = {
    "Qwen2ForCausalLM": _estimate_qwen2_flops,
    "Qwen2_5_VLForConditionalGeneration": _estimate_qwen3_vl_flops,  # Qwen2.5-VL uses similar structure
    "Qwen3VLForConditionalGeneration": _estimate_qwen3_vl_flops,
    "Qwen3VLMoEForConditionalGeneration": _estimate_qwen3_vl_moe_flops,
}


# Prefix token drop indices for different task types (tokens dropped before passing to DiT)
TEXT_ENCODER_PREFIX_DROP_INDICES = {
    "t2i": 34,
    "edit": 64,
    "customize": 0,
}


def _resize_for_qwen3vl(height: int, width: int, target_area: int = 384 * 384) -> tuple[int, int]:
    """
    Calculate resized dimensions for Qwen3VL style resize.
    Maintains aspect ratio while targeting a specific area.
    """
    import math

    aspect_ratio = width / height
    new_width = int(math.sqrt(target_area * aspect_ratio))
    new_height = int(math.sqrt(target_area / aspect_ratio))
    return new_height, new_width


def _resize_for_qwen25vl(
    height: int,
    width: int,
    min_tokens: int = 256,
    max_tokens: int = 1280,
    image_factor: int = 28,
) -> tuple[int, int]:
    """
    Calculate resized dimensions for Qwen2.5VL style resize.
    Resize to be within min/max token area and aligned to image_factor.
    """
    import math

    min_area = min_tokens * image_factor * image_factor
    max_area = max_tokens * image_factor * image_factor
    origin_area = height * width

    target_width = round(width / image_factor) * image_factor
    target_height = round(height / image_factor) * image_factor

    if origin_area > max_area:
        scale = math.sqrt(origin_area / max_area)
        target_width = math.floor(width / scale / image_factor) * image_factor
        target_height = math.floor(height / scale / image_factor) * image_factor
    elif origin_area < min_area:
        scale = math.sqrt(min_area / origin_area)
        target_width = math.ceil(width * scale / image_factor) * image_factor
        target_height = math.ceil(height * scale / image_factor) * image_factor

    return target_height, target_width


def _calculate_vit_seq_len(
    height: int,
    width: int,
    patch_size: int = 14,
    merge_size: int = 1,
) -> int:
    """
    Calculate ViT sequence length from image dimensions.

    Args:
        height: Image height after resize.
        width: Image width after resize.
        patch_size: ViT patch size (typically 14).
        merge_size: Merge size for spatial pooling (Qwen2.5VL uses 2).

    Returns:
        Sequence length for ViT.
    """
    seq_len = (height // patch_size) * (width // patch_size)
    if merge_size > 1:
        seq_len = seq_len // (merge_size * merge_size)
    return seq_len


def estimate_text_encoder_flops_from_config(  # noqa: C901
    config: PretrainedConfig,
    txt_seq_lens: list[int],
    task_types: list[str] | None = None,
    ref_image_sizes: list[tuple[int, int]] | None = None,
    is_training: bool = False,
    vit_target_area: int | None = None,
    vit_min_tokens: int | None = None,
    vit_max_tokens: int | None = None,
) -> float:
    """
    Estimate text encoder FLOPs using the model config.

    Args:
        config: HuggingFace PretrainedConfig of the text encoder.
        txt_seq_lens: List of text sequence lengths (DiT input, with prefix dropped).
        task_types: List of task types for each sequence ("t2i", "edit", "customize").
                   Used to add back the dropped prefix tokens for accurate FLOPs calculation.
        ref_image_sizes: List of (height, width) tuples for reference images processed by ViT.
        is_training: If True, use training FLOPs (6x); else inference (2x).
        vit_target_area: Target area for Qwen3VL style resize (default: 384*384).
        vit_min_tokens: Min tokens for Qwen2.5VL style resize.
        vit_max_tokens: Max tokens for Qwen2.5VL style resize.

    Returns:
        Total FLOPs as a float.
    """
    model_type = config.architectures[0] if hasattr(config, "architectures") and config.architectures else None

    if model_type is None:
        logger.warning("Cannot determine model type from config, skipping text encoder FLOPs")
        return 0.0

    func = ESTIMATE_TEXT_ENCODER_FUNC.get(model_type)
    if func is None:
        logger.warning(f"No FLOPs estimator for model type: {model_type}, skipping text encoder FLOPs")
        return 0.0

    # Add back the dropped prefix tokens for each sequence based on task type
    if task_types is not None and len(task_types) == len(txt_seq_lens):
        actual_txt_seq_lens = [
            seq_len + TEXT_ENCODER_PREFIX_DROP_INDICES.get(task_type, 0)
            for seq_len, task_type in zip(txt_seq_lens, task_types)
        ]
    else:
        actual_txt_seq_lens = txt_seq_lens

    # Calculate ViT image sequence lengths for ref_images
    # Need to consider the resize operation before ViT
    images_seqlens = None
    if ref_image_sizes:
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            patch_size = getattr(vision_config, "patch_size", 14)

            # Determine resize strategy based on model type and parameters
            is_qwen25vl = model_type == "Qwen2_5_VLForConditionalGeneration"
            is_qwen3vl = model_type in ("Qwen3VLForConditionalGeneration", "Qwen3VLMoEForConditionalGeneration")

            images_seqlens = []
            for h, w in ref_image_sizes:
                if is_qwen25vl and vit_min_tokens is not None and vit_max_tokens is not None:
                    # Qwen2.5VL style resize with min/max tokens
                    resized_h, resized_w = _resize_for_qwen25vl(h, w, vit_min_tokens, vit_max_tokens)
                    # Qwen2.5VL uses merge_size=2
                    seq_len = _calculate_vit_seq_len(resized_h, resized_w, patch_size, merge_size=2)
                elif is_qwen3vl or vit_target_area is not None:
                    # Qwen3VL style resize to target area
                    target_area = vit_target_area if vit_target_area is not None else 384 * 384
                    resized_h, resized_w = _resize_for_qwen3vl(h, w, target_area)
                    seq_len = _calculate_vit_seq_len(resized_h, resized_w, patch_size, merge_size=1)
                else:
                    # Default: no resize, direct calculation
                    seq_len = _calculate_vit_seq_len(h, w, patch_size, merge_size=1)
                images_seqlens.append(seq_len)

    tokens_sum = sum(actual_txt_seq_lens)
    # Use delta_time=1.0 to get raw FLOPs, then adjust for training factor
    if images_seqlens is not None:
        flops_tflops = func(config, tokens_sum, actual_txt_seq_lens, delta_time=1.0, images_seqlens=images_seqlens)
    else:
        flops_tflops = func(config, tokens_sum, actual_txt_seq_lens, delta_time=1.0)
    flops = flops_tflops * 1e12  # Convert back to FLOPs

    # The existing functions assume training (6x), adjust if inference only
    if not is_training:
        flops = flops / 3  # 6x -> 2x

    return flops


class VGOFlopsCounter:
    """
    FLOPs counter for VGO (VAE + DiT + Text Encoder) training.

    Example:
        counter = VGOFlopsCounter(dit_params, vae_params, text_encoder_config)
        flops_achieved, flops_promised = counter.estimate_flops(
            image_sizes=[(512, 512), (768, 512)],
            txt_seq_lens=[128, 256],
            delta_time=1.5,
        )
    """

    def __init__(
        self,
        dit_params: dict,
        vae_params: dict | None = None,
        text_encoder_config: PretrainedConfig | None = None,
        include_vae: bool = True,
        include_text_encoder: bool = True,
        vae_encode_only: bool = True,
        vae_is_training: bool = False,
        dit_is_training: bool = True,
        text_encoder_is_training: bool = False,
        vit_target_area: int | None = None,
        vit_min_tokens: int | None = None,
        vit_max_tokens: int | None = None,
    ):
        """
        Initialize the VGO FLOPs counter.

        Args:
            dit_params: DiT model parameters dict with keys:
                - hidden_size, num_heads, depth, depth_single_blocks
                - Optional: mlp_ratio, in_channels, out_channels, context_in_dim, adaln_dim
            vae_params: VAE model parameters dict (optional).
            text_encoder_config: HuggingFace PretrainedConfig of text encoder (optional).
            include_vae: Whether to include VAE FLOPs.
            include_text_encoder: Whether to include text encoder FLOPs.
            vae_encode_only: If True, only count VAE encoder FLOPs.
            vae_is_training: If True, use training FLOPs (6x) for VAE; else inference (2x).
            dit_is_training: If True, use training FLOPs (6x) for DiT; else inference (2x).
            text_encoder_is_training: If True, use training FLOPs (6x) for text encoder; else inference (2x).
            vit_target_area: Target area for Qwen3VL style ViT resize (default: 384*384).
            vit_min_tokens: Min tokens for Qwen2.5VL style ViT resize.
            vit_max_tokens: Max tokens for Qwen2.5VL style ViT resize.
        """
        self.dit_params = dit_params
        self.vae_params = vae_params
        self.text_encoder_config = text_encoder_config
        self.include_vae = include_vae
        self.include_text_encoder = include_text_encoder
        self.vae_encode_only = vae_encode_only
        self.vae_is_training = vae_is_training
        self.dit_is_training = dit_is_training
        self.text_encoder_is_training = text_encoder_is_training
        self.vit_target_area = vit_target_area
        self.vit_min_tokens = vit_min_tokens
        self.vit_max_tokens = vit_max_tokens

    def estimate_flops(
        self,
        image_sizes: list[tuple[int, int]],
        txt_seq_lens: list[int],
        delta_time: float,
        ref_image_sizes: list[tuple[int, int]] | None = None,
        task_types: list[str] | None = None,
        vae_is_training: bool | None = None,
        dit_is_training: bool | None = None,
        text_encoder_is_training: bool | None = None,
    ) -> dict[str, float]:
        """
        Estimate the achieved and promised FLOPs.

        Args:
            image_sizes: List of (height, width) tuples for each target image.
            txt_seq_lens: List of text sequence lengths (DiT input).
            delta_time: Time taken for the step in seconds.
            ref_image_sizes: List of (height, width) tuples for each reference image.
            task_types: List of task types ("t2i", "edit", "customize") for each sequence.
            vae_is_training: Override VAE training mode (uses init value if None).
            dit_is_training: Override DiT training mode (uses init value if None).
            text_encoder_is_training: Override text encoder training mode (uses init value if None).

        Returns:
            Dict with keys:
                - vae_flops: VAE FLOPs (raw)
                - dit_flops: DiT FLOPs (raw)
                - text_encoder_flops: Text encoder FLOPs (raw)
                - total_flops: Total FLOPs (raw)
                - achieved_tflops: Achieved TFLOPs/s
                - promised_tflops: Promised TFLOPs/s (device peak)
                - elapsed_time_s: Elapsed time in seconds
        """
        _vae_is_training = vae_is_training if vae_is_training is not None else self.vae_is_training
        _dit_is_training = dit_is_training if dit_is_training is not None else self.dit_is_training
        _text_encoder_is_training = (
            text_encoder_is_training if text_encoder_is_training is not None else self.text_encoder_is_training
        )

        vae_flops, dit_flops, _ = estimate_vae_dit_flops(
            image_sizes=image_sizes,
            txt_seq_lens=txt_seq_lens,
            dit_params=self.dit_params,
            vae_params=self.vae_params,
            ref_image_sizes=ref_image_sizes,
            vae_is_training=_vae_is_training,
            dit_is_training=_dit_is_training,
            include_vae=self.include_vae,
            vae_encode_only=self.vae_encode_only,
        )

        # Text encoder FLOPs
        text_encoder_flops = 0.0
        if self.include_text_encoder and self.text_encoder_config is not None:
            text_encoder_flops = estimate_text_encoder_flops_from_config(
                config=self.text_encoder_config,
                txt_seq_lens=txt_seq_lens,
                task_types=task_types,
                ref_image_sizes=ref_image_sizes,
                is_training=_text_encoder_is_training,
                vit_target_area=self.vit_target_area,
                vit_min_tokens=self.vit_min_tokens,
                vit_max_tokens=self.vit_max_tokens,
            )

        total_flops = vae_flops + dit_flops + text_encoder_flops
        achieved_tflops = total_flops / delta_time / 1e12  # TFLOPs/s
        promised_tflops = get_device_flops()

        return {
            "vae_flops": vae_flops,
            "dit_flops": dit_flops,
            "text_encoder_flops": text_encoder_flops,
            "total_flops": total_flops,
            "achieved_tflops": achieved_tflops,
            "promised_tflops": promised_tflops,
            "elapsed_time_s": delta_time,
        }


class FlopsContext:
    """
    Context manager for FLOPs calculation.

    Example:
        with FlopsContext(flops_counter) as ctx:
            # Set batch data directly from pack_data and dit_input
            ctx.set_batch(pack_data, dit_input)

            # ... run forward/backward ...

        # After exiting, ctx.result contains the FLOPs info
        print(ctx.result)
        # {
        #     'vae_flops': ..., 'dit_flops': ..., 'text_encoder_flops': ...,
        #     'total_flops': ..., 'achieved_tflops': ..., 'promised_tflops': ...,
        #     'elapsed_time_s': ..., 'mfu': ...
        # }
    """

    def __init__(self, flops_counter: VGOFlopsCounter | None):
        self.flops_counter = flops_counter
        self.image_sizes: list[tuple[int, int]] = []
        self.ref_image_sizes: list[tuple[int, int]] = []
        self.txt_seq_lens: list[int] = []
        self.task_types: list[str] = []
        self.start_event = None
        self.end_event = None
        self.start_time_s: float | None = None
        self.device_module = None
        self.result: dict[str, float] = {}

    def set_batch(self, pack_data: "PackData", dit_input: "DiTInputOutput"):
        """
        Set batch data by extracting information from pack_data and dit_input.

        Args:
            pack_data: PackData object containing target_images, ref_images, and sequences.
            dit_input: DiTInputOutput object containing txt_lens.
        """
        # Extract target image sizes
        for img in pack_data.target_images:
            self.image_sizes.append((img.shape[-2], img.shape[-1]))

        # Extract reference image sizes
        for ref_img_list in pack_data.ref_images:
            for ref_img in ref_img_list:
                self.ref_image_sizes.append((ref_img.shape[-2], ref_img.shape[-1]))

        # Extract text sequence lengths
        self.txt_seq_lens.extend(dit_input.txt_lens)

        # Extract task types from sequences
        for seq in pack_data.sequences:
            self.task_types.append(seq.task_type)

    def __enter__(self):
        if self.flops_counter is not None:
            self.device_module = get_torch_device()
            event_cls = getattr(self.device_module, "Event", None)
            current_stream = getattr(self.device_module, "current_stream", None)

            if callable(event_cls) and callable(current_stream):
                try:
                    self.start_event = event_cls(enable_timing=True)
                    self.end_event = event_cls(enable_timing=True)
                    self.start_event.record(current_stream())
                except Exception as e:
                    logger.debug(f"Device timing event is unavailable on {get_device_name()}, fallback to perf_counter: {e}")
                    self.start_event = None
                    self.end_event = None

            if self.start_event is None or self.end_event is None:
                synchronize_torch_device(self.device_module)
                self.start_time_s = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time_s: float | None = None
        if self.flops_counter is not None:
            if self.start_event is not None and self.end_event is not None and self.device_module is not None:
                try:
                    current_stream = getattr(self.device_module, "current_stream", None)
                    if callable(current_stream):
                        self.end_event.record(current_stream())
                    synchronize_torch_device(self.device_module)
                    elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
                    elapsed_time_s = elapsed_time_ms / 1000.0
                except Exception as e:
                    logger.debug(f"Device event timing failed on {get_device_name()}, fallback to perf_counter: {e}")

            if elapsed_time_s is None and self.start_time_s is not None:
                synchronize_torch_device(self.device_module)
                elapsed_time_s = time.perf_counter() - self.start_time_s

            if len(self.image_sizes) > 0 and elapsed_time_s is not None and elapsed_time_s > 0:
                try:
                    flops_result = self.flops_counter.estimate_flops(
                        image_sizes=self.image_sizes,
                        txt_seq_lens=self.txt_seq_lens,
                        delta_time=elapsed_time_s,
                        ref_image_sizes=self.ref_image_sizes if self.ref_image_sizes else None,
                        task_types=self.task_types if self.task_types else None,
                    )
                    # Copy all results from estimate_flops
                    self.result = flops_result.copy()
                    # Add MFU calculation
                    promised = flops_result["promised_tflops"]
                    achieved = flops_result["achieved_tflops"]
                    self.result["mfu"] = achieved / promised if promised > 0 else 0.0
                except Exception as e:
                    logger.debug(f"FLOPs calculation failed: {e}")

        return False  # Don't suppress exceptions

    def get_log_dict(
        self, dp_group: "torch.distributed.ProcessGroup | None" = None, num_gpus_in_dp_group: int = 1
    ) -> dict[str, float]:
        """
        Get a dict of FLOPs metrics for logging, including cluster-wide MFU.

        Args:
            dp_group: Data parallel process group for cluster-wide MFU calculation.
                      If None or single GPU, only local metrics are returned.

        Returns:
            Dict with keys like "flops/vae_tflops", "flops/cluster_mfu", etc.
        """
        if not self.result:
            return {}

        log_dict: dict[str, float] = {}

        # Record individual FLOPs (local DP group)
        log_dict["flops/vae_tflops"] = self.result["vae_flops"] / 1e12
        log_dict["flops/dit_tflops"] = self.result["dit_flops"] / 1e12
        log_dict["flops/text_encoder_tflops"] = self.result["text_encoder_flops"] / 1e12
        log_dict["flops/total_tflops"] = self.result["total_flops"] / 1e12
        log_dict["flops/local_achieved_tflops_per_sec"] = self.result["achieved_tflops"] / num_gpus_in_dp_group
        log_dict["flops/promised_tflops_per_sec"] = self.result["promised_tflops"]
        log_dict["flops/local_mfu"] = self.result["mfu"] / num_gpus_in_dp_group

        # Calculate cluster-wide MFU
        num_dp_groups = 1
        if dp_group is not None and torch.distributed.is_initialized():
            num_dp_groups = torch.distributed.get_world_size(dp_group)

        if num_dp_groups > 1:
            device = get_current_torch_device()
            reduce_dtype = torch.float32 if device.type == "npu" else torch.float64
            # Create tensors for all_reduce
            local_total_flops = torch.tensor(self.result["total_flops"], device=device, dtype=reduce_dtype)
            local_elapsed_time = torch.tensor(self.result["elapsed_time_s"], device=device, dtype=reduce_dtype)

            # Sum total_flops across all DP groups
            torch.distributed.all_reduce(local_total_flops, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            # Get max elapsed time across all DP groups (accounts for bubble/wait time)
            torch.distributed.all_reduce(local_elapsed_time, op=torch.distributed.ReduceOp.MAX, group=dp_group)

            cluster_total_flops = local_total_flops.item()
            cluster_max_time = local_elapsed_time.item()
            promised_tflops = self.result["promised_tflops"]

            # Cluster MFU = total_flops / (max_time * num_gpus * promised_tflops_per_gpu * 1e12)
            # Cluster achieved TFLOPs/s = cluster_total_flops / cluster_max_time / 1e12
            # Cluster MFU = cluster_achieved / (num_dp_groups * promised_tflops)
            cluster_achieved_tflops = cluster_total_flops / cluster_max_time / 1e12
            cluster_mfu = (
                cluster_achieved_tflops / (num_dp_groups * promised_tflops * num_gpus_in_dp_group)
                if promised_tflops > 0
                else 0.0
            )

            log_dict["flops/cluster_total_tflops"] = cluster_total_flops / 1e12
            log_dict["flops/cluster_max_time_s"] = cluster_max_time
            log_dict["flops/cluster_achieved_tflops_per_sec"] = cluster_achieved_tflops
            log_dict["flops/cluster_mfu"] = cluster_mfu
        else:
            # Single DP group, cluster MFU = local MFU
            log_dict["flops/cluster_mfu"] = self.result["mfu"] / num_gpus_in_dp_group

        return log_dict

