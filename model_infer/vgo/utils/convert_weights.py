import os

import fire
import torch
from einops import rearrange
from loguru import logger
from safetensors.torch import save_file
from torch.distributed.checkpoint.format_utils import (
    FileSystemReader,
    _EmptyStateDictLoadPlanner,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from vgo.utils.extensions.experiment_registry import _compute_file_hash, register_weight_conversion


def merge_lora_weights(
    state_dict: dict[str, torch.Tensor],
    lora_alpha: int | None = None,
    lora_r: int = 16,
    target_module_patterns: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """
    Merge LoRA weights into base model weights.

    LoRA formula: W_merged = W_base + (lora_B @ lora_A) * scaling
    where scaling = lora_alpha / r (default: lora_alpha = r, so scaling = 1)

    Args:
        state_dict: State dict containing both base weights and LoRA weights.
                   LoRA weights are expected to have keys like:
                   - "{module}.lora_A.weight" and "{module}.lora_B.weight"
                   - or "{module}.lora_A.default.weight" and "{module}.lora_B.default.weight" (PEFT format)
        lora_alpha: LoRA alpha parameter. If None, defaults to 8.
        lora_r: LoRA rank parameter.
        target_module_patterns: List of patterns to match target modules.
                               If None, defaults to ["double_blocks", "single_blocks"].

    Returns:
        State dict with LoRA weights merged into base weights.
    """
    if lora_alpha is None:
        lora_alpha = 8
    scaling = lora_alpha / lora_r

    if target_module_patterns is None:
        target_module_patterns = ["double_blocks", "single_blocks"]

    # Find all LoRA A weights
    lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k and k.endswith(".weight")]  # noqa: SIM118

    merged_count = 0
    keys_to_remove = []

    for lora_a_key in lora_a_keys:
        # Determine the corresponding lora_B key
        lora_b_key = lora_a_key.replace("lora_A", "lora_B")

        if lora_b_key not in state_dict:
            logger.warning(f"Found lora_A but missing lora_B: {lora_a_key}")
            continue

        # Extract the base module name
        # Handle both formats:
        # - "module.lora_A.weight" -> "module.weight"
        # - "module.lora_A.default.weight" -> "module.weight"
        if ".lora_A.default." in lora_a_key:
            # PEFT format: "module.lora_A.default.weight"
            base_key = lora_a_key.replace(".lora_A.default.", ".base_layer.")
        else:
            # Simple format: "module.lora_A.weight"
            base_key = lora_a_key.replace(".lora_A.", ".base_layer.")

        target_base_key = base_key.replace(".base_layer", "")

        # Check if this module should be merged based on target patterns
        should_merge = any(pattern in base_key for pattern in target_module_patterns)

        if not should_merge:
            logger.debug(f"Skipping LoRA merge for {base_key} (not in target modules)")
            raise AssertionError(f"Unexpected LoRA weights: {base_key}")

        if base_key not in state_dict:
            logger.warning(f"Base weight not found for LoRA: {base_key}")
            continue

        # Get weights
        lora_a = state_dict[lora_a_key]  # shape: (r, in_features)
        lora_b = state_dict[lora_b_key]  # shape: (out_features, r)
        base_weight = state_dict[base_key]  # shape: (out_features, in_features)

        # Compute LoRA delta: lora_B @ lora_A
        # lora_B: (out_features, r), lora_A: (r, in_features)
        # Result: (out_features, in_features)
        lora_delta = lora_b.to(torch.float64) @ lora_a.to(torch.float64)

        # Merge: W_merged = W_base + lora_delta * scaling
        merged_weight = base_weight.to(torch.float64) + lora_delta.to(torch.float64) * scaling
        state_dict[target_base_key] = merged_weight.to(base_weight.dtype)

        # Mark LoRA keys for removal
        keys_to_remove.extend([lora_a_key, lora_b_key, base_key])
        merged_count += 1

        logger.debug(f"Merged LoRA into {base_key} (scaling={scaling})")

    # Remove LoRA keys from state_dict
    for key in keys_to_remove:
        del state_dict[key]

    # Also remove any other LoRA-related keys (like lora_embedding_A, lora_embedding_B, etc.)
    remaining_lora_keys = [k for k in state_dict.keys() if "lora_" in k.lower()]  # noqa: SIM118
    for key in remaining_lora_keys:
        logger.debug(f"Removing remaining LoRA key: {key}")
        del state_dict[key]

    state_dict = {k.replace("base_model.model.", "").replace(".base_layer", ""): v for k, v in state_dict.items()}

    logger.info(f"Merged {merged_count} LoRA adapters into base weights")

    return state_dict


def rearrange_weights(state_dict: dict[str, torch.Tensor], hidden_dim=3072, num_heads=24):
    mlp_hidden_dim = int(hidden_dim * 4.0)

    for k, v in state_dict.items():
        if "img_attn.qkv.weight" in k:
            state_dict[k] = rearrange(v, "(H K D) I -> (K H D) I", K=3, H=num_heads)
        if "img_attn.qkv.bias" in k:
            state_dict[k] = rearrange(v, "(H K D) -> (K H D)", K=3, H=num_heads)
        if "txt_attn.qkv.weight" in k:
            state_dict[k] = rearrange(v, "(H K D) I -> (K H D) I", K=3, H=num_heads)
        if "txt_attn.qkv.bias" in k:
            state_dict[k] = rearrange(v, "(H K D) -> (K H D)", K=3, H=num_heads)
        if "linear1.weight" in k:
            state_dict[k] = rearrange(
                v,
                "(H K D) I -> (K H D) I",
                K=mlp_hidden_dim // hidden_dim + 3,
                H=num_heads,
            )
        if "linear1.bias" in k:
            state_dict[k] = rearrange(
                v,
                "(H K D) -> (K H D)",
                K=mlp_hidden_dim // hidden_dim + 3,
                H=num_heads,
            )
        if "linear2.weight" in k:
            state_dict[k] = rearrange(
                v,
                "I (H M D) -> I (M H D)",
                M=mlp_hidden_dim // hidden_dim + 1,
                H=num_heads,
            )

    return state_dict


def main(
    dcp_checkpoint_dir,
    target_path,
    enable_rearrange_weights=True,
    merge_lora=False,
    lora_r=16,
    lora_alpha=None,
    register=True,
    source_exp_hash=None,
    source_run_hash=None,
):
    """
    Convert DCP checkpoint to safetensors format.

    Args:
        dcp_checkpoint_dir: Path to the DCP checkpoint directory.
        target_path: Path to save the converted safetensors file.
        enable_rearrange_weights: Whether to rearrange weights for compatibility.
        merge_lora: Whether to merge LoRA weights into base weights.
        lora_r: LoRA rank (default: 16, matching the config).
        lora_alpha: LoRA alpha. If None, defaults to lora_r.
        register: Whether to register the weight conversion (default: True).
        source_exp_hash: Source experiment hash. If None, will try to read from exp_hash.json in checkpoint dir.
        source_run_hash: Source run hash. If None, will try to read from exp_hash.json in checkpoint dir.
    """
    # Try to read source exp_hash and run_hash from checkpoint directory
    if source_exp_hash is None or source_run_hash is None:
        exp_hash_file = os.path.join(dcp_checkpoint_dir, "exp_hash.json")
        if os.path.exists(exp_hash_file):
            import json

            with open(exp_hash_file) as f:
                exp_hash_data = json.load(f)
                if source_exp_hash is None:
                    source_exp_hash = exp_hash_data.get("exp_hash")
                    logger.info(f"Found source experiment hash: {source_exp_hash}")
                if source_run_hash is None:
                    source_run_hash = exp_hash_data.get("run_hash")
                    logger.info(f"Found source run hash: {source_run_hash}")

    logger.debug(f"begin to load {dcp_checkpoint_dir}")
    sd = {}
    _load_state_dict(
        sd,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    logger.info(sd.keys())

    model_sd = sd["model"]

    # Merge LoRA weights if requested
    if merge_lora:
        logger.info(f"Merging LoRA weights (r={lora_r}, alpha={lora_alpha or lora_r})")
        model_sd = merge_lora_weights(
            model_sd,
            lora_alpha=lora_alpha,
            lora_r=lora_r,
            target_module_patterns=["double_blocks", "single_blocks"],
        )

    # Rearrange weights if requested
    if enable_rearrange_weights:
        logger.info("当前正进行 TP+SP 权重转为非 TP+SP 权重")
        model_sd = rearrange_weights(model_sd)

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    save_file(model_sd, target_path)
    logger.info(f"Saved converted weights to {target_path}")

    # Register weight conversion
    if register:
        conversion_info = {
            "conversion_type": "dcp_to_safetensors",
            "source_path": dcp_checkpoint_dir,
            "source_run_hash": source_run_hash,  # 记录 run_hash 以便溯源
            "enable_rearrange_weights": enable_rearrange_weights,
            "merge_lora": merge_lora,
            "checkpoint_hash": _compute_file_hash(target_path),
        }
        if merge_lora:
            conversion_info["lora_r"] = lora_r
            conversion_info["lora_alpha"] = lora_alpha or lora_r

        conversion_hash = register_weight_conversion(
            source_hash=source_exp_hash,
            target_path=target_path,
            conversion_info=conversion_info,
        )
        if conversion_hash:
            logger.success(f"Registered weight conversion: {conversion_hash}")


if __name__ == "__main__":
    fire.Fire(main)
