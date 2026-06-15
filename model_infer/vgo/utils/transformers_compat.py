"""Compatibility helpers for Hugging Face import layout differences."""

try:
    from transformers import AutoProcessor
except ImportError:
    from transformers.models.auto.processing_auto import AutoProcessor

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration


__all__ = [
    "AutoProcessor",
    "Qwen2_5_VLForConditionalGeneration",
    "Qwen3VLForConditionalGeneration",
]
