#!/usr/bin/env python3
"""
Minimal CRef/SRef demo.

Input:
  1) content/reference image (cref)
  2) style/reference image (sref)
  3) one user prompt

Output:
  - generated image
  - the intermediate Qwen3-VL recaption JSON
  - the final recaption prompt actually fed to the image generator

The command-line interface is intentionally small for open-source demo usage,
while the helper functions below keep the original recaption/generation logic.
"""

from __future__ import annotations

import argparse
import ast
import gc
import json
import os
import re
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torchvision.transforms import functional as TVF
from tqdm import tqdm

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional in lightweight inspection envs
    OmegaConf = None

try:
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover
    AutoProcessor = None
    Qwen3VLForConditionalGeneration = None

WORKDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKDIR))

try:
    from multi_cref_eval_rope_fa import ImageGeneratorRopeFA
except Exception:  # pragma: no cover
    ImageGeneratorRopeFA = None


DEFAULT_DEMO_CREF_IMAGE = str(WORKDIR / "assets/00-cref.jpg")
DEFAULT_DEMO_SREF_IMAGE = str(WORKDIR / "assets/00-sref.jpg")
DEFAULT_DEMO_PROMPT = "迁移图2的风格到图1上，保持图1的整体布局不变。"
DEFAULT_DEMO_OUT_DIR = str(WORKDIR / "outputs/sref_12000_demo")

DEFAULT_DATA_ROOT = "/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content"
DEFAULT_SREF_DATA_ROOT = "/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"
DEFAULT_AE_PATH = "/mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/vae"
DEFAULT_QWENVL_PATH = "/tmp/qwenvl_combined"
DEFAULT_QWEN3_MODEL_PATH = "/mnt/jfs/model_zoo/Qwen3-VL-8B-Instruct"
# ---------------------------------------------------------------------------
# Checkpoint layout — matches the public release on HuggingFace:
#   https://huggingface.co/Blue2Giant/FreeStyle_Checkpoint
#
# Download once, for example:
#   huggingface-cli download Blue2Giant/FreeStyle_Checkpoint \
#       --local-dir ./checkpoints
#
# which yields one `model.safetensors` per preset:
#   <CKPT_ROOT>/
#     freestyle-sref-14000-no-rope/model.safetensors
#     freestyle-sref-12000-no-rope/model.safetensors
#     freestyle-cref-sref-50000-rope/model.safetensors
#     freestyle-cref-sref-40000-no-rope/model.safetensors
#     freestyle-cref-sref-36000-no-rope/model.safetensors
#
# Set FREESTYLE_CKPT_ROOT to wherever you downloaded the repo. The default is
# `./checkpoints` next to this script.
# ---------------------------------------------------------------------------
FREESTYLE_CKPT_ROOT = Path(
    os.environ.get("FREESTYLE_CKPT_ROOT", str(WORKDIR / "checkpoints"))
)

# Preset name -> HuggingFace subdirectory (each holds a single model.safetensors).
PRESET_CKPT_SUBDIR: dict[str, str] = {
    "sref_14000": "freestyle-sref-14000-no-rope",
    "sref_12000": "freestyle-sref-12000-no-rope",
    "cref_sref_rope_50000": "freestyle-cref-sref-50000-rope",
    "cref_sref_40000": "freestyle-cref-sref-40000-no-rope",
    "cref_sref_36000_no_rope": "freestyle-cref-sref-36000-no-rope",
}

# Legacy internal checkpoint paths, kept only as a fallback for the original
# development environment. Public users can ignore these: resolution always
# prefers FREESTYLE_CKPT_ROOT (the HuggingFace layout above) and only falls back
# to a legacy path when it actually exists on disk.
LEGACY_CKPT_PATHS: dict[str, list[str]] = {
    "sref_14000": [
        "/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new"
        "/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors",
    ],
    "sref_12000": [
        "/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors",
    ],
    "cref_sref_rope_50000": [
        "/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base"
        "/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors",
    ],
    "cref_sref_40000": [
        "/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious"
        "/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors",
    ],
    "cref_sref_36000_no_rope": [
        "/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors",
        "/mnt/jfs/model_zoo/checkpoint-36000.safetensors",
    ],
}


def preset_public_ckpt_path(preset_name: str) -> Path:
    """Documented public-layout path for a preset (may not exist yet)."""
    return FREESTYLE_CKPT_ROOT / PRESET_CKPT_SUBDIR[preset_name] / "model.safetensors"


def resolve_preset_dit_path(preset_name: str) -> str:
    """Resolve a preset to a concrete checkpoint path.

    Order:
      1. <FREESTYLE_CKPT_ROOT>/<hf_subdir>/model.safetensors  (public layout)
      2. the first existing legacy internal path (dev environment only)
    If nothing exists on disk, return the public-layout path so the resulting
    "file not found" error points at the documented download location.
    """
    public = preset_public_ckpt_path(preset_name)
    if public.exists():
        return str(public)
    for legacy in LEGACY_CKPT_PATHS.get(preset_name, []):
        if Path(legacy).exists():
            return str(legacy)
    return str(public)

# ---------------------------------------------------------------------------
# Inference-only model config (hard-coded; training configs are not shipped)
# ---------------------------------------------------------------------------
# All released FreeStyle checkpoints share the same DiT architecture and the
# same Qwen2.5-VL text-encoder settings. Only frequency-aware RoPE differs, and
# it is toggled by --use_rope / --no_rope (or the weight preset). These values
# are the inference-relevant subset of the original training configs; the rest
# of the training config (data, optimizer, loss policy, internal paths) is not
# needed to run inference and is intentionally omitted.
DIT_INFERENCE_PARAMS: dict[str, Any] = {
    "in_channels": 64,
    "out_channels": 64,
    "enable_txt_norm": True,
    "vec_in_dim": None,
    "context_in_dim": 3584,
    "hidden_size": 3072,
    "mlp_ratio": 4.0,
    "num_heads": 24,
    "depth": 60,
    "depth_single_blocks": 0,
    "axes_dim": [16, 56, 56],
    "theta": 10000,
    "qkv_bias": True,
    "guidance_embed": False,
    "enable_zero_t_embed": True,
}

# Frequency-aware RoPE modulation parameters for the RoPE checkpoints.
ROPE_FA_INFERENCE_PARAMS: dict[str, Any] = {
    "enabled": True,
    "shf_min": 0.9,
    "slf_min": 1.2,
    "shf_max": 0.9,
    "slf_max": 1.2,
    "beta": 2.0,
    "spatial_axes_only": True,
}

# Shared text-encoder / pipe settings used by vgo.inference.load_models.
LLM_INFERENCE_PARAMS: dict[str, Any] = {
    "llm_encoder_type": "naive",
    "llm_image_min_token": 188,
    "llm_image_max_token": 188,
    "max_length": 2048,
    # ae_path / llm_model_path come from CLI (--ae_path / --qwenvl_path); keep
    # them null here so no internal training paths are embedded.
    "ae_path": None,
    "llm_model_path": None,
    "lora": None,
}

TASK_SREF = "sref"
TASK_CREF_SREF = "cref_sref"


def build_inference_config(use_rope: bool):
    """Build the minimal in-memory config consumed by vgo.inference.load_models.

    This replaces the training YAML: only the inference-relevant
    ``engine_config.pipe`` block is constructed, with RoPE toggled in code.
    """
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required to build the inference config")
    dit = dict(DIT_INFERENCE_PARAMS)
    if use_rope:
        dit["rope_fa"] = dict(ROPE_FA_INFERENCE_PARAMS)
    pipe = {"dit": dit, **LLM_INFERENCE_PARAMS}
    return OmegaConf.create({"engine_config": {"pipe": pipe}})


WEIGHT_PRESETS = {
    "sref_14000": {
        "use_rope": False,
        "task": TASK_SREF,
        "data_root": DEFAULT_SREF_DATA_ROOT,
        "recaption_task_type": "sref",
    },
    "sref_12000": {
        "use_rope": False,
        "task": TASK_SREF,
        "data_root": DEFAULT_SREF_DATA_ROOT,
        "recaption_task_type": "sref",
    },
    "cref_sref_40000": {
        "use_rope": False,
        "task": TASK_CREF_SREF,
        "data_root": DEFAULT_DATA_ROOT,
        "recaption_task_type": "identity_style",
    },
    "cref_sref_36000_no_rope": {
        "use_rope": False,
        "task": TASK_CREF_SREF,
        "data_root": DEFAULT_DATA_ROOT,
        "recaption_task_type": "identity_style",
    },
    "cref_sref_rope_50000": {
        "use_rope": True,
        "task": TASK_CREF_SREF,
        "data_root": DEFAULT_DATA_ROOT,
        "recaption_task_type": "identity_style",
    },
}

NEGATIVE_PROMPT = (
    "worst quality, normal quality, low quality, low res, blurry, text, watermark, "
    "logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, "
    "sketch, duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
)

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672),
]


# ---------------------------------------------------------------------------
# Optimized Qwen3 recaption prompts
# ---------------------------------------------------------------------------

RECAPTION_SYSTEM_PROMPT = "You are a helpful assistant. Return valid JSON only."

QWEN3_CREF_SREF_USER_PROMPT = """
[Inputs]
- Original user prompt (highest-priority target content):
{{prompt}}
- Filename style hint, if present. Treat it as a style label only, never as target content:
{{style_hint}}
- Image 1 / scene_1: content or identity reference. Use only compatible visual details.
- Image 2 / scene_2: style reference. Use style only; concrete content is forbidden.

[Recaption task]
Write a clean generation prompt for the final target image.
The final image must depict the Original user prompt. It should have the abstract
visual style of Image 2, but it must not contain Image 2's concrete subjects,
objects, scene layout, props, actions, or story.

[Style extraction rules]
Allowed style words: art medium, brushwork/line quality, edge treatment, color
palette, contrast, lighting, material/texture, geometric simplification,
rendering technique, decorative pattern density, camera/illustration finish,
overall mood.
Forbidden style-image content: any concrete noun or identity observed only in
Image 2, including characters, animals, products, furniture, buildings, foods,
vehicles, landscapes, props, clothing details, pose, exact composition, readable
text, logos, or narrative events.
If the filename style hint sounds like a character/franchise/object name, convert
it into neutral visual traits instead of naming or inserting that content.

[Output requirements]
- JSON only, exactly one object.
- Chinese text values.
- "scene_3" is the only text that will be fed to the image generator, so make it
  concrete, visual, and standalone.
- In "scene_3", first describe the target content from the user prompt, then add
  abstract style attributes. Do not mention references or scene numbers.
- Keep "scene_3" concise: 1-3 sentences, about 60-160 Chinese characters.

{
  "style_only": {
    "abstract_style_cn": "只描述抽象画风特征，不包含任何具体主体、物体、地点、动作或构图复刻"
  },
  "training_output": {
    "sample_instruction_cn_123": "一句自然中文生成指令：目标内容 + 抽象画风，不出现参考图/场景编号/风格图内容"
  },
  "independent_captions": {
    "scene_3": "最终图像的独立中文描述：严格遵循用户提示词的内容，并带有抽象画风；不得包含风格图里的具体内容"
  }
}
""".strip()

META_FORBIDDEN_RE = re.compile(
    r"(scene\s*[_-]?\s*[12]|场景\s*[12]|cref|sref|reference\s+image|style\s+reference|"
    r"style\s+image|source\s+image|参考图|风格图|源图|内容图|图\s*[12])",
    flags=re.IGNORECASE,
)


@dataclass
class RecaptionResult:
    key: str
    original_prompt: str
    final_prompt: str
    raw_response: str
    parsed: dict[str, Any]
    style_hint: str


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_keys(raw: str) -> list[str]:
    if not raw:
        return []
    # Accept comma, Chinese comma, semicolon, or newlines.
    text = raw.replace("，", ",").replace("；", ";")
    parts = re.split(r"[,;\n\r\t]+", text)
    return [p.strip() for p in parts if p.strip()]


def read_key_txt(path: str | Path) -> list[str]:
    """Read key_txt while preserving key-internal/trailing spaces.

    Some benchmark filenames/JSON keys intentionally contain a trailing space
    before the extension (for example "...__Makoto_Shinkai "). Using
    str.strip() here would silently change the key and make it fail lookup.
    Only newline characters are removed; blank/comment detection still uses a
    stripped view.
    """
    keys: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key = line.rstrip("\n\r")
            view = key.strip()
            if not view or view.startswith("#"):
                continue
            keys.append(key)
    return keys


def resolve_keys(prompts: dict[str, str], keys_arg: str = "", key_txt: str = "") -> list[str]:
    if key_txt:
        keys = read_key_txt(key_txt)
    else:
        keys = split_keys(keys_arg)
    if not keys:
        keys = list(prompts.keys())
    missing = [k for k in keys if k not in prompts]
    if missing:
        print(f"[WARN] {len(missing)} key(s) missing from prompts_json; first few: {missing[:8]}", flush=True)
    return [k for k in keys if k in prompts]


def basename_txt_for_keys(keys: Iterable[str], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            f.write(str(k).strip() + "\n")
    return path


def style_hint_from_key(key: str) -> str:
    if "__" not in key:
        return ""
    hint = key.split("__", 1)[1]
    hint = hint.replace("_", " ").replace("-", " ").strip()
    return re.sub(r"\s+", " ", hint)


def find_image_path(folder: str | Path, key: str) -> Path:
    folder = Path(folder)
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        p = folder / f"{key}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"missing image for key={key} under {folder}")


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def setup_qwenvl_combined(qwenvl_path: str) -> Path:
    path = Path(qwenvl_path)
    if path.name == "text_encoder" and not (path / "config.json").exists():
        path = path.parent
    if (path / "config.json").exists() and (
        (path / "preprocessor_config.json").exists() or (path / "processor_config.json").exists()
    ):
        return path
    path.mkdir(parents=True, exist_ok=True)
    for src_dir in [
        "/mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/text_encoder",
        "/mnt/jfs/model_zoo/qwen/Qwen-Image-Edit-2511/processor",
    ]:
        src = Path(src_dir)
        if not src.exists():
            continue
        for item in src.iterdir():
            target = path / item.name
            if not target.exists():
                target.symlink_to(item)
    return path


def clear_cuda(device: str | torch.device | None = None) -> None:
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        if device is not None:
            dev = torch.device(device)
            if dev.type == "cuda":
                torch.cuda.set_device(dev.index if dev.index is not None else 0)
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    except Exception:
        pass


def set_cuda_device(device: str | torch.device) -> None:
    if not torch.cuda.is_available():
        return
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev.index if dev.index is not None else 0)


def select_bucket_by_aspect(cref: Image.Image) -> tuple[int, int]:
    w0, h0 = cref.size
    aspect = w0 / float(max(h0, 1))
    _, w, h = min((abs(aspect - (w / float(h))), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS)
    return int(w), int(h)


def resolve_noise_output_size(width: int, height: int) -> tuple[int, int]:
    """Resolve the denoising/noise canvas size for normal CRef+SRef/SRef tasks.

    This size is passed to ImageGenerator.generate_image(width=..., height=...).
    For edit tasks with two references, vgo.inference uses this as the initial
    noise/output canvas size; it must not be reused to force-resize cref/sref.
    """
    if width > 0 and height > 0:
        return int(width), int(height)
    return 1024, 1024


def should_keep_cref_output_resolution(recaption_task_type: str) -> bool:
    """Return True for style-transfer output sizing.

    Style-transfer output should match the first input image (cref) resolution.
    Normal CRef+SRef/SRef tasks use --width/--height, defaulting to 1024x1024.
    """
    return normalize_recaption_task_type(recaption_task_type) == RECAPTION_TASK_TYPE_STYLE_TRANSFER


def resolve_ref_bucket_size(cref: Image.Image, resolution_mode: str) -> tuple[int, int, str]:
    """Resolve cref/sref preprocessing bucket independently from output noise.

    The default follows the cref aspect ratio using the same bucket list as
    gradio_vgo_infer.py.  This preserves reference-image ratio instead of
    squashing every input reference into the output canvas.
    """
    mode = (resolution_mode or "follow_cref_aspect").strip().lower()
    if mode in ("follow_cref_aspect", "closest_ratio", "auto", "follow", "cref_aspect"):
        w, h = select_bucket_by_aspect(cref)
        return w, h, "follow_cref_aspect"
    if mode in ("square", "square_1024", "1024", "1024x1024"):
        return 1024, 1024, "square_1024"
    raise ValueError(f"unknown resolution_mode={resolution_mode}")


def maybe_resize_pair(cref: Image.Image, sref: Image.Image, size: tuple[int, int]) -> tuple[Image.Image, Image.Image]:
    if cref.size != size:
        cref = cref.resize(size, Image.Resampling.BICUBIC)
    if sref.size != size:
        sref = sref.resize(size, Image.Resampling.BICUBIC)
    return cref, sref


def resolve_use_rope(use_rope: Any, task: str = "") -> bool:
    """Resolve the RoPE flag to a concrete bool. Defaults to no RoPE."""
    if isinstance(use_rope, bool):
        return use_rope
    if isinstance(use_rope, str):
        return use_rope.strip().lower() in ("1", "true", "yes", "on", "rope")
    return False


# ---------------------------------------------------------------------------
# Qwen3 recaption
# ---------------------------------------------------------------------------


class LocalQwen3Recaptioner:
    def __init__(self, model_path: str, device: str = "cuda:0", max_new_tokens: int = 384, image_long_edge: int = 512, image_tokens: int = 188):
        if AutoProcessor is None or Qwen3VLForConditionalGeneration is None:
            raise ImportError("transformers does not provide Qwen3VLForConditionalGeneration / AutoProcessor")
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.image_long_edge = max(64, int(image_long_edge))
        self.image_tokens = max(64, int(image_tokens))
        self.min_pixels = self.image_tokens * 28 * 28
        self.max_pixels = self.image_tokens * 28 * 28
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model = self._load_model()
        self.model.eval()
        print(
            f"[RECAPTION] loaded qwen3 model={self.model_path} device={self.device} "
            f"dtype={self.dtype} max_new_tokens={self.max_new_tokens} "
            f"image_long_edge={self.image_long_edge} image_tokens={self.image_tokens}",
            flush=True,
        )

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        raw = str(device).strip().lower()
        if raw in ("", "auto"):
            return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        if raw.isdigit():
            return torch.device(f"cuda:{int(raw)}")
        return torch.device(str(device).strip())

    def _load_model(self):
        candidates = ["flash_attention_2", "sdpa"] if self.device.type == "cuda" else ["sdpa"]
        last_exc: Exception | None = None
        for impl in candidates:
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    dtype=self.dtype,
                    attn_implementation=impl,
                ).to(self.device)
                if impl != candidates[0]:
                    print(f"[RECAPTION] fallback attn_implementation={impl}", flush=True)
                return model
            except Exception as exc:
                last_exc = exc
                print(f"[RECAPTION] qwen3 load failed impl={impl}: {exc}", flush=True)
        raise RuntimeError(f"failed to load qwen3 model: {self.model_path}") from last_exc

    def _load_recaption_image(self, path: Path) -> Image.Image:
        img = Image.open(path).convert("RGB")
        # Downsample for VLM recaption only. This avoids very slow Qwen3-VL visual
        # encoding on 1024px benchmark images while preserving enough style/content.
        if max(img.size) > self.image_long_edge:
            img.thumbnail((self.image_long_edge, self.image_long_edge), Image.Resampling.LANCZOS)
        return img

    def generate(self, cref_path: Path, sref_path: Path, request_text: str) -> str:
        image_kwargs = {"min_pixels": self.min_pixels, "max_pixels": self.max_pixels}
        cref_img = self._load_recaption_image(cref_path)
        sref_img = self._load_recaption_image(sref_path)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": RECAPTION_SYSTEM_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": cref_img, **image_kwargs},
                    {"type": "image", "image": sref_img, **image_kwargs},
                    {"type": "text", "text": request_text},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    def close(self) -> None:
        for attr in ("model", "processor"):
            if hasattr(self, attr):
                delattr(self, attr)
        clear_cuda(self.device)


RECAPTION_TASK_TYPE_STYLE_TRANSFER = "style_transfer"
RECAPTION_TASK_TYPE_IDENTITY_STYLE = "identity_style"
RECAPTION_TASK_TYPE_CREF_SREF = "cref_sref"
RECAPTION_TASK_TYPE_SREF = "sref"

# Recaption prompts by weight family, with one explicit source of truth each:
#   - SRef / style transfer -> recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER
#     (read verbatim so the full canonical style-transfer template is used)
#   - CRef+SRef (identity_style) -> local QWEN3_CREF_SREF_USER_PROMPT
REC_STYLE_TRANSFER_CONST = "PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER"
RECAPTION_PROMPT_NAME_STYLE = REC_STYLE_TRANSFER_CONST
RECAPTION_PROMPT_NAME_CREF_SREF = "QWEN3_CREF_SREF_USER_PROMPT"


def normalize_recaption_task_type(task_type: str) -> str:
    raw = str(task_type or "").strip().lower().replace("-", "_")
    if raw in ("", "identity", "identity_style", "cref_sref", "cref+sref", "cref_sref_identity"):
        return RECAPTION_TASK_TYPE_IDENTITY_STYLE
    if raw in ("sref", "sref_only", "sref_infer", "sref_inference"):
        return RECAPTION_TASK_TYPE_SREF
    if raw in ("style", "style_transfer", "transfer", "sref_transfer"):
        return RECAPTION_TASK_TYPE_STYLE_TRANSFER
    raise ValueError(
        f"unknown recaption_task_type={task_type!r}; expected sref, identity_style/cref_sref, or style_transfer"
    )


def read_recaption_prompt_constant(const_name: str) -> str:
    """Read a string constant from recaption.py without importing it.

    recaption.py imports optional API-client packages we do not need for offline
    inference, so AST parsing avoids that dependency while still using the exact,
    complete template defined there.
    """
    recaption_py = WORKDIR / "recaption.py"
    if not recaption_py.exists():
        raise FileNotFoundError(f"missing recaption.py next to core infer script: {recaption_py}")
    tree = ast.parse(recaption_py.read_text(encoding="utf-8"), filename=str(recaption_py))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == const_name:
                value = ast.literal_eval(node.value)
                if not isinstance(value, str):
                    raise TypeError(f"{const_name} in {recaption_py} is not a string")
                return value
    raise KeyError(f"{const_name} not found in {recaption_py}")


_STYLE_TRANSFER_PROMPT_CACHE: dict[str, str] = {}


def get_style_transfer_recaption_prompt() -> str:
    if "p" not in _STYLE_TRANSFER_PROMPT_CACHE:
        _STYLE_TRANSFER_PROMPT_CACHE["p"] = read_recaption_prompt_constant(REC_STYLE_TRANSFER_CONST)
    return _STYLE_TRANSFER_PROMPT_CACHE["p"]


def recaption_uses_style_template(recaption_task_type: str) -> bool:
    """SRef and style-transfer both use the full style-transfer template."""
    return normalize_recaption_task_type(recaption_task_type) in (
        RECAPTION_TASK_TYPE_SREF,
        RECAPTION_TASK_TYPE_STYLE_TRANSFER,
    )


def get_recaption_prompt_template_name(recaption_task_type: str) -> str:
    if recaption_uses_style_template(recaption_task_type):
        return RECAPTION_PROMPT_NAME_STYLE
    return RECAPTION_PROMPT_NAME_CREF_SREF


def get_recaption_prompt_template(recaption_task_type: str) -> str:
    if recaption_uses_style_template(recaption_task_type):
        return get_style_transfer_recaption_prompt()
    return QWEN3_CREF_SREF_USER_PROMPT


def validate_recaption_task_type_for_task(
    task: str, recaption_task_type: str, weight_preset: str = ""
) -> None:
    """Refuse prompt/weight combinations that would mix the two families.

    A pure SRef weight must not run the CRef+SRef recaption prompt, and a
    CRef+SRef weight must not run the pure-SRef recaption prompt. style_transfer
    is allowed on either family.
    """
    rtt = normalize_recaption_task_type(recaption_task_type)
    where = f" (weight_preset={weight_preset})" if weight_preset else ""
    if task == TASK_SREF and rtt == RECAPTION_TASK_TYPE_IDENTITY_STYLE:
        raise ValueError(
            f"recaption_task_type={rtt!r} is a CRef+SRef prompt and cannot run on an "
            f"SRef weight{where}; use 'sref' or 'style_transfer'."
        )
    if task == TASK_CREF_SREF and rtt == RECAPTION_TASK_TYPE_SREF:
        raise ValueError(
            f"recaption_task_type={rtt!r} is a pure-SRef prompt and cannot run on a "
            f"CRef+SRef weight{where}; use 'identity_style'/'cref_sref' or 'style_transfer'."
        )


def render_recaption_request(
    original_prompt: str,
    style_hint: str = "",
    recaption_task_type: str = RECAPTION_TASK_TYPE_IDENTITY_STYLE,
) -> str:
    template = get_recaption_prompt_template(recaption_task_type)
    rendered = template.replace("{{prompt}}", str(original_prompt).strip())
    # Kept for compatibility if future templates add this placeholder.
    rendered = rendered.replace("{{style_hint}}", str(style_hint or "").strip() or "N/A")
    return rendered


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    m = re.match(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", s, flags=re.IGNORECASE)
    return (m.group(1).strip() if m else s)


def extract_json_object(text: str) -> dict[str, Any]:
    s = strip_code_fences(text)
    try:
        obj = json.loads(s)
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(s[start:end + 1])
        if isinstance(obj, str):
            obj = json.loads(obj)
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"could not extract JSON object from response: {s[:300]}")


def clean_final_prompt(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\bscene\s*[_-]?\s*3\b\s*[:：]?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"场景\s*3\s*[:：]?", "", text)
    # Keep scene_1/scene_2 wording from recaption.py training instructions.
    # The downstream generator is conditioned on two reference images, and the
    # requested final prompt is the model JSON fields scene_3 +
    # primary_instruction_cn_123, so we should not delete those field contents.
    text = re.sub(r"\s{2,}", " ", text).strip(" ，,。:：")
    return text


def extract_json_string_field(raw_response: str, field_names: Iterable[str]) -> str:
    """Best-effort field extraction from imperfect model JSON.

    Qwen3-VL occasionally returns nearly-valid JSON with a missing comma or a
    trailing explanation.  For demo robustness, pull important string fields by
    name before falling back to the original prompt.
    """
    raw = str(raw_response or "")
    for name in field_names:
        pattern = rf'"{re.escape(str(name))}"\s*:\s*"((?:\\.|[^"\\])*)"'
        m = re.search(pattern, raw, flags=re.DOTALL)
        if not m:
            continue
        value = m.group(1)
        try:
            return json.loads('"' + value + '"')
        except Exception:
            return value.replace('\\n', ' ').replace('\\"', '"')
    return ""


def parse_recaption_response(raw_response: str, original_prompt: str) -> tuple[str, dict[str, Any]]:
    try:
        parsed = extract_json_object(raw_response)
        parse_error = ""
    except Exception as exc:
        # Keep the pipeline runnable even if the demo recaption model emits
        # slightly malformed JSON.  The structured output will still contain
        # raw_response plus this parse error for debugging/open-source users.
        parse_error = f"{type(exc).__name__}: {exc}"
        print(f"[WARN] recaption JSON parse failed, using best-effort field extraction: {parse_error}", flush=True)
        parsed = {
            "_parse_error": parse_error,
            "_raw_response": raw_response,
        }

    captions = parsed.get("independent_captions") or {}
    training = parsed.get("training_output") or {}
    if not isinstance(captions, dict):
        captions = {}
    if not isinstance(training, dict):
        training = {}

    # New recaption contract: feed the generator with scene_3 +
    # primary_instruction_cn_123.  Keep conservative fallbacks for partially
    # valid model JSON, but prefer the exact Chinese fields requested.
    scene_3 = clean_final_prompt(captions.get("scene_3") or captions.get("scene_3_en") or "")
    primary = clean_final_prompt(
        training.get("primary_instruction_cn_123")
        or training.get("sample_instruction_cn_123")
        or training.get("primary_instruction_en_123")
        or training.get("sample_instruction_en_123")
        or ""
    )

    if not scene_3 and parse_error:
        scene_3 = clean_final_prompt(extract_json_string_field(raw_response, ["scene_3", "scene_3_en"]))
    if not primary and parse_error:
        primary = clean_final_prompt(
            extract_json_string_field(
                raw_response,
                [
                    "primary_instruction_cn_123",
                    "sample_instruction_cn_123",
                    "primary_instruction_en_123",
                    "sample_instruction_en_123",
                ],
            )
        )

    parts = [x for x in (scene_3, primary) if x]
    final = "，".join(parts)
    if not final:
        final = clean_final_prompt(original_prompt)
    return final, parsed


def recaption_one(
    recaptioner: LocalQwen3Recaptioner,
    key: str,
    original_prompt: str,
    cref_path: Path,
    sref_path: Path,
    recaption_task_type: str = RECAPTION_TASK_TYPE_IDENTITY_STYLE,
) -> RecaptionResult:
    style_hint = style_hint_from_key(key)
    request = render_recaption_request(original_prompt, style_hint, recaption_task_type=recaption_task_type)
    raw = recaptioner.generate(cref_path, sref_path, request)
    final, parsed = parse_recaption_response(raw, original_prompt)
    return RecaptionResult(
        key=key,
        original_prompt=original_prompt,
        final_prompt=final,
        raw_response=raw,
        parsed=parsed,
        style_hint=style_hint,
    )


def recaption_many(
    keys: list[str],
    prompts: dict[str, str],
    cref_dir: str | Path,
    sref_dir: str | Path,
    recaption_json: str | Path,
    structured_json: str | Path,
    model_path: str,
    device: str,
    max_new_tokens: int,
    image_long_edge: int,
    image_tokens: int,
    overwrite: bool,
    recaption_task_type: str = RECAPTION_TASK_TYPE_IDENTITY_STYLE,
    fallback_original_on_error: bool = True,
) -> dict[str, str]:
    recaption_path = Path(recaption_json)
    structured_path = Path(structured_json)
    recaptioned: dict[str, str] = {}
    structured: dict[str, Any] = {}
    if recaption_path.exists() and not overwrite:
        try:
            existing = load_json(recaption_path)
            if isinstance(existing, dict):
                recaptioned.update({str(k): str(v) for k, v in existing.items()})
        except Exception:
            pass
    if structured_path.exists() and not overwrite:
        try:
            existing_s = load_json(structured_path)
            if isinstance(existing_s, dict):
                structured.update(existing_s)
        except Exception:
            pass

    pending = [k for k in keys if overwrite or k not in recaptioned]
    print(f"[RECAPTION] task_type={normalize_recaption_task_type(recaption_task_type)} total={len(keys)} cached={len(keys)-len(pending)} pending={len(pending)}", flush=True)
    if not pending:
        return {k: recaptioned[k] for k in keys if k in recaptioned}

    recaptioner = LocalQwen3Recaptioner(
        model_path=model_path,
        device=device,
        max_new_tokens=max_new_tokens,
        image_long_edge=image_long_edge,
        image_tokens=image_tokens,
    )
    try:
        for key in tqdm(pending, desc="qwen3 recaption", dynamic_ncols=True):
            cref_path = find_image_path(cref_dir, key)
            sref_path = find_image_path(sref_dir, key)
            original = str(prompts[key])
            try:
                result = recaption_one(recaptioner, key, original, cref_path, sref_path, recaption_task_type=recaption_task_type)
                recaptioned[key] = result.final_prompt
                structured[key] = {
                    "key": result.key,
                    "original_prompt": result.original_prompt,
                    "final_prompt": result.final_prompt,
                    "style_hint": result.style_hint,
                    "recaption_task_type": normalize_recaption_task_type(recaption_task_type),
                    "raw_response": result.raw_response,
                    "parsed": result.parsed,
                }
                print(f"[RECAPTION DONE] {key}: {result.final_prompt}", flush=True)
            except Exception as exc:
                print(f"[RECAPTION FAIL] {key}: {type(exc).__name__}: {exc}", flush=True)
                if not fallback_original_on_error:
                    raise
                recaptioned[key] = clean_final_prompt(original)
                structured[key] = {
                    "key": key,
                    "original_prompt": original,
                    "final_prompt": recaptioned[key],
                    "style_hint": style_hint_from_key(key),
                    "recaption_task_type": normalize_recaption_task_type(recaption_task_type),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            dump_json(recaptioned, recaption_path)
            dump_json(structured, structured_path)
    finally:
        recaptioner.close()
    return {k: recaptioned[k] for k in keys if k in recaptioned}


def run_recaption_subprocess(args: argparse.Namespace, recaption_json: Path, structured_json: Path, basename_txt: Path) -> None:
    """Run Qwen3-VL recaption in a child process before loading VGO.

    In long-running CUDA processes, the local Qwen3-VL recaption model can leave
    allocator/driver state behind even after ``del`` + ``empty_cache``.  On the
    single-GPU rlaunch workers used here this made the subsequent VGO generator
    load unstable.  A short-lived child process gives the OS/CUDA driver a hard
    boundary: when recaption exits, all Qwen3-VL GPU memory is definitely gone.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data_root", str(args.data_root),
        "--out_dir", str(args.out_dir),
        "--recaption_json", str(recaption_json),
        "--structured_json", str(structured_json),
        "--basename_txt", str(basename_txt),
        "--recaption_model_path", str(args.recaption_model_path),
        "--recaption_device", str(args.recaption_device),
        "--recaption_max_new_tokens", str(args.recaption_max_new_tokens),
        "--recaption_image_long_edge", str(args.recaption_image_long_edge),
        "--recaption_image_tokens", str(args.recaption_image_tokens),
        "--recaption_task_type", str(args.recaption_task_type),
        "--recaption_only",
        "--no_recaption_subprocess",
    ]
    if args.prompts_json:
        cmd += ["--prompts_json", str(args.prompts_json)]
    if args.cref_dir:
        cmd += ["--cref_dir", str(args.cref_dir)]
    if args.sref_dir:
        cmd += ["--sref_dir", str(args.sref_dir)]
    if args.keys:
        cmd += ["--keys", str(args.keys)]
    if args.key_txt:
        cmd += ["--key_txt", str(args.key_txt)]
    if args.overwrite:
        cmd += ["--overwrite"]

    print("[RECAPTION SUBPROCESS] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# VGO generation
# ---------------------------------------------------------------------------


def load_generator(dit_path: str, use_rope: bool, ae_path: str, qwenvl_path: str, device: str):
    from vgo.inference import ImageGenerator

    qwenvl_path = str(setup_qwenvl_combined(qwenvl_path))
    use_rope = bool(use_rope)
    config_obj = build_inference_config(use_rope)
    generator_cls = ImageGeneratorRopeFA if use_rope and ImageGeneratorRopeFA is not None else ImageGenerator
    if use_rope and ImageGeneratorRopeFA is None:
        raise RuntimeError("--use_rope requested but ImageGeneratorRopeFA failed to import")
    print(
        f"[GENERATOR] loading {generator_cls.__name__} ckpt={dit_path} "
        f"rope_fa={use_rope} device={device}",
        flush=True,
    )
    set_cuda_device(device)
    gen = generator_cls(
        world_mesh=None,
        dit_path=dit_path,
        config_obj=config_obj,
        ae_path=ae_path or None,
        qwenvl_model_path=qwenvl_path or None,
        device=device,
    )
    print("[GENERATOR] ready", flush=True)
    return gen


def generate_many(
    keys: list[str],
    prompts: dict[str, str],
    cref_dir: str | Path,
    sref_dir: str | Path,
    out_dir: str | Path,
    dit_path: str,
    use_rope: bool,
    ae_path: str,
    qwenvl_path: str,
    device: str,
    width: int,
    height: int,
    resolution_mode: str,
    steps: int,
    cfg: float,
    seed: int,
    overwrite: bool,
    recaption_task_type: str = RECAPTION_TASK_TYPE_IDENTITY_STYLE,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = load_generator(dit_path, use_rope, ae_path, qwenvl_path, device=device)
    try:
        for key in tqdm(keys, desc="vgo generate", dynamic_ncols=True):
            out_path = out_dir / f"{key}.png"
            if out_path.exists() and not overwrite:
                print(f"[SKIP] exists {out_path}", flush=True)
                continue
            cref_path = find_image_path(cref_dir, key)
            sref_path = find_image_path(sref_dir, key)
            cref = load_rgb(cref_path)
            sref = load_rgb(sref_path)
            orig_cref_size = cref.size
            orig_sref_size = sref.size
            keep_cref_resolution = should_keep_cref_output_resolution(recaption_task_type)
            if keep_cref_resolution:
                noise_w, noise_h = orig_cref_size
                output_size_mode = "style_transfer_keep_cref_resolution"
            else:
                noise_w, noise_h = resolve_noise_output_size(width, height)
                output_size_mode = "user_width_height"
            ref_w, ref_h, ref_mode = resolve_ref_bucket_size(cref, resolution_mode)
            cref, sref = maybe_resize_pair(cref, sref, (ref_w, ref_h))
            prompt = str(prompts[key]).strip()
            print(
                f"[GENERATE] key={key} output_noise={noise_w}x{noise_h} mode={output_size_mode} "
                f"ref_bucket={ref_w}x{ref_h} ref_mode={ref_mode} "
                f"cref={orig_cref_size[0]}x{orig_cref_size[1]}->{ref_w}x{ref_h} "
                f"sref={orig_sref_size[0]}x{orig_sref_size[1]}->{ref_w}x{ref_h} "
                f"prompt={prompt}",
                flush=True,
            )
            set_cuda_device(device)
            with torch.inference_mode():
                images = gen.generate_image(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    width=int(noise_w),
                    height=int(noise_h),
                    num_steps=int(steps),
                    cfg_guidance=float(cfg),
                    seed=[int(seed)],
                    ref_image=[cref, sref],
                    task_type="edit",
                )
            image = TVF.to_pil_image(images[0].float())
            # vgo.inference rounds generation dimensions up to multiples of 16.
            # For style transfer, save an exact CRef-resolution image.
            if keep_cref_resolution and image.size != orig_cref_size:
                print(
                    f"[RESIZE OUTPUT] {image.size[0]}x{image.size[1]} -> "
                    f"{orig_cref_size[0]}x{orig_cref_size[1]}",
                    flush=True,
                )
                image = image.resize(orig_cref_size, Image.Resampling.LANCZOS)
            image.save(out_path)
            print(f"[SAVED] {out_path}", flush=True)
    finally:
        try:
            del gen
        except Exception:
            pass
        clear_cuda(device)


# ---------------------------------------------------------------------------
# Minimal single-sample demo
# ---------------------------------------------------------------------------


def generate_one(
    cref_path: str | Path,
    sref_path: str | Path,
    prompt: str,
    output_path: str | Path,
    dit_path: str,
    use_rope: bool,
    ae_path: str,
    qwenvl_path: str,
    device: str,
    width: int,
    height: int,
    resolution_mode: str,
    steps: int,
    cfg: float,
    seed: int,
    overwrite: bool,
    recaption_task_type: str = RECAPTION_TASK_TYPE_IDENTITY_STYLE,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        print(f"[SKIP] exists {output_path}", flush=True)
        return

    gen = load_generator(dit_path, use_rope, ae_path, qwenvl_path, device=device)
    try:
        cref = load_rgb(cref_path)
        sref = load_rgb(sref_path)
        orig_cref_size = cref.size
        orig_sref_size = sref.size
        keep_cref_resolution = should_keep_cref_output_resolution(recaption_task_type)
        if keep_cref_resolution:
            noise_w, noise_h = orig_cref_size
            output_size_mode = "style_transfer_keep_cref_resolution"
        else:
            noise_w, noise_h = resolve_noise_output_size(width, height)
            output_size_mode = "user_width_height"
        ref_w, ref_h, ref_mode = resolve_ref_bucket_size(cref, resolution_mode)
        cref, sref = maybe_resize_pair(cref, sref, (ref_w, ref_h))
        prompt = str(prompt).strip()
        print(
            f"[GENERATE] output_noise={noise_w}x{noise_h} mode={output_size_mode} "
            f"ref_bucket={ref_w}x{ref_h} ref_mode={ref_mode} "
            f"cref={orig_cref_size[0]}x{orig_cref_size[1]}->{ref_w}x{ref_h} "
            f"sref={orig_sref_size[0]}x{orig_sref_size[1]}->{ref_w}x{ref_h} "
            f"prompt={prompt}",
            flush=True,
        )
        set_cuda_device(device)
        with torch.inference_mode():
            images = gen.generate_image(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                width=int(noise_w),
                height=int(noise_h),
                num_steps=int(steps),
                cfg_guidance=float(cfg),
                seed=[int(seed)],
                ref_image=[cref, sref],
                task_type="edit",
            )
        image = TVF.to_pil_image(images[0].float())
        # vgo.inference rounds generation dimensions up to multiples of 16.
        # For style transfer, save an exact CRef-resolution image.
        if keep_cref_resolution and image.size != orig_cref_size:
            print(
                f"[RESIZE OUTPUT] {image.size[0]}x{image.size[1]} -> "
                f"{orig_cref_size[0]}x{orig_cref_size[1]}",
                flush=True,
            )
            image = image.resize(orig_cref_size, Image.Resampling.LANCZOS)
        image.save(output_path)
        print(f"[SAVED] {output_path}", flush=True)
    finally:
        try:
            del gen
        except Exception:
            pass
        clear_cuda(device)


def run_demo_recaption_subprocess(args: argparse.Namespace, recaption_json: Path, structured_json: Path) -> None:
    """Run only Qwen3-VL recaption in a short-lived child process.

    This keeps the recaption model out of memory before loading the image
    generator.  It is the default path for the one-sample demo.
    """
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(args.cref_image),
        str(args.sref_image),
        str(args.prompt),
        "--weight_preset", str(args.weight_preset or ""),
        "--out_dir", str(args.out_dir),
        "--recaption_json", str(recaption_json),
        "--structured_json", str(structured_json),
        "--recaption_model_path", str(args.recaption_model_path),
        "--recaption_device", str(args.recaption_device),
        "--recaption_max_new_tokens", str(args.recaption_max_new_tokens),
        "--recaption_image_long_edge", str(args.recaption_image_long_edge),
        "--recaption_image_tokens", str(args.recaption_image_tokens),
        "--recaption_task_type", str(args.recaption_task_type),
        "--recaption_only",
        "--no_recaption_subprocess",
    ]
    if args.overwrite:
        cmd += ["--overwrite"]
    print("[RECAPTION SUBPROCESS] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_single_demo(args: argparse.Namespace) -> None:
    cref_path = Path(args.cref_image)
    sref_path = Path(args.sref_image)
    if not cref_path.exists():
        raise FileNotFoundError(f"cref image not found: {cref_path}")
    if not sref_path.exists():
        raise FileNotFoundError(f"sref image not found: {sref_path}")

    out_dir = Path(args.out_dir or DEFAULT_DEMO_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output_path) if args.output_path else out_dir / "result.png"
    recaption_json = Path(args.recaption_json) if args.recaption_json else out_dir / "final_prompt.json"
    structured_json = Path(args.structured_json) if args.structured_json else out_dir / "recaption_result.json"
    final_prompt_txt = out_dir / "final_prompt.txt"
    summary_json = out_dir / "demo_summary.json"
    key = "demo"

    print("=" * 80, flush=True)
    print("Minimal CRef/SRef demo", flush=True)
    print(f"cref_image      : {cref_path}", flush=True)
    print(f"sref_image      : {sref_path}", flush=True)
    print(f"user_prompt     : {args.prompt}", flush=True)
    print(f"out_dir         : {out_dir}", flush=True)
    print(f"output_path     : {output_path}", flush=True)
    print(f"recaption_json  : {recaption_json}", flush=True)
    print(f"structured_json : {structured_json}", flush=True)
    print(f"weight_preset   : {args.weight_preset or '<custom>'}", flush=True)
    print(f"recaption_task  : {normalize_recaption_task_type(args.recaption_task_type)}", flush=True)
    print(f"recaption_prompt: {get_recaption_prompt_template_name(args.recaption_task_type)}", flush=True)
    print(f"dit_path        : {args.dit_path}", flush=True)
    print(f"task            : {args.task}", flush=True)
    print(f"use_rope        : {bool(args.use_rope)}", flush=True)
    print("=" * 80, flush=True)

    if args.skip_recaption:
        prompt_data = load_json(recaption_json)
        if isinstance(prompt_data, dict) and key in prompt_data:
            final_prompt = str(prompt_data[key])
        elif isinstance(prompt_data, dict) and "final_prompt" in prompt_data:
            final_prompt = str(prompt_data["final_prompt"])
        else:
            raise TypeError(f"cannot find final prompt in {recaption_json}")
    else:
        if bool(args.recaption_subprocess) and not bool(args.recaption_only):
            run_demo_recaption_subprocess(args, recaption_json, structured_json)
            prompt_data = load_json(recaption_json)
            final_prompt = str(prompt_data[key] if isinstance(prompt_data, dict) and key in prompt_data else prompt_data)
        else:
            recaptioner = LocalQwen3Recaptioner(
                model_path=args.recaption_model_path,
                device=args.recaption_device,
                max_new_tokens=args.recaption_max_new_tokens,
                image_long_edge=args.recaption_image_long_edge,
                image_tokens=args.recaption_image_tokens,
            )
            try:
                result = recaption_one(
                    recaptioner,
                    key=key,
                    original_prompt=str(args.prompt),
                    cref_path=cref_path,
                    sref_path=sref_path,
                    recaption_task_type=str(args.recaption_task_type),
                )
            finally:
                try:
                    del recaptioner
                except Exception:
                    pass
                clear_cuda(args.recaption_device)

            final_prompt = result.final_prompt
            dump_json({key: final_prompt}, recaption_json)
            dump_json(
                {
                    "key": key,
                    "cref_image": str(cref_path),
                    "sref_image": str(sref_path),
                    "original_prompt": result.original_prompt,
                    "final_prompt": result.final_prompt,
                    "raw_response": result.raw_response,
                    "parsed": result.parsed,
                    "style_hint": result.style_hint,
                    "recaption_task_type": normalize_recaption_task_type(args.recaption_task_type),
                },
                structured_json,
            )

    final_prompt_txt.write_text(str(final_prompt).strip() + "\n", encoding="utf-8")
    print(f"[RECAPTION FINAL PROMPT] {final_prompt}", flush=True)
    print(f"[SAVED] {recaption_json}", flush=True)
    print(f"[SAVED] {structured_json}", flush=True)
    print(f"[SAVED] {final_prompt_txt}", flush=True)

    if args.recaption_only:
        print("[DONE] recaption_only", flush=True)
        return

    generate_one(
        cref_path=cref_path,
        sref_path=sref_path,
        prompt=final_prompt,
        output_path=output_path,
        dit_path=args.dit_path,
        use_rope=bool(args.use_rope),
        ae_path=args.ae_path,
        qwenvl_path=args.qwenvl_path,
        device=args.generator_device,
        width=int(args.width),
        height=int(args.height),
        resolution_mode=str(args.resolution_mode),
        steps=int(args.steps),
        cfg=float(args.cfg),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
        recaption_task_type=str(args.recaption_task_type),
    )
    dump_json(
        {
            "cref_image": str(cref_path),
            "sref_image": str(sref_path),
            "original_prompt": str(args.prompt),
            "final_prompt": str(final_prompt),
            "output_image": str(output_path),
            "recaption_json": str(recaption_json),
            "structured_json": str(structured_json),
            "weight_preset": str(args.weight_preset or "<custom>"),
            "dit_path": str(args.dit_path),
            "task": str(args.task),
            "use_rope": bool(args.use_rope),
            "recaption_task_type": normalize_recaption_task_type(args.recaption_task_type),
        },
        summary_json,
    )
    print(f"[SAVED] {summary_json}", flush=True)
    print("[DONE] all", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal CRef/SRef demo: input <cref_image> <sref_image> <prompt>, output image + recaption JSON"
    )
    parser.add_argument("cref_image", nargs="?", default=DEFAULT_DEMO_CREF_IMAGE, help="content/reference image; pass this first")
    parser.add_argument("sref_image", nargs="?", default=DEFAULT_DEMO_SREF_IMAGE, help="style/reference image; pass this second")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_DEMO_PROMPT, help="user prompt")
    parser.add_argument("--weight_preset", default="sref_12000", choices=["", *sorted(WEIGHT_PRESETS.keys())], help="built-in weight/config preset; demo default is sref_12000")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT, help="legacy batch-mode input root; not needed for the minimal demo")
    parser.add_argument("--prompts_json", default="", help="defaults to {data_root}/prompts.json")
    parser.add_argument("--cref_dir", default="", help="defaults to {data_root}/cref")
    parser.add_argument("--sref_dir", default="", help="defaults to {data_root}/sref")
    parser.add_argument("--out_dir", default=DEFAULT_DEMO_OUT_DIR, help="demo output dir")
    parser.add_argument("--output_path", default="", help="defaults to {out_dir}/result.png")
    parser.add_argument("--batch_mode", action="store_true", help="legacy prompts.json + key batch mode")
    parser.add_argument("--keys", default="", help="comma/newline separated keys")
    parser.add_argument("--key_txt", default="", help="txt with one key per line")
    parser.add_argument("--basename_txt", default="", help="write selected basenames here for metric scripts")

    parser.add_argument("--recaption_model_path", default=DEFAULT_QWEN3_MODEL_PATH)
    parser.add_argument("--recaption_device", default="cuda:0")
    parser.add_argument("--recaption_max_new_tokens", type=int, default=1024)
    parser.add_argument("--recaption_image_long_edge", type=int, default=512)
    parser.add_argument("--recaption_image_tokens", type=int, default=188)
    parser.add_argument("--recaption_task_type", default=RECAPTION_TASK_TYPE_IDENTITY_STYLE, help="Recaption prompt family. sref/style_transfer -> recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER (full style-transfer template); identity_style/cref_sref -> QWEN3_CREF_SREF_USER_PROMPT (CRef+SRef). SRef weights accept sref/style_transfer; CRef+SRef weights accept identity_style/cref_sref/style_transfer.")
    parser.add_argument("--skip_recaption", action="store_true", help="use existing --recaption_json/prompts directly")
    parser.add_argument("--recaption_only", action="store_true")
    parser.add_argument("--recaption_subprocess", dest="recaption_subprocess", action="store_true", default=True, help="run Qwen3 recaption in a short-lived child process before VGO generation (default)")
    parser.add_argument("--no_recaption_subprocess", "--no-recaption-subprocess", dest="recaption_subprocess", action="store_false", help="run recaption and generation in the same Python process")
    parser.add_argument("--recaption_json", default="", help="defaults to {out_dir}/recaption_prompts.json")
    parser.add_argument("--structured_json", default="", help="defaults to {out_dir}/recaption_structured.json")

    parser.add_argument("--dit_path", default="", help="DiT checkpoint (.safetensors). Normally set by --weight_preset (resolved under FREESTYLE_CKPT_ROOT); pass this only for a custom weight without a preset")
    parser.add_argument("--task", default=None, choices=[TASK_SREF, TASK_CREF_SREF], help="sref or cref_sref; usually set by --weight_preset. Controls the default recaption prompt/data root")
    parser.add_argument("--use_rope", dest="use_rope", action="store_true", default=None, help="enable frequency-aware RoPE modulation (for RoPE-trained weights); usually set by --weight_preset")
    parser.add_argument("--no_rope", "--no-rope", dest="use_rope", action="store_false", help="disable frequency-aware RoPE modulation")
    parser.add_argument("--ae_path", default=DEFAULT_AE_PATH)
    parser.add_argument("--qwenvl_path", default=DEFAULT_QWENVL_PATH)
    parser.add_argument("--generator_device", default="cuda:0")
    parser.add_argument("--width", type=int, default=1024, help="normal CRef+SRef/SRef output width; ignored by style_transfer, which follows cref image width")
    parser.add_argument("--height", type=int, default=1024, help="normal CRef+SRef/SRef output height; ignored by style_transfer, which follows cref image height")
    parser.add_argument("--resolution_mode", default="follow_cref_aspect", choices=["square_1024", "follow_cref_aspect", "closest_ratio", "auto"], help="Reference image bucket mode; output size uses --width/--height except style_transfer follows cref resolution")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def apply_weight_preset(args: argparse.Namespace) -> None:
    preset_name = str(getattr(args, "weight_preset", "") or "").strip()
    if preset_name:
        preset = WEIGHT_PRESETS[preset_name]
        args.dit_path = resolve_preset_dit_path(preset_name)
        # Only fall back to preset values when the caller did not set them on the CLI.
        if getattr(args, "use_rope", None) is None:
            args.use_rope = preset["use_rope"]
        if getattr(args, "task", None) is None:
            args.task = preset["task"]
        # If the caller did not explicitly move away from the default CRef+SRef
        # root, switch sref presets to the sref benchmark root.
        if getattr(args, "data_root", DEFAULT_DATA_ROOT) == DEFAULT_DATA_ROOT:
            args.data_root = preset["data_root"]
        # Use the task's recaption prompt unless the caller explicitly supplied a
        # different task type on the CLI.
        if "--recaption_task_type" not in sys.argv and "--recaption-task-type" not in sys.argv:
            args.recaption_task_type = preset["recaption_task_type"]

    # Resolve final defaults so a custom run (no preset) is still well-defined.
    if getattr(args, "task", None) is None:
        args.task = (
            TASK_SREF
            if normalize_recaption_task_type(args.recaption_task_type) == RECAPTION_TASK_TYPE_SREF
            else TASK_CREF_SREF
        )
    args.use_rope = resolve_use_rope(getattr(args, "use_rope", None), args.task)
    # Guard against running the wrong recaption prompt family on a weight.
    validate_recaption_task_type_for_task(
        args.task,
        args.recaption_task_type,
        str(getattr(args, "weight_preset", "") or ""),
    )


def run_batch(args: argparse.Namespace) -> None:
    """Legacy prompts.json + key batch runner, kept for internal compatibility."""
    os.chdir(WORKDIR)

    data_root = Path(args.data_root)
    prompts_json = Path(args.prompts_json) if args.prompts_json else data_root / "prompts.json"
    cref_dir = Path(args.cref_dir) if args.cref_dir else data_root / "cref"
    sref_dir = Path(args.sref_dir) if args.sref_dir else data_root / "sref"
    out_dir = Path(args.out_dir) if args.out_dir else data_root / "qwen3_style_guard_ckpt40000_selected"
    out_dir.mkdir(parents=True, exist_ok=True)
    recaption_json = Path(args.recaption_json) if args.recaption_json else out_dir / "recaption_prompts.json"
    structured_json = Path(args.structured_json) if args.structured_json else out_dir / "recaption_structured.json"
    basename_txt = Path(args.basename_txt) if args.basename_txt else out_dir / "selected_keys.txt"

    prompts = load_json(prompts_json)
    if not isinstance(prompts, dict):
        raise TypeError(f"prompts_json must contain a dict: {prompts_json}")
    prompts = {str(k): str(v) for k, v in prompts.items()}
    keys = resolve_keys(prompts, keys_arg=args.keys, key_txt=args.key_txt)
    if not keys:
        raise RuntimeError("no valid keys to process")
    basename_txt_for_keys(keys, basename_txt)

    print("=" * 80, flush=True)
    print(f"data_root       : {data_root}", flush=True)
    print(f"prompts_json    : {prompts_json}", flush=True)
    print(f"cref_dir        : {cref_dir}", flush=True)
    print(f"sref_dir        : {sref_dir}", flush=True)
    print(f"out_dir         : {out_dir}", flush=True)
    print(f"weight_preset   : {args.weight_preset or '<custom>'}", flush=True)
    print(f"keys            : {len(keys)} -> {keys[:10]}", flush=True)
    print(f"recaption_json  : {recaption_json}", flush=True)
    print(f"structured_json : {structured_json}", flush=True)
    print(f"recaption_task  : {normalize_recaption_task_type(args.recaption_task_type)}", flush=True)
    print(f"recaption_prompt: {get_recaption_prompt_template_name(args.recaption_task_type)}", flush=True)
    print(f"basename_txt    : {basename_txt}", flush=True)
    print(f"dit_path        : {args.dit_path}", flush=True)
    print(f"task            : {args.task}", flush=True)
    print(f"use_rope        : {bool(args.use_rope)}", flush=True)
    print("=" * 80, flush=True)

    if args.skip_recaption:
        recaptioned = load_json(recaption_json)
        if not isinstance(recaptioned, dict):
            raise TypeError(f"recaption_json must contain dict: {recaption_json}")
        final_prompts = {k: str(recaptioned[k]) for k in keys if k in recaptioned}
    else:
        if bool(args.recaption_subprocess) and not bool(args.recaption_only):
            # Isolate Qwen3-VL recaption from VGO loading so CUDA memory is fully
            # returned to the OS before the generator is constructed.
            run_recaption_subprocess(args, recaption_json, structured_json, basename_txt)
            recaptioned = load_json(recaption_json)
            if not isinstance(recaptioned, dict):
                raise TypeError(f"recaption_json must contain dict after subprocess: {recaption_json}")
            final_prompts = {k: str(recaptioned[k]) for k in keys if k in recaptioned}
        else:
            final_prompts = recaption_many(
                keys=keys,
                prompts=prompts,
                cref_dir=cref_dir,
                sref_dir=sref_dir,
                recaption_json=recaption_json,
                structured_json=structured_json,
                model_path=args.recaption_model_path,
                device=args.recaption_device,
                max_new_tokens=args.recaption_max_new_tokens,
                image_long_edge=args.recaption_image_long_edge,
                image_tokens=args.recaption_image_tokens,
                overwrite=bool(args.overwrite),
                recaption_task_type=str(args.recaption_task_type),
            )

    missing_prompt = [k for k in keys if k not in final_prompts]
    if missing_prompt:
        raise RuntimeError(f"missing final prompts for keys: {missing_prompt[:10]}")

    # Persist a compact prompt file suitable for multi_cref_eval.py if desired.
    dump_json({k: final_prompts[k] for k in keys}, recaption_json)

    if args.recaption_only:
        print("[DONE] recaption_only", flush=True)
        return

    generate_many(
        keys=keys,
        prompts=final_prompts,
        cref_dir=cref_dir,
        sref_dir=sref_dir,
        out_dir=out_dir,
        dit_path=args.dit_path,
        use_rope=bool(args.use_rope),
        ae_path=args.ae_path,
        qwenvl_path=args.qwenvl_path,
        device=args.generator_device,
        width=int(args.width),
        height=int(args.height),
        resolution_mode=str(args.resolution_mode),
        steps=int(args.steps),
        cfg=float(args.cfg),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
        recaption_task_type=str(args.recaption_task_type),
    )
    print("[DONE] all", flush=True)


def main() -> None:
    args = parse_args()
    apply_weight_preset(args)
    os.chdir(WORKDIR)
    if args.batch_mode:
        run_batch(args)
    else:
        run_single_demo(args)


if __name__ == "__main__":
    main()
