#!/usr/bin/env bash
# Minimal CRef+SRef inference demo: two images + one prompt -> generated image.
#
# Usage:
#   conda activate Sref
#   cd model_infer
#   bash run_cref_sref_infer.sh
#
# Weights: download the checkpoint repo once and point FREESTYLE_CKPT_ROOT at it.
#   huggingface-cli download Blue2Giant/FreeStyle_Checkpoint --local-dir ./checkpoints
# Layout (one model.safetensors per preset):
#   checkpoints/freestyle-cref-sref-50000-rope/model.safetensors      <- cref_sref_rope_50000
#   checkpoints/freestyle-cref-sref-40000-no-rope/model.safetensors   <- cref_sref_40000
#   checkpoints/freestyle-cref-sref-36000-no-rope/model.safetensors   <- cref_sref_36000_no_rope
# FREESTYLE_CKPT_ROOT defaults to ./checkpoints, so if you downloaded there you
# can leave the export below commented out.
#
# Choose one --weight_preset below (RoPE is selected automatically by the preset):
#   cref_sref_rope_50000      : RoPE-trained weight  (runs ImageGeneratorRopeFA)
#   cref_sref_40000           : no-RoPE 40000 weight
#   cref_sref_36000_no_rope   : no-RoPE 36000 weight
#
# Input order is important:
#   1) assets/02-cref.png  : content/layout reference image
#   2) assets/02-sref.png  : style reference image
#   3) Chinese prompt below
#
# Common edits:
#   --recaption_task_type: identity_style for CRef+SRef, style_transfer for style-transfer
#   --out_dir: output directory
#   --steps: denoising steps; use 8 for a quick smoke test, 28 for normal inference

set -e
cd "$(dirname "$0")"

# export FREESTYLE_CKPT_ROOT=./checkpoints   # uncomment/edit if weights live elsewhere

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
VGO_STREAM_LOAD_SAFETENSORS=1 \
VGO_STREAM_LOAD_DTYPE=bfloat16 \
VGO_STREAM_LOAD_DEVICE=cuda:0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  assets/02-cref.png \
  assets/02-sref.png \
  '猫咪在一个壁炉前面趴着，迁移图2的风格到图1上' \
  --weight_preset cref_sref_rope_50000 \
  --out_dir outputs/cref_sref_rope_50000_demo \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
