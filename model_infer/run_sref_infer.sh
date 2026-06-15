#!/usr/bin/env bash
# Minimal style-transfer demo: input two images + one prompt, then output
# the generated image and the intermediate Qwen3-VL recaption result.
#
# Usage:
#   conda activate Sref
#   cd /data/vgo/opensource_cref_sref_core_infer_0615
#   bash run_sref_infer.sh
#
# Input order is important:
#   1) assets/00-cref.jpg  : content/layout reference image
#   2) assets/00-sref.jpg  : style reference image
#   3) Chinese prompt below
#
# Style-transfer output automatically keeps the same resolution as assets/00-cref.jpg.
# For normal CRef+SRef/SRef tasks, edit --width/--height; default is 1024x1024.
#
# To try another SRef weight, change --weight_preset to sref_14000.
# Current setting uses the 12000 SRef checkpoint:
#   /mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors

set -e
cd "$(dirname "$0")"

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
VGO_STREAM_LOAD_SAFETENSORS=1 \
VGO_STREAM_LOAD_DTYPE=bfloat16 \
VGO_STREAM_LOAD_DEVICE=cuda:0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
  --weight_preset sref_12000 \
  --out_dir outputs/sref_12000_style_transfer_demo \
  --recaption_task_type style_transfer \
  --width 1024 \
  --height 1024 \
  --steps 8 \
  --cfg 8 \
  --seed 42 \
  --overwrite
