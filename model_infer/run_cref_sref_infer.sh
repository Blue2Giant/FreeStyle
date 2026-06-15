#!/usr/bin/env bash
# Minimal CRef+SRef inference example.
#
# Usage:
#   1) conda activate Sref
#   2) cd /data/vgo/opensource_cref_sref_core_infer_0615
#   3) bash run_cref_sref_infer.sh
#
# IMPORTANT about RoPE:
#   RoPE is NOT hard-coded in Python. It is enabled by the YAML config.
#   The config must contain engine_config.pipe.dit.rope_fa.enabled: true.
#
# Choose one of these presets below:
#   cref_sref_rope_50000      : RoPE weight + RoPE config
#   cref_sref_36000_no_rope   : no-RoPE weight + normal config
#   cref_sref_40000           : normal 40000 weight + normal config
#
# Common edits:
#   --weight_preset: choose one preset above
#   --recaption_task_type: identity_style for CRef+SRef, style_transfer for style-transfer
#   --data_root: directory containing prompts.json, cref/, sref/
#   --out_dir: output directory
#   --keys: comma-separated sample keys; delete this line to run all prompts.json keys
#   --steps: denoising steps; use 8 for a quick smoke test, 28 for normal inference

set -e
cd "$(dirname "$0")"

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset cref_sref_rope_50000 \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content \
  --out_dir /data/benchmark_metrics/cref_sref_infer_rope_50000 \
  --keys 'football__sticker_figure' \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
