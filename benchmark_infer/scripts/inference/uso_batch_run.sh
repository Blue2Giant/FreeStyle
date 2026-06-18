#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -euo pipefail

# USO single-GPU batch inference for the cref+sref benchmark.
#   - output resolution is forced to 1024x1024
#   - input reference images are passed at their original pixel size (--no-preprocess-ref)
# Model weight paths are configured at the top of src/inference/USO/batch_simple_demo.py.

# Data root should be downloaded from our open-source benchmark.
sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
out_dir=$sref_root/uso

# After this finishes, the Qwen dual judge wrapper can evaluate the result with:
#   DATA_ROOT=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content RESULT_NAME=uso bash /data/FreeStyle/benchmark_infer/scripts/metrics/triplet_qwen_dual_judge.sh
# The metrics wrapper will adapt the standard cref/sref/uso layout to the new JSONL judge input.

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 "$REPO_ROOT/src/inference/USO/batch_simple_demo.py" \
  --input-dir "$sref_root" \
  --prompts-json "$sref_root/prompts.json" \
  --out-dir "$out_dir" \
  --overwrite \
  --width 1024 \
  --height 1024 \
  --no-preprocess-ref \
  --sref-only \
  --use-siglip
