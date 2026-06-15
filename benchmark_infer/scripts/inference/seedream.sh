#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export SEEDREAM_API_KEY=YOUR_API_KEY

#cref dir and cref_sref_dir should be download from our opensource benchmark
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
cref_sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
python $REPO_ROOT/src/inference/seeddream_batch.py \
  --cref_dir $cref_sref_dir/cref \
  --sref_dir $cref_sref_dir/sref \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/prompts.json \
  --out_dir $cref_sref_dir/seedream_1024x1024 \
  --resolution 2048x2048 \
  --save_resolution 1024x1024 \
  --workers 2
