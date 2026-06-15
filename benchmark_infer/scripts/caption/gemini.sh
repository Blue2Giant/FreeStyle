#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export GEMINI_API_KEY=YOUR_API_KEY
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content #it should be download from our opensource benchmark
python $REPO_ROOT/src/caption/gemini_image_min_batch.py \
  --prompts_json $sref_dir/prompts.json \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/gemini-edit \
  --model_id gemini-2.5-flash-image-native \
  --num_procs 8 \
  --num_generate 3 \
cref_sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content #it should be download from our opensource benchmark
python $REPO_ROOT/src/caption/gemini_image_min_batch.py \
  --prompts_json $cref_sref_dir/prompts.json \
  --cref_dir $cref_sref_dir/cref \
  --sref_dir $cref_sref_dir/sref \
  --out_dir $cref_sref_dir/gemini-edit-new \
  --model_id gemini-2.5-flash-image-native \
  --overwrite \
  --num_procs 8