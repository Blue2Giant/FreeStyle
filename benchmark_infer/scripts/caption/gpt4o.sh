#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export OPENAI_API_KEY=YOUR_API_KEY
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content
sref_dir=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
python $REPO_ROOT/src/caption/gpt-4o-haoling_core_batch.py \
  --prompts_json $sref_dir/prompts.json \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/gpt4o-edit \
  --model gpt-4o-all \
  --base_url https://YOUR_OPENAI_COMPAT_ENDPOINT/v1 \
  --num_procs 16