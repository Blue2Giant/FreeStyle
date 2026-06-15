#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTENT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
STYLE_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
RESULT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
SREF_PROMPT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json"
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture"
OUT_SCORE_JSON="$SREF_ROOT/follow_scores.json"
OUT_REASON_JSON="$SREF_ROOT/follow_reasons.json"
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): base_url + served-model-name
xingpeng_ip=http://YOUR_QWEN_VLM_HOST:22002/v1
xingpeng_model=Qwen3-VL-30B-A3B-Instruct
python3 $REPO_ROOT/src/metrics/vlm/edit_instruction_follow_dir.py \
  --image_dir $RESULT_DIR \
  --prompt_json $SREF_PROMPT \
  --out_score_json $OUT_SCORE_JSON \
  --out_reason_json $OUT_REASON_JSON \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --api_key YOUR_KEY \
  --instruction_text_mode first_sentence \
  --num_procs 128 \
  --overwrite