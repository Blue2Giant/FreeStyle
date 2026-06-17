#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ref_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
output_json_content_leakage_score="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_leakage_score.json"
output_json_content_leakage_reason="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_leakage_reason.json"
ip=http://YOUR_QWEN_VLM_HOST:22002/v1
model=Qwen3-VL-30B-A3B-Instruct

python3 $REPO_ROOT/src/metrics/vlm/content_leakage_dir.py \
  --ref_dir $ref_dir \
  --output_dir $result_dir \
  --out_score_json $output_json_content_leakage_score \
  --out_reason_json $output_json_content_leakage_reason \
  --base_url $ip \
  --model $model \
  --num_procs 32 \
  --overwrite
