#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
style_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
output_json_style_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_style_descrete.json"
reason_json_style_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_style_reason_descrete.json"
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): base_url + served-model-name
xingpeng_ip=http://YOUR_QWEN_VLM_HOST:22002/v1
xingpeng_model=Qwen3-VL-30B-A3B-Instruct
python3 $REPO_ROOT/src/metrics/vlm/style_similarity_dir.py \
  --style_dir $style_dir \
  --output_dir $result_dir \
  --out_score_json $output_json_style_discrete \
  --out_reason_json $reason_json_style_discrete \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --num_procs 32 \
  --overwrite