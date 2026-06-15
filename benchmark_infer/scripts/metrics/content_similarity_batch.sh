#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
output_json_content_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_descrete.json"
reason_json_content_discrete="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit/qwen_resize_output_content_reason_descrete.json"
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): base_url + served-model-name
xingpeng_ip=http://YOUR_QWEN_VLM_HOST:22002/v1
xingpeng_model=Qwen3-VL-30B-A3B-Instruct
python3 $REPO_ROOT/src/metrics/vlm/content_similarity_dir.py \
  --content_dir $content_dir \
  --output_dir $result_dir \
  --out_json $output_json_content_discrete \
  --out_reason_json $reason_json_content_discrete \
  --base_url $xingpeng_ip \
  --model $xingpeng_model \
  --num_procs 64 \
  --overwrite