#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
content_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/cref"
style_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/sref"
result_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit"
output_json_content="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit/qwen_reject_cref.json"
output_json_style="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/qwen-edit/qwen_reject_sref.json" 
python3 $REPO_ROOT/src/metrics/vlm/triplet_qwen_dual_judge.py \
    --content_dir "$content_dir" \
    --style_dir "$style_dir" \
    --result_dir "$result_dir" \
    --output_content_json $output_json_content \
    --output_style_json $output_json_style \
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): "<served-model-name>@<base_url>"
    --endpoint "Qwen3-VL-30B-A3B-Instruct@http://YOUR_QWEN_VLM_HOST:22002/v1" \
    --procs_per_endpoint 32 \
    --overwrite