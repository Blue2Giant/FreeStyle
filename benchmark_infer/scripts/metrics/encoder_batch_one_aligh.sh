#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

#!/bin/bash
set -euo pipefail
cd $REPO_ROOT/src/metrics/encoder
RUNNER_PY="$REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py"
GPUS="0"

CONTENT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
STYLE_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
RESULT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
SREF_PROMPT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json"
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture"
#模型权重位置配置
DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"
CSD_MODEL="/data/Sref_Cref/CSD/pretrainedmodels/vit-b-300ep.pth.tar"
CLIPCAP_MODEL="/mnt/jfs/model_zoo/clip-vit-large-patch14"
ONEALIGN_MODEL="/mnt/jfs/model_zoo/one-align"
ONEALIGN_TASK="aesthetics"

#输出的标注路径配置
OUT_DINOV2_JSON="$SREF_ROOT/dinov2_out.json"
OUT_CAS_JSON="$SREF_ROOT/cas_out.json"
OUT_ONEIG_JSON="$SREF_ROOT/oneig_out.json"
OUT_CLIPCAP_JSON="$SREF_ROOT/clipcap_out.json"
OUT_ONEALIGN_JSON="$SREF_ROOT/onealign_out.json"
OUT_CSD_JSON="$SREF_ROOT/csd_out.json"
echo "=== onealign aesthetics ==="
python3 "$RUNNER_PY" onealign \
  --image_dir "$RESULT_DIR" \
  --out_json "$OUT_ONEALIGN_JSON" \
  --model "$ONEALIGN_MODEL" \
  --task "$ONEALIGN_TASK" \
  --gpus "$GPUS"