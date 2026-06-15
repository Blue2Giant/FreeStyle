#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -euo pipefail
cd $REPO_ROOT/src/metrics/encoder
RUNNER_PY="$REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py"
GPUS="0"

CONTENT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/cref"
STYLE_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/sref"
RESULT_DIR="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
SREF_PROMPT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/prompts.json"
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_bench_cref_sref_ture/qwen-edit"
#模型权重位置配置
DINOV2_MODEL="/mnt/jfs/model_zoo/dinov2-with-registers-large"
CAS_MODEL="/mnt/jfs/model_zoo/dinov2-base"
ONEIG_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder"
CSD_MODEL="/mnt/jfs/model_zoo/OneIG-StyleEncoder/csd.pth"
CSD_MODEL_ONLY="/mnt/jfs/model_zoo/OneIG-StyleEncoder/vit-b-300ep.pth.tar"
VIT_L="/mnt/jfs/model_zoo/OneIG-StyleEncoder/ViT-L-14.pt"
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
OUT_LAION_JSON="$SREF_ROOT/laion_scores.json"
OUT_V25_AESTHETIC="$SREF_ROOT/v25_scores.json"
overwrite=1
#风格一致性

# echo "==== CSD ===="
# python3 "$RUNNER_PY" pair \
#   --encoder csd \
#   --dir_a "$STYLE_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --out_json "$OUT_CSD_JSON" \
#   --model dummy \
#   --csd_arch vit_base \
#   --csd_model_path $CSD_MODEL_ONLY \
#   --device cuda \
#   --gpus "$GPUS" \
#   --overwrite $overwrite

# echo "=== oneig ===="
# python3 "$RUNNER_PY" pair \
#   --encoder oneig \
#   --dir_a "$STYLE_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model dummy \
#   --oneig_model_path "$CSD_MODEL" \
#   --oneig_se_model_path "$ONEIG_MODEL" \
#   --oneig_clip_model_path "$VIT_L" \
#   --out_json "$OUT_ONEIG_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite

# #内容一致性
# echo "=== dinov2 ===="
# python3 "$RUNNER_PY" pair \
#   --encoder dinov2 \
#   --dir_a "$CONTENT_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model "$DINOV2_MODEL" \
#   --out_json "$OUT_DINOV2_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite
# echo "=== cas ===="
# python3 "$RUNNER_PY" pair \
#   --encoder cas \
#   --dir_a "$CONTENT_DIR" \
#   --dir_b "$RESULT_DIR" \
#   --model "$CAS_MODEL" \
#   --out_json "$OUT_CAS_JSON" \
#   --gpus "$GPUS" \
#   --overwrite $overwrite

#指令遵循
# echo "=== clip cap ==="
# python3 "$RUNNER_PY" clip_cap \
#   --image_dir "$RESULT_DIR" \
#   --prompt_json "$SREF_PROMPT" \
#   --out_json "$OUT_CLIPCAP_JSON" \
#   --model "$CLIPCAP_MODEL" \
#   --gpus "$GPUS" \
#   --clipcap_text_mode first_sentence \
#   --overwrite $overwrite

# echo "=== clip-t ==="
# python3 "$RUNNER_PY" clip_t \
#   --image_dir "$RESULT_DIR" \
#   --prompt_json "$SREF_PROMPT" \
#   --out_json "$OUT_CLIPCAP_JSON" \
#   --model /mnt/jfs/model_zoo/openai/clip-vit-base-patch32 \
#   --sim_metric cosine \
#   --clipcap_text_mode first_sentence \
#   --overwrite $overwrite

#美学评分
# echo "=== laion aesthetic ==="
# python $REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py aesthetic \
#   --backend laion \
#   --image_dir $RESULT_DIR \
#   --out_json $OUT_LAION_JSON \
#   --laion_clip_model ViT-L-14 \
#   --laion_clip_ckpt /mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors \
#   --laion_linear_path ~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth \
#   --device cuda \
#   --gpus 0 \
#   --overwrite $overwrite

echo "==== aesthetic v25 ===="
python $REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py aesthetic \
  --backend v25 \
  --image_dir $RESULT_DIR \
  --out_json $OUT_V25_AESTHETIC \
  --v25_encoder_model_name /mnt/jfs/model_zoo/siglip-so400m-patch14-384/ \
  --dtype bfloat16 \
  --device cuda \
  --gpus 0 \
  --overwrite $overwrite
