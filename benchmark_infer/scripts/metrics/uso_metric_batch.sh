#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -euo pipefail
cd $REPO_ROOT/src/metrics/encoder
RUNNER_PY="$REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py"
GPUS="0"
#sref
MODELS=("ours" "newnew800_omnistyle" "newnew800_csgo" "newnew800_easyref" "newnew800_flux_9b" "newnew800_omnistyle" "uso" "gpt4o-edit" "gemini-edit")
MODELS=("gemini-edit" "seedream" "TeleStyle") #还没跑的
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content"

#cref sref
MODELS=("ours" "uso" "gpt4o-edit" "gemini-edit" "qwen-edit")
MODELS=("flux_9b_klein" "TeleStyle") #还没跑的
SREF_ROOT="/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content"


CONTENT_DIR="$SREF_ROOT/cref"
STYLE_DIR="$SREF_ROOT/sref"
SREF_PROMPT="$SREF_ROOT/prompts.json"
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

overwrite=1
num_procs=128

for MODEL in "${MODELS[@]}"; do
  RESULT_DIR="$SREF_ROOT/$MODEL"

  OUT_DINOV2_JSON="$RESULT_DIR/dinov2_out.json"
  OUT_CAS_JSON="$RESULT_DIR/cas_out.json"
  OUT_ONEIG_JSON="$RESULT_DIR/oneig_out.json"
  OUT_CLIPCAP_JSON="$RESULT_DIR/clipcap_out.json"
  OUT_ONEALIGN_JSON="$RESULT_DIR/onealign_out.json"
  OUT_CSD_JSON="$RESULT_DIR/csd_out.json"
  OUT_LAION_JSON="$RESULT_DIR/laion_scores.json"
  OUT_V25_AESTHETIC="$RESULT_DIR/v25_scores.json"

  echo "==== CSD ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder csd \
    --dir_a "$STYLE_DIR" \
    --dir_b "$RESULT_DIR" \
    --out_json "$OUT_CSD_JSON" \
    --model dummy \
    --csd_arch vit_base \
    --csd_model_path $CSD_MODEL_ONLY \
    --device cuda \
    --gpus "$GPUS" \
    --overwrite $overwrite

  echo "=== oneig ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder oneig \
    --dir_a "$STYLE_DIR" \
    --dir_b "$RESULT_DIR" \
    --model dummy \
    --oneig_model_path "$CSD_MODEL" \
    --oneig_se_model_path "$ONEIG_MODEL" \
    --oneig_clip_model_path "$VIT_L" \
    --out_json "$OUT_ONEIG_JSON" \
    --gpus "$GPUS" \
    --overwrite $overwrite

  echo "=== dinov2 ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder dinov2 \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$RESULT_DIR" \
    --model "$DINOV2_MODEL" \
    --out_json "$OUT_DINOV2_JSON" \
    --gpus "$GPUS" \
    --overwrite $overwrite

  echo "=== cas ($MODEL) ===="
  python3 "$RUNNER_PY" pair \
    --encoder cas \
    --dir_a "$CONTENT_DIR" \
    --dir_b "$RESULT_DIR" \
    --model "$CAS_MODEL" \
    --out_json "$OUT_CAS_JSON" \
    --gpus "$GPUS" \
    --overwrite $overwrite

  echo "=== clip t ($MODEL) ==="
  python3 "$RUNNER_PY" clip_t \
    --image_dir "$RESULT_DIR" \
    --prompt_json "$SREF_PROMPT" \
    --out_json "$OUT_CLIPCAP_JSON" \
    --model /mnt/jfs/model_zoo/openai/clip-vit-base-patch32 \
    --sim_metric cosine \
    --clipcap_text_mode first_sentence \
    --overwrite $overwrite

  echo "=== laion aesthetic ($MODEL) ==="
  python $REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py aesthetic \
    --backend laion \
    --image_dir $RESULT_DIR \
    --out_json $OUT_LAION_JSON \
    --laion_clip_model ViT-L-14 \
    --laion_clip_ckpt /mnt/jfs/model_zoo/open_clip/open_clip_model_ea4f182e96863ce2a27be5067cdb54d4.safetensors \
    --laion_linear_path ~/.cache/emb_reader/sa_0_4_vit_l_14_linear.pth \
    --device cuda \
    --gpus 0 \
    --overwrite $overwrite

  echo "==== aesthetic v25 ($MODEL) ===="
  python $REPO_ROOT/src/metrics/encoder/encoder_batch_runner.py aesthetic \
    --backend v25 \
    --image_dir $RESULT_DIR \
    --out_json $OUT_V25_AESTHETIC \
    --v25_encoder_model_name /mnt/jfs/model_zoo/siglip-so400m-patch14-384/ \
    --dtype bfloat16 \
    --device cuda \
    --gpus 0 \
    --overwrite $overwrite

  echo "====vlm style ($MODEL)===="
  style_dir="$SREF_ROOT/sref"
  result_dir="$SREF_ROOT/$MODEL"
  output_json_style_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_style_descrete.json"
  reason_json_style_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_style_reason_descrete.json"
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
    --num_procs $num_procs \
    --overwrite

  echo "====vlm content ($MODEL)===="
  content_dir="$SREF_ROOT/cref"
  result_dir="$SREF_ROOT/$MODEL"
  output_json_content_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_content_descrete.json"
  reason_json_content_discrete="$SREF_ROOT/$MODEL/qwen_resize_output_content_reason_descrete.json"
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
    --num_procs $num_procs \
    --overwrite

  echo "=== vlm insturction follow ($MODEL) ==="
  OUT_SCORE_JSON="$SREF_ROOT/$MODEL/follow_scores.json"
  OUT_REASON_JSON="$SREF_ROOT/$MODEL/follow_reasons.json"
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): base_url + served-model-name
  xingpeng_ip=http://YOUR_QWEN_VLM_HOST:22002/v1
  xingpeng_model=Qwen3-VL-30B-A3B-Instruct
  python3 $REPO_ROOT/src/metrics/vlm/edit_instruction_follow_dir.py \
    --image_dir $SREF_ROOT/$MODEL \
    --prompt_json $SREF_PROMPT \
    --out_score_json $OUT_SCORE_JSON \
    --out_reason_json $OUT_REASON_JSON \
    --base_url $xingpeng_ip \
    --model $xingpeng_model \
    --instruction_text_mode first_sentence \
    --num_procs $num_procs \
    --overwrite

  echo "=== triplet qwen dual judge ($MODEL) ==="
  content_dir="$SREF_ROOT/cref"
  style_dir="$SREF_ROOT/sref"
  result_dir="$SREF_ROOT/$MODEL"
  output_json_content="$SREF_ROOT/$MODEL/qwen_reject_cref.json"
  output_json_style="$SREF_ROOT/$MODEL/qwen_reject_sref.json"
  python3 $REPO_ROOT/src/metrics/vlm/triplet_qwen_dual_judge.py \
      --content_dir "$content_dir" \
      --style_dir "$style_dir" \
      --result_dir "$result_dir" \
      --output_content_json $output_json_content \
      --output_style_json $output_json_style \
# Qwen3-VL VLM service you deploy yourself (vLLM, see README): "<served-model-name>@<base_url>"
      --endpoint "Qwen3-VL-30B-A3B-Instruct@http://YOUR_QWEN_VLM_HOST:22002/v1" \
      --procs_per_endpoint $num_procs \
      --overwrite
done
    
