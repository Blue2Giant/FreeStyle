#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd $REPO_ROOT/src/metrics/encoder
python $REPO_ROOT/src/metrics/encoder/true_oneig.py \
  --image_a $REPO_ROOT/assets/stylized.png\
  --image_b $REPO_ROOT/assets/style.webp \
  --model_path $REPO_ROOT/logs/csd.pth  \
  --clip_model_path $REPO_ROOT/logs/ViT-L-14.pt  \
  --se_model_path /mnt/jfs/model_zoo/OneIG-StyleEncoder/ \
  --device cuda:0