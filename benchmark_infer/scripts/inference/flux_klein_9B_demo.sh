#!/usr/bin/env bash
sref_root=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content

python $REPO_ROOT/src/inference/flux_klein_9B.py \
  --prompts_json /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/prompts.json \
  --cref_dir $sref_root/cref \
  --sref_dir $sref_root/sref \
  --out_dir $sref_root/flux-klein-9b_1024x1024 \
  --model_name /mnt/jfs/model_zoo/FLUX.2-klein-9B/ \
  --steps 4 \
  --guidance_scale 1.0 \
  --gpus 0 \
  --output_resolution 1024x1024 \
  --save_jsonl
