#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CSGO_DIR="$REPO_ROOT/src/inference/CSGO"
export PYTHONPATH="$PYTHONPATH:$CSGO_DIR"
cd "$CSGO_DIR"

# Data root should be downloaded from our open-source benchmark.
sref_dir="/mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content/"
python "$CSGO_DIR/infer_csgo_ljh_batch.py" \
  --cref_dir $sref_dir/cref \
  --sref_dir $sref_dir/sref \
  --out_dir $sref_dir/csgo \
  --skip_existing
