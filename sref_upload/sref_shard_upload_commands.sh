#!/usr/bin/env bash
# Manual commands for SREF tar-shard upload.
# Usage:
#   source /data/vgo/.codex_tmp/sref_shard_upload_commands.sh
#   init_sref_upload_env
#   prepare_sref_staging
#   delete_old_sref_once       # only once before first shard upload; marker prevents repeated deletion
#   upload_sref_batch 0 3      # 00-3f
#   upload_sref_batch 4 7      # 40-7f
#   upload_sref_batch 8 b      # 80-bf
#   upload_sref_batch c f      # c0-ff
#
# If interrupted, re-run the same upload_sref_batch command. upload_large_folder resumes using
# /mnt/jfs/gemini_sref_final_hf_upload_root/.cache/huggingface/upload.

export SREF_SOURCE_ROOT=${SREF_SOURCE_ROOT:-/mnt/jfs/gemini_sref_export}
export SREF_SHARD_ROOT=${SREF_SHARD_ROOT:-/mnt/jfs/gemini_sref_export_shards}
export SREF_STAGING_ROOT=${SREF_STAGING_ROOT:-/mnt/jfs/gemini_sref_final_hf_upload_root}
export SREF_REPO_ID=${SREF_REPO_ID:-Blue2Giant/FreeStyle_StyleTransfer}
export SREF_REPO_TYPE=${SREF_REPO_TYPE:-dataset}
export SREF_WORKERS=${SREF_WORKERS:-8}

init_sref_upload_env() {
  if [ -z "${HF_TOKEN:-}" ]; then
    echo '[auth] HF_TOKEN is missing. Input token silently:' >&2
    read -r -s HF_TOKEN
    export HF_TOKEN
  fi
  eval "$(curl -s http://deploy.i.shaipower.com/httpproxy)"
  export HF_XET_HIGH_PERFORMANCE=1
  mkdir -p /mnt/jfs/vgo_hf_upload_logs/sref_shards
  echo "[env] repo=$SREF_REPO_ID workers=$SREF_WORKERS staging=$SREF_STAGING_ROOT"
}

create_sref_shards_all() {
  init_sref_upload_env
  python /data/vgo/.codex_tmp/create_sref_image_shards.py \
    --root "$SREF_SOURCE_ROOT" \
    --out "$SREF_SHARD_ROOT"
}

create_sref_shards_range() {
  # Example: create_sref_shards_range 00 3f
  local start_hex=$1
  local end_hex=$2
  local prefixes
  prefixes=$(python3 - "$start_hex" "$end_hex" <<'PY'
import sys
s=int(sys.argv[1],16); e=int(sys.argv[2],16)
print(','.join(f'{i:02x}' for i in range(s,e+1)))
PY
)
  init_sref_upload_env
  python /data/vgo/.codex_tmp/create_sref_image_shards.py \
    --root "$SREF_SOURCE_ROOT" \
    --out "$SREF_SHARD_ROOT" \
    --prefixes "$prefixes"
}

prepare_sref_staging() {
  init_sref_upload_env
  python3 - <<'PY'
import importlib.util
from pathlib import Path
import os
spec = importlib.util.spec_from_file_location('u', '/data/vgo/.codex_tmp/upload_sref_shards_large_folder.py')
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
u.prepare_staging(Path(os.environ['SREF_SHARD_ROOT']), Path(os.environ['SREF_STAGING_ROOT']), Path(os.environ['SREF_SOURCE_ROOT']))
PY
}

delete_old_sref_once() {
  # Deletes remote sref/ once. Marker file prevents repeated deletion on resume.
  init_sref_upload_env
  python3 - <<'PY'
import importlib.util, os
from pathlib import Path
from huggingface_hub import HfApi
spec = importlib.util.spec_from_file_location('u', '/data/vgo/.codex_tmp/upload_sref_shards_large_folder.py')
u = importlib.util.module_from_spec(spec); spec.loader.exec_module(u)
api = HfApi(token=os.environ['HF_TOKEN'])
u.delete_remote_sref_once(api, os.environ['SREF_REPO_ID'], os.environ['SREF_REPO_TYPE'], Path(os.environ['SREF_STAGING_ROOT']) / '.remote_sref_deleted_ok')
PY
}

upload_sref_all_resumable() {
  # One-shot resumable upload of all staged files. Safe to rerun after interruption.
  init_sref_upload_env
  python /data/vgo/.codex_tmp/upload_sref_shards_large_folder.py \
    --folder "$SREF_SHARD_ROOT" \
    --source-root "$SREF_SOURCE_ROOT" \
    --repo-id "$SREF_REPO_ID" \
    --workers "$SREF_WORKERS"
}

upload_sref_batch() {
  # Upload one hex-nibble batch of tar shards from staging root.
  # Examples:
  #   upload_sref_batch 0 3   # images_00.tar through images_3f.tar
  #   upload_sref_batch 4 7
  #   upload_sref_batch 8 b
  #   upload_sref_batch c f
  local start_nibble=$1
  local end_nibble=$2
  init_sref_upload_env
  prepare_sref_staging
  export BATCH_START_NIBBLE="$start_nibble"
  export BATCH_END_NIBBLE="$end_nibble"
  LOG=/mnt/jfs/vgo_hf_upload_logs/sref_shards/manual_batch_${start_nibble}_${end_nibble}_$(date +%Y%m%d_%H%M%S).log
  echo "[upload_batch] $start_nibble-$end_nibble log=$LOG"
  python3 - <<'PY' 2>&1 | tee -a "$LOG"
import os
from huggingface_hub import HfApi
repo_id=os.environ['SREF_REPO_ID']
repo_type=os.environ['SREF_REPO_TYPE']
folder=os.environ['SREF_STAGING_ROOT']
workers=int(os.environ.get('SREF_WORKERS','8'))
s=int(os.environ['BATCH_START_NIBBLE'],16)
e=int(os.environ['BATCH_END_NIBBLE'],16)
patterns = [
    'sref/README.md',
    'sref/export_summary.json',
    'sref/pairs.csv',
    'sref/metadata.jsonl',
    'sref/shards_manifest.jsonl',
    'sref/shards_summary.json',
]
for i in range(s, e+1):
    patterns.append(f'sref/image_tar_shards/images_{i:x}*.tar')
print('[patterns]')
for p in patterns: print(' ', p)
api=HfApi(token=os.environ['HF_TOKEN'])
print('[auth]', api.whoami(token=os.environ['HF_TOKEN']).get('name'))
api.upload_large_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=folder,
    allow_patterns=patterns,
    ignore_patterns=['**/.cache/**', '**/*.tmp', '**/.remote_sref_deleted_ok'],
    num_workers=workers,
    print_report=True,
    print_report_every=60,
)
print('[done] batch uploaded', os.environ['BATCH_START_NIBBLE'], os.environ['BATCH_END_NIBBLE'])
PY
}

monitor_sref_upload_tmux() {
  tmux capture-pane -pt cpu_vgo_2-27:1.0 -S -220 | tail -n 160
}

monitor_sref_shards() {
  echo '[local shard status]'
  find "$SREF_SHARD_ROOT/image_tar_shards" -maxdepth 1 -name 'images_*.tar' -type f | wc -l
  du -sh "$SREF_SHARD_ROOT/image_tar_shards" 2>/dev/null || true
  echo '[staging/cache status]'
  find "$SREF_STAGING_ROOT" -type f | wc -l 2>/dev/null || true
  du -sh "$SREF_STAGING_ROOT" 2>/dev/null || true
  find "$SREF_STAGING_ROOT/.cache/huggingface/upload" -type f 2>/dev/null | wc -l || true
}

echo '[loaded] SREF upload helper functions:'
echo '  init_sref_upload_env'
echo '  create_sref_shards_all | create_sref_shards_range 00 3f'
echo '  prepare_sref_staging'
echo '  delete_old_sref_once'
echo '  upload_sref_all_resumable'
echo '  upload_sref_batch 0 3 | upload_sref_batch 4 7 | upload_sref_batch 8 b | upload_sref_batch c f'
echo '  monitor_sref_upload_tmux | monitor_sref_shards'