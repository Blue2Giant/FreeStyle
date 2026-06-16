#!/usr/bin/env bash
# ============================================================================
# SREF Tar Shard Upload to Blue2Giant/FreeStyle_StyleTransfer
# ============================================================================
# 用法:
#   1. 先在有 /mnt/jfs 挂载的远程 pod 里执行（通过 brainctl / tmux）
#   2. 输入 HF token（输入时不会回显）
#   3. 上传会自动分批进行，中断后重新运行此脚本即可续传
#
# 监控:
#   tmux attach -t cpu_temp_3    # 实时查看进度，Ctrl-B D 退出不中断
#
# 注意:
#   - 本脚本使用分批上传（upload_sref_batch），每次上传一个 hex 范围
#   - 4 个 batch 依次执行: 00-0f, 10-1f, 20-2f, 30-3f
#   - 每批会自动整理 staging（hardlink，几乎瞬间完成）
#   - 如果某批中断，重新运行脚本会自动续传该批
# ============================================================================

set -euo pipefail

# ---------- 路径配置 ----------
export SREF_SOURCE_ROOT=${SREF_SOURCE_ROOT:-/mnt/jfs/gemini_sref_export}
export SREF_SHARD_ROOT=${SREF_SHARD_ROOT:-/mnt/jfs/gemini_sref_export_shards}
export SREF_STAGING_ROOT=${SREF_STAGING_ROOT:-/mnt/jfs/gemini_sref_final_hf_upload_root}
export SREF_REPO_ID=${SREF_REPO_ID:-Blue2Giant/FreeStyle_StyleTransfer}
export SREF_REPO_TYPE=${SREF_REPO_TYPE:-dataset}
export SREF_WORKERS=${SREF_WORKERS:-8}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="/mnt/jfs/vgo_hf_upload_logs/sref_shards"
mkdir -p "$LOG_DIR"

# ---------- 函数定义 ----------

init_env() {
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "[auth] HF_TOKEN is missing. Input token silently:" >&2
        read -r -s HF_TOKEN
        export HF_TOKEN
    fi

    eval "$(curl -s http://deploy.i.shaipower.com/httpproxy)"
    export HF_XET_HIGH_PERFORMANCE=1

    python3 -c "
from huggingface_hub import HfApi
api = HfApi()
print('[auth] user:', api.whoami().get('name'))
" 2>&1

    echo "[env] repo=$SREF_REPO_ID workers=$SREF_WORKERS staging=$SREF_STAGING_ROOT"
}

check_shards() {
    echo "[shard] checking local shards..."
    local count
    count=$(ls "$SREF_SHARD_ROOT/image_tar_shards/" 2>/dev/null | wc -l)
    echo "[shard] tar shards: $count / 256"
    if [ "$count" -ne 256 ]; then
        echo "[shard] WARNING: expected 256 shards, found $count"
    fi
    du -sh "$SREF_SHARD_ROOT/image_tar_shards" 2>/dev/null || true
}

upload_batch() {
    local start_nibble=$1
    local end_nibble=$2
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local log="$LOG_DIR/batch_${start_nibble}_${end_nibble}_${ts}.log"

    echo "[batch] uploading shards ${start_nibble}-${end_nibble}... (log: $log)"

    source /data/vgo/.codex_tmp/sref_shard_upload_commands.sh >/dev/null 2>&1
    upload_sref_batch "$start_nibble" "$end_nibble" 2>&1 | tee "$log"

    echo "[batch] done: ${start_nibble}-${end_nibble}"
}

run_all_batches() {
    echo "[batch] starting all 4 batches..."
    upload_batch 0 3   # images_00.tar ~ images_3f.tar
    upload_batch 4 7   # images_40.tar ~ images_7f.tar
    upload_batch 8 b   # images_80.tar ~ images_bf.tar
    upload_batch c f   # images_c0.tar ~ images_ff.tar
    echo "[batch] ALL BATCHES COMPLETE"
}

# ---------- 主流程 ----------

echo "============================================"
echo "  SREF Shard Upload -> $SREF_REPO_ID"
echo "============================================"
echo ""

check_shards
echo ""
init_env
echo ""
run_all_batches
