#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
set -euo pipefail

# Qwen3-VL dual judge for triplet data.
#
# Native inputs of src/metrics/vlm/triplet_qwen_dual_judge.py:
#   1) --root mode:
#      <TRIPLET_ROOT>/style_and_content/        generated/main images
#      <TRIPLET_ROOT>/content_1/, content_2/    content references
#      <TRIPLET_ROOT>/style_1/, style_2/        style references
#   2) --input_jsonl mode:
#      each line is JSON with style_and_content + content_* + style_* image paths.
#
# For the open-source benchmark layout used by inference scripts:
#      <DATA_ROOT>/cref, <DATA_ROOT>/sref, <DATA_ROOT>/<RESULT_NAME>
# this wrapper auto-builds a temporary jsonl with content_1/style_1.

DATA_ROOT="${DATA_ROOT:-/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content}"
RESULT_NAME="${RESULT_NAME:-uso}"
CONTENT_DIR="${CONTENT_DIR:-$DATA_ROOT/cref}"
STYLE_DIR="${STYLE_DIR:-$DATA_ROOT/sref}"
RESULT_DIR="${RESULT_DIR:-$DATA_ROOT/$RESULT_NAME}"

# Optional native inputs. If INPUT_JSONL is set, it wins. Otherwise TRIPLET_ROOT wins.
TRIPLET_ROOT="${TRIPLET_ROOT:-}"
INPUT_JSONL="${INPUT_JSONL:-}"

OUT_DIR="${OUT_DIR:-$RESULT_DIR/qwen_dual_judge}"
mkdir -p "$OUT_DIR"

OUT_ALL="${OUT_ALL:-$OUT_DIR/all.json}"
OUT_POS="${OUT_POS:-$OUT_DIR/pos.json}"
OUT_NEG="${OUT_NEG:-$OUT_DIR/neg.json}"
OUT_DETAIL="${OUT_DETAIL:-$OUT_DIR/detail.json}"

QWEN_MODEL="${QWEN_MODEL:-Qwen3-VL-30B-A3B-Instruct}"
QWEN_BASE_URL="${QWEN_BASE_URL:-http://YOUR_QWEN_VLM_HOST:22002/v1}"
PROCS_PER_ENDPOINT="${PROCS_PER_ENDPOINT:-32}"
NUM_PROCS="${NUM_PROCS:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"   # <=0 means full set

# Legacy/simple layout -> jsonl adapter.
if [[ -z "$INPUT_JSONL" && -z "$TRIPLET_ROOT" ]]; then
  INPUT_JSONL="${GENERATED_INPUT_JSONL:-$OUT_DIR/input_pairs.jsonl}"
  python3 - "$CONTENT_DIR" "$STYLE_DIR" "$RESULT_DIR" "$INPUT_JSONL" <<'EOF_PY'
import json
import sys
from pathlib import Path

content_dir, style_dir, result_dir, out_jsonl = sys.argv[1:5]
exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

def image_map(d):
    out = {}
    root = Path(d)
    if not root.is_dir():
        raise RuntimeError(f"image dir not found: {d}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.setdefault(p.stem, str(p))
    return out

content = image_map(content_dir)
style = image_map(style_dir)
result = image_map(result_dir)
keys = sorted(set(content) & set(style) & set(result))
Path(out_jsonl).parent.mkdir(parents=True, exist_ok=True)
with open(out_jsonl, "w", encoding="utf-8") as f:
    for k in keys:
        f.write(json.dumps({
            "style_and_content": result[k],
            "content_1": content[k],
            "style_1": style[k],
        }, ensure_ascii=False) + "\n")
print(f"[jsonl] wrote {len(keys)} samples -> {out_jsonl}", flush=True)
if not keys:
    raise RuntimeError("no common image basename among CONTENT_DIR/STYLE_DIR/RESULT_DIR")
EOF_PY
fi

PY_ARGS=(
  "$REPO_ROOT/src/metrics/vlm/triplet_qwen_dual_judge.py"
  --out_all "$OUT_ALL"
  --out_pos "$OUT_POS"
  --out_neg "$OUT_NEG"
  --out_detail "$OUT_DETAIL"
  --num_samples "$NUM_SAMPLES"
  --content_judge_times "${CONTENT_JUDGE_TIMES:-3}"
  --content_min_true "${CONTENT_MIN_TRUE:-2}"
  --content_ratio "${CONTENT_RATIO:-0.66}"
  --content_conf_thr "${CONTENT_CONF_THR:-0.5}"
  --style_judge_times "${STYLE_JUDGE_TIMES:-3}"
  --style_min_true "${STYLE_MIN_TRUE:-2}"
  --style_ratio "${STYLE_RATIO:-0.66}"
  --style_conf_thr "${STYLE_CONF_THR:-0.5}"
)

if (( PROCS_PER_ENDPOINT > 0 )); then
  PY_ARGS+=(--endpoint "${QWEN_MODEL}@${QWEN_BASE_URL}" --procs_per_endpoint "$PROCS_PER_ENDPOINT")
else
  PY_ARGS+=(--model "$QWEN_MODEL" --base_url "$QWEN_BASE_URL")
  if (( NUM_PROCS > 0 )); then
    PY_ARGS+=(--num_procs "$NUM_PROCS")
  fi
fi

if [[ -n "$INPUT_JSONL" ]]; then
  PY_ARGS+=(--input_jsonl "$INPUT_JSONL")
else
  PY_ARGS+=(--root "$TRIPLET_ROOT")
fi

if [[ "${STYLE_REPEAT_ONLY_STYLE1:-0}" == "1" ]]; then
  PY_ARGS+=(--style_repeat_only_style1)
fi

if [[ -n "${CONTENT_ID_TXT:-}" ]]; then
  PY_ARGS+=(--content_id_txt "$CONTENT_ID_TXT")
fi

if [[ -n "${STYLE_ID_TXT:-}" ]]; then
  PY_ARGS+=(--style_id_txt "$STYLE_ID_TXT")
fi

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  PY_ARGS+=(--overwrite)
fi

echo "[cmd] python3 ${PY_ARGS[*]}"
python3 "${PY_ARGS[@]}"
