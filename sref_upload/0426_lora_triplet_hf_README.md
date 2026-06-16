# 0426 Lora Triplet Dataset

This directory contains the normalized export of the lora-triplet portion of:

- `/data/vgo/xingpeng/new_vgo/Sref_Cref_MiniVGO/configs/data/0426_cref_sref_full_diffusion.yaml`

It covers three nonzero-weight lora-triplet sources:

- `cref_sref_qwen_lora_part1`
- `cref_sref_flux_lora_part1`
- `cref_sref_illustrious_lora_part1`

## Directory Layout

For the Hugging Face release, the dataset card stays at repository root and the exported data is placed under `cref_sref/`:

```text
<repo-root>/
  README.md
  cref_sref/
    HF_UPLOAD_CHECKLIST.md
    README.md
    qwen/
    flux/
    illustrious/
```

The working export root may also contain `logs/`. That directory is internal and can be excluded from the Hugging Face upload.

Each source subdirectory under `cref_sref/` has the same structure:

```text
<source-name>/
  README.md
  summary.json
  triplets.csv
  content_images.csv
  style_images.csv
  target_images.csv
  images/
    content/...
    style/...
    target/...
  _state/
    manifest.json
    triplets.jsonl
    content_images.jsonl
    style_images.jsonl
    target_images.jsonl
```

## What A Triplet Means

Each triplet row corresponds to one vault training sequence and three training images:

- `content`: the image used for `cref_0`
- `style`: the image used for `sref_0`
- `target`: the image used for the combined content+style target

So the key relationship is:

- `triplets.csv` = one row per training sequence
- `content_images.csv` = one row per unique content image
- `style_images.csv` = one row per unique style image
- `target_images.csv` = one row per unique target image

The images are deduplicated. The same exported image path can appear in many triplet rows.

## How To Read The Files

### 1. `triplets.csv`

Use this file when you want to understand the training example itself.

Important columns:

- `sequence_id`: unique id of the vault sequence
- `base_model`: one of `qwen`, `flux`, `illustrious`
- `pair_key`: pair identifier
- `content_model_id`
- `style_model_id`
- `content_image_path`
- `style_image_path`
- `target_image_path`
- `content_original_path`
- `style_original_path`
- `target_original_path`
- `content_match_status`
- `style_match_status`
- `target_match_status`
- `content_prompt_status`
- `style_prompt_status`
- `target_prompt_status`
- `content_generation_prompt`
- `style_generation_prompt`
- `target_generation_prompt`
- `vault_texts_json`

### 2. `content_images.csv` / `style_images.csv` / `target_images.csv`

Use these files when you want image-level metadata.

Important columns:

- `exported_image_path`: relative path under the source directory
- `original_path`: recovered original generation image path when matched
- `match_status`: whether original-path matching succeeded
- `prompt_status`: whether the original generation prompt was recovered
- `generation_prompt`
- `base_prompt`
- `sequence_count`: how many triplets reuse this exported image
- `sequence_ids_json`: which triplets reuse this image

## How To View One Triplet

### Method 1: inspect one row from `triplets.csv`

```bash
python3 - <<'PY'
import csv
path = '/path/to/repo/cref_sref/qwen/triplets.csv'
with open(path, 'r', encoding='utf-8', newline='') as fh:
    row = next(csv.DictReader(fh))
for key in [
    'sequence_id',
    'pair_key',
    'content_image_path',
    'style_image_path',
    'target_image_path',
    'content_match_status',
    'style_match_status',
    'target_match_status',
    'content_generation_prompt',
    'style_generation_prompt',
    'target_generation_prompt',
]:
    print(f'{key}: {row[key]}')
PY
```

### Method 2: load the three images for a given sequence

```bash
python3 - <<'PY'
import csv
from pathlib import Path

base = Path('/path/to/repo/cref_sref/qwen')
with open(base / 'triplets.csv', 'r', encoding='utf-8', newline='') as fh:
    row = next(csv.DictReader(fh))

print('sequence_id:', row['sequence_id'])
print('content image:', base / row['content_image_path'])
print('style image:', base / row['style_image_path'])
print('target image:', base / row['target_image_path'])
PY
```

### Method 3: join a triplet row to image-level metadata

Join:

- `triplets.csv.content_image_path` -> `content_images.csv.exported_image_path`
- `triplets.csv.style_image_path` -> `style_images.csv.exported_image_path`
- `triplets.csv.target_image_path` -> `target_images.csv.exported_image_path`

This lets you answer:

- Which triplets reuse the same image?
- What is the recovered original path?
- Was the original prompt recovered?

## How To Interpret Match And Prompt Status

### `match_status`

- `matched`: exact visual-key match found in the original candidate pool
- `unmatched`: candidate pool exists, but no exact unique match was found
- `ambiguous`: more than one candidate matched the same visual key
- `no_candidates`: no candidate pool was available for that lookup

### `prompt_status`

- `resolved`: generation prompt metadata was recovered
- `unmatched_original`: original image path was not matched
- `missing_prompt_payload`: prompt sidecar json was missing
- `missing_prompt_entry`: prompt file existed, but the specific image entry was missing
- `missing_prompt_index`: image filename could not be mapped to a prompt index

## Important Semantics

- Exported images are vault training images, not raw copies of original one-lora or dual-lora PNG files.
- `original_path` and prompt recovery fields are best-effort provenance fields.
- Some rows intentionally remain unmatched rather than risk incorrect prompt assignment.
- `_state/` is internal resume state used during export; it is not required for ordinary dataset consumption.

## Source-Level Summary

Final exported sequence counts:

- `qwen`: `33,582`
- `flux`: `273,682`
- `illustrious`: `172,589`

For detailed per-source match counts, see each source's `summary.json`.
