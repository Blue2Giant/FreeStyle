# HF Upload Checklist

This checklist is for uploading `/mnt/jfs/vgo_hf_exports/0426_lora_triplet_normalized` to Hugging Face.

Recommended repository layout:

- repository root: `README.md`
- dataset payload: `cref_sref/`

## Export Status

All three source exports completed successfully.

- `qwen`: `33,582 / 33,582` sequences
- `flux`: `273,682 / 273,682` sequences
- `illustrious`: `172,589 / 172,589` sequences

## Files To Upload

Required:

- `README.md`
- `cref_sref/HF_UPLOAD_CHECKLIST.md`
- `cref_sref/README.md`
- `cref_sref/qwen/`
- `cref_sref/flux/`
- `cref_sref/illustrious/`

Do not upload:

- `logs/`

Inside each source directory under `cref_sref/`, keep:

- `README.md`
- `summary.json`
- `triplets.csv`
- `content_images.csv`
- `style_images.csv`
- `target_images.csv`
- `images/`

## Internal State

`_state/` is not required for ordinary dataset use.

You have two valid choices:

1. Keep `_state/` in the upload if you want resume/debug provenance bundled with the release.
2. Exclude `_state/` from the upload if you want a cleaner consumer-facing dataset layout.

Recommended:

- Keep `_state/` in the backup/archive copy.
- Exclude `_state/` from the public Hugging Face dataset unless you explicitly want to expose internal resume state.
- Exclude top-level `logs/` from the public Hugging Face dataset.

## Pre-Upload Sanity Checks

- `triplets.csv` row count matches `summary.json.exported_sequences`
- `content_images.csv` row count matches `summary.json.unique_content_images`
- `style_images.csv` row count matches `summary.json.unique_style_images`
- `target_images.csv` row count matches `summary.json.unique_target_images`
- each source `summary.json` exists
- each source `README.md` exists
- image files are present under `images/content`, `images/style`, `images/target`

## Consumer Guidance

Tell users to start from:

- source-level `README.md`
- `triplets.csv`

Tell users that:

- `triplets.csv` is sequence-level
- `*_images.csv` files are deduplicated image-level metadata
- image paths inside `triplets.csv` are relative to the source directory
- `original_path` and prompt recovery fields are best-effort provenance, not guaranteed for every row

## Match Coverage Notes

The dataset intentionally keeps unmatched rows rather than forcing incorrect provenance.

Expected status values include:

- `match_status=matched`
- `match_status=unmatched`
- `match_status=ambiguous`
- `match_status=no_candidates`

Expected prompt states include:

- `prompt_status=resolved`
- `prompt_status=unmatched_original`
- `prompt_status=missing_prompt_payload`
- `prompt_status=missing_prompt_entry`

## Suggested Public Description

Suggested short description:

> Normalized lora-triplet training export from the 0426 cref/sref diffusion configuration, with deduplicated images, sequence-level triplets, and best-effort original prompt provenance.
