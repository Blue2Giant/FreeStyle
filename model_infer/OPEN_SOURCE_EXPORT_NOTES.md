# CRef/SRef Core Inference Export

This directory is a standalone code export for `cref_sref_core_infer.py`.
It contains only project code/configs needed to run the core CRef+SRef inference flow:

- Core entrypoint: `cref_sref_core_infer.py`
- Recaption prompt templates: `recaption.py`
  - `PROMPT_WITH_INSTUCTION_CREF_SREF` for CRef+SRef / identity-style tasks
  - `PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER` for style-transfer tasks
- RoPE-FA inference wrapper: `multi_cref_eval_rope_fa.py`
- Core VGO package: `vgo/`
- Local lightweight `torchvision/` compatibility shim used by this project

The inference model config is hard-coded in `cref_sref_core_infer.py`; no training
YAMLs are shipped. RoPE and task selection are controlled by `--use_rope`/`--no_rope`
and `--task` (or via `--weight_preset`).

The final generator prompt is parsed from the Qwen3-VL JSON as:

```text
independent_captions.scene_3 + training_output.primary_instruction_cn_123
```

Not included intentionally:

- model checkpoints / safetensors
- Qwen/Qwen3 model weights
- benchmark datasets or generated images
- credentials, logs, caches, tmux/rlaunch helper state

Validated smoke tests in `Sref` conda env on 2026-06-15:

1. Base full pipeline with recaption subprocess (new recaption.py prompt, 8 denoise steps):
   `/data/benchmark_metrics/core_infer_newprompt_pipeline_0615/football__sticker_figure.png`
2. RoPE-FA generator path (`ImageGeneratorRopeFA`, `rope_fa=True`, 8 steps):
   `/data/benchmark_metrics/core_infer_rope_smoke_0615/football__sticker_figure.png`
3. Style-transfer full pipeline using `PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER` (8 denoise steps):
   `/data/benchmark_metrics/core_infer_newprompt_style_transfer_pipeline_0615/football__sticker_figure.png`

For usage, see `CREF_SREF_CORE_INFER_USAGE.md`.

## Launcher scripts

Two minimal launcher scripts are included. They intentionally use hard-coded Python arguments for readability; edit the `--weight_preset`, `--data_root`, `--out_dir`, and `--keys` lines directly before running.

- `run_sref_infer.sh`: SRef example, choose `sref_14000` or `sref_12000`; uses `--recaption_task_type sref`.
- `run_cref_sref_infer.sh`: CRef+SRef example, choose one of:
  - `cref_sref_rope_50000`: RoPE weight (preset sets `--use_rope`).
  - `cref_sref_40000`: no-RoPE 40000 weight (preset sets `--no_rope`).
  - `cref_sref_36000_no_rope`: no-RoPE 36000 weight at `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` (preset sets `--no_rope`).

The Python entrypoint supports the same presets via `--weight_preset`; full examples are in `WEIGHT_PRESETS_USAGE.md`.
