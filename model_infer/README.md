# CRef / SRef Core Inference Minimal Demo

This directory is a minimal inference demo: feed in two images and one prompt, and get back the final generated image plus the intermediate recaption result.

## 1. Minimal invocation

```bash

python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  'Transfer the style of image 2 onto image 1' \
  --weight_preset sref_14000 \
  --out_dir outputs/sref_14000_demo \
  --recaption_task_type style_transfer \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

The input order is fixed:

1. `cref_image`: content / subject / layout reference image;
2. `sref_image`: style reference image;
3. `prompt`: the user instruction.

Output files default to `--out_dir`:

```text
result.png              # final generated image
final_prompt.json       # the recaption prompt actually fed to the generator, JSON
final_prompt.txt        # the recaption prompt actually fed to the generator, plain text
recaption_result.json   # Qwen3-VL recaption intermediate result, incl. raw_response / parsed
demo_summary.json       # summary of this run's configuration
```

You can also just run the bundled example script:

```bash
bash run_sref_infer.sh
```

---

## 2. Download the weights and put them in place (HuggingFace)

All weights are released at **[Blue2Giant/FreeStyle_Checkpoint](https://huggingface.co/Blue2Giant/FreeStyle_Checkpoint)**. Download them to a local `checkpoints/` directory with a single command:

```bash
cd model_infer
huggingface-cli download Blue2Giant/FreeStyle_Checkpoint --local-dir ./checkpoints
```

After downloading, the directory layout is as follows — **each preset maps to one subdirectory containing a single `model.safetensors`**:

```text
checkpoints/
  freestyle-sref-14000-no-rope/model.safetensors          # preset: sref_14000
  freestyle-cref-sref-40000-no-rope/model.safetensors      # preset: cref_sref_40000
  freestyle-cref-sref-36000-no-rope/model.safetensors      # preset: cref_sref_36000_no_rope
```

The script reads from `./checkpoints` by default (i.e. the `checkpoints/` directory next to `cref_sref_core_infer.py`). If you keep the weights somewhere else, just set the environment variable:

```bash
export FREESTYLE_CKPT_ROOT=/path/to/your/checkpoints
```

Then at inference time you only pass `--weight_preset`; the script automatically builds `$FREESTYLE_CKPT_ROOT/<subdir>/model.safetensors` and at the same time sets `--task` (sref / cref_sref) and whether RoPE is enabled (`--use_rope` / `--no_rope`).

| Preset | Task | RoPE? | Subdirectory (under `$FREESTYLE_CKPT_ROOT/`) |
|---|---|---:|---|
| `sref_14000` | SRef | No | `freestyle-sref-14000-no-rope/model.safetensors` |
| `cref_sref_40000` | CRef+SRef | Yes | `freestyle-cref-sref-40000-no-rope/model.safetensors` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `freestyle-cref-sref-36000-no-rope/model.safetensors` |

Notes:

- Whether RoPE is used is controlled by `--use_rope` / `--no_rope` (the preset sets this automatically).
- `sref_14000` and `cref_sref_36000_no_rope` run the plain `ImageGenerator` (no RoPE); `cref_sref_40000` uses `ImageGeneratorRopeFA`.
- To use your own weight (without a preset), specify `--dit_path`, `--task`, and `--use_rope` / `--no_rope` manually.

---

## 3. How to pass the recaption task type

| Argument | Use case | Recaption prompt source | Output size rule |
|---|---|---|---|
| `--recaption_task_type sref` | SRef inference (SRef weights only) | local `SREF_RECAPTION_TEMPLATE_MINIMAL` | uses `--width/--height`, default `1024x1024` |
| `--recaption_task_type identity_style` | normal CRef+SRef inference (CRef+SRef weights only) | local `QWEN3_CREF_SREF_USER_PROMPT` | uses `--width/--height`, default `1024x1024` |
| `--recaption_task_type style_transfer` | style transfer (works with both weight families) | local `SREF_RECAPTION_TEMPLATE_MINIMAL` | automatically keeps the output at the CRef image resolution |

The pairing between weight and recaption prompt is **strictly validated**: a mismatch raises an error immediately, preventing different weights from getting their inference mixed up:

- SRef weights (`sref_14000`) only accept `sref` or `style_transfer`; passing `identity_style` / `cref_sref` raises an error.
- CRef+SRef weights (`cref_sref_*`) only accept `identity_style` / `cref_sref` or `style_transfer`; passing `sref` raises an error.

Both recaption prompt templates are **hard-coded in `cref_sref_core_infer.py`** (`SREF_RECAPTION_TEMPLATE_MINIMAL` and `QWEN3_CREF_SREF_USER_PROMPT`); they are no longer read as constants from `recaption.py`. The runtime log prints `recaption_prompt: <template name>` so you can confirm which one this run used.

The final prompt sent to the generator is built from the JSON returned by Qwen3-VL as:

```text
independent_captions.scene_3 + training_output.sample_instruction_cn_123
```

---

## 4. Full invocation scripts for the three weights

Below, each preset comes with a **complete, copy-paste-runnable command**, already prefixed with the recommended environment variables. Before running:

```bash
conda activate Sref
cd /data/FreeStyle/model_infer
```

The minimal-demo input order for all commands is fixed: `cref_image` (content/layout reference), `sref_image` (style reference), `prompt` (user instruction).

### 4.1 SRef 14000 (no RoPE)

For pure SRef inference, use `--recaption_task_type sref`:

```bash
VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
VGO_STREAM_LOAD_SAFETENSORS=1 \
VGO_STREAM_LOAD_DTYPE=bfloat16 \
VGO_STREAM_LOAD_DEVICE=cuda:0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  'Transfer the style of image 2 onto image 1, keeping image 1 overall layout unchanged.' \
  --weight_preset sref_14000 \
  --out_dir outputs/sref_14000_demo \
  --recaption_task_type sref \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

To run a style-transfer demo with this SRef weight, replace `--recaption_task_type sref` with `--recaption_task_type style_transfer`.

### 4.2 CRef+SRef 40000

The normal no-RoPE 40000 CRef+SRef weight; the preset automatically set rope modulated:

```bash
VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
VGO_STREAM_LOAD_SAFETENSORS=1 \
VGO_STREAM_LOAD_DTYPE=bfloat16 \
VGO_STREAM_LOAD_DEVICE=cuda:0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  assets/02-cref.png \
  assets/02-sref.png \
  'A cat lying in front of a fireplace, transfer the style of image 2 onto image 1' \
  --weight_preset cref_sref_40000 \
  --out_dir outputs/cref_sref_40000_demo \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

To use this no-RoPE weight for style transfer, replace `--recaption_task_type identity_style` with `--recaption_task_type style_transfer`.

### 4.3 CRef+SRef 36000 no-RoPE (no RoPE)

This weight has no RoPE modulation; the preset automatically sets `--no_rope`:

```bash
VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
VGO_STREAM_LOAD_SAFETENSORS=1 \
VGO_STREAM_LOAD_DTYPE=bfloat16 \
VGO_STREAM_LOAD_DEVICE=cuda:0 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  assets/02-cref.png \
  assets/02-sref.png \
  '一只猫趴在壁炉前，迁移图2的风格到图1上' \
  --weight_preset cref_sref_36000_no_rope \
  --out_dir outputs/cref_sref_36000_no_rope_demo \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

To use this no-RoPE weight for style transfer, replace `--recaption_task_type identity_style` with `--recaption_task_type style_transfer`. This weight corresponds to `$FREESTYLE_CKPT_ROOT/freestyle-cref-sref-36000-no-rope/model.safetensors` (see the download/placement instructions in Section 2 above).

---

## 4. Loading large safetensors weights

Weights like 14000 / 36000 are large (~40-77G), so the demo scripts support streaming load by default to avoid loading the full weight into CPU memory at once:

```bash
VGO_STREAM_LOAD_SAFETENSORS=1
VGO_STREAM_LOAD_DTYPE=bfloat16
VGO_STREAM_LOAD_DEVICE=cuda:0
```

These environment variables are already set in `run_sref_infer.sh`.

---

## 5. Legacy prompts.json + keys batch mode

The default entrypoint is now the "two images + one prompt" minimal demo. The old benchmark batch mode is still available, but you need to explicitly add:

```bash
--batch_mode
```

Only batch mode reads:

```text
--data_root
--prompts_json
--cref_dir
--sref_dir
--keys / --key_txt
```
