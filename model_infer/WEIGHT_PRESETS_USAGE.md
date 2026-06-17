# Weight Presets and Inference Usage

This repo provides one Python entrypoint:

```bash
python3 cref_sref_core_infer.py ...
```

and two minimal launcher examples:

```bash
bash run_sref_infer.sh
bash run_cref_sref_infer.sh
```

The easiest way to switch weights is to change `--weight_preset`.

---

## 1. Available weight presets

| Preset | Task | RoPE? | Weight path |
|---|---|---:|---|
| `sref_14000` | SRef | No | `/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors` |
| `cref_sref_40000` | CRef+SRef | Yes | `/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` |

Each preset sets three things for you: the weight path (`--dit_path`), the task (`--task sref` or `--task cref_sref`), and whether frequency-aware RoPE is enabled (`--use_rope` / `--no_rope`).

### Note for `cref_sref_36000_no_rope`

This is the CRef+SRef **no-RoPE** 36000 checkpoint. The primary checkpoint path is the converted copy:

```bash
/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors
```

This file was visible in the `gpu_temp_1` / `Sref` environment on 2026-06-15 as a 77G checkpoint. If that converted copy is not visible on another worker, the Python script can fall back to the legacy visible copy:

```bash
/mnt/jfs/model_zoo/checkpoint-36000.safetensors
```

---

## 2. Task and RoPE control (no training config needed)

The inference model config is **hard-coded in `cref_sref_core_infer.py`** — the
original training YAMLs are not shipped. Two flags control behavior:

| Flag | Values | Meaning |
|---|---|---|
| `--task` | `sref`, `cref_sref` | selects the task (default recaption prompt + benchmark data root) |
| `--use_rope` / `--no_rope` | — | enables / disables frequency-aware RoPE modulation |

`--weight_preset` already sets both flags for each released checkpoint, so you
normally don't pass them. They matter when you run **your own** weight without a
preset, e.g.:

```bash
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg assets/00-sref.jpg "your prompt" \
  --weight_preset "" \
  --dit_path /path/to/your/checkpoint.safetensors \
  --task cref_sref \
  --use_rope \
  --recaption_task_type identity_style \
  --out_dir outputs/custom
```

When `--use_rope` is set, the script builds the DiT with the RoPE-FA modulation
parameters baked into `ROPE_FA_INFERENCE_PARAMS` and runs `ImageGeneratorRopeFA`.
Otherwise it runs the plain `ImageGenerator`. A preset's explicit `--use_rope` /
`--no_rope` / `--task` on the command line always overrides the preset value.

---

## 3. Recaption task types

The script supports these recaption task types:

| `--recaption_task_type` | Use case | Prompt template |
|---|---|---|
| `sref` | SRef inference | built-in SRef minimal prompt |
| `identity_style` | CRef+SRef / identity-style generation | `recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF` |
| `style_transfer` | Style-transfer generation | `recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER` |

The final generator prompt is built from the Qwen3-VL JSON as:

```text
independent_captions.scene_3 + training_output.primary_instruction_cn_123
```

For `sref`, the prompt JSON may not contain `primary_instruction_cn_123`; in that case the script falls back to `sample_instruction_cn_123` or `scene_3`.

---

## 4. SRef inference examples

### 4.1 SRef checkpoint 14000

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset sref_14000 \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
  --out_dir /data/benchmark_metrics/sref_infer_sref_14000 \
  --keys 'football__sticker_figure' \
  --recaption_task_type sref \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

You can also edit and run:

```bash
bash run_sref_infer.sh
```

---

## 5. CRef+SRef inference examples

### 5.1 CRef+SRef RoPE checkpoint 40000

This is the CRef+SRef **RoPE** 40000 checkpoint. It uses RoPE-FA frequency-aware modulation:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset cref_sref_40000 \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content \
  --out_dir /data/benchmark_metrics/cref_sref_infer_40000_no_rope \
  --keys 'football__sticker_figure' \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

### 5.2 CRef+SRef no-RoPE checkpoint 36000

Use this when the weight was **not** trained with RoPE-FA:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset cref_sref_36000_no_rope \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content \
  --out_dir /data/benchmark_metrics/cref_sref_infer_36000_no_rope \
  --keys 'football__sticker_figure' \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

You can also edit and run:

```bash
bash run_cref_sref_infer.sh
```

---

## 6. Running all samples

To run all keys in `prompts.json`, remove the `--keys ...` line from the command.

For a faster smoke test, use fewer denoising steps:

```bash
--steps 8
```

For normal inference, use:

```bash
--steps 28
```

---

## 7. Output files

Each run writes:

```text
$out_dir/{key}.png
$out_dir/recaption_prompts.json
$out_dir/recaption_structured.json
$out_dir/selected_keys.txt
```
