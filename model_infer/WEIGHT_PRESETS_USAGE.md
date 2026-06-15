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

| Preset | Task | RoPE? | Weight path | Config path |
|---|---|---:|---|---|
| `sref_14000` | SRef | No | `/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors` | `configs/train/0415_qwen_image_sref_noise_query.yaml` |
| `sref_12000` | SRef | No | `/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors` | `configs/train/0415_qwen_image_sref_noise_query.yaml` |
| `cref_sref_rope_50000` | CRef+SRef | Yes | `/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors` | `configs/train/0506_qwen_cref_sref_from40000_no_illutrious_rope.yaml` |
| `cref_sref_40000` | CRef+SRef | No | `/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors` | `configs/train/0426_qwen_cref_sref_full_diffusion_no_illustrious.yaml` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` | `configs/train/0426_qwen_cref_sref_full_diffusion_no_illustrious.yaml` |

### Note for `cref_sref_36000_no_rope`

This is the CRef+SRef **no-RoPE** 36000 checkpoint. It should use the normal CRef+SRef config, not the RoPE config:

```bash
configs/train/0426_qwen_cref_sref_full_diffusion_no_illustrious.yaml
```

The primary checkpoint path is the converted copy:

```bash
/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors
```

This file was visible in the `gpu_temp_1` / `Sref` environment on 2026-06-15 as a 77G checkpoint. If that converted copy is not visible on another worker, the Python script can fall back to the legacy visible copy:

```bash
/mnt/jfs/model_zoo/checkpoint-36000.safetensors
```

---

## 2. RoPE config behavior

RoPE is **not hard-coded** in Python. It is controlled by the YAML config.

For RoPE inference, use:

```bash
--weight_preset cref_sref_rope_50000
```

This preset uses:

```bash
configs/train/0506_qwen_cref_sref_from40000_no_illutrious_rope.yaml
```

That config contains RoPE settings under:

```yaml
engine_config:
  pipe:
    dit:
      rope_fa:
        enabled: true
        shf_min: ...
        slf_min: ...
        shf_max: ...
        slf_max: ...
        beta: ...
        spatial_axes_only: ...
```

When `rope_fa.enabled: true`, `cref_sref_core_infer.py` automatically uses:

```python
ImageGeneratorRopeFA
```

For no-RoPE weights, use a no-RoPE preset such as:

```bash
--weight_preset cref_sref_36000_no_rope
```

which uses the normal config:

```bash
configs/train/0426_qwen_cref_sref_full_diffusion_no_illustrious.yaml
```

and automatically uses:

```python
ImageGenerator
```

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

### 4.2 SRef checkpoint 12000

Only change `--weight_preset` and `--out_dir`:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset sref_12000 \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_sref_200_content \
  --out_dir /data/benchmark_metrics/sref_infer_sref_12000 \
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

### 5.1 CRef+SRef RoPE checkpoint 50000

Use this when the weight was trained with RoPE-FA:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

VGO_DISABLE_TORCH_COMPILE=1 \
VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
TRANSFORMERS_VERBOSITY=error \
CUDA_VISIBLE_DEVICES=0 \
python3 cref_sref_core_infer.py \
  --weight_preset cref_sref_rope_50000 \
  --data_root /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content \
  --out_dir /data/benchmark_metrics/cref_sref_infer_rope_50000 \
  --keys 'football__sticker_figure' \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

### 5.2 CRef+SRef no-RoPE checkpoint 40000

This is the normal no-RoPE 40000 CRef+SRef checkpoint. It uses the same normal config as the 36000 no-RoPE checkpoint:

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

### 5.3 CRef+SRef no-RoPE checkpoint 36000

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
