# CRef / SRef Core Inference Minimal Demo

这个目录是一个最小推理 demo：输入两张图片和一个 prompt，输出最终生成图以及中间 recaption 结果。

## 1. 最小调用格式

```bash

python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上' \
  --weight_preset sref_12000 \
  --out_dir outputs/sref_12000_style_transfer_demo \
  --recaption_task_type style_transfer \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

输入顺序固定为：

1. `cref_image`：内容 / 主体 / 布局参考图；
2. `sref_image`：风格参考图；
3. `prompt`：用户指令。

输出文件默认在 `--out_dir` 下：

```text
result.png              # 最终生成图
final_prompt.json       # 真正送入生成模型的 recaption prompt，JSON 格式
final_prompt.txt        # 真正送入生成模型的 recaption prompt，纯文本
recaption_result.json   # Qwen3-VL recaption 中间结果，包含 raw_response / parsed
demo_summary.json       # 本次推理配置汇总
```

也可以直接跑当前示例脚本：

```bash
bash run_sref_infer.sh
```

---

## 2. 五个常用权重位置和对应 preset

推理时推荐直接使用 `--weight_preset`，脚本会自动设置 `--dit_path`、`--task`（sref / cref_sref）以及是否启用 RoPE（`--use_rope` / `--no_rope`）。

| Preset | 任务类型 | RoPE? | 权重位置 |
|---|---|---:|---|
| `sref_14000` | SRef | No | `/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors` |
| `sref_12000` | SRef | No | `/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors` |
| `cref_sref_rope_50000` | CRef+SRef | Yes | `/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors` |
| `cref_sref_40000` | CRef+SRef | No | `/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` |

说明：

- 推理用的模型 config 已经**硬编码在 `cref_sref_core_infer.py` 里**，仓库不再附带训练 YAML。是否使用 RoPE 由 `--use_rope` / `--no_rope` 控制（preset 已自动设置）。
- `cref_sref_rope_50000` 会自动启用 RoPE，并走 `ImageGeneratorRopeFA`；其余 preset 走普通 `ImageGenerator`。
- 用自己的权重（不带 preset）时，可手动指定 `--dit_path`、`--task` 和 `--use_rope` / `--no_rope`。
- `cref_sref_36000_no_rope` 如果主路径不可见，代码会尝试 fallback 到 `/mnt/jfs/model_zoo/checkpoint-36000.safetensors`。

---

## 3. Recaption task type 该怎么传

| 参数 | 使用场景 | Recaption prompt 来源 | 输出尺寸规则 |
|---|---|---|---|
| `--recaption_task_type sref` | SRef 推理（仅 SRef 权重） | 本地 `SREF_RECAPTION_TEMPLATE_MINIMAL` | 使用 `--width/--height`，默认 `1024x1024` |
| `--recaption_task_type identity_style` | CRef+SRef 普通推理（仅 CRef+SRef 权重） | 本地 `QWEN3_CREF_SREF_USER_PROMPT` | 使用 `--width/--height`，默认 `1024x1024` |
| `--recaption_task_type style_transfer` | 风格迁移（两类权重都可用） | 本地 `SREF_RECAPTION_TEMPLATE_MINIMAL` | 自动保持输出图和 CRef 图同分辨率 |

权重与 recaption prompt 的对应关系是**硬性校验**的，写错会直接报错，避免不同权重推理弄混：

- SRef 权重（`sref_14000` / `sref_12000`）只接受 `sref` 或 `style_transfer`；传 `identity_style` / `cref_sref` 会报错。
- CRef+SRef 权重（`cref_sref_*`）只接受 `identity_style` / `cref_sref` 或 `style_transfer`；传 `sref` 会报错。

两个 recaption prompt 模板都**硬编码在 `cref_sref_core_infer.py` 里**（`SREF_RECAPTION_TEMPLATE_MINIMAL` 与 `QWEN3_CREF_SREF_USER_PROMPT`），不再从 `recaption.py` 读取常量。运行时日志会打印 `recaption_prompt: <模板名>`，方便确认本次用的是哪个。

最终送给生成模型的 prompt 来自 Qwen3-VL 返回 JSON 中的：

```text
independent_captions.scene_3 + training_output.sample_instruction_cn_123
```

---

## 4. 五个权重的完整调用脚本

下面每个 preset 都给出一条**可直接复制运行的完整命令**，已经带上推荐的环境变量前缀。运行前先：

```bash
conda activate Sref
cd /data/FreeStyle/model_infer
```

所有命令的最小 demo 输入顺序固定为：`cref_image`（内容/布局参考图）、`sref_image`（风格参考图）、`prompt`（用户指令）。

### 4.1 SRef 14000（无 RoPE）

纯 SRef 推理时用 `--recaption_task_type sref`：

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
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
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

如果要用这个 SRef 权重做 style transfer demo，把 `--recaption_task_type sref` 换成 `--recaption_task_type style_transfer`。

### 4.2 SRef 12000（无 RoPE）

当前 `run_sref_infer.sh` 使用的就是这个权重，并设置为 style transfer demo：

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
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
  --weight_preset sref_12000 \
  --out_dir outputs/sref_12000_style_transfer_demo \
  --recaption_task_type style_transfer \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

注意：这里虽然写了 `--width 1024 --height 1024`，但因为 `recaption_task_type=style_transfer`，最终保存的 `result.png` 会自动 resize 回 CRef 图分辨率。

### 4.3 CRef+SRef RoPE 50000（启用 RoPE）

这个权重使用 RoPE 调制，preset 会自动启用 `--use_rope` 并走 `ImageGeneratorRopeFA`：

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
  '猫咪在一个壁炉前面趴着，迁移图2的风格到图1上' \
  --weight_preset cref_sref_rope_50000 \
  --out_dir outputs/cref_sref_rope_50000_demo \
  --recaption_task_type identity_style \
  --width 1024 \
  --height 1024 \
  --steps 28 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

如果这个 RoPE 权重用于 style transfer，把 `--recaption_task_type identity_style` 换成 `--recaption_task_type style_transfer`。

### 4.4 CRef+SRef 40000（无 RoPE）

普通 no-RoPE 的 40000 CRef+SRef 权重，preset 会自动设置 `--no_rope`：

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
  '猫咪在一个壁炉前面趴着，迁移图2的风格到图1上' \
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

如果这个 no-RoPE 权重用于 style transfer，把 `--recaption_task_type identity_style` 换成 `--recaption_task_type style_transfer`。

### 4.5 CRef+SRef 36000 no-RoPE（无 RoPE）

这个权重没有 RoPE 调制，preset 会自动设置 `--no_rope`：

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
  '猫咪在一个壁炉前面趴着，迁移图2的风格到图1上' \
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

如果这个 no-RoPE 权重用于 style transfer，把 `--recaption_task_type identity_style` 换成 `--recaption_task_type style_transfer`。如果主路径 `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` 在某个 worker 上不可见，代码会自动 fallback 到 `/mnt/jfs/model_zoo/checkpoint-36000.safetensors`。

---

## 5. 大 safetensors 权重加载设置

12000 / 36000 等权重较大，demo 脚本默认支持 streaming load，避免一次性把 70G+ 权重全部加载到 CPU 内存：

```bash
VGO_STREAM_LOAD_SAFETENSORS=1
VGO_STREAM_LOAD_DTYPE=bfloat16
VGO_STREAM_LOAD_DEVICE=cuda:0
```

`run_sref_infer.sh` 里已经写好了这些环境变量。

---

## 6. 老的 prompts.json + keys 批量模式

默认入口现在是“两张图 + 一个 prompt”的最小 demo。旧的 benchmark 批量模式仍保留，需要显式加：

```bash
--batch_mode
```

批量模式才会读取：

```text
--data_root
--prompts_json
--cref_dir
--sref_dir
--keys / --key_txt
```
