# CRef / SRef Core Inference Minimal Demo

这个目录是一个最小推理 demo：输入两张图片和一个 prompt，输出最终生成图以及中间 recaption 结果。

## 1. 最小调用格式

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
conda activate Sref

python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上' \
  --weight_preset sref_12000 \
  --out_dir outputs/sref_12000_style_transfer_demo \
  --recaption_task_type style_transfer \
  --steps 8 \
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

## 2. 四个常用权重位置和对应 preset

推理时推荐直接使用 `--weight_preset`，脚本会自动设置对应的 `dit_path` 和 `config_path`。

| Preset | 任务类型 | RoPE? | 权重位置 | 自动使用的 config |
|---|---|---:|---|---|
| `sref_14000` | SRef | No | `/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors` | `configs/train/0415_qwen_image_sref_noise_query.yaml` |
| `sref_12000` | SRef | No | `/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors` | `configs/train/0415_qwen_image_sref_noise_query.yaml` |
| `cref_sref_rope_50000` | CRef+SRef | Yes | `/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors` | `configs/train/0506_qwen_cref_sref_from40000_no_illutrious_rope.yaml` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` | `configs/train/0426_qwen_cref_sref_full_diffusion_no_illustrious.yaml` |

说明：

- RoPE 不是在 Python 里硬编码的；是否使用 RoPE 由 config 中的 `engine_config.pipe.dit.rope_fa.enabled` 决定。
- `cref_sref_rope_50000` 会自动使用 RoPE config，并走 `ImageGeneratorRopeFA`。
- `cref_sref_36000_no_rope` 是无 RoPE 调制权重，应使用普通 CRef+SRef config。
- `cref_sref_36000_no_rope` 如果主路径不可见，代码会尝试 fallback 到 `/mnt/jfs/model_zoo/checkpoint-36000.safetensors`。

---

## 3. Recaption task type 该怎么传

| 参数 | 使用场景 | Recaption prompt 来源 | 输出尺寸规则 |
|---|---|---|---|
| `--recaption_task_type sref` | SRef 推理 | 内置 SRef prompt | 使用 `--width/--height`，默认 `1024x1024` |
| `--recaption_task_type identity_style` | CRef+SRef 普通推理 | `recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF` | 使用 `--width/--height`，默认 `1024x1024` |
| `--recaption_task_type style_transfer` | 风格迁移 | `recaption.py:PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER` | 自动保持输出图和 CRef 图同分辨率 |

最终送给生成模型的 prompt 来自 Qwen3-VL 返回 JSON 中的：

```text
independent_captions.scene_3 + training_output.primary_instruction_cn_123
```

---

## 4. 四个权重的推理参数示例

下面命令都使用最小 demo 输入：

```bash
assets/00-cref.jpg
assets/00-sref.jpg
'迁移图2的风格到图1上，保持图1的整体布局不变。'
```

### 4.1 SRef 14000

纯 SRef 推理时用 `--recaption_task_type sref`：

```bash
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

如果要用这个 SRef 权重做 style transfer demo，把 recaption 参数换成：

```bash
--recaption_task_type style_transfer
```

### 4.2 SRef 12000

当前 `run_sref_infer.sh` 使用的就是这个权重，并设置为 style transfer demo：

```bash
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
  --weight_preset sref_12000 \
  --out_dir outputs/sref_12000_style_transfer_demo \
  --recaption_task_type style_transfer \
  --width 1024 \
  --height 1024 \
  --steps 8 \
  --cfg 8 \
  --seed 42 \
  --overwrite
```

注意：这里虽然写了 `--width 1024 --height 1024`，但因为 `recaption_task_type=style_transfer`，最终保存的 `result.png` 会自动 resize 回 CRef 图分辨率。

### 4.3 CRef+SRef RoPE 50000

这个权重需要 RoPE config。直接传 preset 即可，不需要手动传 `--config_path`：

```bash
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
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

如果这个 RoPE 权重用于 style transfer，则把：

```bash
--recaption_task_type identity_style
```

换成：

```bash
--recaption_task_type style_transfer
```

### 4.4 CRef+SRef 36000 no-RoPE

这个权重没有 RoPE 调制，直接使用普通 CRef+SRef config：

```bash
python3 cref_sref_core_infer.py \
  assets/00-cref.jpg \
  assets/00-sref.jpg \
  '迁移图2的风格到图1上，保持图1的整体布局不变。' \
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

如果这个 no-RoPE 权重用于 style transfer，同样把 `--recaption_task_type` 改成：

```bash
--recaption_task_type style_transfer
```

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
