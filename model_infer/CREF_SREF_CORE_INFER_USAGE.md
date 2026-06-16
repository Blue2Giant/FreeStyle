# `cref_sref_core_infer.py` 使用说明

> **当前 open-source demo 入口已简化。** 默认模式不再要求 `prompts.json`，直接输入两张图和一个 prompt：
>
> ```bash
> python3 cref_sref_core_infer.py assets/00-cref.jpg assets/00-sref.jpg '迁移图2的风格到图1上，保持图1的整体布局不变。' \
>   --weight_preset sref_12000 \
>   --out_dir outputs/sref_12000_style_transfer_demo \
>   --recaption_task_type style_transfer \
>   --steps 8 \
>   --overwrite
> ```
>
> 输出文件：`result.png`、`final_prompt.json`、`final_prompt.txt`、`recaption_result.json`、`demo_summary.json`。
>
> 分辨率规则：`style_transfer` 会自动让最终保存的 `result.png` 与第一张输入图 / CRef 图分辨率一致；普通 CRef+SRef/SRef 任务使用 `--width/--height` 指定输出尺寸，默认 `1024x1024`。
>
> 老的 `prompts.json + keys` 批量模式仍保留，需要显式加 `--batch_mode`。


脚本路径：

```bash
cref_sref_core_infer.py
```

这个脚本是从 Gradio 服务中抽出的 **CRef + SRef 核心推理流程**，主要用于：

1. 从原始 `prompts.json` 读取用户 prompt；
2. 用 Qwen3-VL 对 `cref + sref + user prompt` 做 recaption；
3. 将 recaption 后的最终生成 prompt 保存成 `recaption_prompts*.json`；
4. 加载 VGO / MiniVGO 权重；
5. 按 `cref + sref + recaption prompt` 生成最终图片。

---

## 1. 默认输入目录

如果不显式指定参数，脚本默认使用：

```bash
DATA_ROOT=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
PROMPTS_JSON=$DATA_ROOT/prompts.json
CREF_DIR=$DATA_ROOT/cref
SREF_DIR=$DATA_ROOT/sref
```

其中：

- `prompts.json`：原始用户 prompt，格式为 `key -> prompt`；
- `cref/`：内容/主体参考图，文件名通常为 `{key}.png`；
- `sref/`：风格参考图，文件名通常为 `{key}.png`。

---

## 2. 模型与任务 / RoPE 控制（不需要训练 config）

推理用的模型 config 已经**硬编码在 `cref_sref_core_infer.py` 里**，仓库不再附带训练 YAML。两个 flag 控制行为：

| Flag | 取值 | 含义 |
|---|---|---|
| `--task` | `sref`, `cref_sref` | 选择任务（默认 recaption prompt 和 benchmark 数据根目录） |
| `--use_rope` / `--no_rope` | — | 启用 / 关闭 frequency-aware RoPE 调制 |

当 `--use_rope` 时，脚本会用硬编码在 `ROPE_FA_INFERENCE_PARAMS` 里的 RoPE-FA 调制参数构建 DiT，并走 `ImageGeneratorRopeFA` 推理路径；否则走普通 `ImageGenerator`。

默认 40000 base 权重为：

```bash
/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors
```

如果要跑 RoPE 加权版本，显式传 `--use_rope` 和对应权重：

```bash
--use_rope \
--dit_path /mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors
```


### 推荐：用 `--weight_preset` 切换常用权重

脚本里已经内置了常用权重 preset，推荐直接传 `--weight_preset`，这样权重路径、`--task`、`--use_rope`/`--no_rope`、默认数据根目录和默认 recaption 类型会一起切换：

| Preset | 任务 | RoPE? | 权重路径 |
|---|---|---:|---|
| `sref_14000` | SRef | No | `/mnt/jfs/debug_sre_enrichment_new_0415_h100_from_12000-new/0415_qwen_image_sref_noise_query/converted/checkpoint-14000/model.safetensors` |
| `sref_12000` | SRef | No | `/mnt/jfs/model_zoo/checkpoint-12000_converted/model.safetensors` |
| `cref_sref_rope_50000` | CRef+SRef | Yes | `/mnt/jfs/debug_sref_entropy_0429_cref_sref_full_diffusion_from36000_rope_fa_8gpu_from_no_illutrious_base/0505_qwen_cref_sref_full_diffusion_from40000_rope_fa/converted/checkpoint-50000/model.safetensors` |
| `cref_sref_40000` | CRef+SRef | No | `/mnt/jfs/debug_sref_entropy_0426_cref_sref_full_diffusion_no_illustrious/0426_qwen_cref_sref_full_diffusion/converted/checkpoint-40000/model.safetensors` |
| `cref_sref_36000_no_rope` | CRef+SRef | No | `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` |

每个 preset 帮你设置三件事：权重路径（`--dit_path`）、任务（`--task`）、以及是否启用 RoPE（`--use_rope` / `--no_rope`）。命令行上显式传的 `--task` / `--use_rope` / `--no_rope` 会覆盖 preset 的默认值。

`cref_sref_36000_no_rope` 是 CRef+SRef 的 **无 RoPE 调制** 36000 权重，preset 会自动设置 `--no_rope`。如果 `/mnt/jfs/model_zoo/checkpoint-36000_converted/checkpoint-36000.safetensors` 在某个 worker 上不可见，脚本会尝试 fallback 到 `/mnt/jfs/model_zoo/checkpoint-36000.safetensors`。

---

## 3. 输出文件

假设指定：

```bash
--out_dir $OUT_DIR
```

脚本会输出：

```text
$OUT_DIR/{key}.png                         # 最终生成图片
$OUT_DIR/recaption_prompts.json            # key -> final prompt，真正用于推理
$OUT_DIR/recaption_structured.json         # recaption debug 信息，包括 raw_response / parsed / original_prompt
$OUT_DIR/selected_keys.txt                 # 本次处理的 key 列表
```

并行分片时建议不要让多个进程写同一个 recaption JSON，而是使用：

```text
$OUT_DIR/recaption_prompts_shard0.json
$OUT_DIR/recaption_prompts_shard1.json
...
$OUT_DIR/recaption_structured_shard0.json
$OUT_DIR/recaption_structured_shard1.json
...
```

---

> Recaption prompt now comes from `recaption.py`. For CRef+SRef/identity-style tasks use
> `PROMPT_WITH_INSTUCTION_CREF_SREF`; for style-transfer tasks use
> `PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER`. The final generator prompt is
> `independent_captions.scene_3 + training_output.primary_instruction_cn_123` from the
> model-returned JSON.


> **重要：输入参考图尺寸与输出 noise 尺寸是分离的。** 现在脚本中 `--width/--height` 只控制 `ImageGenerator.generate_image(width, height)` 的 denoising/noise canvas，也就是输出图尺寸；不会再把 `cref/sref` 强行 resize 到该尺寸。`cref/sref` 会根据 `--resolution_mode follow_cref_aspect` 按 cref 宽高比选择 Gradio 同款 ratio bucket 后再作为参考图输入。

---

## 4. 常用参数

### 数据与 key

```bash
--data_root PATH       # 数据根目录，默认 sample_800_cref_sref_200_content
--prompts_json PATH    # 原始 prompt JSON，默认 {data_root}/prompts.json
--cref_dir PATH        # cref 目录，默认 {data_root}/cref
--sref_dir PATH        # sref 目录，默认 {data_root}/sref
--keys "k1,k2"         # 直接传 key，逗号/换行均可
--key_txt PATH         # 每行一个 key 的文本文件
```

### Recaption

```bash
--recaption_model_path PATH       # 默认 /mnt/jfs/model_zoo/Qwen3-VL-8B-Instruct
--recaption_device cuda:0
--recaption_max_new_tokens 1024
--recaption_image_long_edge 512   # Qwen3-VL recaption 阶段缩图，避免 1024 图过慢
--recaption_image_tokens 188
--recaption_task_type identity_style    # 默认：使用 recaption.py 的 PROMPT_WITH_INSTUCTION_CREF_SREF
--recaption_task_type style_transfer    # 风格迁移任务：使用 PROMPT_WITH_INSTUCTION_CREF_SREF_STYLE_TRANSFER
--recaption_only                  # 只做 recaption，不做生成
--skip_recaption                  # 跳过 recaption，直接读 --recaption_json
--recaption_subprocess            # 默认开启：recaption 子进程隔离，避免 Qwen3-VL 显存残留影响后续 VGO 加载
--no_recaption_subprocess         # 关闭子进程隔离，recaption 和生成在同一 Python 进程内执行
--recaption_json PATH             # key -> final prompt
--structured_json PATH            # recaption debug json
```

### 生成

```bash
--dit_path PATH
--task {sref,cref_sref}    # 选择任务；preset 会自动设置
--use_rope                 # 启用 frequency-aware RoPE 调制
--no_rope                  # 关闭 RoPE 调制
--ae_path PATH
--qwenvl_path PATH
--generator_device cuda:0
--width 1024
--height 1024
--steps 28
--cfg 8
--seed 42
--overwrite
```

---

## 5. 单 key / 少量 key 示例

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615

export VGO_DISABLE_TORCH_COMPILE=1
export VGO_DISABLE_VARLEN_OPS_COMPILE=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32'

python3 cref_sref_core_infer.py \
  --batch_mode \
  --keys '272737929__low_poly,bear__Statue' \
  --out_dir /mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/debug_rope50000 \
  --weight_preset cref_sref_rope_50000 \
  --recaption_device cuda:0 \
  --generator_device cuda:0 \
  --width 1024 --height 1024 \
  --steps 28 --cfg 8 \
  --overwrite
```

---

## 6. 两阶段用法：先 recaption，再生成

### 6.1 只做 recaption

```bash
python3 cref_sref_core_infer.py \
  --key_txt /path/to/keys.txt \
  --out_dir $OUT_DIR \
  --recaption_only \
  --overwrite
```

生成：

```text
$OUT_DIR/recaption_prompts.json
$OUT_DIR/recaption_structured.json
```

### 6.2 跳过 recaption，直接用已有 recaption prompt 生成

```bash
python3 cref_sref_core_infer.py \
  --batch_mode \
  --key_txt /path/to/keys.txt \
  --out_dir $OUT_DIR \
  --skip_recaption \
  --recaption_json $OUT_DIR/recaption_prompts.json \
  --weight_preset cref_sref_rope_50000 \
  --generator_device cuda:0 \
  --width 1024 --height 1024 \
  --steps 28 --cfg 8 \
  --overwrite
```

---

## 7. 4 进程并行全量推理推荐方式

多个进程并行时：

- 先把 `prompts.json` 中所有有效 key round-robin 切成 4 份；
- 每个进程绑定一张 GPU；
- 每个进程使用独立的：
  - `--key_txt`
  - `--recaption_json`
  - `--structured_json`
- 4 个进程可以共用同一个 `--out_dir`，因为图片文件名的 key 不重叠。

示例：

```bash
DATA_ROOT=/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content
WORKDIR=/data/vgo/opensource_cref_sref_core_infer_0615
OUT_DIR=$DATA_ROOT/qwen3_style_guard_rope50000_full_4proc_0611

mkdir -p $OUT_DIR/_keys $OUT_DIR/logs

python3 - <<'PY'
import json
from pathlib import Path
root = Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content')
out = root / 'qwen3_style_guard_rope50000_full_4proc_0611'
prompts = json.load(open(root / 'prompts.json', encoding='utf-8'))
keys = [k for k in prompts if (root/'cref'/f'{k}.png').exists() and (root/'sref'/f'{k}.png').exists()]
for i in range(4):
    shard = keys[i::4]
    with open(out/'_keys'/f'shard_{i}.txt', 'w', encoding='utf-8') as f:
        for k in shard:
            f.write(k + '\n')
print('total_valid', len(keys), 'shards', [len(keys[i::4]) for i in range(4)])
PY

for i in 0 1 2 3; do
  GPU=$i
  CUDA_VISIBLE_DEVICES=$GPU \
  VGO_DISABLE_TORCH_COMPILE=1 \
  VGO_DISABLE_VARLEN_OPS_COMPILE=1 \
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True,max_split_size_mb:32' \
  PYTHONPATH=$WORKDIR \
  nohup python3 $WORKDIR/cref_sref_core_infer.py \
    --batch_mode \
    --key_txt $OUT_DIR/_keys/shard_${i}.txt \
    --out_dir $OUT_DIR \
    --recaption_json $OUT_DIR/recaption_prompts_shard${i}.json \
    --structured_json $OUT_DIR/recaption_structured_shard${i}.json \
    --weight_preset cref_sref_rope_50000 \
    --recaption_device cuda:0 \
    --generator_device cuda:0 \
    --width 1024 --height 1024 \
    --steps 28 --cfg 8 \
    --overwrite \
    > $OUT_DIR/logs/shard_${i}.log 2>&1 &
  echo $! > $OUT_DIR/logs/shard_${i}.pid
done
```

查看进度：

```bash
tail -f $OUT_DIR/logs/shard_0.log
watch -n 30 'ls $OUT_DIR/*.png 2>/dev/null | wc -l'
```

合并分片 recaption prompt：

```bash
python3 - <<'PY'
import json
from pathlib import Path
out = Path('/mnt/jfs/bench-bucket/sref_bench/sample_800_cref_sref_200_content/qwen3_style_guard_rope50000_full_4proc_0611')
merged = {}
for p in sorted(out.glob('recaption_prompts_shard*.json')):
    merged.update(json.load(open(p, encoding='utf-8')))
json.dump(merged, open(out/'recaption_prompts_merged.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('merged', len(merged))
PY
```

## 8. Launcher scripts for common weights

两个 launcher 都是极简示例，参数已经硬编码在 `.sh` 里；如果要换权重/数据/输出目录，直接编辑脚本里的 `python3 cref_sref_core_infer.py ...` 参数即可。

SRef inference:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
bash run_sref_infer.sh
# 在脚本里把 --weight_preset 改成 sref_14000 或 sref_12000
```

CRef+SRef inference:

```bash
cd /data/vgo/opensource_cref_sref_core_infer_0615
bash run_cref_sref_infer.sh
# 在脚本里把 --weight_preset 改成：
#   cref_sref_rope_50000       # RoPE 权重 + RoPE config
#   cref_sref_40000            # no-RoPE 40000 权重 + normal config
#   cref_sref_36000_no_rope    # no-RoPE 36000 权重 + normal config
```

The same presets can be used directly with Python through `--weight_preset`:

```bash
--weight_preset sref_14000
--weight_preset sref_12000
--weight_preset cref_sref_rope_50000
--weight_preset cref_sref_40000
--weight_preset cref_sref_36000_no_rope
```
