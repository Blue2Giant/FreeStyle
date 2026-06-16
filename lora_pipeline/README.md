# Opensource LoRA Pipeline

本仓库用于开源 LoRA 模型的**批量数据制作**：通过 ComfyUI 对 Civitai 上的开源 LoRA 进行批量推理出图，并提供相关的 meta 信息（模型 ID、触发词、prompt 模板、ComfyUI workflow 等）。

---

## 目录结构

```
opensource_lora_pipeline/
├── README.md                          # 本文件
├── comfykit/                          # ComfyUI Python SDK（异步执行引擎）
│   ├── comfyui/                       #   HTTP/WebSocket executor、workflow parser
│   └── utils/                         #   工具函数（配置、网络、图像、文件等）
├── meta/                              # 元数据与配置
│   ├── comfyui_start_new_server.sh    #   ComfyUI 多 GPU 服务启动脚本
│   ├── gemini_trigger.txt             #   人工核验的稳定风格触发词列表
│   ├── model_ids/                     #   各架构 LoRA 的 Civitai model ID 列表
│   ├── prompts/                       #   prompt 模板与触发词词表
│   ├── triplet_keys/                  #   triplet JSONL 中提取的 lora key 列表
│   └── workflows/                     #   ComfyUI workflow JSON 模板
├── scripts/                           # 批量推理启动脚本（多机多卡分发）
├── generate_word_table/               # 从 CSV 生成 prompt 词表的工具
├── illustrious_one_lora_diverse.py    # Illustrious 单 LoRA 推理主逻辑（基类）
├── one_lora_flux.py                   # Flux 单 LoRA 推理
├── one_lora_qwen.py                   # Qwen 单 LoRA 推理
├── dual_lora_flux.py                  # Flux 双 LoRA 推理
├── dual_lora_illustrious.py           # Illustrious 双 LoRA 推理
├── dual_lora_qwen.py                  # Qwen 双 LoRA 推理
├── probe_comfy_ports.py               # 批量探测 ComfyUI 实例端口可达性
├── comfykit_demo.py                   # ComfyKit SDK 使用示例
├── comfykit_demo.sh                   # demo 启动脚本
└── comfykit_demo_stress.py            # 压力测试示例
```

---

## 一、ComfyUI 服务启动

脚本位于：`meta/comfyui_start_new_server.sh`

### 1. 创建独立输出目录

- 在 `/mnt/jfs/comfyui_output/` 下按 `{主机名}_{时间戳}_{PID}` 格式创建本次运行的唯一输出目录
- 避免多机/多次启动产生冲突

### 2. 准备运行环境（在 `/workspace/ComfyUI` 下）

| 操作 | 说明 |
|------|------|
| 拷贝 custom_nodes | 从 `/mnt/jfs/comfyui_nodes/custom_nodes` 全量拷贝到 `/workspace/ComfyUI/custom_nodes`，确保插件版本一致 |
| 超分模型软链 | `models/upscale_models` → `/mnt/jfs/model_zoo/comfyui/` |
| 额外模型路径配置 | 从 `/data/ComfyUI/extra_model_paths.yaml` 拷贝到工作目录 |
| 输入目录软链 | `models/input` → `/mnt/jfs/comfyui_input/` |
| 输出目录软链 | `output` → 本次运行的独立输出目录 |

### 3. 多 GPU 并行启动

- 自动检测 GPU 数量（通过 `nvidia-smi`）
- 每张 GPU 启动一个 ComfyUI 实例，端口从 8188 递增（GPU0→8188，GPU1→8189，...）
- 所有实例后台运行，脚本用 `wait` 阻塞直到所有实例退出

### 4. 端口分配示例

```
GPU 0 → http://0.0.0.0:8188
GPU 1 → http://0.0.0.0:8189
GPU 2 → http://0.0.0.0:8190
...
```

---

## 二、额外模型路径配置（extra_model_paths.yaml）

文件位置：`/data/ComfyUI/extra_model_paths.yaml`

```yaml
comfyui:
    checkpoints: /mnt/jfs/model_zoo/comfyui/
    text_encoders: /mnt/jfs/model_zoo/comfyui/
    clip_vision: /mnt/jfs/model_zoo/comfyui/
    diffusion_models: /mnt/jfs/model_zoo/comfyui/
    controlnet: models/controlnet/
    loras: |
        /mnt/jfs/model_zoo/style_lora/  #downloaded community lora place in it
    vae: /mnt/jfs/model_zoo/comfyui/
    audio_encoders: models/audio_encoders/
    LLM: /mnt/jfs/model_zoo/comfyui/LLM/
    unet: /mnt/jfs/model_zoo/comfyui/unet/
    clip: /mnt/jfs/model_zoo/comfyui/clip/
```

**要点：**
- 大部分模型统一放在 `/mnt/jfs/model_zoo/comfyui/`
- LoRA 配置了多个来源目录，覆盖 Flux.1 D、Qwen、Illustrious、SDXL 1.0 等架构
- LLM 和 unet/clip 有独立子目录

### LoRA 权重下载说明

你需要从 [Civitai](https://civitai.com) 下载对应的开源 LoRA 权重，并放置到上述 loras 目录中。

**下载方式：** 通过 `https://civitai.com/models/{model_id}` 访问对应模型页面下载权重文件。

**放置规则：**

| 架构 | 目标目录 | Model ID 列表文件 |
|------|---------|-----------------|
| Flux.1 D | `/mnt/jfs/all_loras/civitai/'Flux.1 D'/` | `meta/model_ids/flux_content_one_lora.txt`、`meta/model_ids/flux_style_one_lora.txt` |
| Qwen | `/mnt/jfs/all_loras/civitai/Qwen/` | `meta/model_ids/qwen_content_one_lora.txt`、`meta/model_ids/qwen_style_one_lora.txt` |
| Illustrious | `/mnt/jfs/all_loras/civitai/Illustrious/` | `meta/model_ids/illustrious_content_one_lora.txt`、`meta/model_ids/illustrious_style_one_lora.txt` |

每个 txt 文件中每行一个 model_id，例如 `1041877` 对应下载链接为：
```
https://civitai.com/models/1041877
```

下载后将 `.safetensors` 权重文件放入对应架构的目录即可。

---

## 三、Custom Nodes 列表

以下是我们使用的 ComfyUI 自定义节点（源自 `/data/ComfyUI/custom_nodes/`）：

| 节点 | 用途简述 |
|------|---------|
| **ComfyUI-Manager** | 插件管理器，安装/更新其他节点 |
| **ComfyUI-Impact-Pack** | 检测、分割、细节增强等综合工具包 |
| **comfyui-inspire-pack** | 灵感工具包，提供额外采样/控制节点 |
| **ComfyUI_Comfyroll_CustomNodes** | 通用工具集（数学、文本、图像处理） |
| **comfyui-easy-use** | 简化工作流的快捷节点 |
| **comfyui_essentials** | 核心增强节点（裁剪、混合、遮罩等） |
| **ComfyUI_ADV_CLIP_emb** | 高级 CLIP 文本编码控制 |
| **ComfyUI-Custom-Scripts** | 自定义脚本集合（预览、自动排列等） |
| **comfyui-kjnodes** | KJ 工具节点（批处理、条件等） |
| **ComfyUI_LayerStyle** | 图层样式处理 |
| **ComfyLiterals** | 字面量输入节点（字符串、数字等） |
| **rgthree-comfy** | 工作流效率工具（Reroute、Mute 等） |
| **comfyui-image-saver** | 图像保存节点（多格式、元数据） |
| **comfyui-saveimage-plus** | 增强版图像保存 |
| **ComfyUI-EsesImageCompare** | 图像对比节点 |
| **eden_comfy_pipelines** | Eden 工作流管线 |
| **comfyui-various** | 杂项工具节点 |
| **comfyui-yaser-nodes** | Yaser 自定义节点 |
| **ComfyUI-Jjk-Nodes** | JJK 自定义节点 |
| **ComfyUI-HunyuanVideoMultiLora** | 混元视频多 LoRA 支持 |
| **ComfyUI_QwenVL** | Qwen VL 模型集成 |
| **ComfyUI_Swwan** | Swwan 自定义节点 |
| **SCG_LocalVLM** | 本地 VLM 推理节点 |
| **qweneditutils** | Qwen Edit 辅助工具 |
| **ysc_highresfix** | 高分辨率修复 |
| **websocket_image_save.py** | WebSocket 图像保存脚本 |

---

## 四、Meta 信息

### 4.1 风格触发词列表（gemini_trigger.txt）

文件位置：`meta/gemini_trigger.txt`

该文件是经过**人工核验**的稳定风格触发词列表，供 Gemini 模型使用。列表中每行一个触发词

**用途：** 在使用 Gemini 进行风格判断/分类时，以此列表作为标准风格词汇表。这些触发词已经验证能被 Gemini 稳定识别和区分，避免使用未经验证的触发词导致判断结果不稳定。

**数量：** 共 622 个风格触发词，涵盖传统绘画（油画、水彩、版画）、现代艺术流派（超现实主义、波普、极简）、数字艺术、摄影风格、游戏美术、动画风格、民族/地域艺术等大类。

### 4.2 Model ID 列表（meta/model_ids/）

按架构和 LoRA 类型分文件存放 Civitai model ID：

| 文件 | 数量 | 说明 |
|------|------|------|
| `flux_content_one_lora.txt` | 91 | Flux 内容型单 LoRA |
| `flux_style_one_lora.txt` | 1460 | Flux 风格型单 LoRA |
| `flux__dual_lora.txt` | 23130 | Flux 双 LoRA 组合 |
| `illustrious_content_one_lora.txt` | 799 | Illustrious 内容型单 LoRA |
| `illustrious_style_one_lora.txt` | 191 | Illustrious 风格型单 LoRA |
| `illustrious_dual_lora.txt` | 24646 | Illustrious 双 LoRA 组合 |
| `qwen_content_one_lora.txt` | 19 | Qwen 内容型单 LoRA |
| `qwen_style_one_lora.txt` | 53 | Qwen 风格型单 LoRA |
| `qwen_dual_lora.txt` | 608 | Qwen 双 LoRA 组合 |

单 LoRA key 格式：`1041877`；双 LoRA key 格式：`1041877__1001511`（两个 ID 用双下划线连接）。

### 4.3 Prompt 模板（meta/prompts/）

| 文件 | 说明 |
|------|------|
| `diverse_prompts_100.txt` | 100 条多样化场景 prompt（用于单 LoRA 推理） |
| `STYLE_UNIVERSE_TRIGGER.csv/.txt` | 风格 LoRA 通用触发词词表 |
| `CHARACTER_UNIVERSE_TRIGGER.csv/.txt` | 角色 LoRA 通用触发词词表 |
| `OTHER_UNIVERSE_TRIGGER.csv/.txt` | 其他类型通用触发词词表 |
| `TRIPLET_UNIVERSE_TRIGGER.txt` | Triplet 评测通用触发词 |

### 4.4 Workflow 模板（meta/workflows/）

| 文件 | 说明 |
|------|------|
| `flux_dual_lora.json` | Flux 双 LoRA workflow |
| `flux_full_lora-2.json` | Flux 完整 LoRA workflow |
| `illustrious_simple.json` | Illustrious 简化 workflow |
| `qwen_dual_lora.json` | Qwen 双 LoRA workflow |
| `qwen_one_lora0320.json` | Qwen 单 LoRA workflow |
| `sdxl_dual_lora_ljh.json` | SDXL 双 LoRA workflow |

---

## 五、批量生成lora数据

### 5.1 推理脚本（Python）

各推理脚本基于 `comfykit` SDK 异步调用 ComfyUI 实例：

- `illustrious_one_lora_diverse.py` — 基类，包含通用推理逻辑（遍历 LoRA、注入 workflow、收集输出）
- `one_lora_flux.py` / `one_lora_qwen.py` — 单 LoRA 推理，继承基类并注入对应 workflow
- `dual_lora_flux.py` / `dual_lora_illustrious.py` / `dual_lora_qwen.py` — 双 LoRA 推理

### 5.2 启动脚本（`scripts/`）

Shell 脚本是**批量跑数据**的入口，每个脚本通过 `while true` 无限循环持续调用 Python 推理脚本，遍历 meta/model_ids/ 下的 model ID 列表和 meta/prompts/ 下的 prompt 模板，在多台 ComfyUI 服务器上并发出图。

脚本内定义了本次任务的关键参数（目标服务器 IP、LoRA 根目录、prompt 文件、model ID 文件、workflow 模板等），只需修改对应变量即可启动不同类型的批量推理任务。

| 脚本 | 任务类型 |
|------|---------|
| `scripts/one_lora_flux.sh` | Flux 单 LoRA 批量出图（style / character / other 三组循环） |
| `scripts/one_lora_qwen.sh` | Qwen 单 LoRA 批量出图（style / character / other 三组循环） |
| `scripts/illustrious_one_lora_diverse.sh` | Illustrious 单 LoRA 批量出图（style / character / other 三组循环） |
| `scripts/dual_lora_flux.sh` | Flux 双 LoRA 批量出图 |
| `scripts/dual_lora_illustrious.sh` | Illustrious 双 LoRA 批量出图 |
| `scripts/dual_lora_qwen.sh` | Qwen 双 LoRA 批量出图 |

```bash
# 示例：启动 Flux 单 LoRA 批量推理（将持续运行直到手动中断）
bash scripts/one_lora_flux.sh

# 示例：启动 Illustrious 双 LoRA 批量推理
bash scripts/dual_lora_illustrious.sh
```

### 5.3 端口探测工具

在推理前可用 `probe_comfy_ports.py` 检查各机器 ComfyUI 端口是否可达：

```bash
python probe_comfy_ports.py \
  --shell-file scripts/dual_lora_flux.sh \
  --start-port 8188 \
  --port-count 8 \
  --timeout-sec 2 \
  --concurrency 256
```

---

## 六、ComfyKit SDK

`comfykit/` 是一个异步 ComfyUI 执行引擎，支持：

- HTTP/WebSocket 两种通信方式
- 连接池管理（`session_pool_size`）
- Workflow JSON 动态参数注入
- RunningHub 远程执行支持

**环境配置与详细文档：** 请参考 [ComfyKit GitHub 仓库](https://github.com/puke3615/ComfyKit) 进行环境安装和更多用法理解。

**基本用法：**

```python
from comfykit import ComfyKit

async with ComfyKit(comfyui_url="http://host:8188", session_pool_size=2) as kit:
    result = await kit.execute("workflow.json")
    print(result.images)
```

---

## 七、快速开始

```bash
# 1. 启动 ComfyUI 服务（多GPU并行）
bash meta/comfyui_start_new_server.sh

# 2. 探测端口可达性
python probe_comfy_ports.py --shell-file scripts/one_lora_flux.sh --start-port 8188 --port-count 8

# 3. 启动批量推理
bash scripts/one_lora_flux.sh
```

启动后通过 `http://<host>:8188` 访问第一个实例的 Web UI。

---

## 八、生成图片的相似性判别

LoRA 批量生成的图片需要通过相似性判别来筛选，我们使用 VLM（Qwen3-VL）进行内容与画风的双重判别。

### 判别脚本

位置：`/data/FreeStyle/benchmark_infer/scripts/metrics/triplet_qwen_dual_judge.sh`

该脚本对每张生成图片同时做两路判别：

- **内容判别**：与 content 参考图比较主体内容和主题是否一致
- **风格判别**：与 style 参考图比较画风/视觉风格是否一致

### 判别方式：Logits 拒绝采样

与传统的离散打分（如 1-5 分制）不同，我们采用 **logits 拒绝采样**方式：

1. VLM 以 `temperature=0.0` 生成，请求 `top_logprobs` 获取输出 token 的 logit 分布
2. 从 logit 分布中提取 "0" 和 "1" 两个 token 的 log probability（logp0、logp1）
3. 通过 `sigmoid(logp1 - logp0)` 计算连续相似度分数（0~1）

这样即使 VLM 的离散输出是 "1"，如果 logp1 和 logp0 很接近（置信度低），分数也会偏低，相当于一种**软拒绝采样**——对置信度不足的样本自动降低权重或丢弃。

实测效果准确率要好于离散的打分判别方式。
