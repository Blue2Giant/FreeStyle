<!-- 中文 | [English](README.en.md) -->

# Sref/Cref 风格-内容参考图像生成与评测工具集

围绕「内容参考图（cref）+ 风格参考图（sref）+ 文本指令」的图像生成与一致性评测流水线。
包含三类入口：**推理生成（inference）**、**图像打标（caption）**、**指标评测（metrics）**。

---

## 目录结构

```
.
├── scripts/            # 所有可执行入口脚本（.sh），按用途分三类
│   ├── inference/      # 跑各模型生成图像（flux / qwen / telestyle / seedream / csgo / uso / omnistyle）
│   ├── caption/        # 用 Gemini / GPT-4o 给参考图打标
│   └── metrics/        # 一致性 / 美学 / 指令遵循评测（encoder 指标 + VLM 裁判，uso_metric_batch.sh 为一站式入口）
├── src/                # Python 实现，与 scripts/ 一一对应
│   ├── inference/      # 推理脚本 + CSGO/ USO/ OmniStyle/ 三个模型仓库
│   ├── caption/        # 打标脚本
│   └── metrics/
│       ├── encoder/    # 编码器指标实现（CSD/OneIG/DINOv2/CAS/CLIP-T/美学，含 CSD/ 子模块）
│       └── vlm/        # 基于 VLM 的相似度 / 裁判实现
├── requirements.txt    # 依赖（从 Sref conda 环境导出）
└── README.md / README.en.md
```

每个 `.sh` 顶部都会自动定位仓库根目录，无需手动改脚本里对 Python 文件的引用：

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
```

---

## 环境安装

环境基于 **Python 3.10**，依赖见 `requirements.txt`（从 `Sref` conda 环境导出）。

```bash
conda create -n sref python=3.10 -y
conda activate sref

# torch 系列建议用对应 CUDA 的官方 wheel（requirements 里 pin 的是 cu124）：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

说明：
- `clip` 来自 OpenAI 官方 git 仓库，已在 `requirements.txt` 中以 git URL 形式给出。
- `flash-attn` 与 `step-vault` 在导出文件中被注释掉（原环境是本地 wheel）。二者对本工具集**非必需**，需要时自行安装。
- `requirements.txt` 是从 conda 环境完整导出的（含全部传递依赖），可按需精简。

---

## 需要自行准备的外部资源

这些是环境相关的路径，仓库内**未包含**，请按需下载并修改脚本中的路径常量。

### 1. 模型权重（默认指向 `/mnt/jfs/model_zoo/...`）

| 用途 | 脚本中变量 | 模型 |
|---|---|---|
| 风格一致性 CSD | `CSD_MODEL_ONLY` | `vit-b-300ep.pth.tar`（参考 [learn2phoenix/CSD](https://github.com/learn2phoenix/CSD)） |
| 风格一致性 OneIG | `ONEIG_MODEL` / `VIT_L` | `OneIG-StyleEncoder`、`ViT-L-14.pt`（参考 [OneIG-Bench/OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark)） |
| 内容一致性 DINOv2 | `DINOV2_MODEL` | `dinov2-with-registers-large` |
| 内容一致性 CAS | `CAS_MODEL` | `dinov2-base` |
| 指令遵循 CLIP-T | — | `clip-vit-base-patch32` |
| 美学 LAION | `--laion_clip_ckpt` | open_clip `ViT-L-14` + linear 头（参考 [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)） |
| 美学 v2.5 | `--v25_encoder_model_name` | `siglip-so400m-patch14-384`（参考 [discus0434/aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5)） |
| 美学 OneAlign | `ONEALIGN_MODEL` | `one-align` |
| 推理 FLUX | `--model_name` | `FLUX.2-klein-9B` |
| 推理 Qwen | `--model_name` | `Qwen-Image-Edit-2511` |
| 推理 TeleStyle | `TELESTYLE_DIR` | `Tele-AI/TeleStyle` |
| 推理 CSGO | 脚本内 `.py` 常量 | `stable-diffusion-xl-base-1.0` + IP-Adapter + `csgo_4_32.bin` + `sdxl-vae-fp16-fix` + ControlNet tile + BLIP |
| 推理 USO | `batch_simple_demo.py` 顶部常量 | `FLUX.1-dev`（`flux1-dev.safetensors` + `ae.safetensors`）+ `t5-xxl` + `clip-vit-large-patch14` + `siglip` |
| 推理 OmniStyle | `T5`/`CLIP`/`FLUX_DEV`/`AE`/`LORA`/`SIGLIP_PATH` | `FLUX.1-dev` + `t5-xxl` + `clip-vit-large-patch14` + `siglip` + `dit_lora.safetensors` |

#### CSGO / USO / OmniStyle 权重放置位置

这三个模型的权重路径目前**硬编码**在各自的推理代码 / 入口脚本里（指向我们机器上的本地路径）。
下表说明每个权重在代码中的位置，以及对应的官方仓库（请到官方仓库 / HuggingFace 按说明下载，再把下面的路径改成你自己的下载目录）。

**CSGO** —— 官方仓库 [instantX-research/CSGO](https://github.com/instantX-research/CSGO)，权重 [HuggingFace: InstantX/CSGO](https://huggingface.co/InstantX/CSGO)。
路径常量在 `src/inference/CSGO/infer_csgo_ljh_batch.py`：

| 权重 | 代码位置（默认值） |
|---|---|
| SDXL base | `base_model_path = /mnt/jfs/model_zoo/stable-diffusion-xl-base-1.0` |
| IP-Adapter image encoder | `image_encoder_path = /mnt/jfs/model_zoo/IP-Adapter/sdxl_models/image_encoder` |
| CSGO ckpt | `csgo_ckpt = /mnt/jfs/model_zoo/CSGO/csgo_4_32.bin` |
| SDXL VAE | `pretrained_vae_name_or_path = /mnt/jfs/model_zoo/sdxl-vae-fp16-fix` |
| ControlNet Tile | `controlnet_path = /mnt/jfs/model_zoo/TTPLanet_SDXL_Controlnet_Tile_Realistic` |
| BLIP caption | `/mnt/jfs/model_zoo/blip-image-captioning-large` |

**USO** —— 官方仓库 [bytedance/USO](https://github.com/bytedance/USO)，权重 [HuggingFace: bytedance-research/USO](https://huggingface.co/bytedance-research/USO)。
路径以环境变量形式写在 `src/inference/USO/batch_simple_demo.py` 顶部：

| 权重 | 代码位置（默认值） |
|---|---|
| FLUX.1-dev | `FLUX_DEV = /data/USO/weights/FLUX.1-dev/flux1-dev.safetensors` |
| FLUX VAE | `AE = /data/USO/weights/FLUX.1-dev/ae.safetensors` |
| USO LoRA | `LORA = /data/USO/weights/USO/uso_flux_v1.0/dit_lora.safetensors` |
| USO projector | `PROJECTION_MODEL = /data/USO/weights/USO/uso_flux_v1.0/projector.safetensors` |
| SigLIP | `SIGLIP_PATH = /data/USO/weights/siglip` |
| T5-XXL | `T5 = /data/USO/weights/t5-xxl` |
| CLIP | `CLIP = /mnt/jfs/model_zoo/clip-vit-large-patch14` |

**OmniStyle** —— 官方仓库 [StyleX-Research/OmniStyle](https://github.com/StyleX-Research/OmniStyle)（作者镜像 [wangyePHD/OmniStyle](https://github.com/wangyePHD/OmniStyle)），权重见仓库内的 HuggingFace 链接。
路径通过入口脚本 `scripts/inference/omnistyle_demo.sh` 里的环境变量传入（也可改 `run_omnistyle.py` 顶部默认值）：

| 权重 | 入口脚本变量（默认值） |
|---|---|
| FLUX.1-dev | `FLUX_DEV = /data/USO/weights/FLUX.1-dev/flux1-dev.safetensors` |
| FLUX VAE | `AE = /data/USO/weights/FLUX.1-dev/ae.safetensors` |
| OmniStyle LoRA | `LORA = /data/Sref_Cref/OmniStyle/pretrained/dit_lora.safetensors` |
| T5-XXL | `T5 = /data/USO/weights/t5-xxl` |
| CLIP | `CLIP = /mnt/jfs/model_zoo/clip-vit-large-patch14/` |
| SigLIP | `SIGLIP_PATH = /data/USO/weights/siglip` |

### 2. API Key 与服务端点（占位符）

脚本中以下占位符需替换为你自己的值：
- `YOUR_API_KEY` —— OpenAI / Gemini / Seedream 等服务的密钥（可改为从环境变量读取）。
- `YOUR_QWEN_VLM_HOST:22002` —— **你自己部署的 Qwen3-VL 服务地址**（见下方「部署 Qwen3-VL 服务」），所有 VLM 评测脚本（`*_similarity_batch.sh`、`instruction_score.sh`、`triplet_qwen_dual_judge.sh`、`uso_metric_batch.sh`）都通过它访问模型。脚本里对应变量为 `xingpeng_ip`（base_url）和 `xingpeng_model`（服务的模型名，默认 `Qwen3-VL-30B-A3B-Instruct`，需与启动命令的 `--served-model-name` 一致）。
- `YOUR_OPENAI_COMPAT_ENDPOINT` —— GPT-4o / Gemini 等的 OpenAI 兼容代理地址（仅 GPT-4o 裁判 / 打标脚本使用）。

#### 部署 Qwen3-VL 服务（VLM 评测前置）

VLM 相关的判别脚本需要**先**起一个 Qwen3-VL（Instruct）服务，提供 OpenAI 兼容接口。
可以参考 [Qwen 官方仓库](https://github.com/QwenLM/Qwen3-VL) 部署，也可以直接用我们使用的
vLLM 启动方式（本项目即以此方式提供服务）：

```bash
vllm serve /mnt/jfs/model_zoo/Qwen3-VL-30B-A3B-Instruct/ \
  --tensor-parallel-size 4 \
  --enable-prefix-caching False \
  --async-scheduling \
  --host 0.0.0.0 \
  --port 22002 \
  --gpu-memory-utilization 0.8 \
  --served-model-name "Qwen3-VL-30B-A3B-Instruct" \
  --mm-processor-cache-gb 0
```

启动后服务的 OpenAI 兼容地址即为 `http://<本机IP>:22002/v1`。把判别脚本里的
`YOUR_QWEN_VLM_HOST` 改成该地址的主机（端口默认与上面的 `--port` 一致），
`xingpeng_model` 与 `--served-model-name` 保持一致，即可用于后续 VLM 评测。

### 3. 数据布局

评测 / 推理用的 benchmark 已开源在 HuggingFace：
[**Blue2Giant/FreeStyle_Bench**](https://huggingface.co/datasets/Blue2Giant/FreeStyle_Bench)。
其中包含两种任务的 benchmark：

- **sref**：风格参考任务（仅风格参考图 + 文本指令）。
- **cref_sref**：内容 + 风格参考任务（内容参考图 + 风格参考图 + 文本指令）。

每个任务的 benchmark 目录结构如下，脚本默认数据根目录即指向其中某一个任务：

```
<DATA_ROOT>/            # 某个任务的 benchmark 根目录
├── cref/            # 内容参考图（编号 000000.png ...）
├── sref/            # 风格参考图
├── prompts.json     # {"000000": "指令文本", ...}
└── <model_name>/    # 各模型生成结果（评测输出的 *.json 也写在这里）
```

脚本里数据根目录目前硬编码为 `/mnt/jfs/bench-bucket/sref_bench/...`，请下载上面的 benchmark 后改成你的路径。

---

## 使用

> 运行前请先按上一节替换模型路径、API key、端点和数据根目录。

### 推理生成

```bash
bash scripts/inference/Qwen_2511_demo.sh        # 单模型 demo
bash scripts/inference/flux_klein_9B_demo.sh
bash scripts/inference/TeleStyle_demo.sh
bash scripts/inference/seedream.sh              # Seedream（API）

bash scripts/inference/csgo_infer.sh            # CSGO（SDXL + IP-Adapter）
bash scripts/inference/uso_batch_run.sh         # USO（FLUX.1-dev + LoRA），单卡
bash scripts/inference/omnistyle_demo.sh        # OmniStyle（FLUX.1-dev + LoRA）
```

### 图像打标

```bash
bash scripts/caption/gemini.sh
bash scripts/caption/gpt4o.sh
```

### 指标评测

一站式批量评测（最常用，覆盖编码器指标 + VLM 指标）：

```bash
bash scripts/metrics/uso_metric_batch.sh
```

也可单独跑某一类：

```bash
bash scripts/metrics/encoder_batch.sh           # CSD/OneIG/DINOv2/CAS/CLIP-T/美学
bash scripts/metrics/style_similarity_batch.sh  # VLM 风格一致性
bash scripts/metrics/content_similarity_batch.sh
bash scripts/metrics/instruction_score.sh
bash scripts/metrics/triplet_qwen_dual_judge.sh
```

评测指标概览：

| 维度 | 实现 | 指标 |
|---|---|---|
| 风格一致性 | encoder | CSD、OneIG |
| 内容一致性 | encoder | DINOv2、CAS |
| 指令遵循 | encoder / VLM | CLIP-T、VLM 打分 |
| 美学质量 | encoder | LAION、Aesthetic v2.5、OneAlign |
| 综合裁判 | VLM | Qwen3-VL / GPT-4o 三元组双裁判 |

---

## 备注

- `src/**/` 下的 `*_demo.py` 含示例图片路径（`assets/`、`logs/`）等占位，仅作单文件 demo，运行批量入口脚本时不依赖它们。
- VLM 相关脚本通过 OpenAI 兼容接口访问模型，可对接任意兼容服务。
