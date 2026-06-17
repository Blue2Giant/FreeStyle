<!-- English | [中文](README.md) -->

# Sref/Cref Style–Content Reference Image Generation & Evaluation Toolkit

A generation and consistency-evaluation pipeline built around **content reference (cref) + style reference (sref) + text instruction**.
It exposes three kinds of entry points: **inference** (generate images), **caption** (label reference images), and **metrics** (evaluate results).

---

## Directory Layout

```
.
├── scripts/            # All executable entry scripts (.sh), grouped by purpose
│   ├── inference/      # Generate images per model (flux / qwen / telestyle / seedream / csgo / uso / omnistyle)
│   ├── caption/        # Caption reference images via Gemini / GPT-4o
│   └── metrics/        # Consistency / aesthetic / instruction-following eval (encoder metrics + VLM judges; uso_metric_batch.sh is the one-stop entry)
├── src/                # Python implementation, mirroring scripts/
│   ├── inference/      # Inference scripts + CSGO/ USO/ OmniStyle/ model repos
│   ├── caption/        # Captioning scripts
│   └── metrics/
│       ├── encoder/    # Encoder-metric impl (CSD/OneIG/DINOv2/CAS/CLIP-T/aesthetic, incl. CSD/ submodule)
│       └── vlm/        # VLM-based similarity / judge impl
├── requirements.txt    # Dependencies (exported from the Sref conda env)
└── README.md / README.en.md
```

Every `.sh` resolves the repo root automatically, so you don't need to edit the Python-file
references inside the scripts:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
```

---

## Installation

The environment is based on **Python 3.10**; dependencies are listed in `requirements.txt`
(exported from the `Sref` conda environment).

```bash
conda create -n sref python=3.10 -y
conda activate sref

# Install the torch family from the official CUDA wheels (requirements pins cu124):
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Notes:
- `clip` comes from OpenAI's official git repo and is listed as a git URL in `requirements.txt`.
- `flash-attn` and `step-vault` are commented out in the export (they were local wheels in the
  original environment). Neither is **required** for this toolkit; install them yourself if needed.
- `requirements.txt` is a full export from the conda environment (including all transitive
  dependencies) and can be trimmed as needed.

---

## External Resources You Must Provide

These are environment-specific paths that are **not** included in the repo. Download them as
needed and update the path constants in the scripts.

### 1. Model weights (default paths point to `/mnt/jfs/model_zoo/...`)

| Purpose | Script variable | Model |
|---|---|---|
| Style consistency CSD | `CSD_MODEL_ONLY` | `vit-b-300ep.pth.tar` (ref. [learn2phoenix/CSD](https://github.com/learn2phoenix/CSD)) |
| Style consistency OneIG | `ONEIG_MODEL` / `VIT_L` | `OneIG-StyleEncoder`, `ViT-L-14.pt` (ref. [OneIG-Bench/OneIG-Benchmark](https://github.com/OneIG-Bench/OneIG-Benchmark)) |
| Content consistency DINOv2 | `DINOV2_MODEL` | `dinov2-with-registers-large` |
| Content consistency CAS | `CAS_MODEL` | `dinov2-base` |
| Instruction following CLIP-T | — | `clip-vit-base-patch32` |
| Aesthetic LAION | `--laion_clip_ckpt` | open_clip `ViT-L-14` + linear head (ref. [LAION-AI/aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)) |
| Aesthetic v2.5 | `--v25_encoder_model_name` | `siglip-so400m-patch14-384` (ref. [discus0434/aesthetic-predictor-v2-5](https://github.com/discus0434/aesthetic-predictor-v2-5)) |
| Aesthetic OneAlign | `ONEALIGN_MODEL` | `one-align` |
| Inference FLUX | `--model_name` | `FLUX.2-klein-9B` |
| Inference Qwen | `--model_name` | `Qwen-Image-Edit-2511` |
| Inference TeleStyle | `TELESTYLE_DIR` | `Tele-AI/TeleStyle` |
| Inference CSGO | `.py` constants in script | `stable-diffusion-xl-base-1.0` + IP-Adapter + `csgo_4_32.bin` + `sdxl-vae-fp16-fix` + ControlNet tile + BLIP |
| Inference USO | constants at top of `batch_simple_demo.py` | `FLUX.1-dev` (`flux1-dev.safetensors` + `ae.safetensors`) + `t5-xxl` + `clip-vit-large-patch14` + `siglip` |
| Inference OmniStyle | `T5`/`CLIP`/`FLUX_DEV`/`AE`/`LORA`/`SIGLIP_PATH` | `FLUX.1-dev` + `t5-xxl` + `clip-vit-large-patch14` + `siglip` + `dit_lora.safetensors` |

#### Where to put CSGO / USO / OmniStyle weights

The weight paths for these three models are currently **hardcoded** in their inference code /
entry scripts (pointing to local paths on our machine). The tables below show where each weight
is referenced in the code and the corresponding upstream repo — download the weights from the
official repo / HuggingFace, then change the paths to your own download directory.

**CSGO** — official repo [instantX-research/CSGO](https://github.com/instantX-research/CSGO), weights [HuggingFace: InstantX/CSGO](https://huggingface.co/InstantX/CSGO).
Path constants live in `src/inference/CSGO/infer_csgo_ljh_batch.py`:

| Weight | Location in code (default) |
|---|---|
| SDXL base | `base_model_path = /mnt/jfs/model_zoo/stable-diffusion-xl-base-1.0` |
| IP-Adapter image encoder | `image_encoder_path = /mnt/jfs/model_zoo/IP-Adapter/sdxl_models/image_encoder` |
| CSGO ckpt | `csgo_ckpt = /mnt/jfs/model_zoo/CSGO/csgo_4_32.bin` |
| SDXL VAE | `pretrained_vae_name_or_path = /mnt/jfs/model_zoo/sdxl-vae-fp16-fix` |
| ControlNet Tile | `controlnet_path = /mnt/jfs/model_zoo/TTPLanet_SDXL_Controlnet_Tile_Realistic` |
| BLIP caption | `/mnt/jfs/model_zoo/blip-image-captioning-large` |

**USO** — official repo [bytedance/USO](https://github.com/bytedance/USO), weights [HuggingFace: bytedance-research/USO](https://huggingface.co/bytedance-research/USO).
Paths are set as environment variables at the top of `src/inference/USO/batch_simple_demo.py`:

| Weight | Location in code (default) |
|---|---|
| FLUX.1-dev | `FLUX_DEV = /data/USO/weights/FLUX.1-dev/flux1-dev.safetensors` |
| FLUX VAE | `AE = /data/USO/weights/FLUX.1-dev/ae.safetensors` |
| USO LoRA | `LORA = /data/USO/weights/USO/uso_flux_v1.0/dit_lora.safetensors` |
| USO projector | `PROJECTION_MODEL = /data/USO/weights/USO/uso_flux_v1.0/projector.safetensors` |
| SigLIP | `SIGLIP_PATH = /data/USO/weights/siglip` |
| T5-XXL | `T5 = /data/USO/weights/t5-xxl` |
| CLIP | `CLIP = /mnt/jfs/model_zoo/clip-vit-large-patch14` |

**OmniStyle** — official repo [StyleX-Research/OmniStyle](https://github.com/StyleX-Research/OmniStyle) (author mirror [wangyePHD/OmniStyle](https://github.com/wangyePHD/OmniStyle)); weights via the HuggingFace link in that repo.
Paths are passed via environment variables in the entry script `scripts/inference/omnistyle_demo.sh` (or edit the defaults at the top of `run_omnistyle.py`):

| Weight | Entry-script variable (default) |
|---|---|
| FLUX.1-dev | `FLUX_DEV = /data/USO/weights/FLUX.1-dev/flux1-dev.safetensors` |
| FLUX VAE | `AE = /data/USO/weights/FLUX.1-dev/ae.safetensors` |
| OmniStyle LoRA | `LORA = /data/Sref_Cref/OmniStyle/pretrained/dit_lora.safetensors` |
| T5-XXL | `T5 = /data/USO/weights/t5-xxl` |
| CLIP | `CLIP = /mnt/jfs/model_zoo/clip-vit-large-patch14/` |
| SigLIP | `SIGLIP_PATH = /data/USO/weights/siglip` |

### 2. API keys and service endpoints (placeholders)

Replace the following placeholders in the scripts with your own values:
- `YOUR_API_KEY` — API key for OpenAI / Gemini / Seedream, etc. (can be switched to read from env vars).
- `YOUR_QWEN_VLM_HOST:22002` — **the Qwen3-VL service you deploy yourself** (see "Deploy the Qwen3-VL service" below). All VLM evaluation scripts (`*_similarity_batch.sh`, `instruction_score.sh`, `triplet_qwen_dual_judge.sh`, `uso_metric_batch.sh`) reach the model through it. The relevant variables are `xingpeng_ip` (base_url) and `xingpeng_model` (the served model name, default `Qwen3-VL-30B-A3B-Instruct`, which must match `--served-model-name` in the launch command).
- `YOUR_OPENAI_COMPAT_ENDPOINT` — OpenAI-compatible proxy address for GPT-4o / Gemini, etc. (used only by the GPT-4o judge / captioning scripts).

#### Deploy the Qwen3-VL service (prerequisite for VLM evaluation)

The VLM judge scripts require a running Qwen3-VL (Instruct) service that exposes an
OpenAI-compatible interface, started **before** evaluation. You can deploy it following the
[official Qwen repo](https://github.com/QwenLM/Qwen3-VL), or use the exact vLLM command we use
(this project serves the model this way):

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

Once started, the OpenAI-compatible base URL is `http://<your-host-ip>:22002/v1`. Set
`YOUR_QWEN_VLM_HOST` in the judge scripts to that host (the port matches `--port` above), keep
`xingpeng_model` in sync with `--served-model-name`, and the VLM evaluation scripts are ready to run.

### 3. Data layout

The benchmark used for inference / evaluation is open-sourced on HuggingFace:
[**Blue2Giant/FreeStyle_Bench**](https://huggingface.co/datasets/Blue2Giant/FreeStyle_Bench).
It contains benchmarks for two tasks:

- **sref**: style-reference task (content and style reference image + text instruction).
- **cref_sref**: content + style reference task (content reference + style reference + text instruction).

Each task's benchmark has the structure below; the scripts' data root points to one such task:

```
<DATA_ROOT>/            # benchmark root of one task
├── cref/            # Content reference images (000000.png ...)
├── sref/            # Style reference images
├── prompts.json     # {"000000": "instruction text", ...}
└── <model_name>/    # Per-model generated outputs (evaluation *.json files are written here too)
```

The data root is currently hardcoded as `/mnt/jfs/bench-bucket/sref_bench/...`; download the benchmark above and change it to your path.

---

## Usage

> Before running, replace model paths, API keys, endpoints, and the data root as described above.

### Inference

```bash
bash scripts/inference/Qwen_2511_demo.sh        # single-model demo
bash scripts/inference/flux_klein_9B_demo.sh
bash scripts/inference/TeleStyle_demo.sh
bash scripts/inference/seedream.sh              # Seedream (API)

bash scripts/inference/csgo_infer.sh            # CSGO (SDXL + IP-Adapter)
bash scripts/inference/uso_batch_run.sh         # USO (FLUX.1-dev + LoRA), single GPU
bash scripts/inference/omnistyle_demo.sh        # OmniStyle (FLUX.1-dev + LoRA)
```

### Captioning

```bash
bash scripts/caption/gemini.sh
bash scripts/caption/gpt4o.sh
```

### Metrics

One-stop batch evaluation (most common; covers encoder metrics + VLM metrics):

```bash
bash scripts/metrics/uso_metric_batch.sh
```

Or run a single category:

```bash
bash scripts/metrics/encoder_batch.sh           # CSD/OneIG/DINOv2/CAS/CLIP-T/aesthetic
bash scripts/metrics/style_similarity_batch.sh  # VLM style consistency
bash scripts/metrics/content_leakage_batch.sh    # VLM content leakage
bash scripts/metrics/content_similarity_batch.sh
bash scripts/metrics/instruction_score.sh
bash scripts/metrics/triplet_qwen_dual_judge.sh
```

Metric overview:

| Dimension | Implementation | Metrics |
|---|---|---|
| Style consistency | encoder | CSD, OneIG |
| Content leakage | VLM | Qwen3-VL content leakage score (0-10) |
| Content consistency | encoder | DINOv2, CAS |
| Instruction following | encoder / VLM | CLIP-T, VLM scoring |
| Aesthetic quality | encoder | LAION, Aesthetic v2.5, OneAlign |
| Holistic judge | VLM | Qwen3-VL / GPT-4o triplet dual judge |

---

## Notes

- The `*_demo.py` files under `src/**/` contain placeholder image paths (`assets/`, `logs/`) and
  are single-file demos only; the batch entry scripts do not depend on them.
- VLM-related scripts access models through an OpenAI-compatible interface and can target any
  compatible service.
