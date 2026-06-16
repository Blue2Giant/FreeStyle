# Open-Source LoRA Pipeline

This repository is used for **batch data generation** of open-source LoRA models: it runs batch inference through ComfyUI on open-source LoRAs from Civitai, producing images along with associated meta information (model IDs, trigger words, prompt templates, ComfyUI workflows, etc.).

---

## Directory Structure

```
opensource_lora_pipeline/
├── README.md                              # This file
├── comfykit/                              # ComfyUI Python SDK (async execution engine)
│   ├── comfyui/                           #   HTTP/WebSocket executor, workflow parser
│   └── utils/                             #   Utilities (config, networking, image, file I/O)
├── meta/                                  # Metadata and configuration
│   ├── comfyui_start_new_server.sh        #   Multi-GPU ComfyUI server startup script
│   ├── gemini_trigger.txt                 #   Manually verified stable style trigger word list
│   ├── model_ids/                         #   Civitai model ID lists for each model family
│   ├── prompts/                           #   Prompt templates and trigger word vocabularies
│   ├── triplet_keys/                      #   LoRA key lists extracted from triplet JSONL
│   └── workflows/                         #   ComfyUI workflow JSON templates
├── scripts/                               # Batch inference entry points (multi-host multi-GPU dispatch)
├── generate_word_table/                   # Tools for generating prompt vocabularies from CSV
├── illustrious_one_lora_diverse.py        # Illustrious single-LoRA inference (base class)
├── one_lora_flux.py                       # Flux single-LoRA inference
├── one_lora_qwen.py                       # Qwen single-LoRA inference
├── dual_lora_flux.py                      # Flux dual-LoRA inference
├── dual_lora_illustrious.py               # Illustrious dual-LoRA inference
├── dual_lora_qwen.py                      # Qwen dual-LoRA inference
├── probe_comfy_ports.py                   # Batch port reachability probe for ComfyUI instances
├── comfykit_demo.py                       # ComfyKit SDK usage example
├── comfykit_demo.sh                       # Demo launcher
└── comfykit_demo_stress.py                # Stress test example
```

---

## 1. Starting the ComfyUI Service

Script location: `meta/comfyui_start_new_server.sh`

### 1.1 Create a unique output directory

- Under `/mnt/jfs/comfyui_output/`, create a uniquely named directory for each run using the format `{hostname}_{timestamp}_{PID}`
- This prevents conflicts across multiple hosts or repeated launches

### 1.2 Prepare the runtime environment (under `/workspace/ComfyUI`)

| Operation | Description |
|-----------|-------------|
| Copy custom_nodes | Copy the entire `/mnt/jfs/comfyui_nodes/custom_nodes` to `/workspace/ComfyUI/custom_nodes` to ensure plugin versions are consistent |
| Upscale model symlink | `models/upscale_models` → `/mnt/jfs/model_zoo/comfyui/` |
| Extra model path config | Copy `/data/ComfyUI/extra_model_paths.yaml` to the workspace |
| Input directory symlink | `models/input` → `/mnt/jfs/comfyui_input/` |
| Output directory symlink | `output` → the unique output directory for this run |

### 1.3 Multi-GPU parallel launch

- Automatically detect the number of GPUs (via `nvidia-smi`)
- Launch one ComfyUI instance per GPU, with ports incrementing from 8188 (GPU0→8188, GPU1→8189, ...)
- All instances run in the background; the script blocks with `wait` until all instances exit

### 1.4 Port assignment example

```
GPU 0 → http://0.0.0.0:8188
GPU 1 → http://0.0.0.0:8189
GPU 2 → http://0.0.0.0:8190
...
```

---

## 2. Extra Model Path Configuration (`extra_model_paths.yaml`)

File location: `/data/ComfyUI/extra_model_paths.yaml`

```yaml
comfyui:
    checkpoints: /mnt/jfs/model_zoo/comfyui/
    text_encoders: /mnt/jfs/model_zoo/comfyui/
    clip_vision: /mnt/jfs/model_zoo/comfyui/
    diffusion_models: /mnt/jfs/model_zoo/comfyui/
    controlnet: models/controlnet/
    loras: |
        /mnt/jfs/model_zoo/style_lora/  # downloaded community lora place in it
    vae: /mnt/jfs/model_zoo/comfyui/
    audio_encoders: models/audio_encoders/
    LLM: /mnt/jfs/model_zoo/comfyui/LLM/
    unet: /mnt/jfs/model_zoo/comfyui/unet/
    clip: /mnt/jfs/model_zoo/comfyui/clip/
```

**Key points:**

- Most models are stored under `/mnt/jfs/model_zoo/comfyui/`
- LoRA paths cover multiple source directories for Flux.1 D, Qwen, Illustrious, SDXL 1.0, etc.
- LLM and unet/clip have their own subdirectories

### LoRA Weight Download Instructions

Download the open-source LoRA weights from [Civitai](https://civitai.com) and place them in the loras directories above.

**Download method:** Visit `https://civitai.com/models/{model_id}` for each model.

**Placement rules:**

| Architecture | Target Directory | Model ID List File |
|-------------|-----------------|-------------------|
| Flux.1 D | `/mnt/jfs/all_loras/civitai/'Flux.1 D'/` | `meta/model_ids/flux_content_one_lora.txt`, `meta/model_ids/flux_style_one_lora.txt` |
| Qwen | `/mnt/jfs/all_loras/civitai/Qwen/` | `meta/model_ids/qwen_content_one_lora.txt`, `meta/model_ids/qwen_style_one_lora.txt` |
| Illustrious | `/mnt/jfs/all_loras/civitai/Illustrious/` | `meta/model_ids/illustrious_content_one_lora.txt`, `meta/model_ids/illustrious_style_one_lora.txt` |

Each line in a txt file is one `model_id`. For example, `1041877` corresponds to:

```
https://civitai.com/models/1041877
```

After downloading, place the `.safetensors` weight files into the corresponding architecture directory.

---

## 3. Custom Nodes List

Below are the ComfyUI custom nodes we use (from `/data/ComfyUI/custom_nodes/`):

| Node | Description |
|------|-------------|
| **ComfyUI-Manager** | Plugin manager for installing/updating other nodes |
| **ComfyUI-Impact-Pack** | Detection, segmentation, detail enhancement toolkit |
| **comfyui-inspire-pack** | Inspiration toolkit with extra sampling/control nodes |
| **ComfyUI_Comfyroll_CustomNodes** | General utility set (math, text, image processing) |
| **comfyui-easy-use** | Simplified workflow shortcut nodes |
| **comfyui_essentials** | Core enhancement nodes (crop, blend, mask) |
| **ComfyUI_ADV_CLIP_emb** | Advanced CLIP text encoding control |
| **ComfyUI-Custom-Scripts** | Custom script collection (preview, auto-layout) |
| **comfyui-kjnodes** | KJ utility nodes (batch, conditioning) |
| **ComfyUI_LayerStyle** | Layer style processing |
| **ComfyLiterals** | Literal input nodes (strings, numbers) |
| **rgthree-comfy** | Workflow efficiency tools (Reroute, Mute, etc.) |
| **comfyui-image-saver** | Image save node (multi-format, metadata) |
| **comfyui-saveimage-plus** | Enhanced image save node |
| **ComfyUI-EsesImageCompare** | Image comparison node |
| **eden_comfy_pipelines** | Eden workflow pipeline |
| **comfyui-various** | Miscellaneous utility nodes |
| **comfyui-yaser-nodes** | Yaser custom nodes |
| **ComfyUI-Jjk-Nodes** | JJK custom nodes |
| **ComfyUI-HunyuanVideoMultiLora** | Hunyuan video multi-LoRA support |
| **ComfyUI_QwenVL** | Qwen VL model integration |
| **ComfyUI_Swwan** | Swwan custom nodes |
| **SCG_LocalVLM** | Local VLM inference node |
| **qweneditutils** | Qwen Edit helper utilities |
| **ysc_highresfix** | High-resolution fix |
| **websocket_image_save.py** | WebSocket image save script |

---

## 4. Meta Information

### 4.1 Style Trigger Word List (`gemini_trigger.txt`)

File location: `meta/gemini_trigger.txt`

This is a list of **manually verified** stable style trigger words for use with the Gemini model. One trigger word per line.

**Purpose:** Serves as the standard style vocabulary when using Gemini for style classification. These trigger words have been verified to be stably recognized and distinguished by Gemini, avoiding instability caused by unverified trigger words.

**Count:** 622 style trigger words in total, covering traditional painting (oil, watercolor, printmaking), modern art movements (surrealism, pop art, minimalism), digital art, photography styles, game art, animation styles, ethnic/regional art, and more.

### 4.2 Model ID Lists (`meta/model_ids/`)

Civitai model IDs are organized by architecture and LoRA type:

| File | Count | Description |
|------|-------|-------------|
| `flux_content_one_lora.txt` | 91 | Flux content-type single LoRA |
| `flux_style_one_lora.txt` | 1,460 | Flux style-type single LoRA |
| `flux__dual_lora.txt` | 23,130 | Flux dual-LoRA combinations |
| `illustrious_content_one_lora.txt` | 799 | Illustrious content-type single LoRA |
| `illustrious_style_one_lora.txt` | 191 | Illustrious style-type single LoRA |
| `illustrious_dual_lora.txt` | 24,646 | Illustrious dual-LoRA combinations |
| `qwen_content_one_lora.txt` | 19 | Qwen content-type single LoRA |
| `qwen_style_one_lora.txt` | 53 | Qwen style-type single LoRA |
| `qwen_dual_lora.txt` | 608 | Qwen dual-LoRA combinations |

Single-LoRA key format: `1041877`; dual-LoRA key format: `1041877__1001511` (two IDs joined by double underscore).

### 4.3 Prompt Templates (`meta/prompts/`)

| File | Description |
|------|-------------|
| `diverse_prompts_100.txt` | 100 diverse scene prompts (for single-LoRA inference) |
| `STYLE_UNIVERSE_TRIGGER.csv/.txt` | Universal trigger word vocabulary for style LoRAs |
| `CHARACTER_UNIVERSE_TRIGGER.csv/.txt` | Universal trigger word vocabulary for character LoRAs |
| `OTHER_UNIVERSE_TRIGGER.csv/.txt` | Universal trigger word vocabulary for other LoRA types |
| `TRIPLET_UNIVERSE_TRIGGER.txt` | Universal trigger words for triplet evaluation |

### 4.4 Workflow Templates (`meta/workflows/`)

| File | Description |
|------|-------------|
| `flux_dual_lora.json` | Flux dual-LoRA workflow |
| `flux_full_lora-2.json` | Flux full LoRA workflow |
| `illustrious_simple.json` | Illustrious simplified workflow |
| `qwen_dual_lora.json` | Qwen dual-LoRA workflow |
| `qwen_one_lora0320.json` | Qwen single-LoRA workflow |
| `sdxl_dual_lora_ljh.json` | SDXL dual-LoRA workflow |

---

## 5. Batch LoRA Data Generation

### 5.1 Inference Scripts (Python)

Each inference script asynchronously calls ComfyUI instances via the `comfykit` SDK:

- `illustrious_one_lora_diverse.py` — Base class containing the generic inference logic (LoRA traversal, workflow injection, output collection)
- `one_lora_flux.py` / `one_lora_qwen.py` — Single-LoRA inference, inherits the base class and injects the corresponding workflow
- `dual_lora_flux.py` / `dual_lora_illustrious.py` / `dual_lora_qwen.py` — Dual-LoRA inference

### 5.2 Launch Scripts (`scripts/`)

Shell scripts are the **batch data generation entry points**. Each script runs an infinite `while true` loop that continuously calls the Python inference scripts, iterating over the model ID lists in `meta/model_ids/` and the prompt templates in `meta/prompts/`, distributing inference across multiple ComfyUI servers.

Key parameters for each task are defined inside the scripts (target server IPs, LoRA root directory, prompt files, model ID files, workflow templates, etc.) — simply edit the corresponding variables to launch different types of batch inference tasks.

| Script | Task Type |
|--------|-----------|
| `scripts/one_lora_flux.sh` | Flux single-LoRA batch image generation (style / character / other, three loops) |
| `scripts/one_lora_qwen.sh` | Qwen single-LoRA batch image generation (style / character / other, three loops) |
| `scripts/illustrious_one_lora_diverse.sh` | Illustrious single-LoRA batch image generation (style / character / other, three loops) |
| `scripts/dual_lora_flux.sh` | Flux dual-LoRA batch image generation |
| `scripts/dual_lora_illustrious.sh` | Illustrious dual-LoRA batch image generation |
| `scripts/dual_lora_qwen.sh` | Qwen dual-LoRA batch image generation |

```bash
# Example: start Flux single-LoRA batch inference (runs continuously until manually stopped)
bash scripts/one_lora_flux.sh

# Example: start Illustrious dual-LoRA batch inference
bash scripts/dual_lora_illustrious.sh
```

### 5.3 Port Probe Utility

Before inference, use `probe_comfy_ports.py` to check whether ComfyUI ports on each machine are reachable:

```bash
python probe_comfy_ports.py \
  --shell-file scripts/dual_lora_flux.sh \
  --start-port 8188 \
  --port-count 8 \
  --timeout-sec 2 \
  --concurrency 256
```

---

## 6. ComfyKit SDK

`comfykit/` is an async ComfyUI execution engine supporting:

- HTTP and WebSocket communication modes
- Connection pool management (`session_pool_size`)
- Dynamic workflow JSON parameter injection
- RunningHub remote execution support

**Environment setup and full documentation:** Please refer to the [ComfyKit GitHub repository](https://github.com/puke3615/ComfyKit) for installation and advanced usage.

**Basic usage:**

```python
from comfykit import ComfyKit

async with ComfyKit(comfyui_url="http://host:8188", session_pool_size=2) as kit:
    result = await kit.execute("workflow.json")
    print(result.images)
```

---

## 7. Quick Start

```bash
# 1. Start ComfyUI service (multi-GPU parallel)
bash meta/comfyui_start_new_server.sh

# 2. Probe port reachability
python probe_comfy_ports.py --shell-file scripts/one_lora_flux.sh --start-port 8188 --port-count 8

# 3. Start batch inference
bash scripts/one_lora_flux.sh
```

After starting, access the first instance's Web UI at `http://<host>:8188`.

---

## 8. Similarity Judgment for Generated Images

LoRA batch-generated images need to be filtered through similarity judgment. We use a VLM (Qwen3-VL) for dual-axis content and style discrimination.

### Judgment Script

Location: `/data/FreeStyle/benchmark_infer/scripts/metrics/triplet_qwen_dual_judge.sh`

The script performs two parallel judgments on each generated image:

- **Content judgment**: Compares the subject and theme against a content reference image
- **Style judgment**: Compares the painting style / visual style against a style reference image

### Judgment Method: Logits-based Rejection Sampling

Unlike traditional discrete scoring (e.g., 1-5 scale), we use **logits-based rejection sampling**:

1. The VLM generates with `temperature=0.0`, requesting `top_logprobs` to obtain the output token logit distribution
2. Extract the log probabilities (logp0, logp1) of the "0" and "1" tokens from the logit distribution
3. Compute a continuous similarity score (0~1) via `sigmoid(logp1 - logp0)`

This means that even if the VLM's discrete output is "1", if logp1 and logp0 are close (low confidence), the score will still be low — functioning as a **soft rejection sampling** that automatically down-weights or discards samples with insufficient confidence.

Empirically, this approach achieves higher accuracy than discrete scoring methods.
