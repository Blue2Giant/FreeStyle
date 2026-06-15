#!/usr/bin/env bash
set -euo pipefail

# ===== 1. 为本次运行创建独立输出目录 =====
BASE_OUTPUT_ROOT="/mnt/jfs/comfyui_output"

# 简短主机名，防止多机冲突
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname || echo "host")
# 时间戳（年月日时分秒）
TS=$(date +%Y%m%d_%H%M%S)
# 再加上当前进程 PID，进一步避免同一秒多次启动冲突
RUN_NAME="${HOSTNAME_SHORT}_${TS}_$$"

RUN_OUTPUT_DIR="${BASE_OUTPUT_ROOT}/${RUN_NAME}"

# 在 /mnt/jfs 下创建本次运行的独立目录
mkdir -p "${RUN_OUTPUT_DIR}"

echo "本次 ComfyUI 输出目录: ${RUN_OUTPUT_DIR}"

# ===== 2. 在 /workspace/ComfyUI 下建立软链 output -> RUN_OUTPUT_DIR =====
cd /workspace/ComfyUI

# 拷贝所有的custom nodes
rm -rf /workspace/ComfyUI/custom_nodes
cp -r /mnt/jfs/comfyui_nodes/custom_nodes /workspace/ComfyUI/

# 创建超分模型的软链接
rm -rf /workspace/ComfyUI/models/upscale_models
ln -s  /mnt/jfs/model_zoo/comfyui/ /workspace/ComfyUI/models/upscale_models

# 拷贝我们最新的配置文件
rm -rf /workspace/ComfyUI/extra_model_paths.yaml
cp  /data/ComfyUI/extra_model_paths.yaml /workspace/ComfyUI/extra_model_paths.yaml

#建立软链接路径
rm -rf /workspace/ComfyUI/input

if [ ! -d /mnt/jfs/comfyui_input ]; then
  mkdir -p /mnt/jfs/comfyui_input
fi

ln -s /mnt/jfs/comfyui_input/  /workspace/ComfyUI/models/input

# 如果之前已经有 output（目录/软链/文件），先删掉
if [ -L output ] || [ -d output ] || [ -e output ]; then
  rm -rf output
fi

# 建立输出目录新的软链
ln -s "${RUN_OUTPUT_DIR}" /workspace/ComfyUI/output

# 顺便导出一个环境变量，方便后续脚本使用（可选）
# export COMFY_OUTPUT_DIR="${RUN_OUTPUT_DIR}"

# ===== 3. 启动多个 ComfyUI 实例 =====
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
START_PORT=8188

echo "GPU 数量: ${NUM_GPUS}"

for ((i=0; i<NUM_GPUS; i++)); do
  PORT=$((START_PORT + i))
  echo "启动 GPU ${i} -> 端口 ${PORT}"

  # 全部后台启动
  CUDA_VISIBLE_DEVICES=${i} python main.py --listen 0.0.0.0 --port="${PORT}" &
done

echo "所有 ComfyUI 实例已启动，输出目录: ${RUN_OUTPUT_DIR}"

# ===== 4. 等所有子进程退出 =====
wait
echo "所有 ComfyUI 实例已退出，本次运行输出目录仍为: ${RUN_OUTPUT_DIR}"
