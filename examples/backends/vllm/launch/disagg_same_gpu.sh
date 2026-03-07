#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ./disagg_same_gpu.sh
# Automatically calculates GPU memory fraction so each worker gets 4GB

# Get total and free GPU memory
GPU_MEM_INFO=$(python3 -c "import torch; free, total = torch.cuda.mem_get_info(); print(f'{free/1024**3:.2f} {total/1024**3:.2f}')" 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: Failed to check GPU memory. Is PyTorch with CUDA available?"
  exit 1
fi

FREE_GPU_GB=$(echo $GPU_MEM_INFO | awk '{print $1}')
TOTAL_GPU_GB=$(echo $GPU_MEM_INFO | awk '{print $2}')

# Each worker needs 4GB
REQUIRED_GB_PER_WORKER=4
REQUIRED_GB_TOTAL=8

# Calculate fraction needed per worker (4GB / total GPU memory)
GPU_MEM_FRACTION=$(python3 -c "print(f'{$REQUIRED_GB_PER_WORKER / $TOTAL_GPU_GB:.3f}')")

# Check if we have enough free memory
if python3 -c "import sys; sys.exit(0 if float('$FREE_GPU_GB') >= $REQUIRED_GB_TOTAL else 1)"; then
  echo "GPU memory check passed: ${FREE_GPU_GB}GB free / ${TOTAL_GPU_GB}GB total (required: ${REQUIRED_GB_TOTAL}GB)"
  echo "Using ${GPU_MEM_FRACTION} memory fraction per worker (${REQUIRED_GB_PER_WORKER}GB each)"
else
  echo "Error: Insufficient GPU memory. Required: ${REQUIRED_GB_TOTAL}GB, Available: ${FREE_GPU_GB}GB"
  echo "Please free up GPU memory before running disaggregated mode on single GPU."
  exit 1
fi

MODEL="Qwen/Qwen3-0.6B"

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated on Same GPU (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "GPU Mem:     ${GPU_MEM_FRACTION} per worker (${REQUIRED_GB_PER_WORKER}GB each)"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run decode worker with metrics on port 8081
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# For disaggregated deployments we standardize on DYN_SYSTEM_PORT1/2 instead of
# *_PREFILL/*_DECODE env names so test harnesses can set one simple pair.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization ${GPU_MEM_FRACTION} \
  --max-model-len 16384 &

# Wait for decode worker to initialize before starting prefill worker
# This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
# The decode worker needs time to:
# 1. Load model weights and allocate its memory fraction
# 2. Initialize KV cache
# 3. Register with NATS service discovery so prefill worker can find it
echo "Waiting for decode worker to initialize..."
sleep 10

# run prefill worker with metrics on port 8082
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization ${GPU_MEM_FRACTION} \
  --max-model-len 16384 \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
