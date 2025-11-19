#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ./disagg_same_gpu.sh [GPU_MEM_FRACTION]
#   GPU_MEM_FRACTION: Fraction of GPU memory to use per worker (default: 0.45)
#   Example: ./disagg_same_gpu.sh 0.45

# GPU memory fraction to use per worker (default: 0.45 = 45% each = 90% total for both workers)
GPU_MEM_FRACTION="${1:-0.45}"

# Check GPU memory before starting disaggregated mode on single GPU
FREE_GPU_GB=$(python3 -c "import torch; print(torch.cuda.mem_get_info()[0]/1024**3)" 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "Error: Failed to check GPU memory. Is PyTorch with CUDA available?"
  exit 1
fi

REQUIRED_GB=16
# Use Python for floating-point comparison to avoid bc dependency
if python3 -c "import sys; sys.exit(0 if float('$FREE_GPU_GB') >= $REQUIRED_GB else 1)"; then
  echo "GPU memory check passed: ${FREE_GPU_GB}GB available (required: ${REQUIRED_GB}GB)"
else
  echo "Error: Insufficient GPU memory. Required: ${REQUIRED_GB}GB, Available: ${FREE_GPU_GB}GB"
  echo "Please free up GPU memory before running disaggregated mode on single GPU."
  exit 1
fi

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run ingress with KV router mode for disaggregated setup
python3 -m dynamo.frontend --router-mode kv --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker with metrics on port 8081
DYN_SYSTEM_PORT=8081 \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  --mem-fraction-static ${GPU_MEM_FRACTION} \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests 2 \
  --enable-metrics &
PREFILL_PID=$!

# Wait for prefill worker to initialize before starting decode worker
# This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
# The prefill worker needs time to:
# 1. Load model weights and allocate its memory fraction
# 2. Initialize KV cache with --delete-ckpt-after-loading to free checkpoint memory
# 3. Register with NATS service discovery so decode worker can find it
echo "Waiting for prefill worker to initialize..."
sleep 5

# run decode worker with metrics on port 8082 (foreground)
DYN_SYSTEM_PORT=8082 \
python3 -m dynamo.sglang \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port 12345 \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  --mem-fraction-static ${GPU_MEM_FRACTION} \
  --chunked-prefill-size 4096 \
  --max-prefill-tokens 4096 \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests 2 \
  --enable-metrics

