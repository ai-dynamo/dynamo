#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Disaggregated mode on single GPU - for testing only
# Both prefill and decode workers share the same GPU with reduced memory

# Check GPU memory availability
FREE_GPU_GB=$(python3 -c "import torch; print(torch.cuda.mem_get_info()[0]/1024**3)" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to check GPU memory. Is PyTorch with CUDA available?"
    exit 1
fi

REQUIRED_GB=16
# Use bash arithmetic instead of bc to avoid external dependency
FREE_GPU_INT=$(python3 -c "print(int(float('$FREE_GPU_GB')))" 2>/dev/null)
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse GPU memory value."
    exit 1
fi

if (( FREE_GPU_INT < REQUIRED_GB )); then
    echo "Error: Insufficient GPU memory. Required: ${REQUIRED_GB}GB, Available: ${FREE_GPU_GB}GB"
    echo "Please free up GPU memory before running disaggregated mode on single GPU."
    exit 1
fi

echo "GPU memory check passed: ${FREE_GPU_GB}GB available (required: ${REQUIRED_GB}GB)"

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/tests/serve/configs/trtllm/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/tests/serve/configs/trtllm/decode.yaml"}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
export MODALITY=${MODALITY:-"text"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# run prefill worker (shares GPU with decode)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8081 \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode prefill &
PREFILL_PID=$!

# run decode worker (shares GPU with prefill)
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_ENABLED=true DYN_SYSTEM_PORT=8082 \
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  --disaggregation-mode decode

