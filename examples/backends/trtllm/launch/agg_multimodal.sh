#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct-fp8/agg.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $WORKER1_PID $WORKER2_PID 2>/dev/null || true
    wait $DYNAMO_PID $WORKER1_PID $WORKER2_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend --router-mode kv &
DYNAMO_PID=$!

# run worker 1
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics &
WORKER1_PID=$!

# run worker 2
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics &
WORKER2_PID=$!

wait $DYNAMO_PID
