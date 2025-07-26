#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"engine_configs/agg.yaml"}
export ENABLE_KV_EVENTS=${ENABLE_KV_EVENTS:-"true"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# Build frontend extra args
FRONTEND_EXTRA_ARGS=()
if [[ "${ENABLE_KV_EVENTS,,}" == "false" ]]; then
  FRONTEND_EXTRA_ARGS+=(--no-kv-events)
fi

# run frontend
python3 -m dynamo.frontend --router-mode kv --http-port 8000 "${FRONTEND_EXTRA_ARGS[@]}" &
DYNAMO_PID=$!

# Build extra args
EXTRA_ARGS=()
if [[ "${ENABLE_KV_EVENTS,,}" == "true" ]]; then
  EXTRA_ARGS+=(--publish-events-and-metrics)
fi

# run worker
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  "${EXTRA_ARGS[@]}"
