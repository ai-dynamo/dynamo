#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3/agg.yaml"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID 2>/dev/null || true
    wait $DYNAMO_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend with KV router in approximate mode (i.e. no KV events)
python3 -m dynamo.frontend --router-mode kv --no-kv-events &
DYNAMO_PID=$!

# run worker (no event publishing needed - frontend handles routing with predictive approx kv mode)
python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS"
