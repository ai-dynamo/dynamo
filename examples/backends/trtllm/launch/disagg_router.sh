#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-0.6B"}
export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"prefill_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/recipes/qwen3/trtllm/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/recipes/qwen3/trtllm/decode.yaml"}
export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"0"}
export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"1"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
python3 -m dynamo.frontend --router-mode kv --http-port 8000 &
DYNAMO_PID=$!


EXTRA_PREFILL_ARGS=()
EXTRA_DECODE_ARGS=()
if [ "$DISAGGREGATION_STRATEGY" == "prefill_first" ]; then
  EXTRA_PREFILL_ARGS+=(--publish-events-and-metrics)
else
  EXTRA_DECODE_ARGS+=(--publish-events-and-metrics)
fi

# run prefill worker
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --disaggregation-mode prefill \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
  "${EXTRA_PREFILL_ARGS[@]}" &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --disaggregation-mode decode \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
  "${EXTRA_DECODE_ARGS[@]}"
