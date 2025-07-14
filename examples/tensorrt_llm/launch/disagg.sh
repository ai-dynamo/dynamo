#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"decode_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"engine_configs/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"engine_configs/decode.yaml"}
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

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run ingress
dynamo run in=http out=dyn --http-port=8000 &
DYNAMO_PID=$!

# run prefill worker
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 components/worker.py \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$PREFILL_ENGINE_ARGS" \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
  --disaggregation-mode prefill &
PREFILL_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 components/worker.py \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args  "$DECODE_ENGINE_ARGS" \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" \
  --disaggregation-mode decode