#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export MODEL_PATH=${MODEL_PATH:-"orangina"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"orangina"}
export DISAGGREGATION_STRATEGY=${DISAGGREGATION_STRATEGY:-"prefill_first"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"engine_configs/orangina/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"engine_configs/orangina/decode.yaml"}

# run clear_namespace
python3 utils/clear_namespace.py --namespace dynamo

# run frontend
python3 -m dynamo.frontend --router-mode round-robin --http-port 8000

# run prefill worker
CUDA_VISIBLE_DEVICES=0,1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --disaggregation-mode prefill \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" &

# run decode workers
CUDA_VISIBLE_DEVICES=2,3 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --disaggregation-mode decode \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" &

CUDA_VISIBLE_DEVICES=4,5 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --disaggregation-mode decode \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY" &

CUDA_VISIBLE_DEVICES=6,7 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --disaggregation-mode decode \
  --disaggregation-strategy "$DISAGGREGATION_STRATEGY"