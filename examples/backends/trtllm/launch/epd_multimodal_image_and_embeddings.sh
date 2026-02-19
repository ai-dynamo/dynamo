#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/decode.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llava-v1.6-mistral-7b-hf/encode.yaml"}


export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_CUDA_VISIBLE_DEVICES_2=${ENCODE_CUDA_VISIBLE_DEVICES_2:-"3"}

export PREFILL_CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-"1"}
export PREFILL_CUDA_VISIBLE_DEVICES_2=${DECODE_CUDA_VISIBLE_DEVICES_2:-"3"}

export DECODE_CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-"2"}
export DECODE_CUDA_VISIBLE_DEVICES_2=${DECODE_CUDA_VISIBLE_DEVICES_2:-"3"}

export MODALITY=${MODALITY:-"multimodal"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID $DECODE_PID $ENCODE_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID $DECODE_PID $ENCODE_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# run encode worker
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode encode &
ENCODE_PID=$!

CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode encode &
ENCODE_PID=$!

# # run encode worker
# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# # run encode worker
# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# # run encode worker
# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!


# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# # run encode worker
# CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$ENCODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode encode &
# ENCODE_PID=$!

# run prefill worker
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill \
  --encode-endpoint "$ENCODE_ENDPOINT" &
PREFILL_PID=$!

# run prefill worker
CUDA_VISIBLE_DEVICES=$PREFILL_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill \
  --encode-endpoint "$ENCODE_ENDPOINT" &
PREFILL_PID=$!

# # run decode worker
# CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
#   --model-path "$MODEL_PATH" \
#   --served-model-name "$SERVED_MODEL_NAME" \
#   --extra-engine-args "$DECODE_ENGINE_ARGS" \
#   --modality "$MODALITY" \
#   --disaggregation-mode decode &
# DECODE_PID=$!

# run decode worker
CUDA_VISIBLE_DEVICES=$DECODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode decode &
DECODE_PID=$!

wait $DYNAMO_PID
