#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E + Agg (2 Workers) launch script for Qwen3-VL-30B-A3B multimodal.
# Separate encode workers handle vision encoding; 2 aggregated workers
# handle both prefill and decode (no KV cache transfer overhead).
#
# GPU layout (2 GPUs):
#   GPU 0: encode worker 1 + agg worker 1
#   GPU 1: encode worker 2 + agg worker 2

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct-fp8/agg.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct-fp8/encode.yaml"}
export AGG_CUDA_VISIBLE_DEVICES_1=${AGG_CUDA_VISIBLE_DEVICES_1:-"0"}
export AGG_CUDA_VISIBLE_DEVICES_2=${AGG_CUDA_VISIBLE_DEVICES_2:-"1"}
export ENCODE_CUDA_VISIBLE_DEVICES_1=${ENCODE_CUDA_VISIBLE_DEVICES_1:-"0"}
export ENCODE_CUDA_VISIBLE_DEVICES_2=${ENCODE_CUDA_VISIBLE_DEVICES_2:-"1"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $ENCODE1_PID $ENCODE2_PID $AGG1_PID $AGG2_PID 2>/dev/null || true
    wait $DYNAMO_PID $ENCODE1_PID $ENCODE2_PID $AGG1_PID $AGG2_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run frontend
python3 -m dynamo.frontend --http-port 8000 &
DYNAMO_PID=$!

# --- Encode workers (vision encoder, lightweight) ---

# encode worker 1 on GPU 0
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE1_PID=$!

# encode worker 2 on GPU 1
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE2_PID=$!

# --- Aggregated workers (prefill + decode combined) ---

# agg worker 1 on GPU 0
CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES_1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --publish-events-and-metrics &
AGG1_PID=$!

# agg worker 2 on GPU 1
CUDA_VISIBLE_DEVICES=$AGG_CUDA_VISIBLE_DEVICES_2 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --publish-events-and-metrics &
AGG2_PID=$!

wait $DYNAMO_PID

