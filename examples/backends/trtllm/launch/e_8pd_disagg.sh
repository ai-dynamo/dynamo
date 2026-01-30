#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 1 Encode + 1 Prefill-Decode worker (TP=8)
# GPU 0: Encode
# GPU 1-8: PD worker (tensor parallel across 8 GPUs)

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-4-Scout-17B-16E-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"meta-llama/Llama-4-Scout-17B-16E-Instruct"}
export PD_ENGINE_ARGS=${PD_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llama4/multimodal/llama4-Scout/agg.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/llama4/multimodal/llama4-Scout/encode.yaml"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export PD_CUDA_VISIBLE_DEVICES=${PD_CUDA_VISIBLE_DEVICES:-"1,2,3,4,5,6,7,8"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $ENCODE_PID $PD_PID 2>/dev/null || true
    wait $DYNAMO_PID $ENCODE_PID $PD_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# run encode worker
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE_PID=$!

# run PD worker (tensor parallel across 8 GPUs)
CUDA_VISIBLE_DEVICES=$PD_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PD_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --dyn-encoder-cache-capacity-gb 4 &
PD_PID=$!

wait $DYNAMO_PID
