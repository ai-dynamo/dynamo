#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 1 Encode + 3 PD workers for Qwen2-VL-7B-Instruct
# GPU 0: Encode (vision encoder)
# GPU 1-3: PD workers (prefill + decode, TP=1 each)

# Environment variables with defaults
export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2-VL-7B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen2-VL-7B-Instruct"}
export PD_ENGINE_ARGS=${PD_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen2-vl-7b-instruct/agg.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen2-vl-7b-instruct/encode.yaml"}
export ENCODE_CUDA_VISIBLE_DEVICES=${ENCODE_CUDA_VISIBLE_DEVICES:-"0"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $ENCODE_PID $PD_PID_1 $PD_PID_2 $PD_PID_3 $NATS_PID $ETCD_PID 2>/dev/null || true
    wait $DYNAMO_PID $ENCODE_PID $PD_PID_1 $PD_PID_2 $PD_PID_3 $NATS_PID $ETCD_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM


# =============================================================================
# Start infrastructure services (NATS + etcd)
# =============================================================================
echo "Starting NATS server..."
nats-server -js &
NATS_PID=$!

echo "Starting etcd..."
etcd --listen-client-urls http://0.0.0.0:2379 \
     --advertise-client-urls http://0.0.0.0:2379 \
     --data-dir /tmp/etcd &
ETCD_PID=$!

# Wait for infrastructure to start
echo "Waiting for infrastructure services to start..."
sleep 3

# =============================================================================
# Start dynamo services
# =============================================================================

# run frontend
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# run encode worker (vision encoder on GPU 0)
CUDA_VISIBLE_DEVICES=$ENCODE_CUDA_VISIBLE_DEVICES python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$ENCODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode encode &
ENCODE_PID=$!

# run PD worker 1 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PD_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --dyn-encoder-cache-capacity-gb 4 &
PD_PID_1=$!

# run PD worker 2 (GPU 2)
CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PD_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --dyn-encoder-cache-capacity-gb 4 &
PD_PID_2=$!

# run PD worker 3 (GPU 3)
CUDA_VISIBLE_DEVICES=3 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PD_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill_and_decode \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  --dyn-encoder-cache-capacity-gb 4 &
PD_PID_3=$!

wait $DYNAMO_PID
