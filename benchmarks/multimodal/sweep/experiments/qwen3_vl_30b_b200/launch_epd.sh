#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EPD (Encode-Prefill-Decode) multimodal launch for Qwen3-VL-30B on 4xB200.
#
# Supports configurable encoder count via first positional argument.
# Encoders are split evenly across GPU 0 and GPU 3.
# Prefill always on GPU 1, Decode always on GPU 2.
#
# GPU layout:
#   1E:  GPU 0: 1 encoder  | GPU 1: prefill | GPU 2: decode | GPU 3: —
#   2E:  GPU 0: 1 encoder  | GPU 1: prefill | GPU 2: decode | GPU 3: 1 encoder
#   4E:  GPU 0: 2 encoders | GPU 1: prefill | GPU 2: decode | GPU 3: 2 encoders
#   6E:  GPU 0: 3 encoders | GPU 1: prefill | GPU 2: decode | GPU 3: 3 encoders
#
# Usage:
#   bash launch_epd.sh <num_encoders> --model-path Qwen/Qwen3-VL-2B-Instruct

NUM_ENCODERS=${1:-1}
shift

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-2B-Instruct"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-2B-Instruct"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/decode.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-2b-instruct/encode.yaml"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Remaining arguments forwarded from sweep orchestrator
EXTRA_ARGS=("$@")

# Compute encoder distribution across GPU 0 and GPU 3
GPU0_ENCODERS=$(( (NUM_ENCODERS + 1) / 2 ))  # ceil(N/2)
GPU3_ENCODERS=$(( NUM_ENCODERS / 2 ))         # floor(N/2)

echo "EPD config: ${NUM_ENCODERS} encoder(s) — GPU 0: ${GPU0_ENCODERS}, GPU 3: ${GPU3_ENCODERS}"

# Collect PIDs for cleanup
PIDS=()

cleanup() {
    echo "Cleaning up background processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run frontend
python3 -m dynamo.frontend &
PIDS+=($!)
DYNAMO_PID=$!

# spawn encoders on GPU 0
for i in $(seq 1 "$GPU0_ENCODERS"); do
    CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$ENCODE_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
      --max-file-size-mb "$MAX_FILE_SIZE_MB" \
      --disaggregation-mode encode \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  Encoder $i on GPU 0 (PID $!)"
done

# spawn encoders on GPU 3
for i in $(seq 1 "$GPU3_ENCODERS"); do
    CUDA_VISIBLE_DEVICES=3 python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$ENCODE_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
      --max-file-size-mb "$MAX_FILE_SIZE_MB" \
      --disaggregation-mode encode \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  Encoder $((GPU0_ENCODERS + i)) on GPU 3 (PID $!)"
done

# run prefill worker on GPU 1
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$PREFILL_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --disaggregation-mode prefill \
  --encode-endpoint "$ENCODE_ENDPOINT" \
  "${EXTRA_ARGS[@]}" &
PIDS+=($!)
echo "  Prefill on GPU 1 (PID $!)"

# run decode worker on GPU 2
CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$DECODE_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
  --max-file-size-mb "$MAX_FILE_SIZE_MB" \
  --disaggregation-mode decode \
  "${EXTRA_ARGS[@]}" &
PIDS+=($!)
echo "  Decode on GPU 2 (PID $!)"

wait $DYNAMO_PID
