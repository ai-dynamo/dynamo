#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# EPD (Encode-Prefill-Decode) multimodal launch for Qwen3-VL-30B on 8xB200.
#
# Each encoder gets its own GPU. Prefill on GPU 1, Decode on GPU 2.
# Available encoder GPUs: 0, 3, 4, 5, 6, 7 (max 6 encoders on 8 GPUs).
#
# GPU layout:
#   1E:  GPU 0: encoder  | GPU 1: prefill | GPU 2: decode
#   2E:  GPU 0: encoder  | GPU 1: prefill | GPU 2: decode | GPU 3: encoder
#   4E:  GPU 0,3,4,5: encoders | GPU 1: prefill | GPU 2: decode
#   6E:  GPU 0,3,4,5,6,7: encoders | GPU 1: prefill | GPU 2: decode
#
# Usage:
#   bash launch_epd.sh <num_encoders> --model-path Qwen/Qwen3-VL-30B-A3B-Instruct

NUM_ENCODERS=${1:-1}
shift

# GPUs available for encoders (GPU 1=prefill, GPU 2=decode)
ENCODER_GPUS=(0 3 4 5 6 7)

if (( NUM_ENCODERS > ${#ENCODER_GPUS[@]} )); then
    echo "ERROR: Requested ${NUM_ENCODERS} encoders but only ${#ENCODER_GPUS[@]} encoder GPUs available (${ENCODER_GPUS[*]})"
    exit 1
fi

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
export PREFILL_ENGINE_ARGS=${PREFILL_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/prefill.yaml"}
export DECODE_ENGINE_ARGS=${DECODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/decode.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/encode.yaml"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

# Remaining arguments forwarded from sweep orchestrator
EXTRA_ARGS=("$@")

# Parse --model-path from extra args to keep SERVED_MODEL_NAME in sync
for (( i=0; i<${#EXTRA_ARGS[@]}; i++ )); do
    if [[ "${EXTRA_ARGS[$i]}" == "--model-path" || "${EXTRA_ARGS[$i]}" == "--model" ]]; then
        MODEL_PATH="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done
export SERVED_MODEL_NAME="$MODEL_PATH"

echo "EPD config: ${NUM_ENCODERS} encoder(s) — 1 per GPU"

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

# spawn encoders — one per GPU
for (( i=0; i<NUM_ENCODERS; i++ )); do
    gpu=${ENCODER_GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$gpu python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$ENCODE_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH" \
      --max-file-size-mb "$MAX_FILE_SIZE_MB" \
      --disaggregation-mode encode \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  Encoder $((i+1)) on GPU $gpu (PID $!)"
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
