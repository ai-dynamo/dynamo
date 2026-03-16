#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 7E+7PD: 7 co-located encoder + PD worker pairs on 7 GPUs.
# Each GPU runs 1 encoder + 1 aggregated PD worker sharing GPU memory.
#
# GPU layout:
#   GPU 0: encoder + PD worker (co-located)
#   GPU 1: encoder + PD worker (co-located)
#   ...
#   GPU 6: encoder + PD worker (co-located)
#
# Usage:
#   bash launch_7e7pd.sh --model-path Qwen/Qwen3-VL-30B-A3B-Instruct

NUM_PAIRS=7

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/agg_colocated.yaml"}
export ENCODE_ENGINE_ARGS=${ENCODE_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/encode.yaml"}
export ENCODE_ENDPOINT=${ENCODE_ENDPOINT:-"dyn://dynamo.tensorrt_llm_encode.generate"}
export MODALITY=${MODALITY:-"multimodal"}
export ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/tmp"}
export MAX_FILE_SIZE_MB=${MAX_FILE_SIZE_MB:-50}

EXTRA_ARGS=("$@")

for (( i=0; i<${#EXTRA_ARGS[@]}; i++ )); do
    if [[ "${EXTRA_ARGS[$i]}" == "--model-path" || "${EXTRA_ARGS[$i]}" == "--model" ]]; then
        MODEL_PATH="${EXTRA_ARGS[$((i+1))]}"
        break
    fi
done
export SERVED_MODEL_NAME="$MODEL_PATH"

echo "7E+7PD config: ${NUM_PAIRS} co-located encoder+PD pairs (${NUM_PAIRS} GPUs)"

PIDS=()

cleanup() {
    echo "Cleaning up background processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Frontend with KV-aware routing across PD workers
python3 -m dynamo.frontend --router-mode kv &
PIDS+=($!)
DYNAMO_PID=$!

# Spawn 7 co-located encoder+PD pairs
for (( gpu=0; gpu<NUM_PAIRS; gpu++ )); do
    # Encoder on this GPU
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
    echo "  GPU $gpu: encoder (PID $!)"

    # PD worker on same GPU (co-located, shares GPU memory)
    CUDA_VISIBLE_DEVICES=$gpu python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$AGG_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --encode-endpoint "$ENCODE_ENDPOINT" \
      --publish-events-and-metrics \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  GPU $gpu: PD worker (PID $!)"
done

wait $DYNAMO_PID
