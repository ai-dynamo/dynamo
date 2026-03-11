#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Data Parallel multimodal workers for Qwen3-VL-30B on B200.
# Spawns N independent aggregated workers on separate GPUs behind
# the Dynamo frontend's round-robin router — the DP baseline for
# fair comparison against EPD (same GPU budget).
#
# Usage:
#   bash launch_dp.sh <num_workers> --model-path Qwen/Qwen3-VL-30B-A3B-Instruct
#
# GPU layout:
#   DP-3: GPU 0,1,2 — 3 independent agg workers
#   DP-4: GPU 0,1,2,3 — 4 independent agg workers

NUM_WORKERS=${1:-3}
shift

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct/agg.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

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

echo "DP config: ${NUM_WORKERS} independent aggregated worker(s)"

# Collect PIDs for cleanup
PIDS=()

cleanup() {
    echo "Cleaning up background processes..."
    kill "${PIDS[@]}" 2>/dev/null || true
    wait "${PIDS[@]}" 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run frontend with round-robin routing (default for multiple workers)
python3 -m dynamo.frontend --router-mode round-robin &
PIDS+=($!)
DYNAMO_PID=$!

# spawn N independent agg workers — one per GPU
for (( i=0; i<NUM_WORKERS; i++ )); do
    CUDA_VISIBLE_DEVICES=$i python3 -m dynamo.trtllm \
      --model-path "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL_NAME" \
      --extra-engine-args "$AGG_ENGINE_ARGS" \
      --modality "$MODALITY" \
      --publish-events-and-metrics \
      "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
    echo "  DP worker $((i+1)) on GPU $i (PID $!)"
done

wait $DYNAMO_PID
