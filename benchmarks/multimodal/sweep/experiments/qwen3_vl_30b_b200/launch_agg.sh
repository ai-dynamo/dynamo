#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Aggregated (single-GPU) multimodal worker for Qwen3-VL-30B on B200.
#
# Usage:
#   bash launch_agg.sh --model-path Qwen/Qwen3-VL-30B-A3B-Instruct-FP8

export DYNAMO_HOME=${DYNAMO_HOME:-"/workspace"}
export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"}
export AGG_ENGINE_ARGS=${AGG_ENGINE_ARGS:-"$DYNAMO_HOME/examples/backends/trtllm/engine_configs/qwen3-vl-30b-a3b-instruct-fp8/agg.yaml"}
export MODALITY=${MODALITY:-"multimodal"}

# Extra arguments forwarded from sweep orchestrator (e.g. --model-path <model>)
EXTRA_ARGS=("$@")

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $WORKER_PID 2>/dev/null || true
    wait $DYNAMO_PID $WORKER_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# run frontend
python3 -m dynamo.frontend --router-mode kv &
DYNAMO_PID=$!

# run aggregated worker on GPU 0
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.trtllm \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --extra-engine-args "$AGG_ENGINE_ARGS" \
  --modality "$MODALITY" \
  --publish-events-and-metrics \
  "${EXTRA_ARGS[@]}" &
WORKER_PID=$!

wait $DYNAMO_PID
