#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Qwen2.5 whole-pipeline benchmark control: one synchronous encoder + vLLM batch.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen2_5_vl_benchmark_encoder.Qwen2_5VLBenchmarkEncoder}"
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${DYN_MAX_NUM_SEQS:-64}"
CONTROL_MAX_BATCH_ITEMS="${DYN_CONTROL_MAX_BATCH_ITEMS:-8}"
CONTROL_MAX_QUEUE_DELAY_US="${DYN_CONTROL_MAX_QUEUE_DELAY_US:-1000}"
CUSTOM_JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$SCRIPT_DIR/../templates/qwen_vl.jinja}"

export DYN_QWEN2_VL_ENCODER_MODEL="${DYN_QWEN2_VL_ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE="${DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE:-1536}"
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY="${DYN_QWEN2_VL_PREPROCESS_CONCURRENCY:-64}"
export DYN_QWEN2_VL_MAX_BATCH_PATCHES="${DYN_QWEN2_VL_MAX_BATCH_PATCHES:-10368}"
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS="${DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS:-1,2,4,8}"
export DYN_QWEN2_VL_MAX_BATCH_ITEMS="${DYN_QWEN2_VL_MAX_BATCH_ITEMS:-8}"
export DYN_QWEN2_VL_MAX_QUEUE_DELAY_US="${DYN_QWEN2_VL_MAX_QUEUE_DELAY_US:-1000}"
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES="${DYN_QWEN2_VL_GRAPH_IMAGE_SIZES:-300x300,500x500}"
export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export DYN_HTTP_PORT="$HTTP_PORT"

if [[ -n "${DYN_VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
    GPU_MEM_ARGS="--gpu-memory-utilization $DYN_VLLM_GPU_MEMORY_UTILIZATION"
else
    GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
    [[ -z "$GPU_MEM_ARGS" ]] && GPU_MEM_ARGS="--gpu-memory-utilization 0.4"
fi

print_launch_banner --no-curl \
    "Qwen2.5 CustomEncoder — Synchronous Batched Control" \
    "$MODEL" \
    "$HTTP_PORT" \
    "Worker GPU:       $WORKER_GPU" \
    "Encoder:          $ENCODER_CLASS" \
    "Outer batch:      $CONTROL_MAX_BATCH_ITEMS" \
    "Outer queue hold: ${CONTROL_MAX_QUEUE_DELAY_US}us" \
    "Vision graphs:    300x300,500x500 x 1,2,4,8" \
    "WARNING: performance-only 2048-to-1536 vision-output truncation; no quality claim."

python -m dynamo.frontend &

JINJA_ARG=()
[[ -n "$CUSTOM_JINJA_TEMPLATE" ]] && \
    JINJA_ARG=(--custom-jinja-template "$CUSTOM_JINJA_TEMPLATE")

CUDA_VISIBLE_DEVICES=$WORKER_GPU \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python -m examples.custom_encoder.benchmark.batched_control_worker \
    --model "$MODEL" \
    --encoder-class "$ENCODER_CLASS" \
    --control-max-batch-items "$CONTROL_MAX_BATCH_ITEMS" \
    --control-max-queue-delay-us "$CONTROL_MAX_QUEUE_DELAY_US" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-prompt-embeds \
    $GPU_MEM_ARGS \
    "${JINJA_ARG[@]}" \
    "$@" &

wait_any_exit
