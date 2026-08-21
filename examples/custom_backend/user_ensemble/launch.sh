#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../..")"
source "$REPO_ROOT/examples/common/gpu_utils.sh"
source "$REPO_ROOT/examples/common/launch_utils.sh"

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder}"
CUSTOM_JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$REPO_ROOT/examples/custom_encoder/templates/qwen_vl.jinja}"
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -z "$GPU_MEM_ARGS" ]] && \
    GPU_MEM_ARGS="--gpu-memory-utilization ${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

print_launch_banner --no-curl "User Ensemble Worker" "$MODEL" "$HTTP_PORT" \
    "Worker GPU: $WORKER_GPU" \
    "Encoder:    $ENCODER_CLASS"

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

python -m dynamo.frontend &

CUDA_VISIBLE_DEVICES=$WORKER_GPU \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python -m examples.custom_backend.user_ensemble.worker \
    --model "$MODEL" \
    --encoder-class "$ENCODER_CLASS" \
    --custom-jinja-template "$CUSTOM_JINJA_TEMPLATE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --disable-kv-routing \
    $GPU_MEM_ARGS \
    "$@" &

wait_any_exit
