#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
source "$REPO_ROOT/examples/common/gpu_utils.sh"
source "$REPO_ROOT/examples/common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder}"
CUSTOM_JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$REPO_ROOT/examples/custom_encoder/templates/qwen_vl.jinja}"
DECODER_GPU="${DYN_DECODER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -n "$GPU_MEM_ARGS" ]] || \
    GPU_MEM_ARGS="--gpu-memory-utilization ${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

print_launch_banner --no-curl "Remote User Ensemble" "$MODEL" "$HTTP_PORT" \
    "Encoder:     $ENCODER_CLASS" \
    "Decoder GPU: $DECODER_GPU"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export DYN_MODEL="$MODEL"
export DYN_CUSTOM_JINJA_TEMPLATE="$CUSTOM_JINJA_TEMPLATE"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --workflow-provider \
    examples.custom_backend.user_ensemble.remote.provider:provide_workflow &

CUDA_VISIBLE_DEVICES= \
DYN_SYSTEM_PORT="${DYN_ENCODER_SYSTEM_PORT:-8081}" \
python3 -m examples.custom_backend.user_ensemble.remote.worker encoder \
    --model "$MODEL" \
    --encoder-class "$ENCODER_CLASS" &

CUDA_VISIBLE_DEVICES= \
DYN_SYSTEM_PORT="${DYN_CLASSIFIER_SYSTEM_PORT:-8082}" \
python3 -m examples.custom_backend.user_ensemble.remote.worker classifier &

CUDA_VISIBLE_DEVICES="$DECODER_GPU" \
DYN_SYSTEM_PORT="${DYN_DECODER_SYSTEM_PORT:-8083}" \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --endpoint dyn://user-ensemble.generator.generate \
    --enable-prompt-embeds \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "$@" &

wait_any_exit
