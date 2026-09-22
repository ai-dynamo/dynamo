#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../..")"
source "$REPO_ROOT/examples/common/gpu_utils.sh"
source "$REPO_ROOT/examples/common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PUBLIC_MODEL_NAME="${DYN_SERVED_MODEL_NAME:-remote-custom-encoder}"
DECODER_MODEL_NAME="${DYN_DECODER_MODEL_NAME:-remote-custom-encoder-decoder}"
NAMESPACE="${DYN_NAMESPACE:-remote-custom-encoder}"
GENERATOR_ENDPOINT="$NAMESPACE.generator.generate"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
DECODER_GPU="${DYN_DECODER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
ENCODER_GPU="${DYN_ENCODER_GPU:-$DECODER_GPU}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -z "$GPU_MEM_ARGS" ]] && GPU_MEM_ARGS="--gpu-memory-utilization 0.8"

export DYN_MODEL="$MODEL"
export DYN_SERVED_MODEL_NAME="$PUBLIC_MODEL_NAME"
export DYN_DECODER_MODEL_NAME="$DECODER_MODEL_NAME"
export DYN_NAMESPACE="$NAMESPACE"
export DYN_REQUEST_PLANE=tcp
export DYN_REQUEST_PLANE_CODEC=msgpack
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

print_launch_banner --no-curl "Remote Custom Encoder" "$MODEL" "$HTTP_PORT" \
    "Inline encoder: ${DYN_ENCODER_CLASS:-HitchhikersVisionEncoder}" \
    "Remote decoder: dyn://$GENERATOR_ENDPOINT"

python -m dynamo.frontend --http-port "$HTTP_PORT" &

CUDA_VISIBLE_DEVICES="$DECODER_GPU" \
DYN_SYSTEM_PORT="${DYN_GENERATOR_SYSTEM_PORT:-8081}" \
python -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$DECODER_MODEL_NAME" \
    --endpoint "dyn://$GENERATOR_ENDPOINT" \
    --enable-prompt-embeds \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "$@" &

CUDA_VISIBLE_DEVICES="$ENCODER_GPU" \
DYN_SYSTEM_PORT="${DYN_ORCHESTRATOR_SYSTEM_PORT:-8082}" \
python -m examples.custom_encoder.remote.orchestrator_worker &

wait_any_exit
