#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../../..")"
source "$REPO_ROOT/examples/common/gpu_utils.sh"
source "$REPO_ROOT/examples/common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
SERVED_MODEL_NAME="${DYN_SERVED_MODEL_NAME:-$MODEL}"
ENCODER_MODEL="${DYN_ENCODER_MODEL:-$MODEL}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder}"
CUSTOM_JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$REPO_ROOT/examples/custom_encoder/templates/qwen_vl.jinja}"
GENERATOR_MODEL_NAME="${DYN_GENERATOR_SERVED_MODEL_NAME:-user-ensemble-generator}"
NIXL_SEND_POOL_CAPACITY="${DYN_NIXL_SEND_POOL_CAPACITY:-0}"
NIXL_SEND_POOL_BYTES="${DYN_NIXL_SEND_POOL_BYTES:-0}"
ENCODER_BATCH_QUEUE_WAIT_MS="${DYN_ENCODER_BATCH_QUEUE_WAIT_MS:-2}"
ENCODER_BATCH_QUEUE_MAX_WAIT_MS="${DYN_ENCODER_BATCH_QUEUE_MAX_WAIT_MS:-50}"
DECODER_GPU="${DYN_DECODER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -n "$GPU_MEM_ARGS" ]] || \
    GPU_MEM_ARGS="--gpu-memory-utilization ${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

print_launch_banner --no-curl "Remote User Ensemble" "$MODEL" "$HTTP_PORT" \
    "Encoder:     $ENCODER_CLASS" \
    "Encoder model: $ENCODER_MODEL" \
    "Decoder GPU: $DECODER_GPU"

export DYN_DISCOVERY_BACKEND="${DYN_DISCOVERY_BACKEND:-file}"
export DYN_EVENT_PLANE="${DYN_EVENT_PLANE:-zmq}"
export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
export DYN_MODEL="$MODEL"
export DYN_CUSTOM_JINJA_TEMPLATE="$CUSTOM_JINJA_TEMPLATE"

python3 -m dynamo.frontend \
    --http-port "$HTTP_PORT" &

CUDA_VISIBLE_DEVICES= \
DYN_SYSTEM_PORT="${DYN_ENCODER_SYSTEM_PORT:-8081}" \
python3 -m dynamo.experimental.workflow.vllm.encoder_worker \
    --endpoint-id user-ensemble.encoder.generate \
    --model "$ENCODER_MODEL" \
    --custom-encoder-class "$ENCODER_CLASS" \
    --nixl-send-pool-capacity "$NIXL_SEND_POOL_CAPACITY" \
    --nixl-send-pool-bytes "$NIXL_SEND_POOL_BYTES" \
    --batch-queue-wait-ms "$ENCODER_BATCH_QUEUE_WAIT_MS" \
    --batch-queue-max-wait-ms "$ENCODER_BATCH_QUEUE_MAX_WAIT_MS" &

CUDA_VISIBLE_DEVICES= \
DYN_SYSTEM_PORT="${DYN_CLASSIFIER_SYSTEM_PORT:-8082}" \
python3 -m examples.experimental.workflow.user_ensemble.remote.classifier_worker &

DYN_MODEL="$MODEL" \
DYN_SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
DYN_CUSTOM_JINJA_TEMPLATE="$CUSTOM_JINJA_TEMPLATE" \
DYN_SYSTEM_PORT="${DYN_ORCHESTRATOR_SYSTEM_PORT:-8084}" \
python3 -m examples.experimental.workflow.user_ensemble.remote.orchestrator_worker &

CUDA_VISIBLE_DEVICES="$DECODER_GPU" \
DYN_SYSTEM_PORT="${DYN_DECODER_SYSTEM_PORT:-8083}" \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$GENERATOR_MODEL_NAME" \
    --endpoint dyn://user-ensemble.generator.generate \
    --enable-prompt-embeds \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "$@" &

wait_any_exit
