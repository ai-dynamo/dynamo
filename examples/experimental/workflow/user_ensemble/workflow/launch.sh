#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../../..")"
source "$REPO_ROOT/examples/common/gpu_utils.sh"
source "$REPO_ROOT/examples/common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PUBLIC_MODEL_NAME="${DYN_SERVED_MODEL_NAME:-user-ensemble}"
DECODER_MODEL_NAME="${DYN_DECODER_MODEL_NAME:-user-ensemble-decoder}"
NAMESPACE="${DYN_NAMESPACE:-workflow-user-ensemble}"
GENERATOR_ENDPOINT="$NAMESPACE.generator.generate"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-4096}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -z "$GPU_MEM_ARGS" ]] && GPU_MEM_ARGS="--gpu-memory-utilization 0.8"

export DYN_NAMESPACE="$NAMESPACE"
export DYN_REQUEST_PLANE=tcp
export DYN_REQUEST_PLANE_CODEC=msgpack
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

print_launch_banner --no-curl "Workflow User Ensemble" "$MODEL" "$HTTP_PORT" \
    "Public model: $PUBLIC_MODEL_NAME" \
    "Inline: encoder, classifier, request adapter, response" \
    "Remote: dyn://$GENERATOR_ENDPOINT"

python -m dynamo.frontend --http-port "$HTTP_PORT" --enable-nvext &

CUDA_VISIBLE_DEVICES="$WORKER_GPU" \
DYN_SYSTEM_PORT="${DYN_GENERATOR_SYSTEM_PORT:-8081}" \
python -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "$DECODER_MODEL_NAME" \
    --endpoint "dyn://$GENERATOR_ENDPOINT" \
    --enable-prompt-embeds \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "$@" &

DYN_MODEL="$MODEL" \
DYN_SERVED_MODEL_NAME="$PUBLIC_MODEL_NAME" \
DYN_SYSTEM_PORT="${DYN_ORCHESTRATOR_SYSTEM_PORT:-8082}" \
python -m \
    examples.experimental.workflow.user_ensemble.workflow.orchestrator_worker &

wait_any_exit
