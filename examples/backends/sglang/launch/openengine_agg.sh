#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL=${MODEL_PATH:-Qwen/Qwen3-0.6B}
HTTP_PORT=${DYN_HTTP_PORT:-8000}
ENGINE_HTTP_PORT=${DYN_ENGINE_HTTP_PORT:-30000}
OPENENGINE_PORT=${DYN_OPENENGINE_PORT:-50051}
SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081}
KV_EVENT_PORT=${DYN_KV_EVENT_PORT:-5557}
OPENENGINE_SCHEMA_RELEASE=b0cf2a4826d246192dc65b055dab6d2b38d2d67e
export OPENENGINE_SCHEMA_RELEASE
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-false}
export DYN_LORA_PATH=${DYN_LORA_PATH:-/tmp/dynamo_loras}
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

print_launch_banner "Launching SGLang through the shared OpenEngine sidecar (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "SGLang HTTP: http://127.0.0.1:$ENGINE_HTTP_PORT" \
    "OpenEngine:  http://127.0.0.1:$OPENENGINE_PORT"

python3 -m dynamo.frontend --router-mode kv &
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name "$MODEL" \
    --host 127.0.0.1 \
    --port "$ENGINE_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$OPENENGINE_PORT" \
    --page-size 16 \
    --enable-lora \
    --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://127.0.0.1:$KV_EVENT_PORT\",\"topic\":\"\"}" \
    $GPU_MEM_ARGS &
DYN_SYSTEM_PORT="$SYSTEM_PORT" dynamo-openengine-sidecar \
    --openengine-endpoint "http://127.0.0.1:$OPENENGINE_PORT" \
    --expected-engine sglang \
    --expected-schema-release "$OPENENGINE_SCHEMA_RELEASE" &

wait_any_exit
