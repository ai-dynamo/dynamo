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
PREFILL_HTTP_PORT=${DYN_ENGINE_HTTP_PORT1:-8101}
DECODE_HTTP_PORT=${DYN_ENGINE_HTTP_PORT2:-8102}
PREFILL_OPENENGINE_PORT=${DYN_OPENENGINE_PORT1:-9101}
DECODE_OPENENGINE_PORT=${DYN_OPENENGINE_PORT2:-9102}
PREFILL_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081}
DECODE_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082}
PREFILL_EVENT_PORT=${DYN_KV_EVENT_PORT1:-5558}
DECODE_EVENT_PORT=${DYN_KV_EVENT_PORT2:-5559}
PREFILL_NIXL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-5600}
DECODE_NIXL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2:-5601}
OPENENGINE_SCHEMA_RELEASE=b0cf2a4826d246192dc65b055dab6d2b38d2d67e
export OPENENGINE_SCHEMA_RELEASE
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_LORA_ENABLED=${DYN_LORA_ENABLED:-false}
export DYN_LORA_PATH=${DYN_LORA_PATH:-/tmp/dynamo_loras}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
: "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

print_launch_banner "Launching vLLM context-first P/D through the shared OpenEngine sidecar (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Prefill OpenEngine: http://127.0.0.1:$PREFILL_OPENENGINE_PORT" \
    "Decode OpenEngine:  http://127.0.0.1:$DECODE_OPENENGINE_PORT"

python3 -m dynamo.frontend --router-mode kv &
CUDA_VISIBLE_DEVICES=${PREFILL_CUDA_VISIBLE_DEVICES:-0} \
VLLM_USE_RUST_FRONTEND=1 \
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
VLLM_NIXL_SIDE_CHANNEL_PORT="$PREFILL_NIXL_PORT" \
    vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$PREFILL_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$PREFILL_OPENENGINE_PORT" \
    --enforce-eager \
    --enable-lora \
    --max-loras 4 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_load_failure_policy":"fail"}' \
    --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$PREFILL_EVENT_PORT\",\"topic\":\"kv-events\"}" \
    $GPU_MEM_ARGS &
CUDA_VISIBLE_DEVICES=${DECODE_CUDA_VISIBLE_DEVICES:-1} \
VLLM_USE_RUST_FRONTEND=1 \
VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 \
VLLM_NIXL_SIDE_CHANNEL_PORT="$DECODE_NIXL_PORT" \
    vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$DECODE_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$DECODE_OPENENGINE_PORT" \
    --enforce-eager \
    --enable-lora \
    --max-loras 4 \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_load_failure_policy":"fail"}' \
    --kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:$DECODE_EVENT_PORT\",\"topic\":\"kv-events\"}" \
    $GPU_MEM_ARGS &

DYN_SYSTEM_PORT="$PREFILL_SYSTEM_PORT" dynamo-openengine-sidecar \
    --openengine-endpoint "http://127.0.0.1:$PREFILL_OPENENGINE_PORT" \
    --expected-engine vllm \
    --expected-schema-release "$OPENENGINE_SCHEMA_RELEASE" &
DYN_SYSTEM_PORT="$DECODE_SYSTEM_PORT" dynamo-openengine-sidecar \
    --openengine-endpoint "http://127.0.0.1:$DECODE_OPENENGINE_PORT" \
    --expected-engine vllm \
    --expected-schema-release "$OPENENGINE_SCHEMA_RELEASE" &

wait_any_exit
