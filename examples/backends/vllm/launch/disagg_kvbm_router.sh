#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Common configuration
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

# Use a deterministic per-worker KV cache budget. The test harness overrides
# this through _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES when a marker supplies a
# measured value.
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

print_launch_banner "Launching Disaggregated + KVBM + KV Routing (4 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Workers:     4 (2 decode + 2 KVBM prefill)"


# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv &

# two decode workers (without KVBM)
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20095} \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    $GPU_MEM_ARGS \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2:-20096} \
CUDA_VISIBLE_DEVICES=1 \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    $GPU_MEM_ARGS \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" &

# two prefill workers with KVBM enabled
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT3:-8083} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT3:-20097} \
DYN_KVBM_LEADER_ZMQ_PUB_PORT=${DYN_KVBM_LEADER_ZMQ_PUB_PORT1:-56001} \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=${DYN_KVBM_LEADER_ZMQ_ACK_PORT1:-56002} \
CUDA_VISIBLE_DEVICES=2 \
DYN_KVBM_CPU_CACHE_GB=${DYN_KVBM_CPU_CACHE_GB:-20} \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
    $GPU_MEM_ARGS \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${DYN_VLLM_KV_EVENT_PORT1:-20081}\",\"enable_kv_cache_events\":true}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT4:-8084} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT4:-20098} \
DYN_KVBM_LEADER_ZMQ_PUB_PORT=${DYN_KVBM_LEADER_ZMQ_PUB_PORT2:-56003} \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=${DYN_KVBM_LEADER_ZMQ_ACK_PORT2:-56004} \
CUDA_VISIBLE_DEVICES=3 \
DYN_KVBM_CPU_CACHE_GB=${DYN_KVBM_CPU_CACHE_GB:-20} \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
    $GPU_MEM_ARGS \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${DYN_VLLM_KV_EVENT_PORT2:-20082}\",\"enable_kv_cache_events\":true}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
