#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + KVBM + KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT"

# run frontend + KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# run workers with KVBM enabled
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of hardcoded fractions
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56002 \
CUDA_VISIBLE_DEVICES=0 DYN_KVBM_CPU_CACHE_GB=2 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' \
    --gpu-memory-utilization 0.4 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 DYN_KVBM_CPU_CACHE_GB=2 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}' \
    --gpu-memory-utilization 0.4 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
