#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
NIXL_BUFFER_DEVICE=xpu
VLLM_NIXL_BACKEND=UCX

# UCX configuration for Intel XPU (Level Zero copy)
export UCX_MEMTYPE_CACHE=0
# export UCX_TLS=shm,ze_copy
# Adjust UCX_NET_DEVICES to match your InfiniBand/RDMA devices, or remove if not using RDMA
# export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving (2 XPUs)" "$MODEL" "$HTTP_PORT"

KV_TRANSFER_CONFIG="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${NIXL_BUFFER_DEVICE}\",\"kv_connector_extra_config\":{\"backends\":[\"${VLLM_NIXL_BACKEND}\"]}}"

# run ingress
python -m dynamo.frontend &

# decode worker on XPU 0
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --disaggregation-mode decode \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" &

# prefill worker on XPU 1
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=1 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --disaggregation-mode prefill \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
