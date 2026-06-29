#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

HTTP_PORT_R1="${DYN_HTTP_PORT_R1:-8000}"
HTTP_PORT_R2="${DYN_HTTP_PORT_R2:-8001}"
print_launch_banner --no-curl "Launching Aggregated + KV Routing + Replicas (2 GPUs)" "$MODEL" "$HTTP_PORT_R1" \
    "Frontend R2: http://localhost:$HTTP_PORT_R2"

# run two routers (different HTTP + system ports)
# Note: use --router-reset-states only on one router to avoid wiping shared state twice.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_R1:-8091} \
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states \
    --http-port ${DYN_HTTP_PORT_R1:-8000} &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT_R2:-8092} \
python -m dynamo.frontend \
    --router-mode kv \
    --http-port ${DYN_HTTP_PORT_R2:-8001} &

# run workers (local indexer is enabled by default, so routers can query on restart)
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
