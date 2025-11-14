#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"

# run frontend + KV router
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

# run workers with KVBM enabled
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56002 \
CUDA_VISIBLE_DEVICES=0 DYN_KVBM_CPU_CACHE_GB=2 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --connector kvbm \
    --gpu-memory-utilization 0.4 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=0 DYN_KVBM_CPU_CACHE_GB=2 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --connector kvbm \
    --gpu-memory-utilization 0.4 \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}'
