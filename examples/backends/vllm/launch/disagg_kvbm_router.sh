#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"

# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# two decode workers (without KVBM)
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-decode-worker &

VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-decode-worker &

# two prefill workers with KVBM enabled
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56002 \
CUDA_VISIBLE_DEVICES=2 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-prefill-worker \
    --connector kvbm nixl \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
CUDA_VISIBLE_DEVICES=3 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-prefill-worker \
    --connector kvbm nixl \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}'
