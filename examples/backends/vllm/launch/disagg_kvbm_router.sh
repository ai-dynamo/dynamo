#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"

# run decode router with kv-overlap-score-weight 0 for pure load balancing
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0 \
    --router-reset-states &

# run standalone router service for prefill workers
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-reset-states \
    --no-track-active-blocks &

# two decode workers (without KVBM)
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager &

CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager &

# two prefill workers with KVBM enabled
# Each worker needs unique ZMQ ports to avoid KVBM coordination conflicts
DYN_KVBM_LEADER_ZMQ_PUB_PORT=56001 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56002 \
CUDA_VISIBLE_DEVICES=2 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-prefill-worker \
    --connector kvbm &

DYN_KVBM_LEADER_ZMQ_PUB_PORT=56003 \
DYN_KVBM_LEADER_ZMQ_ACK_PORT=56004 \
CUDA_VISIBLE_DEVICES=3 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --enforce-eager \
    --is-prefill-worker \
    --connector kvbm
