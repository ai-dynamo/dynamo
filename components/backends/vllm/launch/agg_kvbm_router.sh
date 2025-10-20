#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64

# run frontend + KV router
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

# run workers with KVBM enabled
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# Each worker needs unique barrier ID to avoid KVBM coordination conflicts
DYN_KVBM_BARRIER_ID_PREFIX=kvbm_worker_0 \
CUDA_VISIBLE_DEVICES=0 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --connector kvbm &

DYN_KVBM_BARRIER_ID_PREFIX=kvbm_worker_1 \
CUDA_VISIBLE_DEVICES=1 DYN_KVBM_CPU_CACHE_GB=20 \
    python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --connector kvbm

