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

# run decode router with kv-overlap-score-weight 0 for pure load balancing
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0 \
    --router-reset-states &

# run prefill router service
python -m dynamo.vllm_prefill_router \
    --namespace dynamo \
    --block-size $BLOCK_SIZE &

# two decode workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager &

CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager &

# two prefill workers
CUDA_VISIBLE_DEVICES=2 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --is-prefill-worker &

CUDA_VISIBLE_DEVICES=3 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --is-prefill-worker
