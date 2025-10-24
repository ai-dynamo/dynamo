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

# Start frontend with KV routing
# The frontend will automatically detect prefill workers and activate an internal prefill router
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --router-reset-states &

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
# When registered with --is-prefill-worker, these workers are automatically detected
# by the frontend, which activates an internal prefill router for KV-aware prefill routing
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
