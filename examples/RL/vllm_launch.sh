#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Model configuration
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BLOCK_SIZE=64

echo "Launching Dynamo workers for RL example"
echo "   Model: $MODEL"
echo "   Frontend: http://localhost:$HTTP_PORT/v1"
echo ""

# Start frontend with KV routing
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# Start 7 decode workers on GPUs 0-6
for i in {0..6}; do
    KV_EVENT_PORT=$((20080 + i))
    NIXL_PORT=$((20097 + i))

    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
    CUDA_VISIBLE_DEVICES=$i \
        python3 -m dynamo.vllm \
            --model $MODEL \
            --block-size $BLOCK_SIZE \
            --enforce-eager \
            --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENT_PORT"'","enable_kv_cache_events":true}' &

    echo "✓ Decode worker $i started (GPU $i, metrics port $SYSTEM_PORT)"
done

# Wait for decode workers to initialize before starting prefill
sleep 20

# Start 1 prefill worker on GPU 7
PREFILL_GPU=7
PREFILL_KV_EVENT_PORT=20087
PREFILL_NIXL_PORT=20104

VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_PORT \
CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    python3 -m dynamo.vllm \
        --model $MODEL \
        --block-size $BLOCK_SIZE \
        --enforce-eager \
        --is-prefill-worker \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$PREFILL_KV_EVENT_PORT"'","enable_kv_cache_events":true}'
