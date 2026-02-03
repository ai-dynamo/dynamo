#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Model configuration
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"

# Determine number of GPUs
if [[ -n "$1" ]]; then
    NUM_GPUS=$1
elif [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ "$NUM_GPUS" -lt 2 ]]; then
        echo "Error: Need at least 2 GPUs (1 decode + 1 prefill). Found: $NUM_GPUS"
        exit 1
    fi
fi

# Calculate worker distribution: last GPU for prefill, rest for decode
NUM_DECODE=$((NUM_GPUS - 1))
PREFILL_GPU=$((NUM_GPUS - 1))

# Base ports for workers
BASE_KV_EVENT_PORT="${BASE_KV_EVENT_PORT:-20080}"
BASE_NIXL_PORT="${BASE_NIXL_PORT:-20097}"

# Get host IP address
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname -i 2>/dev/null | awk '{print $1}' || echo "localhost")

echo "Launching Dynamo vLLM workers for multi-turn RL training"
echo "   Model: $MODEL"
echo "   GPUs: $NUM_GPUS total ($NUM_DECODE decode + 1 prefill)"
echo "   Decode workers: GPUs 0-$((NUM_DECODE - 1))"
echo "   Prefill worker: GPU $PREFILL_GPU"
echo "   Frontend: http://${HOST_IP}:${DYN_HTTP_PORT:-8000}/v1"
echo ""

# Start frontend with KV routing
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# Start decode workers on GPUs 0 to (NUM_GPUS-2)
for i in $(seq 0 $((NUM_DECODE - 1))); do
    KV_EVENT_PORT=$((BASE_KV_EVENT_PORT + i))
    NIXL_PORT=$((BASE_NIXL_PORT + i))

    VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT \
    CUDA_VISIBLE_DEVICES=$i \
        python3 -m dynamo.vllm \
            --model $MODEL \
            --block-size $BLOCK_SIZE \
            --is-decode-worker &
done

# Start prefill worker on the last GPU
PREFILL_KV_EVENT_PORT=$((BASE_KV_EVENT_PORT + PREFILL_GPU))
PREFILL_NIXL_PORT=$((BASE_NIXL_PORT + PREFILL_GPU))

VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_NIXL_PORT \
CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    python3 -m dynamo.vllm \
        --model $MODEL \
        --block-size $BLOCK_SIZE \
        --is-prefill-worker \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$PREFILL_KV_EVENT_PORT"'","enable_kv_cache_events":true}' &

wait
