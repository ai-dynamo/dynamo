#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

# Model configuration (matches GRPOConfig in multiturn.py)
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
PAGE_SIZE="${PAGE_SIZE:-64}"

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

NUM_DECODE=$((NUM_GPUS - 1))
PREFILL_GPU=$((NUM_GPUS - 1))
BASE_KV_EVENT_PORT="${BASE_KV_EVENT_PORT:-20080}"
BASE_SYSTEM_PORT="${BASE_SYSTEM_PORT:-8082}"

# Get host IP address
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname -i 2>/dev/null | awk '{print $1}' || echo "localhost")

echo "Launching Dynamo SGLang workers"
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

# Run decode workers on GPUs 0 to (NUM_GPUS-2)
for i in $(seq 0 $((NUM_DECODE - 1))); do
    KV_EVENT_PORT=$((BASE_KV_EVENT_PORT + i))
    SYSTEM_PORT=$((BASE_SYSTEM_PORT + i))

    OTEL_SERVICE_NAME=dynamo-worker-decode-$i \
    DYN_SYSTEM_PORT=$SYSTEM_PORT \
    CUDA_VISIBLE_DEVICES=$i \
        python3 -m dynamo.sglang \
            --model-path $MODEL \
            --served-model-name $MODEL \
            --page-size $PAGE_SIZE \
            --tp 1 \
            --stream-interval 100 \
            --trust-remote-code \
            --disaggregation-mode decode \
            --host 0.0.0.0 \
            --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENT_PORT"'"}' \
            --disaggregation-transfer-backend nixl \
            --enable-metrics &
done

# Run prefill worker on the last GPU
PREFILL_KV_EVENT_PORT=$((BASE_KV_EVENT_PORT + PREFILL_GPU))
PREFILL_SYSTEM_PORT=$((BASE_SYSTEM_PORT + PREFILL_GPU))

OTEL_SERVICE_NAME=dynamo-worker-prefill \
DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT \
CUDA_VISIBLE_DEVICES=$PREFILL_GPU \
    python3 -m dynamo.sglang \
        --model-path $MODEL \
        --served-model-name $MODEL \
        --page-size $PAGE_SIZE \
        --tp 1 \
        --stream-interval 100 \
        --trust-remote-code \
        --disaggregation-mode prefill \
        --host 0.0.0.0 \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$PREFILL_KV_EVENT_PORT"'"}' \
        --disaggregation-transfer-backend nixl \
        --enable-metrics &

wait
