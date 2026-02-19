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

# Router mode: "kv" for cache-aware routing, "round_robin" for simple round-robin
ROUTER_MODE="${ROUTER_MODE:-kv}"

# Determine number of GPUs
if [[ -n "$1" ]]; then
    NUM_GPUS=$1
elif [[ -z "$NUM_GPUS" ]]; then
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [[ "$NUM_GPUS" -lt 1 ]]; then
        echo "Error: Need at least 1 GPU. Found: $NUM_GPUS"
        exit 1
    fi
fi

BASE_KV_EVENT_PORT="${BASE_KV_EVENT_PORT:-20080}"
BASE_SYSTEM_PORT="${BASE_SYSTEM_PORT:-8082}"

# Get host IP address
HOST_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || hostname -i 2>/dev/null | awk '{print $1}' || echo "localhost")

echo "Launching Dynamo SGLang workers for multi-turn RL training (aggregated)"
echo "   Model: $MODEL"
echo "   GPUs: $NUM_GPUS monolithic workers (each does prefill + decode)"
echo "   Router mode: $ROUTER_MODE"
echo "   Frontend: http://${HOST_IP}:${DYN_HTTP_PORT:-8000}/v1"
echo ""

# Start frontend
FRONTEND_ARGS=(--router-mode "$ROUTER_MODE" --router-reset-states)
    if [[ "$ROUTER_MODE" == "round-robin" ]]; then
    FRONTEND_ARGS+=(--no-kv-events)
fi

python -m dynamo.frontend "${FRONTEND_ARGS[@]}" &

# Run monolithic workers (one per GPU)
for i in $(seq 0 $((NUM_GPUS - 1))); do
    KV_EVENT_PORT=$((BASE_KV_EVENT_PORT + i))
    SYSTEM_PORT=$((BASE_SYSTEM_PORT + i))

    # Only include KV events config when using KV-aware routing
    KV_EVENTS_ARGS=()
    if [[ "$ROUTER_MODE" != "round-robin" ]]; then
        KV_EVENTS_ARGS=(--kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:'"$KV_EVENT_PORT"'"}')
    fi

    OTEL_SERVICE_NAME=dynamo-worker-$i \
    DYN_SYSTEM_PORT=$SYSTEM_PORT \
    CUDA_VISIBLE_DEVICES=$i \
        python3 -m dynamo.sglang \
            --model-path $MODEL \
            --served-model-name $MODEL \
            --page-size $PAGE_SIZE \
            --tp 1 \
            --stream-interval 100 \
            --trust-remote-code \
            --host 0.0.0.0 \
            "${KV_EVENTS_ARGS[@]}" \
            --enable-metrics &
done

wait
