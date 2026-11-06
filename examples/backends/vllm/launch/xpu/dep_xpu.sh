#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Common configuration
MODEL="Qwen/Qwen3-30B-A3B"
BLOCK_SIZE=64

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

print_launch_banner "Launching Data Parallel / Expert Parallelism (4 XPUs)" "$MODEL" "$HTTP_PORT"

# run ingress
python -m dynamo.frontend --router-mode kv &

# Data Parallel Attention / Expert Parallelism
# Routing to DP workers managed by Dynamo
# Qwen3-30B-A3B is a small MoE model that fits on smaller GPUs
VLLM_NIXL_SIDE_CHANNEL_PORT=20096 \
ZE_AFFINITY_MASK=0,1,2,3 \
CCL_ZE_IPC_EXCHANGE=sockets \
python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --max_model_len 25.6k \
    --data-parallel-hybrid-lb \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-start-rank 0 \
    --enable-expert-parallel \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}' &

echo "All workers starting. (press Ctrl+C to stop)..."
# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
