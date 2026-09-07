#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with KV routing on 2 XPU GPUs.
#
# GPU assignment:
#   ZE_AFFINITY_MASK - Comma-separated XPU device indices (e.g. "0,1" or "2,3").
#                      First device → worker 1, second → worker 2. Default: "0,1"
#
# Port configuration (set by test framework or defaults):
#   DYN_VLLM_KV_EVENT_PORT1  - ZMQ KV event port for worker 1 (default: 20080)
#   DYN_VLLM_KV_EVENT_PORT2  - ZMQ KV event port for worker 2 (default: 20081)
#   DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1 - NIXL side channel port for worker 1 (default: 20097)
#   DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2 - NIXL side channel port for worker 2 (default: 20098)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# Set deterministic hash for KV event IDs
export PYTHONHASHSEED=0

export VLLM_TARGET_DEVICE=xpu

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Parse ZE_AFFINITY_MASK (comma-separated) into per-worker device indices
IFS=',' read -ra _GPU_IDS <<< "${ZE_AFFINITY_MASK:-0,1}"
GPU_WORKER1="${_GPU_IDS[0]:-0}"
GPU_WORKER2="${_GPU_IDS[1]:-1}"

# KV event ports (configurable to avoid collisions in parallel test runs)
KV_EVENT_PORT1="${DYN_VLLM_KV_EVENT_PORT1:-20080}"
KV_EVENT_PORT2="${DYN_VLLM_KV_EVENT_PORT2:-20081}"

# NIXL side channel ports (per-worker, avoids collisions in parallel test runs)
NIXL_PORT1="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20097}"
NIXL_PORT2="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2:-20098}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + KV Routing (2 GPUs: $GPU_WORKER1, $GPU_WORKER2)" "$MODEL" "$HTTP_PORT"

# run frontend + KV router
python -m dynamo.frontend \
    --router-mode kv &

# run workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
#
# If multiple workers are launched, they must not share the same system/metrics port.
# Use DYN_SYSTEM_PORT{1,2} so tests/launchers can provide a simple numbered port set.
#
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT1 \
ZE_AFFINITY_MASK=$GPU_WORKER1 python3 -m dynamo.vllm \
    --model $MODEL \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    ${GPU_MEM_ARGS:---gpu-memory-utilization 0.75} \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENT_PORT1}\",\"enable_kv_cache_events\":true}" &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$NIXL_PORT2 \
ZE_AFFINITY_MASK=$GPU_WORKER2 python3 -m dynamo.vllm \
    --model $MODEL \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    ${GPU_MEM_ARGS:---gpu-memory-utilization 0.75} \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENT_PORT2}\",\"enable_kv_cache_events\":true}" &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
