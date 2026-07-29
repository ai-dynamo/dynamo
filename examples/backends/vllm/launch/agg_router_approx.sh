#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Aggregated + Approximate KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT"

# run frontend with KV router (--router-mode kv) in approximate mode (--no-kv-events)
python -m dynamo.frontend \
    --router-mode kv \
    --no-kv-events &

# run workers
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
#
# If multiple workers are launched, they must not share the same system/metrics port.
# Use DYN_SYSTEM_PORT{1,2} so tests/launchers can provide a simple numbered port set.
# TODO: use build_vllm_gpu_mem_args to measure VRAM instead of relying on vLLM defaults

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-events-config '{"enable_kv_cache_events": false}' &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 python3 -m dynamo.vllm \
    --model $MODEL \
    --block-size $BLOCK_SIZE \
    --kv-events-config '{"enable_kv_cache_events": false}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
