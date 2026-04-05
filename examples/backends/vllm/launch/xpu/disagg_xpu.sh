#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64
NIXL_BUFFER_DEVICE=cpu

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving (2 XPUs)" "$MODEL" "$HTTP_PORT"

# run ingress
python -m dynamo.frontend &

# decode worker on XPU 0
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --disaggregation-mode decode \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${NIXL_BUFFER_DEVICE}\"}" &

# prefill worker on XPU 1
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=1 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --disaggregation-mode prefill \
    --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"${NIXL_BUFFER_DEVICE}\"}" \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
