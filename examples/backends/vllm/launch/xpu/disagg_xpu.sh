#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

export VLLM_TARGET_DEVICE=xpu

# Common configuration
MODEL="Qwen/Qwen3-0.6B"
BLOCK_SIZE=64

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated Serving XPU (2 GPUs)" "$MODEL" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# --enforce-eager is added for quick deployment. for production use, need to remove this flag
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --disaggregation-mode decode \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=1 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size $BLOCK_SIZE \
    --enforce-eager \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
