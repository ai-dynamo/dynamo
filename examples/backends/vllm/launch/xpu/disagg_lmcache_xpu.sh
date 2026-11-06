#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../../common/launch_utils.sh"

# XPU-specific environment
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn

print_launch_banner "Launching Disaggregated Serving + LMCache on XPU (2 XPUs)" "$MODEL" "$HTTP_PORT"

# run ingress with KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend --router-mode kv &

# run decode worker on XPU 0, without enabling LMCache
ZE_AFFINITY_MASK=0 python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size 64 &

# wait for decode worker to initialize
sleep 20

# run prefill worker on XPU 1 with LMCache
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
ZE_AFFINITY_MASK=1 \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --block-size 64 \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both","kv_buffer_device":"xpu"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
