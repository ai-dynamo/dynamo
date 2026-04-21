#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

print_launch_banner "Launching Disaggregated Serving + KVBM (2 GPUs)" "$MODEL" "$HTTP_PORT"


# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# run decode worker on GPU 0, without enabling KVBM
# NOTE: remove --enforce-eager for production use
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model "$MODEL" --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --enforce-eager &

# run prefill worker on GPU 1 with KVBM enabled using 20GB of CPU cache
# NOTE: remove --enforce-eager for production use
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
DYN_KVBM_CPU_CACHE_GB=20 \
CUDA_VISIBLE_DEVICES=1 \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --disaggregation-mode prefill \
    --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
    --enforce-eager \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
