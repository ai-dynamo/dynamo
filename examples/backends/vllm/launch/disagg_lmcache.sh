#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="Qwen/Qwen3-0.6B"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

print_launch_banner "Launching Disaggregated Serving + LMCache (2 GPUs)" "$MODEL" "$HTTP_PORT"


# run ingress with KV router
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend --router-mode kv &

# decode worker (GPU 0), no LMCache. --disaggregation-mode decode is required
# under --router-mode kv (matches disagg_router.sh); NixlConnector pairs w/ prefill.
# --disable-hybrid-kv-cache-manager must match the prefill worker: the Nixl KV
# block transfer needs identical layouts on both sides (verified: without it the
# decode worker hangs on transfer and the test times out).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 python3 -m dynamo.vllm --model "$MODEL" --disaggregation-mode decode --disable-hybrid-kv-cache-manager --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# Wait for the decode worker to be ready before launching prefill — the prefill
# worker's Nixl transfer needs the decode peer up. Poll its /health endpoint with
# a bounded timeout and fail hard if it never becomes ready (no fixed-sleep race).
DECODE_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
if ! wait_for_ready "http://localhost:${DECODE_SYSTEM_PORT}/health" 60; then
    echo "ERROR: decode worker did not become ready within 60s; aborting." >&2
    exit 1
fi

# prefill worker (GPU 1) with LMCache.
# --disable-hybrid-kv-cache-manager: LMCacheConnectorV1 doesn't support HMA (MultiConnector asserts).
# Distinct DYN_SYSTEM_PORT so the prefill metrics server doesn't collide with decode's.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=1 \
  python3 -m dynamo.vllm \
    --model "$MODEL" \
    --disaggregation-mode prefill \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config '{"kv_connector":"PdConnector","kv_role":"kv_both","kv_connector_extra_config":{"connectors":[{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_both"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
