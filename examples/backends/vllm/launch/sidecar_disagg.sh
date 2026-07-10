#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated 1P1D serving via the OpenEngine sidecar (2 GPUs).
#
# Five processes:
#   1. Dynamo frontend (HTTP ingress)
#   2. Native vLLM decode engine  (kv_role=kv_consumer) + OpenEngine gRPC server
#   3. Dynamo decode sidecar worker -> talks to (2)
#   4. Native vLLM prefill engine (kv_role=kv_producer) + OpenEngine gRPC server
#   5. Dynamo prefill sidecar worker -> talks to (4)
#
# `vllm-rs serve` self-manages a headless Python EngineCore and mounts the
# OpenEngine v1 gRPC service. The engine derives its OpenEngine role from
# kv_transfer_config.kv_role (kv_producer -> PREFILL, kv_consumer -> DECODE),
# so the sidecars receive ONLY the endpoint and discover their role from the
# engine. KV moves over NIXL between the two engines exactly as in the
# in-process disagg path; the sidecars only relay KvSessionRef <->
# kv_transfer_params across the Dynamo boundary.

set -e

# Common configuration
MODEL="Qwen/Qwen3-0.6B"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo "  Launches a 1P1D sidecar deployment (2 GPUs)."
    exit 0
fi

trap 'echo Cleaning up...; kill 0' EXIT

# OpenEngine gRPC ports (one per engine).
DECODE_OE_PORT="${DECODE_OE_PORT:-50051}"
PREFILL_OE_PORT="${PREFILL_OE_PORT:-50052}"
# vllm-rs runs its own OpenAI HTTP frontend per engine; unused by the sidecar
# but still binds, so keep each off the Dynamo frontend's port (8000).
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-8100}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-8101}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Disaggregated Serving (2 GPUs)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Decode engine: vllm-rs serve, kv_consumer, OpenEngine on DECODE_OE_PORT.
# --kv-transfer-config is forwarded to the Python EngineCore.
# --enforce-eager is for quick startup; drop it for production.
CUDA_VISIBLE_DEVICES=0 vllm-rs serve "$MODEL" \
    --port "$DECODE_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$DECODE_OE_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}' &

# 3. Decode sidecar worker (endpoint-only; role discovered as DECODE).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "127.0.0.1:${DECODE_OE_PORT}" &

# 4. Prefill engine: vllm-rs serve, kv_producer, OpenEngine on PREFILL_OE_PORT.
# KV events published so KV-aware routing can observe the prefill cache.
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 vllm-rs serve "$MODEL" \
    --port "$PREFILL_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$PREFILL_OE_PORT" \
    --enforce-eager \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}' \
    --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# 5. Prefill sidecar worker (endpoint-only; role discovered as PREFILL).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "127.0.0.1:${PREFILL_OE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
