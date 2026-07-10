#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated 1P1D serving via the OpenEngine sidecar + KVBM on prefill (2 GPUs).
#
# Same five-process shape as sidecar_disagg.sh (frontend + decode engine/sidecar
# + prefill engine/sidecar), with the prefill engine running the KVBM PdConnector
# (DynamoConnector for G1->G2->G3 tiering, chained with NixlConnector for the P/D
# KV handoff). Decode runs a plain NixlConnector.
#
# Roles are derived from the OUTER kv_role over the OpenEngine handshake
# (kv_producer -> PREFILL, kv_consumer -> DECODE) — the sidecar has no
# `--disaggregation-mode` flag. This differs from the in-process disagg_kvbm.sh,
# which uses kv_role=kv_both + `--disaggregation-mode prefill`. PdConnector itself
# only requires [KVBM-family, NixlConnector] and does not constrain the outer
# kv_role, so the outer role is set to kv_producer here. *** Phase C cluster-
# verify item: confirm vLLM accepts kv_role=kv_producer on the PdConnector outer
# config and that mark_prefill_request's do_remote_decode reaches the inner
# NixlConnector through MultiConnector. ***
#
# KVBM runs inside the prefill engine process (its connector leader/worker live
# in the headless EngineCore / GPU worker), single-node (DYN_RUNTIME_ENABLED_KVBM=0).
# The sidecars relay only the opaque KvSessionRef <-> kv_transfer_params handoff;
# no GPU/torch/NIXL in the Dynamo workers.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Deterministic block-hash seed so KV event IDs line up across processes.
export PYTHONHASHSEED=0

MODEL="Qwen/Qwen3-0.6B"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    echo "Usage: $0"
    echo "  Launches a 1P1D sidecar deployment with KVBM tiering on prefill (2 GPUs)."
    echo "  KVBM tiers via DYN_KVBM_CPU_CACHE_GB (default 20) / DYN_KVBM_DISK_CACHE_GB."
    echo "  For a single-GPU box, co-locate both engines on CUDA_VISIBLE_DEVICES=0"
    echo "  with --gpu-memory-utilization ~0.4 each (see sidecar P/D smoke notes)."
    exit 0
fi

# OpenEngine gRPC ports (one per engine).
DECODE_OE_PORT="${DECODE_OE_PORT:-50051}"
PREFILL_OE_PORT="${PREFILL_OE_PORT:-50052}"
# vllm-rs runs its own OpenAI HTTP frontend per engine; unused by the sidecar.
DECODE_HTTP_PORT="${DECODE_HTTP_PORT:-8100}"
PREFILL_HTTP_PORT="${PREFILL_HTTP_PORT:-8101}"

# ---- Connector configs ----
DECODE_KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
# Prefill: PdConnector chains KVBM tiering (DynamoConnector, kv_both) with the
# NIXL P/D transfer (NixlConnector, kv_producer). Outer kv_role=kv_producer so
# the sidecar derives the PREFILL role.
PREFILL_KV_TRANSFER_CONFIG='{"kv_connector":"PdConnector","kv_role":"kv_producer","kv_connector_extra_config":{"connectors":[{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"},{"kv_connector":"NixlConnector","kv_role":"kv_producer"}]},"kv_connector_module_path":"kvbm.vllm_integration.connector"}'
PREFILL_KV_EVENTS_CONFIG='{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}'

# ---- KVBM config for the prefill engine (single-node tiering) ----
export DYN_RUNTIME_ENABLED_KVBM="${DYN_RUNTIME_ENABLED_KVBM:-0}"
export DYN_KVBM_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-20}"
export DYN_KVBM_LEADER_ZMQ_PUB_PORT="${DYN_KVBM_LEADER_ZMQ_PUB_PORT:-56001}"
export DYN_KVBM_LEADER_ZMQ_ACK_PORT="${DYN_KVBM_LEADER_ZMQ_ACK_PORT:-56002}"
export DYN_KVBM_METRICS="${DYN_KVBM_METRICS:-1}"
export DYN_KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
[[ -n "${DYN_KVBM_DISK_CACHE_GB:-}" ]] && export DYN_KVBM_DISK_CACHE_GB

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Disaggregated + KVBM on prefill (2 GPUs)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Decode engine: vllm-rs serve, kv_consumer, no KVBM. --enforce-eager is for
# quick startup; drop it for production. --disable-hybrid-kv-cache-manager:
# matches the prefill engine's KV-cache layout so the NIXL prefill->decode
# handshake compat check passes (the prefill MUST disable HMA, see below; the
# two instances must have identical KV-cache configs or NIXL refuses the xfer).
CUDA_VISIBLE_DEVICES=0 vllm-rs serve "$MODEL" \
    --port "$DECODE_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$DECODE_OE_PORT" \
    --enforce-eager \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$DECODE_KV_TRANSFER_CONFIG" &

# 3. Decode sidecar worker (endpoint-only; role discovered as DECODE).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "127.0.0.1:${DECODE_OE_PORT}" &

# 4. Prefill engine: vllm-rs serve, kv_producer, KVBM PdConnector + KV events.
# The KVBM connector + consolidator run inside this engine's EngineCore.
# --disable-hybrid-kv-cache-manager is REQUIRED for PdConnector: MultiConnector
# asserts HMA is off unless every sub-connector supports it (DynamoConnector/KVBM
# does not), and the auto-disable that fires for a *single* connector does NOT
# fire through MultiConnector -> startup AssertionError without this flag.
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_PORT=20097 vllm-rs serve "$MODEL" \
    --port "$PREFILL_HTTP_PORT" \
    --openengine-host 127.0.0.1 \
    --openengine-port "$PREFILL_OE_PORT" \
    --enforce-eager \
    --disable-hybrid-kv-cache-manager \
    --kv-transfer-config "$PREFILL_KV_TRANSFER_CONFIG" \
    --kv-events-config "$PREFILL_KV_EVENTS_CONFIG" &

# 5. Prefill sidecar worker (endpoint-only; role discovered as PREFILL).
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "127.0.0.1:${PREFILL_OE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
