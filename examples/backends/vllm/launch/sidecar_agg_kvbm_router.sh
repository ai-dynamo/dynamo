#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the OpenEngine sidecar + KVBM + KV-aware routing (1 GPU).
#
# Builds on sidecar_agg_kvbm.sh by turning on the Dynamo KV router and the
# KVBM KV-event consolidator. The consolidator runs INSIDE the engine process:
# the KVBM connector leader (in the headless EngineCore) starts it from the
# `consolidator_endpoints` that `run_headless` injects into additional_config,
# subscribes to vLLM's raw ZMQ KV-event publisher, and republishes ONE deduped,
# multi-tier (GPU/CPU/disk) event stream. The engine advertises that
# consolidator endpoint in its OpenEngine ready handshake; the sidecar surfaces
# it via GetKvEventSources, and the Dynamo KV router subscribes there (NOT to
# the raw per-rank publishers) to build its prefix-cache radix tree.
#
# Single engine for simplicity: this validates the KV-event indexing pipeline
# (parent linkage / radix-tree build) end-to-end through the sidecar. Add more
# engines + a frontend PrefillRouter for multi-worker selection.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Deterministic block-hash seed so KV event IDs line up across processes.
export PYTHONHASHSEED=0

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default model
MODEL="Qwen/Qwen3-0.6B"

# Parse command line arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <name>       Specify model (default: $MODEL)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "KVBM tiers / consolidator are sized via env vars (defaults in parens):"
            echo "  DYN_KVBM_CPU_CACHE_GB        G2 host pinned-memory cache GB (2)"
            echo "  DYN_KVBM_DISK_CACHE_GB       G3 NVMe cache GB (unset = G3 off)"
            echo "  DYN_KVBM_LEADER_ZMQ_PUB_PORT KVBM leader pub port; consolidator"
            echo "                               output binds at this + 1000 (56001)"
            echo "  KV_EVENTS_PORT               engine raw ZMQ KV-event port (20080)"
            echo ""
            echo "Additional options are passed through to the vllm-rs serve engine."
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
OPENENGINE_HOST="${OPENENGINE_HOST:-127.0.0.1}"
OPENENGINE_PORT="${OPENENGINE_PORT:-50051}"
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"
KV_EVENTS_PORT="${KV_EVENTS_PORT:-20080}"

# ---- KVBM connector + consolidator config (runs inside the engine process) ----
KV_TRANSFER_CONFIG='{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}'
# vLLM's native ZMQ KV-event publisher; the consolidator subscribes to this and
# republishes a deduped, multi-tier stream. enable_kv_cache_events is required.
KV_EVENTS_CONFIG="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${KV_EVENTS_PORT}\",\"enable_kv_cache_events\":true}"
# Single-node tiering: no DistributedRuntime in the standalone engine.
export DYN_RUNTIME_ENABLED_KVBM="${DYN_RUNTIME_ENABLED_KVBM:-0}"
export DYN_KVBM_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-2}"
# Consolidator output binds at DYN_KVBM_LEADER_ZMQ_PUB_PORT + 1000 (so 57001).
export DYN_KVBM_LEADER_ZMQ_PUB_PORT="${DYN_KVBM_LEADER_ZMQ_PUB_PORT:-56001}"
export DYN_KVBM_LEADER_ZMQ_ACK_PORT="${DYN_KVBM_LEADER_ZMQ_ACK_PORT:-56002}"
export DYN_KVBM_METRICS="${DYN_KVBM_METRICS:-1}"
export DYN_KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
[[ -n "${DYN_KVBM_DISK_CACHE_GB:-}" ]] && export DYN_KVBM_DISK_CACHE_GB

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Aggregated + KVBM + KV Routing (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend + KV router. The router subscribes to the engine's KV-event
# source discovered over OpenEngine (the consolidator endpoint) and builds its
# prefix-cache radix tree.
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states &

# 2. Native vLLM engine with the OpenEngine gRPC server, the KVBM connector, and
# the raw ZMQ KV-event publisher the consolidator feeds on. --enforce-eager is
# for quick startup; drop it for production.
vllm-rs serve "$MODEL" \
    --port "$VLLM_RS_HTTP_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --openengine-host "$OPENENGINE_HOST" \
    --openengine-port "$OPENENGINE_PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --kv-events-config "$KV_EVENTS_CONFIG" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no vllm/kvbm import; OpenEngine client only). Its
# kv_event_sources() relays the engine's consolidator endpoint to the router.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
