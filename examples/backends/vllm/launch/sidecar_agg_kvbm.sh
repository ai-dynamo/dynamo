#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving via the OpenEngine sidecar + KVBM multi-tier KV cache (1 GPU).
#
# Same three-process shape as sidecar_agg.sh (Dynamo frontend + `vllm-rs serve`
# OpenEngine engine + dynamo-vllm-sidecar worker), with KVBM offload enabled.
#
# KVBM runs ENTIRELY inside the engine process: it is a vLLM KV connector
# (DynamoConnector) whose leader lives in the headless Python EngineCore and
# whose worker lives in the GPU worker. So KVBM is turned on exactly as in the
# in-process path (agg_kvbm.sh) — via `--kv-transfer-config` and `DYN_KVBM_*`
# env — except those are attached to `vllm-rs serve` (which spawns and inherits
# down to the EngineCore), NOT to the Dynamo sidecar (which has no GPU/torch and
# never imports kvbm). `DYN_RUNTIME_ENABLED_KVBM=0` keeps KVBM single-node
# (G1 GPU -> G2 host -> G3 disk), with no DistributedRuntime in the standalone
# engine process.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

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
            echo "KVBM tiers are sized via env vars (defaults in parens):"
            echo "  DYN_KVBM_CPU_CACHE_GB   G2 host pinned-memory cache GB (20)"
            echo "  DYN_KVBM_DISK_CACHE_GB  G3 NVMe cache GB (unset = G3 off)"
            echo "  DYN_KVBM_METRICS        Prometheus metrics on/off (1)"
            echo "  DYN_KVBM_METRICS_PORT   Prometheus port on the engine (6880)"
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
# vllm-rs runs its own OpenAI HTTP frontend; it is unused by the sidecar but
# still binds, so keep it off the Dynamo frontend's port (8000).
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"

# ---- KVBM connector config (runs inside the engine process) ----
KV_TRANSFER_CONFIG='{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}'
# Single-node tiering: no DistributedRuntime in the standalone engine.
export DYN_RUNTIME_ENABLED_KVBM="${DYN_RUNTIME_ENABLED_KVBM:-0}"
export DYN_KVBM_CPU_CACHE_GB="${DYN_KVBM_CPU_CACHE_GB:-20}"
export DYN_KVBM_METRICS="${DYN_KVBM_METRICS:-1}"
export DYN_KVBM_METRICS_PORT="${DYN_KVBM_METRICS_PORT:-6880}"
# Optional G3 (NVMe) tier — export only when the caller set a size.
[[ -n "${DYN_KVBM_DISK_CACHE_GB:-}" ]] && export DYN_KVBM_DISK_CACHE_GB

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Sidecar Aggregated Serving + KVBM (1 GPU)" "$MODEL" "$HTTP_PORT"

# 1. Dynamo frontend (HTTP ingress)
python -m dynamo.frontend &

# 2. Native vLLM engine (Rust frontend + managed Python EngineCore) with the
# OpenEngine gRPC server and the KVBM connector. `--kv-transfer-config` and the
# DYN_KVBM_* env are forwarded to / inherited by the headless EngineCore, where
# the KVBM leader+worker live. --enforce-eager is for quick startup; drop it for
# production.
vllm-rs serve "$MODEL" \
    --port "$VLLM_RS_HTTP_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --openengine-host "$OPENENGINE_HOST" \
    --openengine-port "$OPENENGINE_PORT" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    "${EXTRA_ARGS[@]}" &

# 3. Dynamo sidecar worker (no vllm/kvbm import; OpenEngine client only). It
# receives ONLY the endpoint; model, role, and KV config are discovered from the
# engine. KVBM is invisible here — its effect (a larger effective KV cache) is
# reported via the engine's total_kv_blocks in the OpenEngine handshake.
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    dynamo-vllm-sidecar \
    --openengine-endpoint "${OPENENGINE_HOST}:${OPENENGINE_PORT}" &

# Exit on first process failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
