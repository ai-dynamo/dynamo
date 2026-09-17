#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One node of an externally managed SGLang attention-DP group with local KV events.
# Requires SGLang #39659 and this Dynamo sidecar on every node. Start the frontend separately.
# Defaults: two nodes, one GPU per node, global TP=2 / attention DP=2.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo "Usage: NODE_RANK=<rank> DIST_INIT_ADDR=<leader-ip:port> bash $0 [SGLang options...]"
    echo "Run once per node; use the same Dynamo namespace, discovery and event services."
    echo "Required: NODE_RANK, DIST_INIT_ADDR"
    echo "Topology: NNODES=2, TP_SIZE=NNODES, DP_SIZE=NNODES (global sizes; PP=1)"
    echo "          DP_SIZE must be divisible by NNODES so every node owns publishers."
    echo "Model:    MODEL=Qwen/Qwen3-0.6B, MAX_MODEL_LEN=4096, MAX_CONCURRENT_SEQS=32"
    echo "Role:     ROLE=aggregated|prefill|decode (default: aggregated)"
    echo "PD:       SGLANG_BOOTSTRAP_HOST=<reachable prefill leader IP> on prefill node 0"
    echo "Ports:    SGLANG_HTTP_PORT=30000, SGLANG_GRPC_PORT=30001, SGLANG_KV_EVENT_PORT=5557"
    echo "          SGLANG_DISAGGREGATION_BOOTSTRAP_PORT=8998, DYN_SYSTEM_PORT=8081"
    echo "          DYN_HTTP_PORT=8000 is the separately launched frontend's port."
    echo "Other:    SGLANG_PYTHON=python3, SGLANG_HOST=0.0.0.0, SGLANG_PAGE_SIZE=64"
    echo "          CUDA_VISIBLE_DEVICES is inherited. Extra arguments go to SGLang only."
    exit 0
fi

: "${NODE_RANK:?Set NODE_RANK to the local node rank within the engine group}"
: "${DIST_INIT_ADDR:?Set DIST_INIT_ADDR to the reachable engine leader host:port}"
NNODES="${NNODES:-2}"
TP_SIZE="${TP_SIZE:-$NNODES}"
DP_SIZE="${DP_SIZE:-$NNODES}"
for size in "$NNODES" "$TP_SIZE" "$DP_SIZE"; do
    if [[ ! "$size" =~ ^[1-9][0-9]*$ ]]; then
        echo "NNODES, TP_SIZE and DP_SIZE must be positive integers" >&2
        exit 1
    fi
done
if [[ ! "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]] || (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK must be in [0, NNODES)" >&2
    exit 1
fi
if (( NNODES < 2 || TP_SIZE % DP_SIZE != 0 || DP_SIZE % NNODES != 0 )); then
    echo "This example requires NNODES >= 2, TP_SIZE divisible by DP_SIZE, and DP_SIZE divisible by NNODES" >&2
    exit 1
fi

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
ROLE="${ROLE:-aggregated}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
SGLANG_HOST="${SGLANG_HOST:-0.0.0.0}"
SGLANG_HTTP_PORT="${SGLANG_HTTP_PORT:-30000}"
SGLANG_GRPC_PORT="${SGLANG_GRPC_PORT:-30001}"
SGLANG_KV_EVENT_PORT="${SGLANG_KV_EVENT_PORT:-5557}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-32}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
ENGINE_ROLE_ARGS=()
SIDECAR_ARGS=()
case "$SGLANG_HOST" in
    0.0.0.0) SIDECAR_HOST=127.0.0.1 ;;
    ::) SIDECAR_HOST='[::1]' ;;
    *:*) SIDECAR_HOST="[$SGLANG_HOST]" ;;
    *) SIDECAR_HOST="$SGLANG_HOST" ;;
esac

case "$ROLE" in
    aggregated) ;;
    prefill|decode)
        ENGINE_ROLE_ARGS=(--disaggregation-mode "$ROLE" --disaggregation-transfer-backend nixl
            --disaggregation-bootstrap-port "${SGLANG_DISAGGREGATION_BOOTSTRAP_PORT:-8998}")
        if [[ "$ROLE" == prefill && "$NODE_RANK" == 0 ]]; then
            : "${SGLANG_BOOTSTRAP_HOST:?Set SGLANG_BOOTSTRAP_HOST to the reachable prefill leader address}"
            SIDECAR_ARGS+=(--bootstrap-host "$SGLANG_BOOTSTRAP_HOST")
        fi
        ;;
    *) echo "ROLE must be aggregated, prefill or decode" >&2; exit 1 ;;
esac
if (( NODE_RANK > 0 )); then
    SIDECAR_ARGS+=(--telemetry-only)
fi

print_launch_banner --no-curl "SGLang multinode KV sidecar ($ROLE, node $NODE_RANK)" "$MODEL" "$HTTP_PORT" \
    "Topology: NNODES=$NNODES, TP_SIZE=$TP_SIZE, DP_SIZE=$DP_SIZE (attention DP)" \
    "Rendezvous: $DIST_INIT_ADDR; local gRPC: $SGLANG_GRPC_PORT" \
    "Start the frontend separately. Restart all nodes and sidecars together after failure."

trap dynamo_exit_trap EXIT

# Sidecars retry gRPC startup; no sleep or inference health check is needed on followers.
# SGLang needs a wildcard bind to advertise a node-local KV-event source.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" --port "$SGLANG_HTTP_PORT" --grpc-port "$SGLANG_GRPC_PORT" \
    --incremental-streaming-output \
    --nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT_ADDR" \
    --tp-size "$TP_SIZE" --dp-size "$DP_SIZE" --enable-dp-attention \
    --kv-events-config "{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_KV_EVENT_PORT}\",\"topic\":\"\"}" \
    --page-size "$SGLANG_PAGE_SIZE" --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    "${ENGINE_ROLE_ARGS[@]}" $GPU_MEM_ARGS "$@" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    python3 -m dynamo.sglang.sidecar \
    --grpc-endpoint "http://$SIDECAR_HOST:$SGLANG_GRPC_PORT" "${SIDECAR_ARGS[@]}" &

wait_any_exit
