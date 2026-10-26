#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Data- and expert-parallel serving through one SGLang native-gRPC sidecar with
# KV-aware routing across its data-parallel ranks (2 GPUs).
# Requires an SGLang build with native gRPC, KV events, and DP attention support.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"

MODEL="${MODEL:-silence09/DeepSeek-R1-Small-2layers}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|--model-path)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for $1"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model|--model-path <name>] [SGLang engine options...]"
            echo
            echo "Additional options are passed to the SGLang engine."
            echo
            echo "Environment overrides:"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            echo "  MODEL                   MoE model to serve (default: silence09/DeepSeek-R1-Small-2layers)"
            echo "  SGLANG_PYTHON           Python with SGLang installed (default: python3)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment, one per rank (default: 0,1)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  SGLANG_HTTP_PORT        SGLang HTTP port (default: 30000)"
            echo "  SGLANG_GRPC_PORT        SGLang gRPC port (default: 30001)"
            echo "  SGLANG_KV_EVENT_PORT    First KV-event port; rank N publishes on this port + N (default: 5557)"
            echo "  SGLANG_DIST_INIT_PORT   Distributed-init port; SGLang also uses the ports after it (default: 29500)"
            echo "  SGLANG_NCCL_PORT        NCCL port (default: 29520)"
            echo "  SGLANG_PARALLEL_SIZE    Tensor, data and expert parallel size (default: 2)"
            echo "  SGLANG_PAGE_SIZE        KV-event block size (default: 64)"
            echo "  MAX_MODEL_LEN           Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS     Maximum concurrent sequences (default: 2)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

resolve_sidecar sglang SIDECAR_CMD

trap dynamo_exit_trap EXIT

SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
SGLANG_HOST="127.0.0.1"
SGLANG_HTTP_PORT=$(dyn_port DYN_ENGINE_HTTP_PORT 1 "${SGLANG_HTTP_PORT:-30000}")
SGLANG_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 1 "${SGLANG_GRPC_PORT:-30001}")
SGLANG_KV_EVENT_PORT=$(dyn_port DYN_VLLM_KV_EVENT_PORT 1 "${SGLANG_KV_EVENT_PORT:-5557}")
SGLANG_DIST_INIT_PORT=$(dyn_port DYN_DIST_INIT_PORT 1 "${SGLANG_DIST_INIT_PORT:-29500}")
SGLANG_NCCL_PORT=$(dyn_port DYN_NCCL_PORT 1 "${SGLANG_NCCL_PORT:-29520}")
SGLANG_PARALLEL_SIZE="${SGLANG_PARALLEL_SIZE:-2}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

KV_EVENTS_CONFIG="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${SGLANG_KV_EVENT_PORT}\"}"

print_launch_banner "Launching SGLang Native-gRPC Sidecar with DEP KV Routing (${SGLANG_PARALLEL_SIZE} GPUs)" "$MODEL" "$HTTP_PORT" \
    "SGLang HTTP: http://${SGLANG_HOST}:${SGLANG_HTTP_PORT}" \
    "SGLang gRPC: ${SGLANG_HOST}:${SGLANG_GRPC_PORT}" \
    "KV events:   tcp://*:${SGLANG_KV_EVENT_PORT} + rank"

python3 -m dynamo.frontend --router-mode kv &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_HTTP_PORT" \
    --grpc-port "$SGLANG_GRPC_PORT" \
    --incremental-streaming-output \
    --tp-size "$SGLANG_PARALLEL_SIZE" \
    --dp-size "$SGLANG_PARALLEL_SIZE" \
    --ep-size "$SGLANG_PARALLEL_SIZE" \
    --enable-dp-attention \
    --page-size "$SGLANG_PAGE_SIZE" \
    --kv-events-config "$KV_EVENTS_CONFIG" \
    --dist-init-addr "${SGLANG_HOST}:${SGLANG_DIST_INIT_PORT}" \
    --nccl-port "$SGLANG_NCCL_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_GRPC_PORT}" &

wait_any_exit
