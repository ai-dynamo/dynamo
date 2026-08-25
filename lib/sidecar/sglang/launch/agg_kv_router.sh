#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two aggregated SGLang native-gRPC sidecars behind Dynamo's KV-aware router.
# Requires two GPUs and an SGLang build that exposes KV-event discovery over GetServerInfo.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"

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
            echo "Additional options are passed to both SGLang engines."
            echo
            echo "Environment overrides:"
            echo "  MODEL                         Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  SGLANG_PYTHON                 Python with SGLang installed (default: python3)"
            echo "  SGLANG_WORKER1_GPU            First GPU assignment (default: 0)"
            echo "  SGLANG_WORKER2_GPU            Second GPU assignment (default: 1)"
            echo "  DYN_HTTP_PORT                 Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1              First sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2              Second sidecar system port (default: 8082)"
            echo "  SGLANG_WORKER1_HTTP_PORT      First SGLang HTTP port (default: 30000)"
            echo "  SGLANG_WORKER1_GRPC_PORT      First SGLang gRPC port (default: 30001)"
            echo "  SGLANG_WORKER1_KV_EVENT_PORT  First SGLang KV-event port (default: 5557)"
            echo "  SGLANG_WORKER2_HTTP_PORT      Second SGLang HTTP port (default: 30010)"
            echo "  SGLANG_WORKER2_GRPC_PORT      Second SGLang gRPC port (default: 30011)"
            echo "  SGLANG_WORKER2_KV_EVENT_PORT  Second SGLang KV-event port (default: 5567)"
            echo "  SGLANG_PAGE_SIZE              KV-event block size (default: 16)"
            echo "  MAX_MODEL_LEN                 Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS           Maximum concurrent sequences per engine (default: 2)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

SGLANG_PYTHON="${SGLANG_PYTHON:-python3}"
SGLANG_HOST="127.0.0.1"
SGLANG_WORKER1_GPU="${SGLANG_WORKER1_GPU:-0}"
SGLANG_WORKER2_GPU="${SGLANG_WORKER2_GPU:-1}"
SGLANG_WORKER1_HTTP_PORT="${SGLANG_WORKER1_HTTP_PORT:-30000}"
SGLANG_WORKER1_GRPC_PORT="${SGLANG_WORKER1_GRPC_PORT:-30001}"
SGLANG_WORKER1_KV_EVENT_PORT="${SGLANG_WORKER1_KV_EVENT_PORT:-5557}"
SGLANG_WORKER2_HTTP_PORT="${SGLANG_WORKER2_HTTP_PORT:-30010}"
SGLANG_WORKER2_GRPC_PORT="${SGLANG_WORKER2_GRPC_PORT:-30011}"
SGLANG_WORKER2_KV_EVENT_PORT="${SGLANG_WORKER2_KV_EVENT_PORT:-5567}"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

KV_EVENTS_CONFIG_1="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_WORKER1_KV_EVENT_PORT}\",\"topic\":\"\"}"
KV_EVENTS_CONFIG_2="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_WORKER2_KV_EVENT_PORT}\",\"topic\":\"\"}"

print_launch_banner "Launching SGLang Native-gRPC Sidecars with KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Worker 1: GPU ${SGLANG_WORKER1_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_WORKER1_GRPC_PORT}, KV events tcp://*:${SGLANG_WORKER1_KV_EVENT_PORT}" \
    "Worker 2: GPU ${SGLANG_WORKER2_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_WORKER2_GRPC_PORT}, KV events tcp://*:${SGLANG_WORKER2_KV_EVENT_PORT}"

python3 -m dynamo.frontend --router-mode kv &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_WORKER1_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_WORKER1_HTTP_PORT" \
    --grpc-port "$SGLANG_WORKER1_GRPC_PORT" \
    --kv-events-config "$KV_EVENTS_CONFIG_1" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_WORKER2_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_WORKER2_HTTP_PORT" \
    --grpc-port "$SGLANG_WORKER2_GRPC_PORT" \
    --kv-events-config "$KV_EVENTS_CONFIG_2" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-1 \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-sglang-sidecar \
    --sglang-endpoint "${SGLANG_HOST}:${SGLANG_WORKER1_GRPC_PORT}" &

OTEL_SERVICE_NAME=dynamo-worker-2 \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-sglang-sidecar \
    --sglang-endpoint "${SGLANG_HOST}:${SGLANG_WORKER2_GRPC_PORT}" &

wait_any_exit
