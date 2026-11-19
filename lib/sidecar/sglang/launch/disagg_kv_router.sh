#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving through four SGLang native-gRPC sidecars with KV-aware routing.
# Requires four GPUs and an SGLang build with native gRPC, KV events, and disaggregation support.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$SCRIPT_DIR/common.sh"

# The sidecar needs SGLang's Rust server (sglang.api.v1).
export SGLANG_RUST_SERVER=1

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
            echo "Additional options are passed to all four SGLang engines."
            echo
            echo "Environment overrides:"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            echo "  MODEL                              Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  SGLANG_PYTHON                      Python with SGLang installed (default: python3)"
            echo "  DYN_HTTP_PORT                      Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1                   First prefill sidecar port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2                   Second prefill sidecar port (default: 8082)"
            echo "  DYN_SYSTEM_PORT3                   First decode sidecar port (default: 8083)"
            echo "  DYN_SYSTEM_PORT4                   Second decode sidecar port (default: 8084)"
            echo "  SGLANG_PREFILL1_GPU                First prefill GPU assignment (default: 0)"
            echo "  SGLANG_PREFILL2_GPU                Second prefill GPU assignment (default: 1)"
            echo "  SGLANG_DECODE1_GPU                 First decode GPU assignment (default: 2)"
            echo "  SGLANG_DECODE2_GPU                 Second decode GPU assignment (default: 3)"
            echo "  SGLANG_PREFILL1_HTTP_PORT          First prefill HTTP port (default: 30000)"
            echo "  SGLANG_PREFILL1_GRPC_PORT          First prefill gRPC port (default: 30010)"
            echo "  SGLANG_PREFILL1_KV_EVENT_PORT      First prefill KV-event port (default: 5557)"
            echo "  SGLANG_PREFILL2_HTTP_PORT          Second prefill HTTP port (default: 30020)"
            echo "  SGLANG_PREFILL2_GRPC_PORT          Second prefill gRPC port (default: 30030)"
            echo "  SGLANG_PREFILL2_KV_EVENT_PORT      Second prefill KV-event port (default: 5567)"
            echo "  SGLANG_DECODE1_HTTP_PORT           First decode HTTP port (default: 30040)"
            echo "  SGLANG_DECODE1_GRPC_PORT           First decode gRPC port (default: 30050)"
            echo "  SGLANG_DECODE1_KV_EVENT_PORT       First decode KV-event port (default: 5577)"
            echo "  SGLANG_DECODE2_HTTP_PORT           Second decode HTTP port (default: 30060)"
            echo "  SGLANG_DECODE2_GRPC_PORT           Second decode gRPC port (default: 30070)"
            echo "  SGLANG_DECODE2_KV_EVENT_PORT       Second decode KV-event port (default: 5587)"
            echo "  SGLANG_BOOTSTRAP_HOST              Advertised prefill bootstrap host (default: 127.0.0.1)"
            echo "  SGLANG_PAGE_SIZE                   KV-event block size (default: 64)"
            echo "  MAX_MODEL_LEN                      Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS                Maximum concurrent sequences per engine (default: 2)"
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
SGLANG_BOOTSTRAP_HOST="${SGLANG_BOOTSTRAP_HOST:-$SGLANG_HOST}"
SGLANG_PREFILL1_GPU="${SGLANG_PREFILL1_GPU:-0}"
SGLANG_PREFILL2_GPU="${SGLANG_PREFILL2_GPU:-1}"
SGLANG_DECODE1_GPU="${SGLANG_DECODE1_GPU:-2}"
SGLANG_DECODE2_GPU="${SGLANG_DECODE2_GPU:-3}"
# Managed ports are indexed like DYN_SYSTEM_PORT{N}: workers 1-2 are prefill, 3-4 are decode.
dyn_worker_ports SGLANG_PREFILL1 1 HTTP=30000 GRPC=30010 KV_EVENT=5557
dyn_worker_ports SGLANG_PREFILL2 2 HTTP=30020 GRPC=30030 KV_EVENT=5567
dyn_worker_ports SGLANG_DECODE1 3 HTTP=30040 GRPC=30050 KV_EVENT=5577
dyn_worker_ports SGLANG_DECODE2 4 HTTP=30060 GRPC=30070 KV_EVENT=5587
SGLANG_LISTENERS=$(sglang_listener_count "${EXTRA_ARGS[@]}")
sglang_check_listener_ports "$SGLANG_PREFILL1_HTTP_PORT" "$SGLANG_PREFILL1_GRPC_PORT" "$SGLANG_LISTENERS"
sglang_check_listener_ports "$SGLANG_PREFILL2_HTTP_PORT" "$SGLANG_PREFILL2_GRPC_PORT" "$SGLANG_LISTENERS"
sglang_check_listener_ports "$SGLANG_DECODE1_HTTP_PORT" "$SGLANG_DECODE1_GRPC_PORT" "$SGLANG_LISTENERS"
sglang_check_listener_ports "$SGLANG_DECODE2_HTTP_PORT" "$SGLANG_DECODE2_GRPC_PORT" "$SGLANG_LISTENERS"
SGLANG_PAGE_SIZE="${SGLANG_PAGE_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)

KV_EVENTS_CONFIG_PREFILL1="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_PREFILL1_KV_EVENT_PORT}\",\"topic\":\"\"}"
KV_EVENTS_CONFIG_PREFILL2="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_PREFILL2_KV_EVENT_PORT}\",\"topic\":\"\"}"
KV_EVENTS_CONFIG_DECODE1="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_DECODE1_KV_EVENT_PORT}\",\"topic\":\"\"}"
KV_EVENTS_CONFIG_DECODE2="{\"publisher\":\"zmq\",\"endpoint\":\"tcp://*:${SGLANG_DECODE2_KV_EVENT_PORT}\",\"topic\":\"\"}"

print_launch_banner "Launching SGLang Native-gRPC Sidecars with Disaggregated KV Routing (4 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Prefill 1: GPU ${SGLANG_PREFILL1_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_PREFILL1_GRPC_PORT}, KV events tcp://*:${SGLANG_PREFILL1_KV_EVENT_PORT}, bootstrap ${SGLANG_BOOTSTRAP_HOST}:${SGLANG_PREFILL1_HTTP_PORT}" \
    "Prefill 2: GPU ${SGLANG_PREFILL2_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_PREFILL2_GRPC_PORT}, KV events tcp://*:${SGLANG_PREFILL2_KV_EVENT_PORT}, bootstrap ${SGLANG_BOOTSTRAP_HOST}:${SGLANG_PREFILL2_HTTP_PORT}" \
    "Decode 1:  GPU ${SGLANG_DECODE1_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_DECODE1_GRPC_PORT}, KV events tcp://*:${SGLANG_DECODE1_KV_EVENT_PORT}" \
    "Decode 2:  GPU ${SGLANG_DECODE2_GPU}, gRPC ${SGLANG_HOST}:${SGLANG_DECODE2_GRPC_PORT}, KV events tcp://*:${SGLANG_DECODE2_KV_EVENT_PORT}"

python3 -m dynamo.frontend --router-mode kv &

# Each prefill engine serves its own bootstrap on its HTTP port.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_PREFILL1_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PREFILL1_HTTP_PORT" \
    --grpc-port "$SGLANG_PREFILL1_GRPC_PORT" \
    --incremental-streaming-output \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --kv-events-config "$KV_EVENTS_CONFIG_PREFILL1" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_PREFILL2_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_PREFILL2_HTTP_PORT" \
    --grpc-port "$SGLANG_PREFILL2_GRPC_PORT" \
    --incremental-streaming-output \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend nixl \
    --kv-events-config "$KV_EVENTS_CONFIG_PREFILL2" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_DECODE1_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_DECODE1_HTTP_PORT" \
    --grpc-port "$SGLANG_DECODE1_GRPC_PORT" \
    --incremental-streaming-output \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --kv-events-config "$KV_EVENTS_CONFIG_DECODE1" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$SGLANG_DECODE2_GPU" \
"$SGLANG_PYTHON" -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$SGLANG_HOST" \
    --port "$SGLANG_DECODE2_HTTP_PORT" \
    --grpc-port "$SGLANG_DECODE2_GRPC_PORT" \
    --incremental-streaming-output \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend nixl \
    --kv-events-config "$KV_EVENTS_CONFIG_DECODE2" \
    --page-size "$SGLANG_PAGE_SIZE" \
    --context-length "$MAX_MODEL_LEN" \
    --max-running-requests "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_PREFILL1_GRPC_PORT}" \
    --bootstrap-host "$SGLANG_BOOTSTRAP_HOST" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_PREFILL2_GRPC_PORT}" \
    --bootstrap-host "$SGLANG_BOOTSTRAP_HOST" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT3:-8083}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_DECODE1_GRPC_PORT}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT4:-8084}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "${SGLANG_HOST}:${SGLANG_DECODE2_GRPC_PORT}" &

wait_any_exit
