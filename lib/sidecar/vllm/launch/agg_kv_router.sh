#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Two aggregated vLLM native-gRPC sidecars behind Dynamo's KV-aware router.
# Requires two GPUs and a vLLM build that exposes KV-event source discovery.

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
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [vLLM engine options...]"
            echo
            echo "Additional options are passed to both managed vLLM engines."
            echo
            echo "Environment overrides:"
            echo "  MODEL                       Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  VLLM_WORKER1_GPU            First GPU assignment (default: 0)"
            echo "  VLLM_WORKER2_GPU            Second GPU assignment (default: 1)"
            echo "  DYN_HTTP_PORT               Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1            First sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2            Second sidecar system port (default: 8082)"
            echo "  VLLM_WORKER1_HTTP_PORT      First vLLM HTTP port (default: 8100)"
            echo "  VLLM_WORKER1_GRPC_PORT      First vLLM gRPC port (default: 50051)"
            echo "  VLLM_WORKER1_KV_EVENT_PORT  First vLLM KV-event port (default: 20080)"
            echo "  VLLM_WORKER2_HTTP_PORT      Second vLLM HTTP port (default: 8110)"
            echo "  VLLM_WORKER2_GRPC_PORT      Second vLLM gRPC port (default: 50052)"
            echo "  VLLM_WORKER2_KV_EVENT_PORT  Second vLLM KV-event port (default: 20081)"
            echo "  VLLM_BLOCK_SIZE             KV-event block size (default: 64)"
            echo "  MAX_MODEL_LEN               Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS         Maximum concurrent sequences per engine (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES      KV cache cap when not profiling (default: 1119388000)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

export PYTHONHASHSEED=0

VLLM_HOST="127.0.0.1"
VLLM_WORKER1_GPU="${VLLM_WORKER1_GPU:-0}"
VLLM_WORKER2_GPU="${VLLM_WORKER2_GPU:-1}"
VLLM_WORKER1_HTTP_PORT="${VLLM_WORKER1_HTTP_PORT:-8100}"
VLLM_WORKER1_GRPC_PORT="${VLLM_WORKER1_GRPC_PORT:-50051}"
VLLM_WORKER1_KV_EVENT_PORT="${VLLM_WORKER1_KV_EVENT_PORT:-20080}"
VLLM_WORKER2_HTTP_PORT="${VLLM_WORKER2_HTTP_PORT:-8110}"
VLLM_WORKER2_GRPC_PORT="${VLLM_WORKER2_GRPC_PORT:-50052}"
VLLM_WORKER2_KV_EVENT_PORT="${VLLM_WORKER2_KV_EVENT_PORT:-20081}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Default KV cache cap from profiling (2x safety over min=560 MiB); ~3.8 GiB peak VRAM.
# The profiler/test framework takes precedence through _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

KV_EVENTS_CONFIG_1="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_WORKER1_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}"
KV_EVENTS_CONFIG_2="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_WORKER2_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching vLLM Native-gRPC Sidecars with KV Routing (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Worker 1: GPU ${VLLM_WORKER1_GPU}, gRPC ${VLLM_HOST}:${VLLM_WORKER1_GRPC_PORT}, KV events tcp://*:${VLLM_WORKER1_KV_EVENT_PORT}" \
    "Worker 2: GPU ${VLLM_WORKER2_GPU}, gRPC ${VLLM_HOST}:${VLLM_WORKER2_GRPC_PORT}, KV events tcp://*:${VLLM_WORKER2_KV_EVENT_PORT}"

python -m dynamo.frontend --router-mode kv &

# vllm-rs manages each headless Python engine and exposes native gRPC on loopback.
# Arguments after -- are forwarded to the managed engine.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_WORKER1_GPU" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_WORKER1_HTTP_PORT" \
    --grpc-port "$VLLM_WORKER1_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-events-config "$KV_EVENTS_CONFIG_1" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_WORKER2_GPU" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_WORKER2_HTTP_PORT" \
    --grpc-port "$VLLM_WORKER2_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-events-config "$KV_EVENTS_CONFIG_2" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

OTEL_SERVICE_NAME=dynamo-worker-1 \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --vllm-endpoint "${VLLM_HOST}:${VLLM_WORKER1_GRPC_PORT}" &

OTEL_SERVICE_NAME=dynamo-worker-2 \
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --vllm-endpoint "${VLLM_HOST}:${VLLM_WORKER2_GRPC_PORT}" &

wait_any_exit
