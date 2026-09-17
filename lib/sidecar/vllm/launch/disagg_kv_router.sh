#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving through four vLLM native-gRPC sidecars with KV-aware routing.
# Requires four GPUs and a vLLM build that exposes KV-event source discovery.
# See ../README.md for the validated vLLM/vllm-rs source state.

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
            echo "Additional options are passed to all four managed vLLM engines."
            echo
            echo "Environment overrides:"
            echo "  MODEL                              Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  DYN_HTTP_PORT                      Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1                   First decode sidecar port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2                   Second decode sidecar port (default: 8082)"
            echo "  DYN_SYSTEM_PORT3                   First prefill sidecar port (default: 8083)"
            echo "  DYN_SYSTEM_PORT4                   Second prefill sidecar port (default: 8084)"
            echo "  VLLM_DECODE1_GPU                   First decode GPU assignment (default: 0)"
            echo "  VLLM_DECODE2_GPU                   Second decode GPU assignment (default: 1)"
            echo "  VLLM_PREFILL1_GPU                  First prefill GPU assignment (default: 2)"
            echo "  VLLM_PREFILL2_GPU                  Second prefill GPU assignment (default: 3)"
            echo "  VLLM_DECODE1_HTTP_PORT             First decode HTTP port (default: 8100)"
            echo "  VLLM_DECODE1_GRPC_PORT             First decode gRPC port (default: 50051)"
            echo "  VLLM_DECODE1_NIXL_SIDE_CHANNEL_PORT First decode NIXL port (default: 20096)"
            echo "  VLLM_DECODE2_HTTP_PORT             Second decode HTTP port (default: 8110)"
            echo "  VLLM_DECODE2_GRPC_PORT             Second decode gRPC port (default: 50052)"
            echo "  VLLM_DECODE2_NIXL_SIDE_CHANNEL_PORT Second decode NIXL port (default: 20097)"
            echo "  VLLM_PREFILL1_HTTP_PORT            First prefill HTTP port (default: 8120)"
            echo "  VLLM_PREFILL1_GRPC_PORT            First prefill gRPC port (default: 50053)"
            echo "  VLLM_PREFILL1_NIXL_SIDE_CHANNEL_PORT First prefill NIXL port (default: 20098)"
            echo "  VLLM_PREFILL1_KV_EVENT_PORT        First prefill KV-event port (default: 20082)"
            echo "  VLLM_PREFILL2_HTTP_PORT            Second prefill HTTP port (default: 8130)"
            echo "  VLLM_PREFILL2_GRPC_PORT            Second prefill gRPC port (default: 50054)"
            echo "  VLLM_PREFILL2_NIXL_SIDE_CHANNEL_PORT Second prefill NIXL port (default: 20099)"
            echo "  VLLM_PREFILL2_KV_EVENT_PORT        Second prefill KV-event port (default: 20083)"
            echo "  VLLM_BLOCK_SIZE                    KV-event block size (default: 64)"
            echo "  MAX_MODEL_LEN                      Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS                Maximum concurrent sequences per engine (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES             KV cache cap when not profiling (default: 1119388000)"
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
VLLM_DECODE1_GPU="${VLLM_DECODE1_GPU:-0}"
VLLM_DECODE2_GPU="${VLLM_DECODE2_GPU:-1}"
VLLM_PREFILL1_GPU="${VLLM_PREFILL1_GPU:-2}"
VLLM_PREFILL2_GPU="${VLLM_PREFILL2_GPU:-3}"
VLLM_DECODE1_HTTP_PORT="${VLLM_DECODE1_HTTP_PORT:-8100}"
VLLM_DECODE1_GRPC_PORT="${VLLM_DECODE1_GRPC_PORT:-50051}"
VLLM_DECODE1_NIXL_SIDE_CHANNEL_PORT="${VLLM_DECODE1_NIXL_SIDE_CHANNEL_PORT:-20096}"
VLLM_DECODE2_HTTP_PORT="${VLLM_DECODE2_HTTP_PORT:-8110}"
VLLM_DECODE2_GRPC_PORT="${VLLM_DECODE2_GRPC_PORT:-50052}"
VLLM_DECODE2_NIXL_SIDE_CHANNEL_PORT="${VLLM_DECODE2_NIXL_SIDE_CHANNEL_PORT:-20097}"
VLLM_PREFILL1_HTTP_PORT="${VLLM_PREFILL1_HTTP_PORT:-8120}"
VLLM_PREFILL1_GRPC_PORT="${VLLM_PREFILL1_GRPC_PORT:-50053}"
VLLM_PREFILL1_NIXL_SIDE_CHANNEL_PORT="${VLLM_PREFILL1_NIXL_SIDE_CHANNEL_PORT:-20098}"
VLLM_PREFILL1_KV_EVENT_PORT="${VLLM_PREFILL1_KV_EVENT_PORT:-20082}"
VLLM_PREFILL2_HTTP_PORT="${VLLM_PREFILL2_HTTP_PORT:-8130}"
VLLM_PREFILL2_GRPC_PORT="${VLLM_PREFILL2_GRPC_PORT:-50054}"
VLLM_PREFILL2_NIXL_SIDE_CHANNEL_PORT="${VLLM_PREFILL2_NIXL_SIDE_CHANNEL_PORT:-20099}"
VLLM_PREFILL2_KV_EVENT_PORT="${VLLM_PREFILL2_KV_EVENT_PORT:-20083}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-64}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"

# Default KV cache cap from profiling (2x safety over min=560 MiB); ~3.8 GiB
# peak VRAM per engine. The profiler/test framework takes precedence through
# _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

KV_TRANSFER_CONFIG='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
KV_EVENTS_CONFIG_1="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_PREFILL1_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}"
KV_EVENTS_CONFIG_2="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_PREFILL2_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching vLLM Native-gRPC Sidecars with Disaggregated KV Routing (4 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Decode 1:  GPU ${VLLM_DECODE1_GPU}, gRPC ${VLLM_HOST}:${VLLM_DECODE1_GRPC_PORT}" \
    "Decode 2:  GPU ${VLLM_DECODE2_GPU}, gRPC ${VLLM_HOST}:${VLLM_DECODE2_GRPC_PORT}" \
    "Prefill 1: GPU ${VLLM_PREFILL1_GPU}, gRPC ${VLLM_HOST}:${VLLM_PREFILL1_GRPC_PORT}, KV events tcp://*:${VLLM_PREFILL1_KV_EVENT_PORT}" \
    "Prefill 2: GPU ${VLLM_PREFILL2_GPU}, gRPC ${VLLM_HOST}:${VLLM_PREFILL2_GRPC_PORT}, KV events tcp://*:${VLLM_PREFILL2_KV_EVENT_PORT}"

python -m dynamo.frontend --router-mode kv &

# vllm-rs manages the headless Python engines and exposes native gRPC on loopback.
# Arguments after -- are forwarded to each managed engine.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_DECODE1_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_DECODE1_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_DECODE1_HTTP_PORT" \
    --grpc-port "$VLLM_DECODE1_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_DECODE2_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_DECODE2_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_DECODE2_HTTP_PORT" \
    --grpc-port "$VLLM_DECODE2_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_PREFILL1_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_PREFILL1_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PREFILL1_HTTP_PORT" \
    --grpc-port "$VLLM_PREFILL1_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --kv-events-config "$KV_EVENTS_CONFIG_1" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_PREFILL2_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_PREFILL2_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PREFILL2_HTTP_PORT" \
    --grpc-port "$VLLM_PREFILL2_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    --kv-events-config "$KV_EVENTS_CONFIG_2" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "${VLLM_HOST}:${VLLM_DECODE1_GRPC_PORT}" \
    --disaggregation-mode decode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "${VLLM_HOST}:${VLLM_DECODE2_GRPC_PORT}" \
    --disaggregation-mode decode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT3:-8083}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "${VLLM_HOST}:${VLLM_PREFILL1_GRPC_PORT}" \
    --component prefill \
    --disaggregation-mode prefill &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT4:-8084}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "${VLLM_HOST}:${VLLM_PREFILL2_GRPC_PORT}" \
    --component prefill \
    --disaggregation-mode prefill &

wait_any_exit
