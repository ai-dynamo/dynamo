#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Data- and expert-parallel serving through one vLLM native-gRPC sidecar with
# KV-aware routing across its data-parallel ranks (2 GPUs).

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
            echo "Additional options are passed to the managed vLLM engine."
            echo
            echo "Environment overrides:"
            echo "  MODEL                   MoE model to serve (default: silence09/DeepSeek-R1-Small-2layers)"
            echo "  CUDA_VISIBLE_DEVICES    GPU assignment, one per rank (default: 0,1)"
            echo "  DYN_HTTP_PORT           Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT         Dynamo sidecar system port (default: 8081)"
            echo "  VLLM_RS_HTTP_PORT       vLLM HTTP port (default: 8100)"
            echo "  VLLM_GRPC_PORT          vLLM gRPC port (default: 30001)"
            echo "  VLLM_KV_EVENT_PORT      First KV-event port; rank N publishes on this port + N (default: 20080)"
            echo "  VLLM_KV_REPLAY_PORT     First KV-event replay port; rank N replays on this port + N (default: 20180)"
            echo "  VLLM_DATA_PARALLEL_SIZE Number of local data-parallel ranks (default: 2)"
            echo "  VLLM_BLOCK_SIZE         KV-event block size (default: 64)"
            echo "  MAX_MODEL_LEN           Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS     Maximum concurrent sequences per rank (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES  KV cache cap when not profiling (default: 1119388000)"
            echo "  DYNAMO_SIDECAR_BIN      Require this absolute native binary path (no fallback)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

resolve_sidecar vllm SIDECAR_CMD

trap dynamo_exit_trap EXIT

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
VLLM_RS_HTTP_PORT=$(dyn_port DYN_ENGINE_HTTP_PORT 1 "${VLLM_RS_HTTP_PORT:-8100}")
VLLM_GRPC_PORT=$(dyn_port DYN_ENGINE_GRPC_PORT 1 "${VLLM_GRPC_PORT:-30001}")
VLLM_KV_EVENT_PORT=$(dyn_port DYN_VLLM_KV_EVENT_PORT 1 "${VLLM_KV_EVENT_PORT:-20080}")
VLLM_KV_REPLAY_PORT=$(dyn_port DYN_VLLM_KV_REPLAY_PORT 1 "${VLLM_KV_REPLAY_PORT:-20180}")
VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-2}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE:-64}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Default KV cache cap from profiling (2x safety over min=560 MiB).
# The profiler/test framework takes precedence through _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

KV_EVENTS_CONFIG="{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_KV_EVENT_PORT}\",\"replay_endpoint\":\"tcp://*:${VLLM_KV_REPLAY_PORT}\",\"enable_kv_cache_events\":true}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching vLLM Native-gRPC Sidecar with DEP KV Routing (${VLLM_DATA_PARALLEL_SIZE} GPUs)" "$MODEL" "$HTTP_PORT" \
    "vLLM HTTP:  http://127.0.0.1:${VLLM_RS_HTTP_PORT}" \
    "vLLM gRPC:  127.0.0.1:${VLLM_GRPC_PORT}" \
    "KV events:  tcp://*:${VLLM_KV_EVENT_PORT} + rank"

python -m dynamo.frontend --router-mode kv &

# vllm-rs manages the headless Python engines and exposes native gRPC on loopback.
# Arguments after -- are forwarded to the managed engines.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_RS_HTTP_PORT" \
    --grpc-port "$VLLM_GRPC_PORT" \
    --data-parallel-size "$VLLM_DATA_PARALLEL_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --reasoning-parser none \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --tensor-parallel-size 1 \
    --enable-expert-parallel \
    --block-size "$VLLM_BLOCK_SIZE" \
    --kv-events-config "$KV_EVENTS_CONFIG" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    "${SIDECAR_CMD[@]}" \
    --grpc-endpoint "127.0.0.1:${VLLM_GRPC_PORT}" &

wait_any_exit
