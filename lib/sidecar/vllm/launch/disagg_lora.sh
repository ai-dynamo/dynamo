#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated serving with LoRA through vLLM's native gRPC servers (2 GPUs).
#
# Requires a vLLM build containing vllm-project/vllm#52840, which adds the
# LoadLora, UnloadLora, and ListLoras control RPCs and per-request lora_name.
#
# Also requires NIXL, which the disaggregated KV transfer depends on. The Dynamo
# containers install it; a bare `pip install vllm` does not, and the engines fail
# with "NIXL is not available" only after the model has finished loading.
#
# Both workers load every adapter eagerly. GenerateRequest carries only lora_name
# and vLLM rejects names it has not already loaded, so an adapter must be loaded on
# each worker that may receive it -- unlike the Python worker, which can defer
# loading because it forwards an adapter path in its internal LoRARequest.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"   # build_vllm_gpu_mem_args
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
LORA_NAME="${LORA_NAME:-codelion/Qwen3-0.6B-accuracy-recovery-lora}"

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
            echo "  MODEL                           Model to serve (default: Qwen/Qwen3-0.6B)"
            echo "  DYN_HTTP_PORT                   Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1                Decode sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2                Prefill sidecar system port (default: 8082)"
            echo "  VLLM_DECODE_HTTP_PORT           Decode vLLM HTTP port (default: 8100)"
            echo "  VLLM_DECODE_GRPC_PORT           Decode vLLM gRPC port (default: 50051)"
            echo "  VLLM_PREFILL_HTTP_PORT          Prefill vLLM HTTP port (default: 8110)"
            echo "  VLLM_PREFILL_GRPC_PORT          Prefill vLLM gRPC port (default: 50052)"
            echo "  VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT  Decode NIXL port (default: 5600)"
            echo "  VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT Prefill NIXL port (default: 20097)"
            echo "  VLLM_PREFILL_KV_EVENT_PORT      Prefill KV event port (default: 20081)"
            echo "  VLLM_DECODE_GPU                 Decode GPU index (default: 0)"
            echo "  VLLM_PREFILL_GPU                Prefill GPU index (default: 1)"
            echo "  MAX_MODEL_LEN                   Maximum model length (default: 4096)"
            echo "  MAX_CONCURRENT_SEQS             Maximum concurrent sequences (default: 2)"
            echo "  DEFAULT_KV_CACHE_BYTES          KV cache cap when not profiling (default: 1119388000)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap dynamo_exit_trap EXIT

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
MAX_LORAS="${MAX_LORAS:-4}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
VLLM_DECODE_HTTP_PORT="${VLLM_DECODE_HTTP_PORT:-8100}"
VLLM_DECODE_GRPC_PORT="${VLLM_DECODE_GRPC_PORT:-50051}"
VLLM_PREFILL_HTTP_PORT="${VLLM_PREFILL_HTTP_PORT:-8110}"
VLLM_PREFILL_GRPC_PORT="${VLLM_PREFILL_GRPC_PORT:-50052}"
VLLM_DECODE_GPU="${VLLM_DECODE_GPU:-0}"
VLLM_PREFILL_GPU="${VLLM_PREFILL_GPU:-1}"
VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT="${VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT:-5600}"
VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT="${VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT:-20097}"
VLLM_PREFILL_KV_EVENT_PORT="${VLLM_PREFILL_KV_EVENT_PORT:-20081}"

export AWS_ENDPOINT="${AWS_ENDPOINT:-http://localhost:9000}"
export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-minioadmin}"
export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-minioadmin}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_ALLOW_HTTP="${AWS_ALLOW_HTTP:-true}"

export DYN_LORA_ENABLED=true
export DYN_LORA_PATH="${DYN_LORA_PATH:-/tmp/dynamo_loras_minio}"
mkdir -p "$DYN_LORA_PATH"

# Both engines must accept the roots the sidecars download into, at identical
# absolute paths.
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES="${VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES:-${DYN_LORA_PATH}:${HF_CACHE}}"

# Default KV cache cap from profiling (2x safety over min=560 MiB); ~3.8 GiB
# peak VRAM per engine. The profiler/test framework takes precedence through
# _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching vLLM Native-gRPC Sidecar Disaggregated Serving + LoRA (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Decode:      GPU ${VLLM_DECODE_GPU}, gRPC 127.0.0.1:${VLLM_DECODE_GRPC_PORT}" \
    "Prefill:     GPU ${VLLM_PREFILL_GPU}, gRPC 127.0.0.1:${VLLM_PREFILL_GRPC_PORT}"

echo ""
echo "Load each adapter on BOTH workers before routing traffic to it:"
echo "  for port in ${DYN_SYSTEM_PORT1:-8081} ${DYN_SYSTEM_PORT2:-8082}; do"
echo "    curl -s -X POST http://localhost:\$port/v1/loras \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"lora_name\": \"${LORA_NAME}\", \"source\": {\"uri\": \"hf://${LORA_NAME}\"}}' | jq ."
echo "  done"
echo ""

python -m dynamo.frontend &

# vllm-rs manages the headless Python engines and exposes native gRPC on
# loopback. Arguments after -- are forwarded to each managed engine.
# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_DECODE_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_DECODE_HTTP_PORT" \
    --grpc-port "$VLLM_DECODE_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --enable-lora \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_PREFILL_GPU" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PREFILL_HTTP_PORT" \
    --grpc-port "$VLLM_PREFILL_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --enable-lora \
    --max-loras "$MAX_LORAS" \
    --max-lora-rank "$MAX_LORA_RANK" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_PREFILL_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_DECODE_GRPC_PORT}" \
    --disaggregation-mode decode &

# Register prefill separately so the frontend routes each disaggregated stage.
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_PREFILL_GRPC_PORT}" \
    --component prefill \
    --disaggregation-mode prefill &

wait_any_exit
