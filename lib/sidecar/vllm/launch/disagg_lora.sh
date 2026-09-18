#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Prefill/decode LoRA serving with NIXL (2 GPUs).
# Requires vLLM #52840 and #54814; load adapters on both workers.

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
            echo "  LORA_NAME                       Example adapter (default: codelion/Qwen3-0.6B-accuracy-recovery-lora)"
            echo "  MAX_LORAS                       GPU-resident adapter capacity (default: 4)"
            echo "  MAX_LORA_RANK                    Maximum adapter rank (default: 64)"
            echo "  DYN_LORA_PATH                   S3 download root (default: /tmp/dynamo_loras_minio)"
            echo "  DYN_HTTP_PORT                   Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1                Decode sidecar system port (default: 8081)"
            echo "  DYN_SYSTEM_PORT2                Prefill sidecar system port (default: 8082)"
            echo "  VLLM_DECODE_HTTP_PORT           Decode vLLM HTTP port (default: 8100)"
            echo "  VLLM_DECODE_GRPC_PORT           Decode vLLM gRPC port (default: 50051)"
            echo "  VLLM_PREFILL_HTTP_PORT          Prefill vLLM HTTP port (default: 8110)"
            echo "  VLLM_PREFILL_GRPC_PORT          Prefill vLLM gRPC port (default: 50052)"
            echo "  VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT  Decode NIXL port (default: 5600)"
            echo "  VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT Prefill NIXL port (default: 20097)"
            echo "  VLLM_PREFILL_KV_EVENT_ENDPOINT  KV event endpoint (default: private local IPC socket)"
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

KV_EVENT_DIR=""
if [[ -z "${VLLM_PREFILL_KV_EVENT_ENDPOINT:-}" ]]; then
    KV_EVENT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-vllm-kv.XXXXXX")"
    VLLM_PREFILL_KV_EVENT_ENDPOINT="ipc://${KV_EVENT_DIR}/events"
fi

lora_exit_trap() {
    local _rc=$?
    if [[ -n "$KV_EVENT_DIR" ]]; then
        rm -rf -- "$KV_EVENT_DIR"
    fi
    dynamo_reap_and_exit "$_rc"
}
trap lora_exit_trap EXIT

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

export DYN_LORA_ENABLED=true
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=true
export DYN_LORA_PATH="${DYN_LORA_PATH:-/tmp/dynamo_loras_minio}"
mkdir -p "$DYN_LORA_PATH"

# Both engines must see the adapter cache at the same absolute location.
if [[ -z "${VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES:-}" ]]; then
    HF_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/hub}"
    mkdir -p "$HF_CACHE"
    export VLLM_RUNTIME_LORA_ALLOWED_PATH_PREFIXES="${DYN_LORA_PATH}:${HF_CACHE}"
fi

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
echo "  # Adapter inference"
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"${LORA_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"What is deep learning?\"}], \"max_tokens\": 300, \"temperature\": 0.0}' | jq ."
echo ""

python -m dynamo.frontend &

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
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"${VLLM_PREFILL_KV_EVENT_ENDPOINT}\",\"enable_kv_cache_events\":true}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_DECODE_GRPC_PORT}" \
    --disaggregation-mode decode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_PREFILL_GRPC_PORT}" \
    --component prefill \
    --disaggregation-mode prefill &

wait_any_exit
