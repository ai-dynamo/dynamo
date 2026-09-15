#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encoder + prefill + decode serving through native vLLM gRPC (3 GPUs).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}"
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh" # build_vllm_gpu_mem_args
# shellcheck disable=SC1091 # Resolved relative to this script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [vLLM engine options...]"
            echo
            echo "Environment overrides:"
            echo "  EC_SHARED_STORAGE_PATH       Existing shared EC directory; otherwise a temporary directory is created"
            echo "  DYN_HTTP_PORT                Dynamo frontend port (default: 8000)"
            echo "  DYN_SYSTEM_PORT1/2/3         Encode/decode/prefill system ports (defaults: 8081/8082/8083)"
            echo "  VLLM_ENCODER_HTTP/GRPC_PORT  Encoder ports (defaults: 8100/50051)"
            echo "  VLLM_PREFILL_HTTP/GRPC_PORT  Prefill ports (defaults: 8110/50052)"
            echo "  VLLM_DECODE_HTTP/GRPC_PORT   Decode ports (defaults: 8120/50053)"
            echo "  VLLM_ENCODER/PREFILL/DECODE_GPU  GPU indices (defaults: 0/1/2)"
            echo "  VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT  Prefill NIXL port (default: 5601)"
            echo "  VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT   Decode NIXL port (default: 5602)"
            echo "  VLLM_PREFILL_KV_EVENT_PORT   Prefill KV event port (default: 20081)"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

EC_STORAGE_OWNED=false
if [[ -z "${EC_SHARED_STORAGE_PATH:-}" ]]; then
    EC_SHARED_STORAGE_PATH="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-vllm-ec.XXXXXX")"
    EC_STORAGE_OWNED=true
else
    mkdir -p "$EC_SHARED_STORAGE_PATH"
fi

epd_exit_trap() {
    local _rc=$?
    if [[ "$EC_STORAGE_OWNED" == true ]]; then
        rm -rf -- "$EC_SHARED_STORAGE_PATH"
    fi
    dynamo_reap_and_exit "$_rc"
}
trap epd_exit_trap EXIT

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
VLLM_ENCODER_HTTP_PORT="${VLLM_ENCODER_HTTP_PORT:-8100}"
VLLM_ENCODER_GRPC_PORT="${VLLM_ENCODER_GRPC_PORT:-50051}"
VLLM_PREFILL_HTTP_PORT="${VLLM_PREFILL_HTTP_PORT:-8110}"
VLLM_PREFILL_GRPC_PORT="${VLLM_PREFILL_GRPC_PORT:-50052}"
VLLM_DECODE_HTTP_PORT="${VLLM_DECODE_HTTP_PORT:-8120}"
VLLM_DECODE_GRPC_PORT="${VLLM_DECODE_GRPC_PORT:-50053}"
VLLM_ENCODER_GPU="${VLLM_ENCODER_GPU:-0}"
VLLM_PREFILL_GPU="${VLLM_PREFILL_GPU:-1}"
VLLM_DECODE_GPU="${VLLM_DECODE_GPU:-2}"
VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT="${VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT:-5601}"
VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT="${VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT:-5602}"
VLLM_PREFILL_KV_EVENT_PORT="${VLLM_PREFILL_KV_EVENT_PORT:-20081}"
ENCODER_GPU_MEMORY_UTILIZATION="${ENCODER_GPU_MEMORY_UTILIZATION:-0.1}"

DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

ENCODER_EC_CONFIG="{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"${EC_SHARED_STORAGE_PATH}\"}}"
CONSUMER_EC_CONFIG="{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"${EC_SHARED_STORAGE_PATH}\"}}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching vLLM Sidecar E+P+D Serving (3 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Encoder: GPU ${VLLM_ENCODER_GPU}, HTTP ${VLLM_ENCODER_HTTP_PORT}, gRPC ${VLLM_ENCODER_GRPC_PORT}" \
    "Prefill: GPU ${VLLM_PREFILL_GPU}, HTTP ${VLLM_PREFILL_HTTP_PORT}, gRPC ${VLLM_PREFILL_GRPC_PORT}, NIXL ${VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT}, events ${VLLM_PREFILL_KV_EVENT_PORT}" \
    "Decode:  GPU ${VLLM_DECODE_GPU}, HTTP ${VLLM_DECODE_HTTP_PORT}, gRPC ${VLLM_DECODE_GRPC_PORT}, NIXL ${VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT}" \
    "EC path: ${EC_SHARED_STORAGE_PATH}"

python -m dynamo.frontend &

# Encoder-only vLLM has no KV cache, so GPU_MEM_ARGS is inapplicable.
# ENCODER_GPU_MEMORY_UTILIZATION sets --gpu-memory-utilization independently.
CUDA_VISIBLE_DEVICES="$VLLM_ENCODER_GPU" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_ENCODER_HTTP_PORT" \
    --grpc-port "$VLLM_ENCODER_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --mm-encoder-only \
    --enforce-eager \
    --no-enable-prefix-caching \
    --max-num-batched-tokens 114688 \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --gpu-memory-utilization "$ENCODER_GPU_MEMORY_UTILIZATION" \
    --ec-transfer-config "$ENCODER_EC_CONFIG" \
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
    --ec-transfer-config "$CONSUMER_EC_CONFIG" \
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_PREFILL_KV_EVENT_PORT}\",\"enable_kv_cache_events\":true}" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

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
    --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_ENCODER_GRPC_PORT}" \
    --disaggregation-mode encode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_DECODE_GRPC_PORT}" \
    --disaggregation-mode decode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT3:-8083}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_PREFILL_GRPC_PORT}" \
    --disaggregation-mode prefill \
    --route-to-encoder &

wait_any_exit
