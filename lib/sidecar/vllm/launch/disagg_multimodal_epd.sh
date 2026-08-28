#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encoder + prefill + decode serving through native vLLM gRPC (3 GPUs).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091 # Shared helper in this directory.
source "$SCRIPT_DIR/_epd_common.sh"

usage() {
    echo "Usage: $0 [--model <name>] [vLLM engine options...]"
    echo
    echo "Environment overrides:"
    epd_print_common_help
    echo "  DYN_SYSTEM_PORT1/2/3         Encode/decode/prefill system ports (defaults: 8081/8082/8083)"
    echo "  VLLM_PREFILL_HTTP_PORT       Prefill vLLM HTTP port (default: 8110)"
    echo "  VLLM_PREFILL_GRPC_PORT       Prefill vLLM gRPC port (default: 50052)"
    echo "  VLLM_DECODE_HTTP_PORT        Decode vLLM HTTP port (default: 8120)"
    echo "  VLLM_DECODE_GRPC_PORT        Decode vLLM gRPC port (default: 50053)"
    echo "  VLLM_PREFILL_GPU             Prefill GPU index (default: 1)"
    echo "  VLLM_DECODE_GPU              Decode GPU index (default: 2)"
    echo "  VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT  Prefill NIXL port (default: 5601)"
    echo "  VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT   Decode NIXL port (default: 5602)"
    echo "  VLLM_PREFILL_KV_EVENT_PORT   Prefill KV event port (default: 20081)"
}

epd_parse_args usage "$@"
epd_init

VLLM_PREFILL_HTTP_PORT="${VLLM_PREFILL_HTTP_PORT:-8110}"
VLLM_PREFILL_GRPC_PORT="${VLLM_PREFILL_GRPC_PORT:-50052}"
VLLM_DECODE_HTTP_PORT="${VLLM_DECODE_HTTP_PORT:-8120}"
VLLM_DECODE_GRPC_PORT="${VLLM_DECODE_GRPC_PORT:-50053}"
VLLM_PREFILL_GPU="${VLLM_PREFILL_GPU:-1}"
VLLM_DECODE_GPU="${VLLM_DECODE_GPU:-2}"
VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT="${VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT:-5601}"
VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT="${VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT:-5602}"
VLLM_PREFILL_KV_EVENT_PORT="${VLLM_PREFILL_KV_EVENT_PORT:-20081}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching vLLM Sidecar E+P+D Serving (3 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Encoder: GPU ${VLLM_ENCODER_GPU}, HTTP ${VLLM_ENCODER_HTTP_PORT}, gRPC ${VLLM_ENCODER_GRPC_PORT}" \
    "Prefill: GPU ${VLLM_PREFILL_GPU}, HTTP ${VLLM_PREFILL_HTTP_PORT}, gRPC ${VLLM_PREFILL_GRPC_PORT}, NIXL ${VLLM_PREFILL_NIXL_SIDE_CHANNEL_PORT}, events ${VLLM_PREFILL_KV_EVENT_PORT}" \
    "Decode:  GPU ${VLLM_DECODE_GPU}, HTTP ${VLLM_DECODE_HTTP_PORT}, gRPC ${VLLM_DECODE_GRPC_PORT}, NIXL ${VLLM_DECODE_NIXL_SIDE_CHANNEL_PORT}" \
    "EC path: ${EC_SHARED_STORAGE_PATH}"

epd_launch_frontend_and_encoder

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
