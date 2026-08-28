#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Encoder + aggregated prefill/decode serving through native vLLM gRPC (2 GPUs).

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck disable=SC1091 # Shared helper in this directory.
source "$SCRIPT_DIR/_epd_common.sh"

usage() {
    echo "Usage: $0 [--model <name>] [vLLM engine options...]"
    echo
    echo "Environment overrides:"
    epd_print_common_help
    echo "  DYN_SYSTEM_PORT1            Encoder sidecar system port (default: 8081)"
    echo "  DYN_SYSTEM_PORT2            PD sidecar system port (default: 8082)"
    echo "  VLLM_PD_HTTP_PORT           PD vLLM HTTP port (default: 8110)"
    echo "  VLLM_PD_GRPC_PORT           PD vLLM gRPC port (default: 50052)"
    echo "  VLLM_PD_GPU                 PD GPU index (default: 1)"
}

epd_parse_args usage "$@"
epd_init

VLLM_PD_HTTP_PORT="${VLLM_PD_HTTP_PORT:-8110}"
VLLM_PD_GRPC_PORT="${VLLM_PD_GRPC_PORT:-50052}"
VLLM_PD_GPU="${VLLM_PD_GPU:-1}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching vLLM Sidecar E+PD Serving (2 GPUs)" "$MODEL" "$HTTP_PORT" \
    "Encoder: GPU ${VLLM_ENCODER_GPU}, HTTP ${VLLM_ENCODER_HTTP_PORT}, gRPC ${VLLM_ENCODER_GRPC_PORT}" \
    "PD:      GPU ${VLLM_PD_GPU}, HTTP ${VLLM_PD_HTTP_PORT}, gRPC ${VLLM_PD_GRPC_PORT}" \
    "EC path: ${EC_SHARED_STORAGE_PATH}"

epd_launch_frontend_and_encoder

# shellcheck disable=SC2086 # GPU_MEM_ARGS intentionally expands into multiple flags.
CUDA_VISIBLE_DEVICES="$VLLM_PD_GPU" \
vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_PD_HTTP_PORT" \
    --grpc-port "$VLLM_PD_GRPC_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- \
    --enforce-eager \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    --ec-transfer-config "$CONSUMER_EC_CONFIG" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_ENCODER_GRPC_PORT}" \
    --disaggregation-mode encode &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
    dynamo-vllm-sidecar \
    --grpc-endpoint "127.0.0.1:${VLLM_PD_GRPC_PORT}" \
    --route-to-encoder &

wait_any_exit
