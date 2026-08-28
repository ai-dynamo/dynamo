#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared setup for the native vLLM gRPC encoder-disaggregation examples.

export DYNAMO_HOME="${DYNAMO_HOME:-$(readlink -f "$SCRIPT_DIR/../../../..")}" # Repository root
# shellcheck disable=SC1091 # Resolved relative to the launch script at runtime.
source "$DYNAMO_HOME/examples/common/gpu_utils.sh" # build_vllm_gpu_mem_args
# shellcheck disable=SC1091 # Resolved relative to the launch script at runtime.
source "$DYNAMO_HOME/examples/common/launch_utils.sh" # print_launch_banner, wait_any_exit

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
EXTRA_ARGS=()

epd_parse_args() {
    local usage_function=$1
    shift

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
                "$usage_function"
                exit 0
                ;;
            *)
                EXTRA_ARGS+=("$1")
                shift
                ;;
        esac
    done
}

epd_print_common_help() {
    echo "  EC_SHARED_STORAGE_PATH       Existing shared EC directory; otherwise a temporary directory is created"
    echo "  DYN_HTTP_PORT                Dynamo frontend port (default: 8000)"
    echo "  VLLM_ENCODER_HTTP_PORT       Encoder vLLM HTTP port (default: 8100)"
    echo "  VLLM_ENCODER_GRPC_PORT       Encoder vLLM gRPC port (default: 50051)"
    echo "  VLLM_ENCODER_GPU             Encoder GPU index (default: 0)"
}

epd_exit_trap() {
    local rc=$?
    if [[ "$EC_STORAGE_OWNED" == true ]]; then
        rm -rf -- "$EC_SHARED_STORAGE_PATH"
    fi
    dynamo_reap_and_exit "$rc"
}

epd_init() {
    EC_STORAGE_OWNED=false
    if [[ -z "${EC_SHARED_STORAGE_PATH:-}" ]]; then
        EC_SHARED_STORAGE_PATH="$(mktemp -d "${TMPDIR:-/tmp}/dynamo-vllm-ec.XXXXXX")"
        EC_STORAGE_OWNED=true
    else
        mkdir -p "$EC_SHARED_STORAGE_PATH"
    fi
    trap epd_exit_trap EXIT

    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
    VLLM_ENCODER_HTTP_PORT="${VLLM_ENCODER_HTTP_PORT:-8100}"
    VLLM_ENCODER_GRPC_PORT="${VLLM_ENCODER_GRPC_PORT:-50051}"
    VLLM_ENCODER_GPU="${VLLM_ENCODER_GPU:-0}"
    ENCODER_GPU_MEMORY_UTILIZATION="${ENCODER_GPU_MEMORY_UTILIZATION:-0.1}"

    local default_kv_cache_bytes="${DEFAULT_KV_CACHE_BYTES:-1119388000}"
    GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
    if [[ -z "$GPU_MEM_ARGS" ]]; then
        GPU_MEM_ARGS="--kv-cache-memory-bytes $default_kv_cache_bytes --gpu-memory-utilization 0.01"
    fi

    ENCODER_EC_CONFIG="{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_producer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"${EC_SHARED_STORAGE_PATH}\"}}"
    # shellcheck disable=SC2034 # Consumed by the launch script that sourced this helper.
    CONSUMER_EC_CONFIG="{\"ec_connector\":\"ECExampleConnector\",\"ec_role\":\"ec_consumer\",\"ec_connector_extra_config\":{\"shared_storage_path\":\"${EC_SHARED_STORAGE_PATH}\"}}"
}

epd_launch_frontend_and_encoder() {
    python -m dynamo.frontend &

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
}
