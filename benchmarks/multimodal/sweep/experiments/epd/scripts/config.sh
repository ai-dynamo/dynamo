#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Shared arguments and runtime configuration for the EPD launchers.

EPD_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DYNAMO_HOME=$(cd -- "$EPD_SCRIPT_DIR/../../../../../.." && pwd)
# shellcheck source=/dev/null
source "$DYNAMO_HOME/examples/common/launch_utils.sh"
# shellcheck source=common.sh
source "$EPD_SCRIPT_DIR/common.sh"

BACKEND=
TOPOLOGY=
BACKEND_LABEL=
MPS_ENV=()

die() { echo "${BACKEND_LABEL:-Dynamo} launcher: $*" >&2; exit 2; }

usage() {
    cat <<EOF
Usage: $(basename "$0") [--image-token-budget N] [OPTIONS]

Options:
  --model MODEL                 Model path or Hugging Face ID
  --served-model-name NAME      Name returned by the OpenAI endpoint
  --image-token-budget N        Per-image cap; -1 leaves processor limits unchanged
  -h, --help

GPU, HTTP port, and logs are configured with DYN_GPU, DYN_HTTP_PORT,
and DYN_LOG_DIR environment values.
EOF
}

configure_vllm() {
    export DYN_MM_IMAGE_CACHE_SIZE=0

    local -a mm_processor_args=()
    if [[ $IMAGE_TOKEN_BUDGET != -1 ]]; then
        local max_pixels=$((IMAGE_TOKEN_BUDGET * 1024))
        local mm_processor_kwargs
        mm_processor_kwargs=$(printf \
            '{"min_pixels":65536,"max_pixels":%s}' "$max_pixels")
        mm_processor_args=(--mm-processor-kwargs "$mm_processor_kwargs")
    fi
    AGG_MEM_ARGS=(--gpu-memory-utilization 0.85)
    PD_MEM_ARGS=(--gpu-memory-utilization 0.78)

    COMMON_ARGS=(
        --enable-multimodal --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
        --limit-mm-per-prompt '{"image":50,"video":0,"audio":0}'
        --frontend-decoding
        --max-num-seqs 128 --max-num-batched-tokens 131072
        --no-enable-prefix-caching
        --mm-processor-cache-gb 0 --moe-backend flashinfer_cutedsl
        "${mm_processor_args[@]}"
    )
    ENCODER_ARGS=(
        --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
        --frontend-decoding
        "${mm_processor_args[@]}"
    )
}

configure_sglang() {
    export DYN_MM_IMAGE_CACHE_SIZE=0
    export DYN_SGLANG_STREAM_INTERVAL=1

    MM_PROCESS_ARGS=()
    if [[ $IMAGE_TOKEN_BUDGET != -1 ]]; then
        local max_pixels=$((IMAGE_TOKEN_BUDGET * 1024))
        local mm_process_config
        mm_process_config=$(printf \
            '{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}},"vision_config":{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}}}}' \
            "$max_pixels" "$max_pixels")
        MM_PROCESS_ARGS=(--mm-process-config "$mm_process_config")
    fi
    AGG_MEM_ARGS=(--mem-fraction-static 0.85)
    PD_MEM_ARGS=(--mem-fraction-static 0.78)

    COMMON_ARGS=(
        --enable-multimodal --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
        --skip-tokenizer-init
        --max-running-requests 128 --chunked-prefill-size 32768
        --max-prefill-tokens 131072 --linear-attn-prefill-backend flashinfer
        --mamba-ssm-dtype bfloat16 --disable-radix-cache
    )
    ENCODER_ARGS=(
        --enable-multimodal --disaggregation-mode encode --frontend-decoding
        --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
        --chat-template qwen2-vl
        "${MM_PROCESS_ARGS[@]}"
    )
}

build_vllm_worker_env() {
    local -n output=$1
    local cache_role=$2 gpu=$3
    prepare_role_cache "$cache_role"
    output=(
        env CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_MAX_CONNECTIONS=1
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}"
    )
    if [[ $TOPOLOGY == epd ]]; then
        output+=(DYN_SPLIT_ENCODE=0)
    fi
    if [[ $cache_role == encoder-* ]]; then
        output+=(
            ENABLE_ENCODER_CACHE=0
            DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1
            DYN_VLLM_ENCODER_KV_CACHE_MEMORY_BYTES=4294967296
        )
    fi
}

build_sglang_worker_env() {
    local -n output=$1
    local cache_role=$2 gpu=$3
    prepare_role_cache "$cache_role"
    output=(
        env CUDA_VISIBLE_DEVICES="$gpu" CUDA_DEVICE_MAX_CONNECTIONS=1
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}"
    )
    if [[ $TOPOLOGY == epd ]]; then
        output+=(DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read)
    fi
}

configure_process_envs() {
    FRONTEND_ENV=(env CUDA_VISIBLE_DEVICES=)
    FRONTEND_ARGS=(--http-port "$HTTP_PORT")
    if [[ $TOPOLOGY == epd ]]; then
        start_mps
        MPS_ENV=(
            CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
            CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log"
        )
    fi

    case "$BACKEND:$TOPOLOGY" in
        vllm:aggregate)
            build_vllm_worker_env AGG_ENV aggregate "$GPU"
            ;;
        vllm:epd)
            build_vllm_worker_env ENCODER0_ENV encoder-0 0
            build_vllm_worker_env ENCODER1_ENV encoder-1 0
            build_vllm_worker_env PD_ENV pd 0
            ;;
        sglang:aggregate)
            build_sglang_worker_env AGG_ENV aggregate "$GPU"
            ;;
        sglang:epd)
            build_sglang_worker_env ENCODER0_ENV encoder-0 0
            build_sglang_worker_env ENCODER1_ENV encoder-1 0
            build_sglang_worker_env PD_ENV pd 0
            ;;
    esac
}

setup_launcher() {
    BACKEND=$1
    TOPOLOGY=$2
    shift 2
    case "$BACKEND:$TOPOLOGY" in
        vllm:aggregate|vllm:epd) BACKEND_LABEL=vLLM ;;
        sglang:aggregate|sglang:epd) BACKEND_LABEL=SGLang ;;
        *) die "unsupported backend/topology: $BACKEND/$TOPOLOGY" ;;
    esac

    MODEL=${MODEL_PATH:-${MODEL:-nvidia/Qwen3.5-122B-A10B-NVFP4}}
    SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-122b-a10b-nvfp4}
    IMAGE_TOKEN_BUDGET=-1
    while (($#)); do
        case "$1" in
            --model|--model-path) MODEL=${2:?}; shift 2 ;;
            --served-model-name) SERVED_MODEL_NAME=${2:?}; shift 2 ;;
            --image-token-budget) IMAGE_TOKEN_BUDGET=${2:?}; shift 2 ;;
            -h|--help) usage; exit 0 ;;
            *) die "unknown option: $1" ;;
        esac
    done
    if [[ $IMAGE_TOKEN_BUDGET != -1 ]]; then
        [[ $IMAGE_TOKEN_BUDGET =~ ^[0-9]+$ ]] && ((IMAGE_TOKEN_BUDGET >= 64)) \
            || die "--image-token-budget must be -1 or an integer >= 64"
    fi

    case ${DYN_RUNTIME_SOURCE_MODE:-worktree} in
        worktree) export PYTHONPATH="$DYNAMO_HOME/components/src:$DYNAMO_HOME/lib/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}" ;;
        installed) export PYTHONPATH=${DYN_INSTALLED_RUNTIME_PYTHONPATH:-} ;;
        *) die "set DYN_RUNTIME_SOURCE_MODE to worktree or installed" ;;
    esac

    HTTP_PORT=${DYN_HTTP_PORT:-8000}
    GPU=${DYN_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
    [[ $GPU != *,* ]] || die "one launch must select exactly one GPU"
    LOG_DIR=${DYN_LOG_DIR:-"$PWD/results/${BACKEND}-${TOPOLOGY}"}
    mkdir -p "$LOG_DIR"

    setup_dynamo_network "epd-${BACKEND}-${TOPOLOGY}-$$"
    export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc,self,sm
    unset DYN_SYSTEM_PORT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE UCX_NET_DEVICES
    setup_nixl_libs
    "configure_$BACKEND"
    python3 -c "import dynamo.frontend.main, dynamo.${BACKEND}.main" \
        || die "selected Dynamo runtime is incompatible with frontend or $BACKEND"
    install_cleanup_traps
    print_launch_banner --no-curl \
        "Launching $BACKEND_LABEL benchmark ${TOPOLOGY}" "$MODEL" "$HTTP_PORT"
    configure_process_envs
}

wait_for_exit() {
    set +e
    wait -n
    local status=$?
    set -e
    exit "$status"
}
