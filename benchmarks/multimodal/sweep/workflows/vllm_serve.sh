#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# vLLM serve wrapper for benchmark sweeps with opt-in Nsight Systems capture.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
SWEEP_REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../../../..")"
source "$SWEEP_REPO_ROOT/examples/common/gpu_utils.sh"
source "$SWEEP_REPO_ROOT/examples/common/launch_utils.sh"

MODEL=""
CAPACITY_GB=0
EXTRA_ARGS=()
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"; shift 2 ;;
        --multimodal-embedding-cache-capacity-gb)
            CAPACITY_GB="$2"; shift 2 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

has_gpu_mem_override=0
has_max_model_len=0
for ((i = 0; i < ${#EXTRA_ARGS[@]}; i++)); do
    case "${EXTRA_ARGS[$i]}" in
        --max-model-len)
            if ((i + 1 >= ${#EXTRA_ARGS[@]})); then
                echo "ERROR: --max-model-len requires a value" >&2
                exit 2
            fi
            has_max_model_len=1
            MAX_MODEL_LEN="${EXTRA_ARGS[$((i + 1))]}"
            ;;
        --max-model-len=*)
            has_max_model_len=1
            ;;
        --gpu-memory-utilization|--gpu-memory-utilization=*|--kv-cache-memory-bytes|--kv-cache-memory-bytes=*)
            has_gpu_mem_override=1
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: --model is required" >&2
    exit 1
fi

EC_ARGS=()
if [[ "$CAPACITY_GB" != "0" ]]; then
    EC_ARGS=(--ec-transfer-config "{
        \"ec_role\": \"ec_both\",
        \"ec_connector\": \"DynamoMultimodalEmbeddingCacheConnector\",
        \"ec_connector_module_path\": \"dynamo.vllm.multimodal_utils.multimodal_embedding_cache_connector\",
        \"ec_connector_extra_config\": {\"multimodal_embedding_cache_capacity_gb\": $CAPACITY_GB}
    }")
fi

GPU_MEM_ARGS=""
if [[ "$has_gpu_mem_override" == "0" ]]; then
    GPU_MEM_ARGS="$(build_vllm_gpu_mem_args)"
    if [[ -z "$GPU_MEM_ARGS" ]]; then
        GPU_MEM_ARGS="--gpu-memory-utilization .9"
    fi
fi
GPU_MEM_ARGV=()
if [[ -n "$GPU_MEM_ARGS" ]]; then
    read -r -a GPU_MEM_ARGV <<< "$GPU_MEM_ARGS"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching standalone vLLM" "$MODEL" "$HTTP_PORT"

VLLM_CMD=(
    vllm serve "$MODEL"
    --port "$HTTP_PORT"
    --enable-log-requests
)
if [[ "$has_max_model_len" == "0" ]]; then
    VLLM_CMD+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ ${#GPU_MEM_ARGV[@]} -gt 0 ]]; then
    VLLM_CMD+=("${GPU_MEM_ARGV[@]}")
fi
if [[ ${#EC_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${EC_ARGS[@]}")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    VLLM_CMD+=("${EXTRA_ARGS[@]}")
fi

LAUNCH_PREFIX=()
if [[ "${DYN_DISABLE_NSYS:-1}" != "1" ]]; then
    NSYS_BIN="${DYN_NSYS_BIN:-/opt/nvidia/nsight-systems-cli/2026.2.1/bin/nsys}"
    if [[ ! -x "$NSYS_BIN" ]]; then
        echo "ERROR: nsys is not executable at $NSYS_BIN" >&2
        exit 1
    fi

    NSYS_DIR="${DYN_NSYS_DIR:-/dynamo-tmp/nsys}"
    NSYS_TMPDIR="${DYN_NSYS_TMPDIR:-/dynamo-tmp/nsys-staging}"
    NSYS_PREFIX="${DYN_NSYS_OUTPUT_PREFIX:-vllm}-${DYN_BENCHMARK_ARM:-standalone}"
    if [[ -n "${DYN_BENCHMARK_SWEEP:-}" ]]; then
        NSYS_PREFIX="${NSYS_PREFIX}-${DYN_BENCHMARK_SWEEP}"
    fi
    mkdir -p "$NSYS_DIR" "$NSYS_TMPDIR"
    export TMPDIR="$NSYS_TMPDIR"

    timestamp="$(date +%Y%m%d_%H%M%S)"
    nsys_output="$NSYS_DIR/${NSYS_PREFIX}_${timestamp}.nsys-rep"
    LAUNCH_PREFIX=(
        "$NSYS_BIN" profile
        --trace="${DYN_NSYS_TRACE:-cuda,nvtx}"
        --sample=none
        --cpuctxsw=none
        --kill=sigterm
        --force-overwrite=true
        -o "$nsys_output"
    )
    echo "[nsys] vllm-serve -> $nsys_output" >&2
fi

server_pid=0
cleanup() {
    local exit_code="${1:-0}"
    trap - EXIT INT TERM
    if [[ "$server_pid" -gt 0 ]] && kill -0 -- "-$server_pid" 2>/dev/null; then
        kill -INT -- "-$server_pid" 2>/dev/null || true
        shutdown_grace="${DYN_SERVER_SHUTDOWN_GRACE_SECONDS:-}"
        if [[ -z "$shutdown_grace" ]]; then
            if [[ "${DYN_DISABLE_NSYS:-1}" == "1" ]]; then
                shutdown_grace=10
            else
                shutdown_grace=150
            fi
        fi
        for _ in $(seq 1 "$shutdown_grace"); do
            kill -0 -- "-$server_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 -- "-$server_pid" 2>/dev/null; then
            kill -KILL -- "-$server_pid" 2>/dev/null || true
        fi
        wait "$server_pid" 2>/dev/null || true
    fi
    exit "$exit_code"
}
trap 'cleanup 0' INT TERM
trap 'cleanup $?' EXIT

if [[ ${#LAUNCH_PREFIX[@]} -gt 0 ]]; then
    setsid "${LAUNCH_PREFIX[@]}" "${VLLM_CMD[@]}" &
else
    setsid "${VLLM_CMD[@]}" &
fi
server_pid=$!
wait "$server_pid"
server_pid=0
trap - EXIT
