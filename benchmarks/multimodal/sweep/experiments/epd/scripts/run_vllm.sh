#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a one-GPU vLLM Aggregated or colocated EPD service.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DYNAMO_HOME=$(cd -- "$SCRIPT_DIR/../../../../../.." && pwd)
# shellcheck source=/dev/null
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"
# shellcheck source=/dev/null
source "$DYNAMO_HOME/examples/common/launch_utils.sh"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

die() { echo "vLLM launcher: $*" >&2; exit 2; }

TOPOLOGY=${1:-}
shift || true
case "$TOPOLOGY" in
    aggregate|epd) ;;
    *) die "topology must be aggregate or epd" ;;
esac

usage() {
    cat <<'EOF'
Usage: run_vllm.sh {aggregate|epd} --image-token-budget N [OPTIONS]

Options:
  --model MODEL                 Model path or Hugging Face ID
  --served-model-name NAME      Name returned by the OpenAI endpoint
  --image-token-budget N        Per-image visual-token upper bound
  -h, --help

GPU, ports, logs, and optional CPU placement are configured with DYN_GPU,
DYN_HTTP_PORT, DYN_*_PORT*, DYN_LOG_DIR, and DYN_CPUSET_* environment values.
EOF
}

MODEL=${MODEL_PATH:-${MODEL:-nvidia/Qwen3.5-122B-A10B-NVFP4}}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen35-122b-a10b-nvfp4}
IMAGE_TOKEN_BUDGET=

while (($#)); do
    case "$1" in
        --model|--model-path) MODEL=${2:?}; shift 2 ;;
        --served-model-name) SERVED_MODEL_NAME=${2:?}; shift 2 ;;
        --image-token-budget) IMAGE_TOKEN_BUDGET=${2:?}; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ "$IMAGE_TOKEN_BUDGET" =~ ^[0-9]+$ ]] && ((IMAGE_TOKEN_BUDGET >= 64)) \
    || die "--image-token-budget must be an integer >= 64"

case ${DYN_RUNTIME_SOURCE_MODE:-} in
    worktree) export PYTHONPATH="$DYNAMO_HOME/components/src:$DYNAMO_HOME/lib/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}" ;;
    installed) export PYTHONPATH=${DYN_INSTALLED_RUNTIME_PYTHONPATH:-} ;;
    *) die "set DYN_RUNTIME_SOURCE_MODE to worktree or installed" ;;
esac

HTTP_PORT=${DYN_HTTP_PORT:-8000}
GPU=${DYN_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
[[ "$GPU" != *,* ]] || die "one launch must select exactly one GPU"
LOG_DIR=${DYN_LOG_DIR:-"$PWD/results/vllm-${TOPOLOGY}-cap${IMAGE_TOKEN_BUDGET}"}
mkdir -p "$LOG_DIR"

setup_dynamo_network "epd-vllm-${TOPOLOGY}-$$"
export DYN_MM_IMAGE_CACHE_SIZE=0 DYN_SPLIT_ENCODE=0
export DYN_VLLM_STREAM_INTERVAL=1
export DYN_VLLM_ENCODER_KV_CACHE_MEMORY_BYTES=4294967296
export ENABLE_ENCODER_CACHE=0 CUDA_DEVICE_MAX_CONNECTIONS=1
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc,self,sm
unset DYN_SYSTEM_PORT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE UCX_DEVICE UCX_NET_DEVICES

setup_nixl_libs
python3 -c 'import dynamo.frontend.main, dynamo.vllm.main' \
    || die "selected Dynamo runtime is incompatible with frontend or vllm"

install_cleanup_traps

MAX_PIXELS=$((IMAGE_TOKEN_BUDGET * 1024))
MM_PROCESSOR_KWARGS=$(printf '{"min_pixels":65536,"max_pixels":%s}' "$MAX_PIXELS")
read -r -a PROFILE_MEM_ARGS <<<"$(build_vllm_gpu_mem_args)"
AGG_MEM_ARGS=(--gpu-memory-utilization 0.85)
ENCODER_MEM_ARGS=(--gpu-memory-utilization 0.20)
PD_MEM_ARGS=(--gpu-memory-utilization 0.78)
if ((${#PROFILE_MEM_ARGS[@]})); then
    AGG_MEM_ARGS=("${PROFILE_MEM_ARGS[@]}")
    ENCODER_MEM_ARGS=("${PROFILE_MEM_ARGS[@]}")
    PD_MEM_ARGS=("${PROFILE_MEM_ARGS[@]}")
fi

COMMON_ARGS=(
    --enable-multimodal --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code --dtype bfloat16 --embedding-transfer-mode nixl-write
    --limit-mm-per-prompt '{"image":50,"video":0,"audio":0}'
    --no-enable-log-requests --frontend-decoding --max-model-len 32768
    --max-num-seqs 128 --max-num-batched-tokens 131072
    --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching
    --quantization modelopt_fp4 --mm-encoder-attn-backend FLASH_ATTN
    --mm-processor-cache-gb 0 --mm-processor-kwargs "$MM_PROCESSOR_KWARGS"
    --moe-backend flashinfer_cutedsl
)

print_launch_banner --multimodal --no-curl \
    "Launching vLLM benchmark ${TOPOLOGY}" "$MODEL" "$HTTP_PORT"
launch frontend "${DYN_CPUSET_FRONTEND:-}" env CUDA_VISIBLE_DEVICES= \
    python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp \
    --event-plane zmq --router-mode round-robin

if [[ $TOPOLOGY == aggregate ]]; then
    prepare_role_cache vllm aggregate
    launch aggregate "${DYN_CPUSET_AGG:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES="$GPU" ROLE=agg \
        DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20097}" \
        "${ROLE_CACHE_ENV[@]}" python3 -m dynamo.vllm \
        "${COMMON_ARGS[@]}" "${AGG_MEM_ARGS[@]}"
else
    start_mps
    MPS_ENV=(CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log")

    prepare_role_cache vllm encoder-0
    launch encoder-0 "${DYN_CPUSET_E0:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 ROLE=encode \
        DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20097}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.vllm \
        "${COMMON_ARGS[@]}" "${ENCODER_MEM_ARGS[@]}" --disaggregation-mode encode
    wait_for_worker_log encoder-0 "Starting to serve the encode worker endpoint"

    prepare_role_cache vllm encoder-1
    launch encoder-1 "${DYN_CPUSET_E1:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 ROLE=encode \
        DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2:-20098}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.vllm \
        "${COMMON_ARGS[@]}" "${ENCODER_MEM_ARGS[@]}" --disaggregation-mode encode
    wait_for_worker_log encoder-1 "Starting to serve the encode worker endpoint"

    prepare_role_cache vllm pd
    launch pd "${DYN_CPUSET_PD:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 ROLE=dedicated_pd \
        DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT3:-8083}" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT3:-20099}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.vllm \
        "${COMMON_ARGS[@]}" "${PD_MEM_ARGS[@]}" \
        --route-to-encoder --disaggregation-mode pd --enable-mm-embeds
fi

set +e
wait -n
status=$?
set -e
exit "$status"
