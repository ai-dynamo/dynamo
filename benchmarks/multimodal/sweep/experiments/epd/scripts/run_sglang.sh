#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Launch a one-GPU SGLang Aggregated or colocated EPD service.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export DYNAMO_HOME=$(cd -- "$SCRIPT_DIR/../../../../../.." && pwd)
# shellcheck source=/dev/null
source "$DYNAMO_HOME/examples/common/gpu_utils.sh"
# shellcheck source=/dev/null
source "$DYNAMO_HOME/examples/common/launch_utils.sh"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

die() { echo "SGLang launcher: $*" >&2; exit 2; }

TOPOLOGY=${1:-}
shift || true
case "$TOPOLOGY" in
    aggregate|epd) ;;
    *) die "topology must be aggregate or epd" ;;
esac

usage() {
    cat <<'EOF'
Usage: run_sglang.sh {aggregate|epd} [--image-token-budget N] [OPTIONS]

Options:
  --model MODEL                 Model path or Hugging Face ID
  --served-model-name NAME      Name returned by the OpenAI endpoint
  --image-token-budget N        Per-image cap; -1 leaves processor limits unchanged
  -h, --help

GPU, ports, logs, and optional CPU placement are configured with DYN_GPU,
DYN_HTTP_PORT, DYN_*_PORT*, DYN_LOG_DIR, and DYN_CPUSET_* environment values.
EOF
}

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
    [[ "$IMAGE_TOKEN_BUDGET" =~ ^[0-9]+$ ]] && ((IMAGE_TOKEN_BUDGET >= 64)) \
        || die "--image-token-budget must be -1 or an integer >= 64"
fi

case ${DYN_RUNTIME_SOURCE_MODE:-worktree} in
    worktree) export PYTHONPATH="$DYNAMO_HOME/components/src:$DYNAMO_HOME/lib/bindings/python/src${PYTHONPATH:+:$PYTHONPATH}" ;;
    installed) export PYTHONPATH=${DYN_INSTALLED_RUNTIME_PYTHONPATH:-} ;;
    *) die "set DYN_RUNTIME_SOURCE_MODE to worktree or installed" ;;
esac

HTTP_PORT=${DYN_HTTP_PORT:-8000}
GPU=${DYN_GPU:-${CUDA_VISIBLE_DEVICES:-0}}
[[ "$GPU" != *,* ]] || die "one launch must select exactly one GPU"
LOG_DIR=${DYN_LOG_DIR:-"$PWD/results/sglang-${TOPOLOGY}"}
mkdir -p "$LOG_DIR"

setup_dynamo_network "epd-sglang-${TOPOLOGY}-$$"
export DYN_MM_IMAGE_CACHE_SIZE=0
export DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read
export DYN_SGLANG_STREAM_INTERVAL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc,self,sm
unset DYN_SYSTEM_PORT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE UCX_DEVICE UCX_NET_DEVICES

setup_nixl_libs
python3 -c 'import dynamo.frontend.main, dynamo.sglang.main' \
    || die "selected Dynamo runtime is incompatible with frontend or sglang"

install_cleanup_traps

MM_PROCESS_ARGS=()
if [[ $IMAGE_TOKEN_BUDGET != -1 ]]; then
    MAX_PIXELS=$((IMAGE_TOKEN_BUDGET * 1024))
    MM_PROCESS_CONFIG=$(printf \
        '{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}},"vision_config":{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}}}}' \
        "$MAX_PIXELS" "$MAX_PIXELS")
    MM_PROCESS_ARGS=(--mm-process-config "$MM_PROCESS_CONFIG")
fi
read -r -a PROFILE_MEM_ARGS <<<"$(build_sglang_gpu_mem_args)"
AGG_MEM_ARGS=(--mem-fraction-static 0.85 "${PROFILE_MEM_ARGS[@]}")
PD_MEM_ARGS=(--mem-fraction-static 0.78 "${PROFILE_MEM_ARGS[@]}")

COMMON_ARGS=(
    --enable-multimodal --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
    --skip-tokenizer-init
    --max-running-requests 128 --chunked-prefill-size 32768 --max-prefill-tokens 131072
    --linear-attn-prefill-backend flashinfer --mamba-ssm-dtype bfloat16
    --disable-radix-cache
)

ENCODER_ARGS=(
    --enable-multimodal --disaggregation-mode encode --frontend-decoding
    --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
    --chat-template qwen2-vl
    "${MM_PROCESS_ARGS[@]}"
)

print_launch_banner --multimodal --no-curl \
    "Launching SGLang benchmark ${TOPOLOGY}" "$MODEL" "$HTTP_PORT"
launch frontend "${DYN_CPUSET_FRONTEND:-}" env CUDA_VISIBLE_DEVICES= \
    python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp \
    --event-plane zmq --router-mode round-robin

if [[ $TOPOLOGY == aggregate ]]; then
    prepare_role_cache sglang aggregate
    launch aggregate "${DYN_CPUSET_AGG:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES="$GPU" DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
        "${ROLE_CACHE_ENV[@]}" python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" --port "${DYN_WORKER_PORT1:-30001}" \
        --nccl-port "${DYN_NCCL_PORT1:-31001}" "${AGG_MEM_ARGS[@]}" \
        --frontend-decoding "${MM_PROCESS_ARGS[@]}"
else
    start_mps
    MPS_ENV=(CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log")

    prepare_role_cache sglang encoder-0
    launch encoder-0 "${DYN_CPUSET_E0:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.sglang \
        "${ENCODER_ARGS[@]}" --port "${DYN_WORKER_PORT1:-30001}" \
        --nccl-port "${DYN_NCCL_PORT1:-31001}"

    prepare_role_cache sglang encoder-1
    launch encoder-1 "${DYN_CPUSET_E1:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT2:-8082}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.sglang \
        "${ENCODER_ARGS[@]}" --port "${DYN_WORKER_PORT2:-30002}" \
        --nccl-port "${DYN_NCCL_PORT2:-31002}"

    prepare_role_cache sglang pd
    launch pd "${DYN_CPUSET_PD:-}" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES=0 DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT3:-8083}" \
        "${ROLE_CACHE_ENV[@]}" "${MPS_ENV[@]}" python3 -m dynamo.sglang \
        "${COMMON_ARGS[@]}" --port "${DYN_WORKER_PORT3:-30003}" \
        --nccl-port "${DYN_NCCL_PORT3:-31003}" "${PD_MEM_ARGS[@]}" \
        --dedicated-mm-encoder --disaggregation-mode pd
fi

set +e
wait -n
status=$?
set -e
exit "$status"
