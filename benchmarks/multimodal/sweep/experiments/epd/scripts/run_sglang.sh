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

die() { echo "SGLang launcher: $*" >&2; exit 2; }

TOPOLOGY=${1:-}
shift || true
case "$TOPOLOGY" in
    aggregate|epd) ;;
    *) die "topology must be aggregate or epd" ;;
esac

usage() {
    cat <<'EOF'
Usage: run_sglang.sh {aggregate|epd} --image-token-budget N [OPTIONS]

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
LOG_DIR=${DYN_LOG_DIR:-"$PWD/results/sglang-${TOPOLOGY}-cap${IMAGE_TOKEN_BUDGET}"}
mkdir -p "$LOG_DIR"

export DYN_NAMESPACE=${DYN_NAMESPACE:-"epd-sglang-${TOPOLOGY}-$$"}
export DYN_DISCOVERY_BACKEND=${DYN_DISCOVERY_BACKEND:-file}
if [[ $DYN_DISCOVERY_BACKEND == file ]]; then
    export DYN_FILE_KV=${DYN_FILE_KV:-"$LOG_DIR/discovery"}
    mkdir -p "$DYN_FILE_KV"
fi
export DYN_REQUEST_PLANE=tcp DYN_EVENT_PLANE=zmq DYN_MM_ALLOW_INTERNAL=1
if [[ -z ${DYN_TCP_RPC_HOST:-} ]]; then
    for host in $(hostname -I); do
        [[ $host == 127.* || $host == ::1 ]] || { DYN_TCP_RPC_HOST=$host; break; }
    done
fi
[[ -n ${DYN_TCP_RPC_HOST:-} ]] || die "set DYN_TCP_RPC_HOST to a non-loopback address"
export DYN_TCP_RPC_HOST
export DYN_MM_IMAGE_CACHE_SIZE=0 DYN_MULTIMODAL_LOADER_CACHE_GB=0
export DYN_MULTIMODAL_EMBEDDING_CACHE_CAPACITY_GB=0
export DYN_MULTIMODAL_EMBEDDING_CACHE_PUBLISHER=0 SGLANG_VLM_CACHE_SIZE_MB=0
export DYN_SGL_EMBEDDING_TRANSFER_MODE=nixl-read
export DYN_SGLANG_STREAM_INTERVAL=1
export SGLANG_VIT_ENABLE_CUDA_GRAPH=0 MM_ATTENTION_BACKEND=fa4
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONNOUSERSITE=1
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc,self,sm
export TRITON_CACHE_DIR="$LOG_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOG_DIR/cache/torchinductor"
export SGLANG_CACHE_DIR="$LOG_DIR/cache/sglang"
export FLASHINFER_WORKSPACE_BASE="$LOG_DIR/cache/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SGLANG_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE"
unset DYN_SYSTEM_PORT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE UCX_DEVICE UCX_NET_DEVICES

NIXL_LIBS=$(python3 - <<'PY'
import importlib.util
from pathlib import Path
spec = importlib.util.find_spec("nixl_cu13")
if spec is None or spec.origin is None:
    raise SystemExit("nixl_cu13 is required for frontend decoding")
root = Path(spec.origin).resolve().parent.parent / ".nixl_cu13.mesonpy.libs"
if not (root / "plugins").is_dir() or not any(root.glob("libnixl.so*")):
    raise SystemExit(f"NIXL libraries/plugins not found under {root}")
print(root)
PY
)
export NIXL_PLUGIN_DIR="$NIXL_LIBS/plugins"
export LD_LIBRARY_PATH="$NIXL_LIBS:$NIXL_PLUGIN_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python3 -c 'import dynamo.frontend.main, dynamo.sglang.main' \
    || die "selected Dynamo runtime is incompatible with frontend or sglang"

PIDS=()
MPS_ROOT=

launch() {
    local name=$1 cpus=$2
    shift 2
    local -a command=("$@")
    [[ -z $cpus ]] || command=(taskset --cpu-list "$cpus" "${command[@]}")
    setsid "${command[@]}" >"$LOG_DIR/$name.log" 2>&1 &
    PIDS+=("$!")
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    set +e
    local pid
    for pid in "${PIDS[@]}"; do kill -TERM -- "-$pid" 2>/dev/null; done
    for pid in "${PIDS[@]}"; do
        for _ in {1..100}; do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        kill -KILL -- "-$pid" 2>/dev/null
    done
    wait 2>/dev/null
    if [[ -n $MPS_ROOT ]]; then
        timeout --signal=TERM --kill-after=2s 10s env \
            CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" \
            CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log" \
            bash -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1
    fi
    [[ $MPS_ROOT == /tmp/dynamo-epd-mps.* ]] && rm -r -- "$MPS_ROOT"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_mps() {
    local minimum=${DYN_EPD_MIN_SHM_BYTES:-68719476736}
    [[ -w /dev/shm && $(stat -f -c %T /dev/shm) == tmpfs ]] \
        || die "EPD requires writable tmpfs at /dev/shm"
    local size
    size=$(df -PB1 /dev/shm | awk 'NR==2 {print $2}')
    ((size >= minimum)) || die "EPD requires /dev/shm >= $minimum bytes"

    MPS_ROOT=$(mktemp -d /tmp/dynamo-epd-mps.XXXXXX)
    mkdir -p "$MPS_ROOT/pipe" "$MPS_ROOT/log"
    local -a command=(env CUDA_VISIBLE_DEVICES="$GPU"
        CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe"
        CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log" nvidia-cuda-mps-control -d)
    [[ -z ${DYN_CPUSET_MPS:-} ]] \
        || command=(taskset --cpu-list "$DYN_CPUSET_MPS" "${command[@]}")
    "${command[@]}"
    for _ in {1..100}; do
        if echo get_server_list | env CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" \
            nvidia-cuda-mps-control >/dev/null 2>&1; then
            return
        fi
        sleep 0.1
    done
    die "CUDA MPS did not become ready"
}

MAX_PIXELS=$((IMAGE_TOKEN_BUDGET * 1024))
MM_PROCESS_CONFIG=$(printf \
    '{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}},"vision_config":{"image":{"size":{"shortest_edge":65536,"longest_edge":%s}}}}' \
    "$MAX_PIXELS" "$MAX_PIXELS")
read -r -a PROFILE_MEM_ARGS <<<"$(build_sglang_gpu_mem_args)"

COMMON_ARGS=(
    --enable-multimodal --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code --skip-tokenizer-init --tp 1 --dtype bfloat16
    --host 0.0.0.0 --log-level warning --context-length 32768 --page-size 64
    --max-running-requests 128 --chunked-prefill-size 32768 --max-prefill-tokens 131072
    --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3
    --attention-backend trtllm_mha --linear-attn-decode-backend flashinfer
    --linear-attn-prefill-backend flashinfer --mamba-scheduler-strategy extra_buffer
    --mamba-track-interval 128 --mamba-ssm-dtype bfloat16 --reasoning-parser qwen3
    --disable-radix-cache --limit-mm-data-per-request '{"image":50,"video":0,"audio":0}'
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":6}'
    --fp4-gemm-backend flashinfer_cutedsl --moe-runner-backend flashinfer_trtllm
)

worker() {
    local role=$1 system_port=$2 worker_port=$3 nccl_port=$4 fraction=$5 cpus=$6
    local visible_gpu=$GPU
    local cache_root="$LOG_DIR/cache/$role"
    mkdir -p "$cache_root"/{home,xdg,triton,torchinductor,sglang,flashinfer,flashinfer_cubin/cubins}
    local -a cache_env=(
        HOME="$cache_root/home" XDG_CACHE_HOME="$cache_root/xdg"
        TRITON_CACHE_DIR="$cache_root/triton"
        TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        SGLANG_CACHE_DIR="$cache_root/sglang"
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
        FLASHINFER_CUBIN_DIR="$cache_root/flashinfer_cubin/cubins")
    local -a args=() mps_env=() mem_args=(--mem-fraction-static "$fraction" "${PROFILE_MEM_ARGS[@]}")
    if [[ $TOPOLOGY == epd ]]; then
        visible_gpu=0
        mps_env=(CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log")
    fi
    case "$role" in
        encoder-*)
            args=(--enable-multimodal --disaggregation-mode encode --frontend-decoding
                --model-path "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
                --chat-template qwen2-vl --skip-tokenizer-init --trust-remote-code
                --encoder-only --tp 1 --dtype bfloat16 --mm-attention-backend fa4
                --mm-process-config "$MM_PROCESS_CONFIG"
                --limit-mm-data-per-request '{"image":50,"video":0,"audio":0}'
                --host 0.0.0.0 --port "$worker_port" --nccl-port "$nccl_port"
                --log-level warning --quantization modelopt_fp4
                --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":4}')
            ;;
        aggregate)
            args=("${COMMON_ARGS[@]}" --port "$worker_port" --nccl-port "$nccl_port"
                "${mem_args[@]}" --frontend-decoding --mm-process-config "$MM_PROCESS_CONFIG")
            ;;
        pd)
            args=("${COMMON_ARGS[@]}" --port "$worker_port" --nccl-port "$nccl_port"
                "${mem_args[@]}" --dedicated-mm-encoder --disaggregation-mode pd)
            ;;
    esac
    launch "$role" "$cpus" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES="$visible_gpu" DYN_SYSTEM_PORT="$system_port" \
        "${cache_env[@]}" "${mps_env[@]}" python3 -m dynamo.sglang "${args[@]}"
}

print_launch_banner --multimodal --no-curl \
    "Launching SGLang benchmark ${TOPOLOGY}" "$MODEL" "$HTTP_PORT"
launch frontend "${DYN_CPUSET_FRONTEND:-}" env CUDA_VISIBLE_DEVICES= \
    python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp \
    --event-plane zmq --router-mode round-robin

if [[ $TOPOLOGY == aggregate ]]; then
    worker aggregate "${DYN_SYSTEM_PORT1:-8081}" "${DYN_WORKER_PORT1:-30001}" \
        "${DYN_NCCL_PORT1:-31001}" 0.85 "${DYN_CPUSET_AGG:-}"
else
    start_mps
    worker encoder-0 "${DYN_SYSTEM_PORT1:-8081}" "${DYN_WORKER_PORT1:-30001}" \
        "${DYN_NCCL_PORT1:-31001}" 0.85 "${DYN_CPUSET_E0:-}"
    worker encoder-1 "${DYN_SYSTEM_PORT2:-8082}" "${DYN_WORKER_PORT2:-30002}" \
        "${DYN_NCCL_PORT2:-31002}" 0.85 "${DYN_CPUSET_E1:-}"
    worker pd "${DYN_SYSTEM_PORT3:-8083}" "${DYN_WORKER_PORT3:-30003}" \
        "${DYN_NCCL_PORT3:-31003}" 0.78 "${DYN_CPUSET_PD:-}"
fi

set +e
wait -n
status=$?
set -e
exit "$status"
