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

export DYN_NAMESPACE=${DYN_NAMESPACE:-"epd-vllm-${TOPOLOGY}-$$"}
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
export DYN_MULTIMODAL_EMBEDDING_CACHE_PUBLISHER=0
export DYN_VLLM_EMBEDDING_TRANSFER_MODE=nixl-write DYN_SPLIT_ENCODE=0
export DYN_VLLM_COALESCE_EMBEDDING_TRANSFER=1
export DYN_VLLM_ENCODE_ROUTING_POLICY=round_robin
export DYN_VLLM_STREAM_INTERVAL=1
export DYN_NIXL_MEDIA_PERSISTENT_ARENA=1 DYN_NIXL_MEDIA_BUFFER_BYTES=2147483648
export DYN_VLLM_GPU_IMAGE_PREPROCESS=0 DYN_VLLM_EXTERNAL_EAGER_VISION_BATCH=0
export DYN_VLLM_EXTERNAL_EAGER_PREPROCESS_BATCH=0
export DYN_VLLM_EXTERNAL_EAGER_PREPROCESS_BATCH_COMBINED_VIT=0
export DYN_VLLM_DISABLE_CROSS_REQUEST_MM_CACHE=1
export DYN_VLLM_ENCODER_KV_CACHE_MEMORY_BYTES=4294967296
export DYN_VLLM_ENCODER_MAX_NUM_SEQS=64 DYN_VLLM_ENCODER_GPU_MEMORY_UTILIZATION=0.20
export DYN_VLLM_ENCODER_MM_ENCODER_ATTN_BACKEND=FLASH_ATTN
export DYN_EWORKER_TORCH_NUM_THREADS=16 MM_ENCODER_ATTN_BACKEND=FLASH_ATTN
export VLLM_MM_PROCESSOR_CACHE_GB=0 ENABLE_ENCODER_CACHE=0
export TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1 PYTHONNOUSERSITE=1
export UCX_TLS=rc,tcp,cuda_copy,cuda_ipc,self,sm
export TRITON_CACHE_DIR="$LOG_DIR/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$LOG_DIR/cache/torchinductor"
export VLLM_CACHE_ROOT="$LOG_DIR/cache/vllm"
export FLASHINFER_WORKSPACE_BASE="$LOG_DIR/cache/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"
unset DYN_SYSTEM_PORT CUDA_MPS_ACTIVE_THREAD_PERCENTAGE UCX_DEVICE UCX_NET_DEVICES

# Frontend decoding loads NIXL directly; the wheel keeps its libraries beside
# the Python package rather than on the default linker path.
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
python3 -c 'import dynamo.frontend.main, dynamo.vllm.main' \
    || die "selected Dynamo runtime is incompatible with frontend or vllm"

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

wait_for_worker_log() {
    local name=$1 pattern=$2 timeout=${DYN_WORKER_READY_TIMEOUT_SECONDS:-1800}
    local pid=${PIDS[$((${#PIDS[@]} - 1))]} deadline=$((SECONDS + timeout))
    while ((SECONDS < deadline)); do
        grep -Fq "$pattern" "$LOG_DIR/$name.log" && return
        kill -0 "$pid" 2>/dev/null || die "$name exited before becoming ready"
        sleep 1
    done
    die "$name was not ready within ${timeout}s"
}

MAX_PIXELS=$((IMAGE_TOKEN_BUDGET * 1024))
MM_PROCESSOR_KWARGS=$(printf '{"min_pixels":65536,"max_pixels":%s}' "$MAX_PIXELS")
read -r -a PROFILE_MEM_ARGS <<<"$(build_vllm_gpu_mem_args)"

COMMON_ARGS=(
    --enable-multimodal --model "$MODEL" --served-model-name "$SERVED_MODEL_NAME"
    --trust-remote-code --dtype bfloat16 --embedding-transfer-mode nixl-write
    --limit-mm-per-prompt '{"image":50,"video":0,"audio":0}'
    --no-enable-log-requests --frontend-decoding --max-model-len 32768
    --max-num-seqs 128 --max-num-batched-tokens 131072
    --kv-cache-dtype fp8_e4m3 --no-enable-prefix-caching
    --quantization modelopt_fp4 --mm-encoder-attn-backend FLASH_ATTN
    --mm-processor-cache-gb 0 --mm-processor-kwargs "$MM_PROCESSOR_KWARGS"
    --moe-backend flashinfer_cutedsl --stream-interval 1
)

worker() {
    local role=$1 system_port=$2 nixl_port=$3 fraction=$4 cpus=$5
    local visible_gpu=$GPU runtime_role=agg
    local cache_root="$LOG_DIR/cache/$role"
    mkdir -p "$cache_root"/{home,xdg,triton,torchinductor,vllm,flashinfer,flashinfer_cubin/cubins}
    local -a cache_env=(
        HOME="$cache_root/home" XDG_CACHE_HOME="$cache_root/xdg"
        TRITON_CACHE_DIR="$cache_root/triton"
        TORCHINDUCTOR_CACHE_DIR="$cache_root/torchinductor"
        VLLM_CACHE_ROOT="$cache_root/vllm"
        FLASHINFER_WORKSPACE_BASE="$cache_root/flashinfer"
        FLASHINFER_CUBIN_DIR="$cache_root/flashinfer_cubin/cubins")
    local -a role_args=() mem_args=(--gpu-memory-utilization "$fraction") mps_env=()
    ((${#PROFILE_MEM_ARGS[@]} == 0)) || mem_args=("${PROFILE_MEM_ARGS[@]}")
    if [[ $TOPOLOGY == epd ]]; then
        visible_gpu=0
        mps_env=(CUDA_MPS_PIPE_DIRECTORY="$MPS_ROOT/pipe" CUDA_MPS_LOG_DIRECTORY="$MPS_ROOT/log")
    fi
    case "$role" in
        encoder-*) runtime_role=encode; role_args=(--disaggregation-mode encode) ;;
        pd) runtime_role=dedicated_pd; role_args=(--route-to-encoder --disaggregation-mode pd --enable-mm-embeds) ;;
    esac
    launch "$role" "$cpus" env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE \
        CUDA_VISIBLE_DEVICES="$visible_gpu" ROLE="$runtime_role" \
        DYN_QWEN36_MOE_ENCODER_FAMILY_PATCH=1 DYN_SYSTEM_PORT="$system_port" \
        VLLM_NIXL_SIDE_CHANNEL_PORT="$nixl_port" "${cache_env[@]}" "${mps_env[@]}" \
        python3 -m dynamo.vllm "${COMMON_ARGS[@]}" "${mem_args[@]}" "${role_args[@]}"
}

print_launch_banner --multimodal --no-curl \
    "Launching vLLM benchmark ${TOPOLOGY}" "$MODEL" "$HTTP_PORT"
launch frontend "${DYN_CPUSET_FRONTEND:-}" env CUDA_VISIBLE_DEVICES= \
    python3 -m dynamo.frontend --http-port "$HTTP_PORT" \
    --discovery-backend "$DYN_DISCOVERY_BACKEND" --request-plane tcp \
    --event-plane zmq --router-mode round-robin

if [[ $TOPOLOGY == aggregate ]]; then
    worker aggregate "${DYN_SYSTEM_PORT1:-8081}" \
        "${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20097}" 0.85 "${DYN_CPUSET_AGG:-}"
else
    start_mps
    worker encoder-0 "${DYN_SYSTEM_PORT1:-8081}" \
        "${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT1:-20097}" 0.20 "${DYN_CPUSET_E0:-}"
    wait_for_worker_log encoder-0 "Starting to serve the encode worker endpoint"
    worker encoder-1 "${DYN_SYSTEM_PORT2:-8082}" \
        "${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT2:-20098}" 0.20 "${DYN_CPUSET_E1:-}"
    wait_for_worker_log encoder-1 "Starting to serve the encode worker endpoint"
    worker pd "${DYN_SYSTEM_PORT3:-8083}" \
        "${DYN_VLLM_NIXL_SIDE_CHANNEL_PORT3:-20099}" 0.78 "${DYN_CPUSET_PD:-}"
fi

set +e
wait -n
status=$?
set -e
exit "$status"
