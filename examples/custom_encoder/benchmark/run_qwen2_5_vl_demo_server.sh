#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Start one isolated side of the Qwen2.5 CustomEncoder live demo.

set -euo pipefail

SIDE="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "$SIDE" in
    control)
        ARM="custom-worker-control"
        LAUNCHER="$SCRIPT_DIR/../launch/agg_qwen2_5_vl_control.sh"
        GPU_INDEX="${DYN_DEMO_GPU_INDEX:-0}"
        HTTP_PORT="${DYN_DEMO_HTTP_PORT:-8000}"
        SYSTEM_PORT="${DYN_DEMO_SYSTEM_PORT:-8081}"
        KV_EVENT_PORT="${DYN_DEMO_KV_EVENT_PORT:-20080}"
        NAMESPACE="${DYN_DEMO_NAMESPACE:-qwen25-demo-control}"
        ;;
    dynamo-vllm)
        ARM="dynamo-vllm-custom"
        LAUNCHER="$SCRIPT_DIR/../launch/agg_qwen2_5_vl_benchmark.sh"
        GPU_INDEX="${DYN_DEMO_GPU_INDEX:-1}"
        HTTP_PORT="${DYN_DEMO_HTTP_PORT:-8001}"
        SYSTEM_PORT="${DYN_DEMO_SYSTEM_PORT:-8082}"
        KV_EVENT_PORT="${DYN_DEMO_KV_EVENT_PORT:-20081}"
        NAMESPACE="${DYN_DEMO_NAMESPACE:-qwen25-demo-dynamo-vllm}"
        ;;
    *)
        echo >&2 "Usage: $0 {control|dynamo-vllm}"
        exit 2
        ;;
esac

CPUSET="${DYN_DEMO_CPUSET:-}"
CACHE_ROOT="${DYN_DEMO_CACHE_ROOT:-/tmp/qwen25-custom-encoder-live-demo/$ARM}"
KV_EVENTS_CONFIG="$(printf \
    '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:%s","enable_kv_cache_events":true}' \
    "$KV_EVENT_PORT")"

for command in nvidia-smi python taskset; do
    command -v "$command" >/dev/null
done
test -x "$LAUNCHER"

gpu_record="$(nvidia-smi -i "$GPU_INDEX" \
    --query-gpu=index,name,memory.total,power.limit,clocks.max.sm,driver_version,uuid,pci.bus_id \
    --format=csv,noheader,nounits)"
IFS=',' read -r _ gpu_name gpu_memory gpu_power gpu_clock _ _ _ <<< "$gpu_record"
gpu_name="${gpu_name# }"
gpu_memory="${gpu_memory# }"
gpu_power="${gpu_power# }"
gpu_clock="${gpu_clock# }"
if [[ "$gpu_name" != "NVIDIA H100 80GB HBM3" \
    || "$gpu_memory" != "81559" \
    || "$gpu_power" != "700.00" \
    || "$gpu_clock" != "1980" ]]; then
    echo >&2 "Unexpected GPU hardware: $gpu_record"
    exit 1
fi
CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    python -c 'import torch; assert torch.cuda.device_count() == 1'

mkdir -p \
    "$CACHE_ROOT/vllm" \
    "$CACHE_ROOT/xdg" \
    "$CACHE_ROOT/torchinductor" \
    "$CACHE_ROOT/triton"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export DYN_WORKER_GPU="$GPU_INDEX"
export DYN_HTTP_PORT="$HTTP_PORT"
export DYN_SYSTEM_PORT="$SYSTEM_PORT"
export DYN_NAMESPACE="$NAMESPACE"
export DYN_MODEL=Qwen/Qwen2.5-1.5B-Instruct
export DYN_ENCODER_CLASS=examples.custom_encoder.qwen2_5_vl_benchmark_encoder.Qwen2_5VLBenchmarkEncoder
export DYN_QWEN2_VL_ENCODER_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
export DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE=1536
export DYN_QWEN2_VL_PREPROCESS_CONCURRENCY=64
export DYN_QWEN2_VL_MAX_BATCH_PATCHES=10368
export DYN_QWEN2_VL_MAX_BATCH_ITEMS=8
export DYN_QWEN2_VL_MAX_QUEUE_DELAY_US=1000
export DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS=1,2,4,8
export DYN_QWEN2_VL_GRAPH_IMAGE_SIZES=300x300,500x500
export DYN_QWEN2_VL_PREPROCESS_CACHE_SIZE=0
export DYN_CUSTOM_ENCODER_DISPATCH_LOG=1
export DYN_CONTROL_MAX_BATCH_ITEMS=8
export DYN_CONTROL_MAX_QUEUE_DELAY_US=1000
export DYN_MAX_MODEL_LEN=2048
export DYN_MAX_NUM_SEQS=64
export DYN_VLLM_GPU_MEMORY_UTILIZATION="${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"

printf 'DEMO_SERVER_SIDE=%s\n' "$SIDE"
printf 'DEMO_SERVER_GPU=%s\n' "$gpu_record"
printf 'DEMO_SERVER_CPUSET=%s\n' "${CPUSET:-unrestricted}"
printf 'DEMO_SERVER_HTTP=http://127.0.0.1:%s\n' "$HTTP_PORT"
printf 'DEMO_SERVER_NAMESPACE=%s\n' "$NAMESPACE"

command=(
    "$LAUNCHER"
    --namespace "$NAMESPACE"
    --enable-prefix-caching
    --kv-events-config "$KV_EVENTS_CONFIG"
)
if [[ -n "$CPUSET" ]]; then
    exec taskset -c "$CPUSET" "${command[@]}"
fi
exec "${command[@]}"
