#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two localhost nodes with one GPU each, a native headless follower, and one sidecar.

set -e

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../../../examples/common/gpu_utils.sh"
# shellcheck source=/dev/null
source "$SCRIPT_DIR/../../../../examples/common/launch_utils.sh"
trap dynamo_exit_trap EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
VLLM_RS_HTTP_PORT="${VLLM_RS_HTTP_PORT:-8100}"
VLLM_GRPC_PORT="${VLLM_GRPC_PORT:-50051}"
VLLM_MASTER_PORT="${VLLM_MASTER_PORT:-29501}"
VLLM_HANDSHAKE_PORT="${VLLM_HANDSHAKE_PORT:-29502}"
VLLM_HEAD_GPU="${VLLM_HEAD_GPU:-0}"
VLLM_FOLLOWER_GPU="${VLLM_FOLLOWER_GPU:-1}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
GPU_MEM_ARGS="${GPU_MEM_ARGS:---kv-cache-memory-bytes 1119388000 --gpu-memory-utilization 0.01}"
read -r -a memory_args <<< "$GPU_MEM_ARGS"
common_args=(
    --tensor-parallel-size 2
    --nnodes 2
    --master-addr 127.0.0.1
    --master-port "$VLLM_MASTER_PORT"
    # Each node exposes one local GPU; global TP ranks cannot index that view.
    --disable-custom-all-reduce
    --enforce-eager
    --max-num-seqs "$MAX_CONCURRENT_SEQS"
    "${memory_args[@]}"
)

print_launch_banner "Launching vLLM sidecar with two-node headless TP" "$MODEL" "${DYN_HTTP_PORT:-8000}"
python3 -m dynamo.frontend &

CUDA_VISIBLE_DEVICES="$VLLM_HEAD_GPU" \
    vllm-rs serve "$MODEL" \
    --host 127.0.0.1 \
    --port "$VLLM_RS_HTTP_PORT" \
    --grpc-port "$VLLM_GRPC_PORT" \
    --data-parallel-rpc-port "$VLLM_HANDSHAKE_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    -- --node-rank 0 "${common_args[@]}" "$@" &

CUDA_VISIBLE_DEVICES="$VLLM_FOLLOWER_GPU" \
    python3 -m vllm.entrypoints.cli.main serve "$MODEL" \
    --headless \
    --node-rank 1 \
    --max-model-len "$MAX_MODEL_LEN" \
    "${common_args[@]}" "$@" &

DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}" \
    dynamo-vllm-sidecar --grpc-endpoint "127.0.0.1:${VLLM_GRPC_PORT}" &

wait_any_exit
