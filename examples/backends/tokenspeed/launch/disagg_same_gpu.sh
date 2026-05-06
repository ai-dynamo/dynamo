#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode TokenSpeed workers on a SINGLE GPU.
#
# Both workers load the model on the same device and share VRAM via a low
# --gpu-memory-utilization (0.3 each, leaving headroom for activations + KV).
# This is meant for integration testing of the bootstrap-info handoff
# between Dynamo and TokenSpeed's Mooncake transport — NOT for performance.
#
# For real workloads use 2+ GPUs (one per worker) or a quantized model.
#
# Prerequisites:
#   - etcd + NATS already running (see disagg_local.sh wrapper)
#   - dynamo.frontend already started on $DYN_HTTP_PORT (default 8000)
#   - mooncake.engine importable in this environment (image
#     aphoh/not-dynamo-tokenspeed:v1 ships it)
#
# Override knobs via env vars: MODEL, BLOCK_SIZE, MAX_NUM_SEQS,
# PREFILL_PORT, DECODE_PORT, BOOTSTRAP_PORT_PREFILL.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
BLOCK_SIZE="${BLOCK_SIZE:-64}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
MEM_UTIL="${MEM_UTIL:-0.3}"
BOOTSTRAP_PORT_PREFILL="${BOOTSTRAP_PORT_PREFILL:-8998}"
PREFILL_SYSTEM_PORT="${PREFILL_SYSTEM_PORT:-8081}"
DECODE_SYSTEM_PORT="${DECODE_SYSTEM_PORT:-8082}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

echo "Launching TokenSpeed disagg (P/D on single GPU=$CUDA_VISIBLE_DEVICES)"
echo "  model=$MODEL block=$BLOCK_SIZE seqs=$MAX_NUM_SEQS gpu_mem_util=$MEM_UTIL"

# Common flags. --attention-backend triton avoids the precompiled flashinfer
# cubin (some GPUs don't have a matching arch). --disable-kvstore turns off
# the CPU offload tier that defaults to ~2x device memory and can blow past
# host RAM for small models on big-VRAM GPUs.
COMMON_FLAGS=(
    --model "$MODEL"
    --served-model-name "$MODEL"
    --disaggregation-transfer-backend mooncake
    --block-size "$BLOCK_SIZE"
    --max-num-seqs "$MAX_NUM_SEQS"
    --gpu-memory-utilization "$MEM_UTIL"
    --attention-backend triton
    --disable-kvstore
    --skip-server-warmup
)

# Prefill worker — owns the Mooncake bootstrap server.
DYN_SYSTEM_PORT=$PREFILL_SYSTEM_PORT python3 -m dynamo.tokenspeed \
    "${COMMON_FLAGS[@]}" \
    --disaggregation-mode prefill \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT_PREFILL" &

# Decode worker — receives KV from prefill via Mooncake.
DYN_SYSTEM_PORT=$DECODE_SYSTEM_PORT python3 -m dynamo.tokenspeed \
    "${COMMON_FLAGS[@]}" \
    --disaggregation-mode decode &

wait
