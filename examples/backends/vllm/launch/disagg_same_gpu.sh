#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is controlled via build_vllm_gpu_mem_args (see gpu_utils.sh).
# Override individual knobs (MAX_MODEL_LEN, MAX_CONCURRENT_SEQS) via env vars.
#
# Measured reference (Qwen/Qwen3-0.6B, --max-model-len 4096, RTX 6000 Ada 48 GiB):
#   estimate (from gpu_utils.sh) : ~4.0 GiB per worker (~8.0 GiB total)
#   actual (nvidia-smi)          : ~3.4 GiB per worker (~6.7 GiB total)
#   fraction per worker (for 48 GiB) : 0.09
#   The ~1.3 GiB pad comes from the overhead term (CUDA ctx + activations).
#   Overestimating is intentional -- better to pad than OOM.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
# Inherit GPU from parent (profiler/test harness sets CUDA_VISIBLE_DEVICES);
# default to GPU 0 for standalone use.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Per-worker KV cache byte cap (deterministic, GPU-size independent).
# Profiled safe value: 1_023_525_000 bytes (~976 MiB, 2x over min 512 MiB).
# --gpu-memory-utilization 0.01 prevents vLLM's startup free-memory check from
# rejecting the launch when a co-resident worker already holds VRAM.
# The profiler/parallel runner overrides via _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES.
DEFAULT_KV_CACHE_BYTES="${DEFAULT_KV_CACHE_BYTES:-1023525000}"

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--kv-cache-memory-bytes $DEFAULT_KV_CACHE_BYTES --gpu-memory-utilization 0.01"
fi

source "$SCRIPT_DIR/../../../common/launch_utils.sh"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated on Same GPU (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run decode worker with metrics on port 8081
# --enforce-eager is added for quick deployment. for production use, need to remove this flag
# For disaggregated deployments we standardize on DYN_SYSTEM_PORT1/2 instead of
# *_PREFILL/*_DECODE env names so test harnesses can set one simple pair.
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  $GPU_MEM_ARGS \
  --max-model-len "$MAX_MODEL_LEN" &

# Wait for decode worker to initialize before starting prefill worker.
# Both workers share one GPU; without this wait they compete for GPU memory
# during model loading and the scheduler OOMs.
# || true: don't let set -e kill the script on timeout (wait_for_ready returns 1).
DECODE_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
wait_for_ready "http://localhost:${DECODE_SYSTEM_PORT}/health" 45 || true

# run prefill worker with metrics on port 8082
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  $GPU_MEM_ARGS \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
