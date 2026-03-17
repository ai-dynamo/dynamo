#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is estimated from model parameters below. Override individual
# knobs (MAX_MODEL_LEN, MAX_CONCURRENT_SEQS) via env vars, or set
# _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE to bypass the calculation entirely.
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

GPU_MEM_FRACTION=$(build_gpu_mem_args vllm --model "$MODEL" --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_CONCURRENT_SEQS" --workers-per-gpu 2)

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
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization "${GPU_MEM_FRACTION}" \
  --max-model-len "$MAX_MODEL_LEN" &

# Wait for decode worker to initialize before starting prefill worker
# This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
# The decode worker needs time to:
# 1. Load model weights and allocate its memory fraction
# 2. Initialize KV cache
# 3. Register with NATS service discovery so prefill worker can find it
echo "Waiting for decode worker to initialize..."
sleep 10

# run prefill worker with metrics on port 8082
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=0 \
python3 -m dynamo.vllm \
  --model "$MODEL" \
  --enforce-eager \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' \
  --gpu-memory-utilization "${GPU_MEM_FRACTION}" \
  --max-model-len "$MAX_MODEL_LEN" \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081","enable_kv_cache_events":true}' &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
