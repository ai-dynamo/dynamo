#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is controlled via build_gpu_mem_args (see gpu_utils.sh).
# Override individual knobs (CONTEXT_LENGTH, MAX_RUNNING_REQUESTS) via env vars.
#
# Measured reference (Qwen/Qwen3-0.6B, --context-length 4096, RTX 6000 Ada 48 GiB):
#   estimate (from gpu_utils.sh) : ~5.7 GiB per worker (w=1.1 + kv=0.9 + oh=3.7)
#   actual (nvidia-smi)          : ~5.3 GiB per worker (~10.9 GiB total)
#   fraction per worker (48 GiB)  : 0.12
#   KV cache                      : 25,536-29,712 tokens per worker
#   Handles full 4096-token context with --max-running-requests 2.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"

MODEL="Qwen/Qwen3-0.6B"

# ---- Tunable (override via env vars) ----
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"

GPU_MEM_ARGS=$(build_gpu_mem_args sglang --workers-per-gpu 2)

source "$SCRIPT_DIR/../../../common/launch_utils.sh"

DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated (same GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)"

# run ingress with KV router mode for disaggregated setup
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend --router-mode kv &

# run prefill worker with metrics on port 8081
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  $GPU_MEM_ARGS \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size "$CONTEXT_LENGTH" \
  --max-prefill-tokens "$CONTEXT_LENGTH" \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --enable-metrics &

# Wait for prefill worker to initialize before starting decode worker
# This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
# The prefill worker needs time to:
# 1. Load model weights and allocate its memory fraction
# 2. Initialize KV cache with --delete-ckpt-after-loading to free checkpoint memory
# 3. Register with NATS service discovery so decode worker can find it
echo "Waiting for prefill worker to initialize..."
sleep 5

# run decode worker with metrics on port 8082
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
python3 -m dynamo.sglang \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  $GPU_MEM_ARGS \
  --context-length "$CONTEXT_LENGTH" \
  --chunked-prefill-size "$CONTEXT_LENGTH" \
  --max-prefill-tokens "$CONTEXT_LENGTH" \
  --enable-memory-saver \
  --delete-ckpt-after-loading \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --enable-metrics &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
