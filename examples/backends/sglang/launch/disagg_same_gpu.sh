#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Disaggregated prefill/decode on a SINGLE GPU.
# Per-worker VRAM is controlled via build_sglang_gpu_mem_args (see gpu_utils.sh).
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
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-25000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--max-total-tokens $MAX_TOTAL_TOKENS"
fi

source "$SCRIPT_DIR/../../../common/launch_utils.sh"

DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated (same GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# NOTE: Each worker picks a random NCCL port (get_free_port) for torch.distributed.
# This has a TOCTOU race — the port can be grabbed before init_process_group binds it,
# causing sporadic EADDRINUSE.  Pass --nccl-port <unique_port> per worker to avoid this.
# run prefill worker with metrics on port 8081
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
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

# Wait for prefill worker to initialize before starting decode worker.
# Both workers share one GPU with --delete-ckpt-after-loading; without this
# wait they compete for GPU memory during model loading and the scheduler OOMs.
# || true: don't let set -e kill the script on timeout (wait_for_ready returns 1).
PREFILL_SYSTEM_PORT="${DYN_SYSTEM_PORT1:-8081}"
wait_for_ready "http://localhost:${PREFILL_SYSTEM_PORT}/health" 45 || true

# run decode worker with metrics on port 8082
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
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
