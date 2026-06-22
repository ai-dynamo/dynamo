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

# --model overrides the default (e.g. a VLM for the multimodal P/D test).
# --single-gpu is a no-op kept for parity with the other launch scripts.
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            if [[ $# -lt 2 || "$2" == -* ]]; then
                echo "Missing value for --model"
                echo "Use --help for usage information"
                exit 1
            fi
            MODEL="$2"
            shift 2
            ;;
        --single-gpu)
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--model <name>] [--single-gpu]"
            echo "  --model <name>  Model to serve (default: $MODEL)"
            echo "  --single-gpu    Accepted no-op; both workers already share GPU 0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ---- Tunable (override via env vars) ----
CONTEXT_LENGTH="${CONTEXT_LENGTH:-4096}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-25000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

GPU_MEM_ARGS=$(build_sglang_gpu_mem_args)
if [[ -z "$GPU_MEM_ARGS" ]]; then
    GPU_MEM_ARGS="--max-total-tokens $MAX_TOTAL_TOKENS"
fi

# torch-memory-saver (--enable-memory-saver/--delete-ckpt-after-loading) packs
# both workers tightly on one GPU but its preload links libcudart.so.12, which
# is absent on CUDA 13 images. Allow opting out where the GPU has enough VRAM
# to hold both workers unpacked.
MEM_SAVER_ARGS="--enable-memory-saver --delete-ckpt-after-loading"
if [[ "${DYN_SGLANG_DISABLE_MEMORY_SAVER:-0}" == "1" ]]; then
    MEM_SAVER_ARGS=""
fi

source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Select legacy vs unified worker entry point. `--unified` routes workers
# through dynamo.sglang.unified_main (the Rust backend-common Worker, which
# owns the prefill drain loop); default stays on the legacy main.
pick_worker_module dynamo.sglang dynamo.sglang.unified_main "$@"

DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner "Launching Disaggregated (same GPU)" "$MODEL" "$HTTP_PORT" \
    "Workers:     2 (prefill + decode, fraction is per worker)"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
# Set DYN_CHAT_PROCESSOR=sglang to exercise the Python pre/post processor instead of Rust.
FRONTEND_ARGS=()
if [[ -n "${DYN_CHAT_PROCESSOR:-}" ]]; then
    FRONTEND_ARGS+=(--dyn-chat-processor "$DYN_CHAT_PROCESSOR")
fi
if [[ -n "${DYN_ROUTER_MODE:-}" ]]; then
    FRONTEND_ARGS+=(--router-mode "$DYN_ROUTER_MODE")
fi
python3 -m dynamo.frontend "${FRONTEND_ARGS[@]}" &

# NOTE: Each worker picks a random NCCL port (get_free_port) for torch.distributed.
# This has a TOCTOU race — the port can be grabbed before init_process_group binds it,
# causing sporadic EADDRINUSE.  Pass --nccl-port <unique_port> per worker to avoid this.
# run prefill worker with metrics on port 8081
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT=${DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT:-60} \
python3 -m "$WORKER_MODULE" \
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
  $MEM_SAVER_ARGS \
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
python3 -m "$WORKER_MODULE" \
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
  $MEM_SAVER_ARGS \
  --max-running-requests "$MAX_RUNNING_REQUESTS" \
  --enable-metrics &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
