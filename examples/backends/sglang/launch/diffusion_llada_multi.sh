#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# All-in-one fleet launcher for LLaDA 2.0 on Dynamo:
# starts one Dynamo frontend (in the requested routing mode) + one SGLang DLLM
# worker per GPU listed in --gpu-ids. Drop-in baseline for §3 of the experiments
# doc — same args as the single-worker launcher, just plural.
#
# Usage:
#   diffusion_llada_multi.sh --gpu-ids 0,1                        # 2 workers + RR frontend
#   diffusion_llada_multi.sh --gpu-ids 0,1,2,3 --mode kv-events   # 4 workers + KV-events frontend
#   diffusion_llada_multi.sh --gpu-ids 0 --worker-only            # 1 worker, no frontend
#
# Use --worker-only when you want to restart the frontend with a different
# --mode without restarting the workers (cheap benchmark sweep iteration).
# In that case launch the frontend separately via frontend_router.sh.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# --- CLI parsing ---
GPU_IDS=""
MODE="round-robin"
WORKER_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-ids)  GPU_IDS="$2"; shift 2 ;;
    --gpu-id)   GPU_IDS="$2"; shift 2 ;;   # alias: single-GPU shorthand
    --mode)     MODE="$2"; shift 2 ;;
    --worker-only) WORKER_ONLY=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$GPU_IDS" ]]; then
  echo "Usage: $0 --gpu-ids 0[,1,2,3] [--mode round-robin|kv-approx|kv-events] [--worker-only]" >&2
  exit 1
fi

IFS=',' read -ra GPUS <<< "$GPU_IDS"

# --- Worker config (env-var overrides, same set as diffusion_llada.sh) ---
MODEL_PATH="${MODEL_PATH:-inclusionAI/LLaDA2.0-mini-preview}"
DLLM_ALGORITHM="${DLLM_ALGORITHM:-LowConfidence}"
DLLM_ALGORITHM_CONFIG="${DLLM_ALGORITHM_CONFIG:-}"
QUANTIZATION="${QUANTIZATION:-}"
NAMESPACE="${NAMESPACE:-dynamo}"
COMPONENT="${COMPONENT:-backend}"
ENDPOINT="${ENDPOINT:-generate}"
TP_SIZE="${TP_SIZE:-1}"

# Opt-in routing-experiment knobs (unset by default to match
# diffusion_llada.sh's bare-SGLang invocation).
PAGE_SIZE="${PAGE_SIZE:-}"
DLLM_LOAD_MULTIPLIER="${DLLM_LOAD_MULTIPLIER:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
DISABLE_RADIX_CACHE="${DISABLE_RADIX_CACHE:-0}"
KV_EVENTS_CONFIG="${KV_EVENTS_CONFIG:-}"

# --- Banner ---
HTTP_PORT="${HTTP_PORT:-8001}"
print_launch_banner --no-curl "Launching LLaDA 2.0 fleet (${#GPUS[@]} worker(s))" "$MODEL_PATH" "$HTTP_PORT" \
    "GPUs:        ${GPUS[*]}" \
    "Mode:        $( [[ $WORKER_ONLY == 1 ]] && echo "<workers only>" || echo "$MODE" )" \
    "Algorithm:   $DLLM_ALGORITHM" \
    "Quantization: ${QUANTIZATION:-bf16}" \
    "Page size:   ${PAGE_SIZE:-<default>}" \
    "Load mult:   ${DLLM_LOAD_MULTIPLIER:-<unset>}"

# --- Frontend (unless --worker-only) ---
if [[ "$WORKER_ONLY" != "1" ]]; then
  echo "Starting Dynamo frontend (mode: $MODE) on :$HTTP_PORT..."
  HTTP_PORT="$HTTP_PORT" bash "$SCRIPT_DIR/frontend_router.sh" --mode "$MODE" &
  sleep 2
fi

# --- Workers (one per GPU, backgrounded into the same process group) ---
build_worker_cmd() {
  local cmd="python -m dynamo.sglang \
      --model-path $MODEL_PATH \
      --tp-size $TP_SIZE \
      --skip-tokenizer-init \
      --trust-remote-code \
      --endpoint dyn://${NAMESPACE}.${COMPONENT}.${ENDPOINT} \
      --enable-metrics \
      --disable-cuda-graph \
      --disable-overlap-schedule \
      --attention-backend triton \
      --dllm-algorithm $DLLM_ALGORITHM"
  [[ -n "$DLLM_ALGORITHM_CONFIG" ]] && cmd="$cmd --dllm-algorithm-config $DLLM_ALGORITHM_CONFIG"
  [[ -n "$PAGE_SIZE" ]]             && cmd="$cmd --page-size $PAGE_SIZE"
  [[ -n "$DLLM_LOAD_MULTIPLIER" ]]  && cmd="$cmd --dllm-load-multiplier $DLLM_LOAD_MULTIPLIER"
  [[ -n "$MEM_FRACTION_STATIC" ]]   && cmd="$cmd --mem-fraction-static $MEM_FRACTION_STATIC"
  [[ -n "$MAX_TOTAL_TOKENS" ]]      && cmd="$cmd --max-total-tokens $MAX_TOTAL_TOKENS"
  [[ -n "$QUANTIZATION" ]]          && cmd="$cmd --quantization $QUANTIZATION"
  [[ "$DISABLE_RADIX_CACHE" == "1" ]] && cmd="$cmd --disable-radix-cache"
  [[ -n "$KV_EVENTS_CONFIG" ]]      && cmd="$cmd --kv-events-config '$KV_EVENTS_CONFIG'"
  echo "$cmd"
}

WORKER_CMD="$(build_worker_cmd)"

for gpu in "${GPUS[@]}"; do
  echo "Starting LLaDA worker on GPU $gpu..."
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"
    eval "$WORKER_CMD"
  ) &
done

# Exit on first child failure; EXIT trap (kill 0) tears down the rest.
wait_any_exit
