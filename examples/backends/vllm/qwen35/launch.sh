#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal serving for Qwen3.5 hybrid models with MM-aware approximate KV routing.
#
# Qwen3.5 is a multimodal hybrid model (GatedDeltaNet + Gated Attention + Vision Encoder).
# It supports images and video but has hybrid architecture constraints:
#
# 1. DO NOT pass --kv-events-config or --enable-kv-cache-events:
#    vLLM disables the Hybrid KV Cache Manager when kv_events_config is set,
#    but Qwen3.5's mixed KV cache specs (GDN + FullAttention) cannot be unified
#    into one type.
#
# 2. Use --mamba-cache-mode align (not "all"):
#    Qwen3.5 raises NotImplementedError with mamba_cache_mode="all".
#
# 3. Approximate KV routing (--no-router-kv-events):
#    Hybrid models cannot emit KV events to the router. The MM Router Worker
#    predicts cache state from its own routing decisions using prefix hashing.
#
# 4. Disaggregated P/D is NOT supported for hybrid models in vLLM:
#    HybridKVCacheCoordinator asserts dcp_world_size == 1.
#
# 5. Use TCP transport for multimodal payloads (NATS has 1MB limit).
#
# Architecture:
#   Frontend (--router-mode round-robin)
#     -> MM Router Worker (approximate KV routing + multimodal hash)
#       -> vLLM Worker (--enable-multimodal --mamba-cache-mode align)

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_CONCURRENT_SEQS="${MAX_CONCURRENT_SEQS:-2}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"

# TCP transport: avoids NATS 1MB payload limit for base64-encoded images
export DYN_REQUEST_PLANE=tcp

print_launch_banner --no-curl "Launching Qwen3.5 Multimodal + MM Router + Approx KV (1 GPU)" "$MODEL" "$HTTP_PORT" \
    "Backend:     dynamo.vllm --enable-multimodal --mamba-cache-mode align" \
    "MM Router:   MM-aware approximate KV routing (--no-router-kv-events)" \
    "Frontend:    round-robin to MM Router" \
    "Transport:   TCP (multimodal payloads)"

print_curl_footer <<CURL
  # Text-only request
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "messages": [{"role": "user", "content": "What is 2+2?"}],
      "max_tokens": 32
    }'

  # Multimodal request (image)
  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{
      "model": "${MODEL}",
      "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the image"},
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"}}
      ]}],
      "max_tokens": 50
    }'
CURL

GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)

# vLLM worker: hybrid multimodal model
# --served-model-name __internal: hides from frontend so traffic goes through MM Router
# --mamba-cache-mode align: required for GDN+Attention hybrid architecture
# --enable-multimodal: enables vision encoder / multimodal data handling
# NOTE: do NOT use --is-decode-worker here. It causes the handler to enter
#   decode-only mode which silently drops image data for models not in
#   QWEN_VL_MODELS (Qwen3.5 is not listed), leading to incorrect prefix
#   cache hits across different images.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
    python -m dynamo.vllm \
    --model "$MODEL" \
    --served-model-name "${MODEL}__internal" \
    --enable-multimodal \
    --mamba-cache-mode align \
    --block-size "$BLOCK_SIZE" \
    --enforce-eager \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_CONCURRENT_SEQS" \
    $GPU_MEM_ARGS \
    "${EXTRA_ARGS[@]}" &

# MM Router Worker: multimodal-aware approximate KV routing
# --namespace dynamo: must match frontend/vllm default (DYN_NAMESPACE defaults to "dynamo")
# --no-router-kv-events: Qwen3.5 hybrid model cannot emit KV events
DYN_SYSTEM_PORT=${MM_ROUTER_SYSTEM_PORT:-8082} \
    python -m examples.backends.vllm.mm_router_worker \
    --model "$MODEL" \
    --namespace dynamo \
    --block-size "$BLOCK_SIZE" \
    --no-router-kv-events &

# Frontend: round-robin dispatch to MM Router (KV routing happens inside MM Router)
python -m dynamo.frontend \
    --router-mode round-robin &

wait_any_exit
