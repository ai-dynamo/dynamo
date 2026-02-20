#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated multimodal serving with LoRA adapter support
#
# Architecture: Single-worker PD (Prefill-Decode) with dynamic LoRA loading
# - Frontend: Rust OpenAIPreprocessor handles image URLs (HTTP and data:// base64)
# - Worker: Standard vLLM worker with vision model + LoRA support
#
# Usage:
#   ./lora_agg.sh                                             # Qwen3-VL-2B (default)
#   ./lora_agg.sh --model llava-hf/llava-1.5-7b-hf           # LLaVA 1.5
#   ./lora_agg.sh -- --enforce-eager                          # Pass extra args to vLLM

set -euo pipefail
trap 'echo "Cleaning up..."; kill 0' EXIT

# ── Configuration ────────────────────────────────────────────────────────

MODEL_NAME="${DYN_MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
LORA_PATH="${DYN_LORA_PATH:-/tmp/dynamo_loras_multimodal}"
MAX_LORA_RANK="${DYN_MAX_LORA_RANK:-64}"
GPU_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"

# ── Parse command-line arguments ─────────────────────────────────────────

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            cat <<USAGE
Usage: $0 [OPTIONS] [-- EXTRA_VLLM_ARGS]

Options:
  --model <model_name>   Vision-language model to serve (default: $MODEL_NAME)
  -h, --help             Show this help message

Environment variables:
  DYN_MODEL_NAME         Base model name (default: Qwen/Qwen3-VL-2B-Instruct)
  DYN_HTTP_PORT          Frontend HTTP port (default: 8000)
  DYN_SYSTEM_PORT        Worker system/admin port (default: 8081)
  DYN_LORA_PATH          Local cache directory for LoRA adapters (default: /tmp/dynamo_loras_multimodal)
  DYN_MAX_LORA_RANK      Maximum LoRA rank supported (default: 64)
  CUDA_VISIBLE_DEVICES   GPU device index (default: 0)

Any arguments after '--' are passed through to the vLLM worker.

After launch, manage LoRA adapters via the system API on port \$DYN_SYSTEM_PORT:
  Load:   curl -X POST http://localhost:${SYSTEM_PORT}/v1/loras \\
            -H "Content-Type: application/json" \\
            -d '{"lora_name": "my-adapter", "source": {"uri": "file:///path/to/adapter"}}'
  List:   curl http://localhost:${SYSTEM_PORT}/v1/loras
  Unload: curl -X DELETE http://localhost:${SYSTEM_PORT}/v1/loras/my-adapter
USAGE
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# ── Banner ───────────────────────────────────────────────────────────────

echo "=================================================="
echo "Aggregated Multimodal Serving with LoRA Support"
echo "=================================================="
echo "Model:        $MODEL_NAME"
echo "Frontend:     http://localhost:$HTTP_PORT"
echo "System API:   http://localhost:$SYSTEM_PORT"
echo "LoRA cache:   $LORA_PATH"
echo "GPU device:   $GPU_DEVICE"
echo "=================================================="

# ── Environment setup ────────────────────────────────────────────────────

# Use TCP transport for multimodal workloads (base64 images can exceed NATS 1MB limit)
export DYN_REQUEST_PLANE=tcp

# Enable dynamic LoRA loading
export DYN_LORA_ENABLED=true
export DYN_LORA_PATH="$LORA_PATH"
mkdir -p "$DYN_LORA_PATH"

# ── Model-specific vLLM settings ────────────────────────────────────────
#
# Qwen VL models use dynamic resolution: a 2560px image can produce 5000+ tokens.
# max-model-len must exceed (text tokens + image tokens).
# Use --mm-processor-kwargs to cap image pixels and reduce token count if OOM.

MODEL_SPECIFIC_ARGS=("--gpu-memory-utilization" "0.85" "--max-model-len" "8192")

case "$MODEL_NAME" in
    Qwen/Qwen2.5-VL-7B-Instruct)
        MODEL_SPECIFIC_ARGS=("--gpu-memory-utilization" "0.85" "--max-model-len" "8192" "--max-num-seqs" "8192")
        ;;
    Qwen/Qwen3-VL-2B-Instruct)
        MODEL_SPECIFIC_ARGS=("--gpu-memory-utilization" "0.85" "--max-model-len" "8192" "--max-num-batched-tokens" "8192")
        ;;
    llava-hf/llava-1.5-7b-hf)
        MODEL_SPECIFIC_ARGS=("--gpu-memory-utilization" "0.85" "--max-model-len" "4096")
        ;;
esac

# ── Start services ──────────────────────────────────────────────────────

echo ""
echo "Starting frontend..."
python -m dynamo.frontend &
FRONTEND_PID=$!

# Wait for frontend to become ready
echo "Waiting for frontend on port $HTTP_PORT..."
for i in $(seq 1 60); do
    if curl -sf "http://localhost:$HTTP_PORT/v1/models" > /dev/null 2>&1; then
        echo "Frontend is ready."
        break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "ERROR: Frontend process exited unexpectedly."
        exit 1
    fi
    sleep 1
done

echo "Starting vLLM worker..."

# --enable-lora: Enable LoRA adapter support in vLLM engine
# --max-lora-rank: Maximum LoRA rank (increase if your adapters have higher rank)
# --connector none: No KV transfer needed for aggregated serving
CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
DYN_SYSTEM_PORT="$SYSTEM_PORT" \
    python -m dynamo.vllm \
        --enable-multimodal \
        --model "$MODEL_NAME" \
        --connector none \
        --enable-lora \
        --max-lora-rank "$MAX_LORA_RANK" \
        "${MODEL_SPECIFIC_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" &

# Wait for all background processes
wait
