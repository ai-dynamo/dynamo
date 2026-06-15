#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# E→PD with FullPromptEncoder: custom vision encoder + text-only LLM PD.
#
# Architecture:
#   Encoder worker: loads a customer-supplied FullPromptEncoder class.
#                   For each request it calls encoder.encode(image_urls,
#                   lm_token_ids, lm_embed_tokens) → (seq_len, lm_hidden_dim),
#                   then transfers the full prompt embedding to the PD via NIXL.
#   PD worker:      aggregated vLLM worker running a text-only LLM.
#                   Receives the tensor as EmbedsPrompt — no multimodal
#                   processing, no token expansion; just transformer layers.
#   Frontend:       standard dynamo.frontend OpenAI-compatible HTTP gateway.
#
# The example encoder (QwenVLExampleEncoder) uses Qwen2.5-VL's ViT and
# prepends image tokens before text.  Replace with your own class for
# production use (learned projector, correct placeholder handling, etc.).
#
# Usage:
#   ./enc_full_prompt_pd.sh [--encoder-model <hf_id>] [--pd-model <hf_id>]
#                           [--encoder-class <dotted.ClassName>]
#                           [--encoder-checkpoint <path>]
#                           [--single-gpu] [--transfer-mode local|nixl_write]
#
# Defaults:
#   Encoder model: Qwen/Qwen2.5-VL-3B-Instruct (ViT backbone)
#   Encoder class: examples.custom_encoder.qwen_vl_example.QwenVLExampleEncoder
#   Encoder ckpt:  same as encoder model
#   PD model:      Qwen/Qwen2.5-1.5B (text-only LLM)
#   GPUs:          encoder=2, pd=1  (avoids GPU 0 which is 4 GB A400)
#   Transfer mode: local

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
ENCODER_MODEL="${DYN_ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
PD_MODEL="${DYN_PD_MODEL:-Qwen/Qwen2.5-1.5B}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.qwen_vl_example.QwenVLExampleEncoder}"
ENCODER_CHECKPOINT="${DYN_ENCODER_CHECKPOINT:-$ENCODER_MODEL}"
DYN_ENCODE_WORKER_GPU="${DYN_ENCODE_WORKER_GPU:-2}"
DYN_PD_WORKER_GPU="${DYN_PD_WORKER_GPU:-1}"
TRANSFER_MODE="${DYN_EMBEDDING_TRANSFER_MODE:-local}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"

EXTRA_ENCODER_ARGS=""
EXTRA_PD_ARGS=""

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --encoder-model)
            ENCODER_MODEL=$2; shift 2 ;;
        --pd-model)
            PD_MODEL=$2; shift 2 ;;
        --encoder-class)
            ENCODER_CLASS=$2; shift 2 ;;
        --encoder-checkpoint)
            ENCODER_CHECKPOINT=$2; shift 2 ;;
        --single-gpu)
            DYN_ENCODE_WORKER_GPU=2
            DYN_PD_WORKER_GPU=2
            EXTRA_ENCODER_ARGS="--enforce-eager"
            EXTRA_PD_ARGS="--enforce-eager --max-model-len 2048"
            shift ;;
        --transfer-mode)
            TRANSFER_MODE=$2; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: enc_full_prompt_pd.sh [OPTIONS]

FullPromptEncoder: custom vision encoder produces full prompt embeddings.

Options:
  --encoder-model <id>       VLM whose ViT is used as backbone (default: Qwen/Qwen2.5-VL-3B-Instruct)
  --pd-model <id>            Text-only LLM for PD (default: Qwen/Qwen2.5-1.5B)
  --encoder-class <path>     Dotted module.ClassName for FullPromptEncoder subclass
  --encoder-checkpoint <p>   Checkpoint passed to FullPromptEncoder.load()
  --single-gpu               Run encoder + PD on the same GPU (GPU 2, enforce-eager)
  --transfer-mode <mode>     local|nixl_write|nixl_read (default: local)
  -h, --help                 Show this help
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=================================================================="
echo "  FullPromptEncoder E→PD"
echo "  Encoder model  : $ENCODER_MODEL (GPU $DYN_ENCODE_WORKER_GPU)"
echo "  Encoder class  : $ENCODER_CLASS"
echo "  Encoder ckpt   : $ENCODER_CHECKPOINT"
echo "  PD model       : $PD_MODEL       (GPU $DYN_PD_WORKER_GPU)"
echo "  Transfer mode  : $TRANSFER_MODE"
echo "  HTTP port      : $HTTP_PORT"
echo "  NOTE: output text is semantically incorrect (random projection);"
echo "        replace QwenVLExampleEncoder with a learned projector for production."
echo "=================================================================="

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[1/3] Starting frontend (port $HTTP_PORT)..."
python -m dynamo.frontend &

# ── Encoder worker ────────────────────────────────────────────────────────────
echo "[2/3] Starting encoder worker..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${VLLM_NIXL_SIDE_CHANNEL_PORT_ENCODE:-20097} \
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
python -m dynamo.vllm \
    --multimodal-encode-worker \
    --enable-multimodal \
    --model "$ENCODER_MODEL" \
    --served-model-name "$PD_MODEL" \
    --full-prompt-encoder-class "$ENCODER_CLASS" \
    --full-prompt-encoder-checkpoint "$ENCODER_CHECKPOINT" \
    --gpu-memory-utilization 0.5 \
    --embedding-transfer-mode "$TRANSFER_MODE" \
    $EXTRA_ENCODER_ARGS &

# ── Text-only PD worker ───────────────────────────────────────────────────────
echo "[3/3] Starting text-only PD worker (model=$PD_MODEL, GPU=$DYN_PD_WORKER_GPU)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${VLLM_NIXL_SIDE_CHANNEL_PORT_PD:-20098} \
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU \
python -m dynamo.vllm \
    --route-to-encoder \
    --enable-multimodal \
    --enable-prompt-embeds \
    --model "$PD_MODEL" \
    --gpu-memory-utilization 0.7 \
    --embedding-transfer-mode "$TRANSFER_MODE" \
    $EXTRA_PD_ARGS &

echo "=================================================================="
echo "All components started. Waiting for initialization (~30-60s)..."
echo "=================================================================="

wait_any_exit
