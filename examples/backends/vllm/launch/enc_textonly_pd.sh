#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# POC: Separate VLM encoder + text-only PD worker (EXTERNAL_PROMPT_EMBEDS path).
#
# Architecture:
#   Encoder: runs the VLM's ViT + LM embed_tokens.  For each request it:
#              1. encodes the image through the ViT → image embeddings
#              2. embeds all text tokens via embed_tokens → text embeddings
#              3. splices image embeddings at the <|image_pad|> placeholder
#              4. transfers the full [N_total, hidden] tensor to the PD via NIXL
#   PD:      aggregated vLLM worker running a text-only LLM (e.g. Qwen2.5-1.5B).
#            Receives the spliced tensor, serialises it as base64 into
#            request["prompt_embeds"], and submits to vLLM as EmbedsPrompt.
#   Frontend: standard dynamo.frontend OpenAI-compatible HTTP gateway.
#
# Key env var: DYN_EXTERNAL_PROMPT_EMBEDS=1 activates the spliced-embedding path.
#
# Usage:
#   ./enc_textonly_pd.sh [--encoder-model <hf_id>] [--pd-model <hf_id>]
#                        [--single-gpu] [--transfer-mode local|nixl_write]
#
# Defaults:
#   Encoder: Qwen/Qwen2.5-VL-3B-Instruct (ViT only, ~1 GB VRAM)
#   PD:      Qwen/Qwen2.5-1.5B (text-only, ~3 GB VRAM)
#   GPUs:    encoder=2, pd=1  (avoids GPU 0 which is 4 GB A400)
#   Transfer: local (fastest, same-node only)

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
ENCODER_MODEL="${DYN_ENCODER_MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
# Default PD: smallest available text-only model so the POC runs on a single GPU.
# Qwen3-0.6B is 0.6B params, hidden_size=1024.  The encoder (Qwen2.5-VL-3B)
# produces 2048-dim embeddings; the handler projects 2048→1024 via a random matrix.
PD_MODEL="${DYN_PD_MODEL:-Qwen/Qwen3-0.6B}"
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
Usage: enc_textonly_pd.sh [OPTIONS]

POC: Encoder (VLM ViT) + Text-only PD (pure LLM) via Dynamo embedding routing.

Options:
  --encoder-model <id>    VLM whose ViT is used as encoder (default: Qwen/Qwen2.5-VL-3B-Instruct)
  --pd-model <id>         Text-only LLM for PD (default: Qwen/Qwen2.5-1.5B)
  --single-gpu            Run encoder + PD on the same GPU (GPU 2, enforce-eager)
  --transfer-mode <mode>  Embedding transfer mode: local|nixl_write|nixl_read (default: local)
  -h, --help              Show this help

Examples:
  ./enc_textonly_pd.sh
  ./enc_textonly_pd.sh --encoder-model Qwen/Qwen3-VL-2B-Instruct --pd-model Qwen/Qwen2.5-1.5B
  ./enc_textonly_pd.sh --single-gpu
EOF
            exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=================================================================="
echo "  EXTERNAL_PROMPT_EMBEDS: VLM Encoder + Text-only PD"
echo "  Encoder model : $ENCODER_MODEL (GPU $DYN_ENCODE_WORKER_GPU)"
echo "  PD model      : $PD_MODEL (GPU $DYN_PD_WORKER_GPU)"
echo "  Transfer mode : $TRANSFER_MODE"
echo "  HTTP port     : $HTTP_PORT"
echo "  NOTE: encoder splices text+image embeddings; random projection"
echo "        bridges encoder hidden_dim → PD hidden_dim, so output text"
echo "        is semantically approximate but the pipeline structure is correct."
echo "=================================================================="

# Use TCP so large base64 images do not hit NATS 1 MB limit.
export DYN_REQUEST_PLANE=tcp
# Increase limits for multimodal payloads.
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200
# Activate spliced-embedding path (encoder produces full text+image tensor).
export DYN_EXTERNAL_PROMPT_EMBEDS=1

# ── Extract VLM chat template ─────────────────────────────────────────────────
# The PD must use the VLM's chat template so prompt_token_ids includes
# <|image_pad|> (token 151655) at the image placeholder position.
# The encoder will splice image embeddings at that position.
VLM_TEMPLATE_FILE="$(mktemp /tmp/vlm_chat_template_XXXXXX.jinja)"
python3 - <<PYEOF
from transformers import AutoTokenizer
import sys
tok = AutoTokenizer.from_pretrained("$ENCODER_MODEL", trust_remote_code=True)
with open("$VLM_TEMPLATE_FILE", "w") as f:
    f.write(tok.chat_template or "")
print(f"Saved VLM chat template from $ENCODER_MODEL → $VLM_TEMPLATE_FILE", file=sys.stderr)
PYEOF

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[1/3] Starting frontend (port $HTTP_PORT)..."
python -m dynamo.frontend &

# ── Encoder worker ────────────────────────────────────────────────────────────
# --multimodal-encode-worker sets --disaggregation-mode=encode.
# For EXTERNAL_PROMPT_EMBEDS the encoder also loads the LM embed_tokens and
# produces a fully-spliced [N_total, hidden] tensor instead of raw image embeddings.
#
# --served-model-name $PD_MODEL is REQUIRED: the PD declares needs=[Encode] in
# the Dynamo model registry when --route-to-encoder is set.
echo "[2/3] Starting encoder worker (model=$ENCODER_MODEL → serves as $PD_MODEL, GPU=$DYN_ENCODE_WORKER_GPU)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${VLLM_NIXL_SIDE_CHANNEL_PORT_ENCODE:-20097} \
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
DYN_EXTERNAL_PROMPT_EMBEDS=1 \
python -m dynamo.vllm \
    --multimodal-encode-worker \
    --enable-multimodal \
    --model "$ENCODER_MODEL" \
    --served-model-name "$PD_MODEL" \
    --gpu-memory-utilization 0.5 \
    --embedding-transfer-mode "$TRANSFER_MODE" \
    $EXTRA_ENCODER_ARGS &

# ── Text-only PD worker ───────────────────────────────────────────────────────
# --route-to-encoder: send images to the encoder worker for embedding extraction.
# --enable-multimodal: required so _extract_multimodal_data enters the EXTERNAL
#                      routing path.
# --enable-prompt-embeds: tells vLLM to accept raw embedding tensors (EmbedsPrompt).
# --custom-jinja-template: VLM template ensures prompt_token_ids has <|image_pad|>
#                          so the encoder can splice at the right position.
# No --disaggregation-mode: defaults to aggregated (both prefill+decode in one process).
echo "[3/3] Starting text-only PD worker (model=$PD_MODEL, GPU=$DYN_PD_WORKER_GPU)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=${VLLM_NIXL_SIDE_CHANNEL_PORT_PD:-20098} \
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU \
DYN_EXTERNAL_PROMPT_EMBEDS=1 \
python -m dynamo.vllm \
    --route-to-encoder \
    --enable-multimodal \
    --enable-prompt-embeds \
    --model "$PD_MODEL" \
    --custom-jinja-template "$VLM_TEMPLATE_FILE" \
    --gpu-memory-utilization 0.7 \
    --embedding-transfer-mode "$TRANSFER_MODE" \
    $EXTRA_PD_ARGS &

echo "=================================================================="
echo "All components started. Waiting for initialization (~30-60s)..."
echo ""
echo "Once ready, test with:"
echo "  python -c \""
echo "import requests, base64, sys"
echo "img_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png'"
echo "resp = requests.post('http://localhost:$HTTP_PORT/v1/chat/completions', json={"
echo "  'model': '$PD_MODEL',"
echo "  'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img_url}}, {'type': 'text', 'text': 'Describe this image.'}]}],"
echo "  'max_tokens': 50"
echo "})"
echo "print(resp.json())"
echo "\""
echo "=================================================================="

wait_any_exit
