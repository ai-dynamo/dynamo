#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Aggregated serving with a CustomEncoder.
#
# Architecture: single aggregated worker
#   - Frontend: dynamo.frontend (OpenAI-compatible HTTP gateway)
#   - Worker:   dynamo.vllm with a CustomEncoder loaded in-process
#
# The CustomEncoder is called for each multimodal request:
#   1. encoder.encode(image_urls) → list[(n_visual_tokens, lm_hidden_dim)]
#   2. Dynamo builds a mixed token-ids/embeds EmbedsPrompt: vLLM embeds the text
#      itself and substitutes the encoder's image embeds at the placeholder span.
#   3. vLLM engine runs transformer layers on the EmbedsPrompt.
#
# No separate encode worker, no NIXL inter-process transfer.
#
# The default model is a text-only LM (Qwen2.5-1.5B-Instruct) — the standard
# topology for this feature (custom encoder + stock LM), served with a minimal
# custom chat template that emits the image placeholder. The default encoder
# (HitchhikersVisionEncoder) fakes an image as the embeddings of a fixed phrase
# so the path can be checked for semantic correctness; replace it with a real
# CustomEncoder subclass for production.
#
# Usage:
#   ./agg_custom.sh [--model <hf_id>] [--encoder-class <dotted.ClassName>]
#                    [--gpu <index>]
#
# Defaults:
#   --model:         Qwen/Qwen2.5-1.5B-Instruct
#   --encoder-class: examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder
#   --gpu:           0

set -e
trap 'echo "Cleaning up..."; kill 0' EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../common/launch_utils.sh"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${DYN_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ENCODER_CLASS="${DYN_ENCODER_CLASS:-examples.custom_encoder.hitchhikers_vision_encoder.HitchhikersVisionEncoder}"
# Precedence: explicit DYN_WORKER_GPU/--gpu > harness-set CUDA_VISIBLE_DEVICES
# (e.g. the pytest profile runner) > device 0 (portable default; on a generic
# host or single-GPU container, GPU 0 is the natural choice).
WORKER_GPU="${DYN_WORKER_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
HTTP_PORT="${DYN_HTTP_PORT:-8000}"
MAX_MODEL_LEN="${DYN_MAX_MODEL_LEN:-16384}"
# A text-only LM's own chat template can't render image content parts, so default
# to the bundled minimal template that emits the <|image_pad|> placeholder.
CUSTOM_JINJA_TEMPLATE="${DYN_CUSTOM_JINJA_TEMPLATE:-$SCRIPT_DIR/../templates/qwen_vl.jinja}"
EXTRA_ARGS=()

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL=$2; shift 2 ;;
        --encoder-class)
            ENCODER_CLASS=$2; shift 2 ;;
        --gpu)
            WORKER_GPU=$2; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage: agg_custom.sh [OPTIONS]

Aggregated serving with a CustomEncoder (no separate encode worker).

Options:
  --model <id>           LLM checkpoint (default: Qwen/Qwen2.5-1.5B-Instruct)
  --encoder-class <path> Dotted module.ClassName for CustomEncoder subclass
  --gpu <index>          GPU index for the worker (default: 0)
  -h, --help             Show this help

Environment variables:
  DYN_MODEL                      LLM model checkpoint
  DYN_ENCODER_CLASS              Dotted class path for the CustomEncoder subclass
  DYN_WORKER_GPU                 GPU index (default: 0)
  DYN_VLLM_GPU_MEMORY_UTILIZATION vLLM memory fraction (default: 0.8)
  DYN_CUSTOM_JINJA_TEMPLATE      Path to a .jinja chat template (defaults to the
                                 bundled templates/qwen_vl.jinja)
  DYN_CUSTOM_PHRASE             Phrase the HitchhikersEncoder embeds as the "image"
EOF
            exit 0 ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# VRAM sizing: honors the profiler/test-harness override
# (_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES) when set, else falls back to a sane
# local default. Required for VRAM-safe parallel test scheduling.
GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
[[ -z "$GPU_MEM_ARGS" ]] && GPU_MEM_ARGS="--gpu-memory-utilization ${DYN_VLLM_GPU_MEMORY_UTILIZATION:-0.8}"

print_launch_banner --no-curl "CustomEncoder — Aggregated Serving" "$MODEL" "$HTTP_PORT" \
    "Worker GPU:  $WORKER_GPU" \
    "Encoder:     $ENCODER_CLASS" \
    "Jinja tmpl:  ${CUSTOM_JINJA_TEMPLATE:-(model default)}"

export DYN_REQUEST_PLANE=tcp
export DYN_TCP_MAX_MESSAGE_SIZE=209715200
export DYN_HTTP_BODY_LIMIT_MB=200

# ── Frontend ──────────────────────────────────────────────────────────────────
echo "[1/2] Starting frontend (port $HTTP_PORT)..."
python -m dynamo.frontend &

# ── Aggregated worker ─────────────────────────────────────────────────────────
echo "[2/2] Starting aggregated worker (model=$MODEL, GPU=$WORKER_GPU)..."
JINJA_ARG=()
[[ -n "$CUSTOM_JINJA_TEMPLATE" ]] && JINJA_ARG=(--custom-jinja-template "$CUSTOM_JINJA_TEMPLATE")
CUDA_VISIBLE_DEVICES=$WORKER_GPU \
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT:-8081} \
python -m dynamo.vllm \
    --model "$MODEL" \
    --custom-encoder-class "$ENCODER_CLASS" \
    --enable-multimodal \
    --enable-prompt-embeds \
    --max-model-len "$MAX_MODEL_LEN" \
    $GPU_MEM_ARGS \
    "${JINJA_ARG[@]}" \
    "${EXTRA_ARGS[@]}" &

echo "=================================================================="
echo "All components started. Waiting for initialization (~30-60s)..."
echo "=================================================================="

wait_any_exit
