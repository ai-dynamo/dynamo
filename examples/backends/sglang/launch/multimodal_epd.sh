#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multimodal E/PD: separate vision encoder (GPU 0) + combined PD worker (GPU 1).
# GPUs: 2 (or 1 with --single-gpu)

# Setup cleanup trap
cleanup() {
    echo "Cleaning up background processes..."
    kill $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    wait $DYNAMO_PID $PREFILL_PID 2>/dev/null || true
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CHAT_TEMPLATE="qwen2-vl"
PROVIDED_CHAT_TEMPLATE=""

# --single-gpu: Packs both workers (encode, PD) onto a single GPU.
# This is intended for functional testing with small models (e.g. 2B) where CI
# only has 1 GPU available. It uses lower mem-fraction-static values to share the GPU
# and enables memory-saving options.
SINGLE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --served-model-name)
            SERVED_MODEL_NAME=$2
            shift 2
            ;;
        --chat-template)
            PROVIDED_CHAT_TEMPLATE=$2
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --served-model-name <served_model_name> Specify the served model name to use (default: empty)"
            echo "  --chat-template <template> Specify the SGLang chat template to use (default: $CHAT_TEMPLATE)"
            echo "  --single-gpu         Pack both workers on 1 GPU (for small models, e.g. 2B)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set CHAT_TEMPLATE if provided
if [[ -n "$PROVIDED_CHAT_TEMPLATE" ]]; then
    CHAT_TEMPLATE="$PROVIDED_CHAT_TEMPLATE"
fi

# Prepare served-model-name argument if provided
SERVED_MODEL_ARG=""
if [[ -n "$SERVED_MODEL_NAME" ]]; then
    SERVED_MODEL_ARG="--served-model-name $SERVED_MODEL_NAME"
fi

# GPU assignments (override via environment variables)
DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
DYN_WORKER_GPU=${DYN_WORKER_GPU:-1}

# GPU memory fractions for workers (used with --mem-fraction-static)
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
DYN_WORKER_GPU_MEM=${DYN_WORKER_GPU_MEM:-0.9}

ENCODE_EXTRA_ARGS=""
WORKER_EXTRA_ARGS=""

if [[ "$SINGLE_GPU" == "true" ]]; then
    ENCODE_EXTRA_ARGS="--mem-fraction-static ${DYN_ENCODE_GPU_MEM}"
    WORKER_EXTRA_ARGS="--mem-fraction-static ${DYN_WORKER_GPU_MEM} --enable-memory-saver --delete-ckpt-after-loading --max-running-requests 2 --chunked-prefill-size 4096 --max-prefill-tokens 4096"
fi

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
echo "=========================================="
if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_LABEL="1 GPU"
else
    GPU_LABEL="2 GPUs"
fi
echo "Launching Multimodal E/PD Workers ($GPU_LABEL)"
echo "=========================================="
echo "Model:       $MODEL_NAME"
echo "Frontend:    http://localhost:$HTTP_PORT"
echo "=========================================="
echo ""
echo "Example test command:"
echo ""
echo "  curl http://localhost:${HTTP_PORT}/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"${MODEL_NAME}\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": ["
echo "        {\"type\": \"text\", \"text\": \"Explain why Roger Federer is considered one of the greatest tennis players of all time\"},"
echo "        {\"type\": \"image_url\", \"image_url\": {\"url\": \"http://images.cocodataset.org/test2017/000000155781.jpg\"}}"
echo "      ]}],"
echo "      \"max_tokens\": 50"
echo "    }'"
echo ""
echo "=========================================="

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &
DYNAMO_PID=$!

# run SGLang multimodal processor
python3 -m dynamo.sglang --multimodal-processor --model-path "$MODEL_NAME" $SERVED_MODEL_ARG --chat-template "$CHAT_TEMPLATE" &

# run SGLang multimodal encode worker
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python3 -m dynamo.sglang --multimodal-encode-worker --model-path "$MODEL_NAME" $SERVED_MODEL_ARG --chat-template "$CHAT_TEMPLATE" $ENCODE_EXTRA_ARGS &

if [[ "$SINGLE_GPU" == "true" ]]; then
    # Wait for encode worker to initialize before starting PD worker.
    # This prevents both workers from competing for GPU memory simultaneously, which can cause OOM.
    echo "Waiting for encode worker to initialize..."
    sleep 5
fi

# run SGLang multimodal inference worker
# TODO: Remove disable-radix-cache once the issue is fixed.
# See https://github.com/sgl-project/sglang/pull/11203.
echo "Starting PD worker on GPU $DYN_WORKER_GPU (GPU mem: $DYN_WORKER_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_WORKER_GPU python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  $SERVED_MODEL_ARG \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disable-radix-cache \
  --disaggregation-transfer-backend nixl \
  $WORKER_EXTRA_ARGS &

# Wait for all background processes to complete
wait
