#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Multimodal E/P/D: separate encoder, prefill, and decode workers.
# Default: 3 GPUs (one per worker). Use --single-gpu to co-locate all on GPU 0.

set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"   # build_gpu_mem_args
source "$SCRIPT_DIR/../../../common/launch_utils.sh" # print_launch_banner, wait_any_exit

# Default values
MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CHAT_TEMPLATE="qwen2-vl"
PROVIDED_CHAT_TEMPLATE=""

# --single-gpu: Packs all workers (encode, prefill, decode) onto a single GPU.
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
            echo "  --single-gpu         Pack all workers on 1 GPU (for small models, e.g. 2B)"
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
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-0}
    DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-0}
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-1}
    DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-2}
fi

# Per-worker CUDA_VISIBLE_DEVICES pinning. In single-gpu mode, inherit from parent
# (the parallel test runner sets CUDA_VISIBLE_DEVICES); overriding would defeat GPU assignment.
if [[ "$SINGLE_GPU" == "true" ]]; then
    _ENCODE_CUDA_PIN=""
    _PREFILL_CUDA_PIN=""
    _DECODE_CUDA_PIN=""
else
    _ENCODE_CUDA_PIN="CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU"
    _PREFILL_CUDA_PIN="CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU"
    _DECODE_CUDA_PIN="CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU"
fi

# GPU memory fractions for workers (used with --mem-fraction-static)
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}

GPU_MEM_ARGS=$(build_gpu_mem_args sglang --workers-per-gpu 3)

ENCODE_EXTRA_ARGS=""
PREFILL_EXTRA_ARGS=""
DECODE_EXTRA_ARGS=""

if [[ "$SINGLE_GPU" == "true" ]]; then
    # 3 workers share one GPU. --max-total-tokens caps the KV cache to a small
    # functional-test size so the last worker can initialize without OOM.
    # --context-length keeps the per-request token pool allocation small.
    ENCODE_EXTRA_ARGS=""
    PREFILL_EXTRA_ARGS="--mem-fraction-static ${DYN_PREFILL_GPU_MEM} --delete-ckpt-after-loading --max-running-requests 2 --context-length 2048 --max-total-tokens 1024 $GPU_MEM_ARGS"
    DECODE_EXTRA_ARGS="--mem-fraction-static ${DYN_DECODE_GPU_MEM} --delete-ckpt-after-loading --max-running-requests 2 --context-length 2048 --max-total-tokens 1024 $GPU_MEM_ARGS"
fi

# Prevent port collisions: the test framework exports DYN_SYSTEM_PORT which all
# child processes would inherit. Unset it so only workers that need it set their own.
unset DYN_SYSTEM_PORT

DISAGG_BOOTSTRAP_PORT="${DYN_DISAGG_BOOTSTRAP_PORT:-12345}"

HTTP_PORT="${DYN_HTTP_PORT:-8000}"
print_launch_banner --multimodal "Launching Disaggregated Multimodal E/P/D" "$MODEL_NAME" "$HTTP_PORT"

# run ingress
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python3 -m dynamo.frontend &

# run SGLang multimodal encode worker (frontend-facing: encodes images, routes to worker)
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
env ${_ENCODE_CUDA_PIN:+"$_ENCODE_CUDA_PIN"} python3 -m dynamo.sglang --multimodal-encode-worker --model-path "$MODEL_NAME" $SERVED_MODEL_ARG --chat-template "$CHAT_TEMPLATE" --skip-tokenizer-init $ENCODE_EXTRA_ARGS &

if [[ "$SINGLE_GPU" == "true" ]]; then
    # Wait for encode worker to initialize before starting prefill worker.
    # This prevents workers from competing for GPU memory simultaneously, which can cause OOM.
    echo "Waiting for encode worker to initialize..."
    sleep 5
fi

# run SGLang multimodal prefill worker
# TODO: Remove disable-radix-cache once the issue is fixed.
# See https://github.com/sgl-project/sglang/pull/11203.
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU (GPU mem: $DYN_PREFILL_GPU_MEM)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
env ${_PREFILL_CUDA_PIN:+"$_PREFILL_CUDA_PIN"} python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  $SERVED_MODEL_ARG \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disable-radix-cache \
  --disaggregation-transfer-backend nixl \
  $PREFILL_EXTRA_ARGS &

if [[ "$SINGLE_GPU" == "true" ]]; then
    # Wait for prefill worker to initialize before starting decode worker.
    echo "Waiting for prefill worker to initialize..."
    sleep 5
fi

# run SGLang multimodal decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU (GPU mem: $DYN_DECODE_GPU_MEM)..."
env ${_DECODE_CUDA_PIN:+"$_DECODE_CUDA_PIN"} python3 -m dynamo.sglang \
  --multimodal-worker \
  --model-path "$MODEL_NAME" \
  $SERVED_MODEL_ARG \
  --page-size 16 \
  --tp 1 \
  --trust-remote-code \
  --skip-tokenizer-init \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port "$DISAGG_BOOTSTRAP_PORT" \
  --host 0.0.0.0 \
  --disaggregation-transfer-backend nixl \
  $DECODE_EXTRA_ARGS &

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
