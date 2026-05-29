#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
source "$SCRIPT_DIR/../../../common/gpu_utils.sh"
source "$SCRIPT_DIR/../../../common/launch_utils.sh"

# Use TCP transport for multimodal workloads (base64 images can exceed NATS 1MB limit)
export DYN_REQUEST_PLANE=tcp

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
SINGLE_GPU=false

# Parse command line arguments
# All extra arguments are passed through to the PD worker's dynamo.vllm
# (which routes them to Dynamo or vLLM as appropriate).
EXTRA_PD_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [EXTRA_ARGS...]"
            echo ""
            echo "Disaggregated multimodal serving with separate Encode and aggregated PD worker"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  --single-gpu                  Run encode and PD workers on the same GPU (for small models, e.g. 2B)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "All additional arguments are passed through to the PD worker's dynamo.vllm."
            echo "Dynamo args (e.g. --multimodal-embedding-cache-capacity-gb) and"
            echo "vLLM engine args (e.g. --no-enable-prefix-caching) are automatically routed."
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
            echo "  $0 --no-enable-prefix-caching --multimodal-embedding-cache-capacity-gb 2"
            echo "  $0 --model Qwen/Qwen2-VL-2B-Instruct --single-gpu"
            echo ""
            exit 0
            ;;
        *)
            EXTRA_PD_ARGS+=("$1")
            shift
            ;;
    esac
done


PD_MAX_MODEL_LEN="16384"


HTTP_PORT="${DYN_HTTP_PORT:-8000}"
if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_LABEL="1 GPU"
else
    GPU_LABEL="2 GPUs"
fi
print_launch_banner --multimodal "Launching Disaggregated Multimodal E+PD ($GPU_LABEL)" "$MODEL_NAME" "$HTTP_PORT"


# Start frontend (no router mode)
echo "Starting frontend..."
python -m dynamo.frontend &

# Each worker needs its own system port when tests inject DYN_SYSTEM_PORT{1,2}.
unset DYN_SYSTEM_PORT

EXTRA_ARGS=""
PD_GPU_MEM_ARGS=""

# GPU assignments (override via environment variables)
# In single-GPU mode both workers share the same GPU.
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-0}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.1}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.7}
    EXTRA_ARGS="--enforce-eager --max-model-len $PD_MAX_MODEL_LEN"
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.9}
fi

# PD worker memory args: when _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES is set,
# build_vllm_gpu_mem_args returns "--kv-cache-memory-bytes N --gpu-memory-utilization 0.01",
# which overrides $DYN_PD_GPU_MEM (the env knob becomes a no-op in that case).
# Without the override env, $DYN_PD_GPU_MEM is the live cap.
PD_GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -z "$PD_GPU_MEM_ARGS" ]]; then
    PD_GPU_MEM_ARGS="--gpu-memory-utilization $DYN_PD_GPU_MEM"
fi

# Start encode worker.
#
# NOTE: encoder VRAM is STATIC, set by the model — $DYN_ENCODE_GPU_MEM
# (--gpu-memory-utilization) is effectively a no-op for non-Qwen-VL models.
# dynamo's EncodeWorkerHandler only consumes engine_args.enforce_eager;
# load_vision_model (components/src/dynamo/vllm/multimodal_utils/model.py)
# branches on family:
#   - Qwen-VL: vLLM mm_encoder_only=True path, hardcoded gpu_memory_utilization=0.2
#     and kv_cache_memory_bytes=64MiB inside the function — only the vision tower
#     loads (small).
#   - Everything else (e.g. LLaVA-1.5-7b): AutoModel.from_pretrained(..., fp16)
#     loads the FULL model weights, then .visual is extracted. The fraction here
#     is ignored.
# Verified empirically for LLaVA-1.5-7b on RTX 6000 Ada (48 GB):
# GPU0 peak ~13.5 GB at DYN_ENCODE_GPU_MEM=0.05, 0.5, 0.9 (identical within jitter).
# The static peak is bounded by the model's fp16 weights (~14 GB), independent
# of GPU size. So sizing for this script: encoder needs ~14 GB free per worker GPU.
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (--gpu-memory-utilization $DYN_ENCODE_GPU_MEM)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
python -m dynamo.vllm \
  --multimodal-encode-worker \
  --enable-multimodal \
  --model "$MODEL_NAME" \
  --gpu-memory-utilization "$DYN_ENCODE_GPU_MEM" \
  $EXTRA_ARGS &

# Start PD worker (aggregated prefill+decode, routes to encoder for embeddings)
echo "Starting PD worker on GPU $DYN_PD_WORKER_GPU (${PD_GPU_MEM_ARGS})..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU \
python -m dynamo.vllm \
  --route-to-encoder \
  --enable-multimodal \
  --enable-mm-embeds \
  --model "$MODEL_NAME" \
  $PD_GPU_MEM_ARGS \
  $EXTRA_ARGS \
  "${EXTRA_PD_ARGS[@]}" &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
