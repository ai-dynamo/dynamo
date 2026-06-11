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
MODEL_NAME="llava-hf/llava-1.5-7b-hf"

# --single-gpu: Packs all 3 workers (encode, prefill, decode) onto a single GPU.
# This is intended for functional testing with small models (e.g. 2B) where CI
# only has 1 GPU available. It reduces performance by:
#   - Enabling --enforce-eager (disables torch.compile and CUDA graph capture)
#   - Hardcoding P/D KV cache to 512 MB (skips all memory profiling)
#   - Limiting --max-model-len to 4096 tokens on P/D workers
#   - Limiting P/D workers to image=3,video=3,audio=0 (--limit-mm-per-prompt)
#   - Using lower gpu-memory-utilization fractions to share the GPU
SINGLE_GPU=false

# --two-gpu: Packs 3 workers onto 2 GPUs.
# Layout: encode + prefill on GPU 0, decode on GPU 1. Preserves the disagg
# semantic — prefill→decode KV transfer still crosses the GPU boundary via
# NIXL — while halving the GPU footprint vs the default 3-GPU mode. Same
# small-KV defaults as --single-gpu (enforce-eager, 512 MB KV, max-model-len
# 4096, limit-mm-per-prompt 3/3/0) so it's a functional-testing knob, not a
# perf config. Override _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES to grow the
# KV cap when profiling.
TWO_GPU=false

# Parse command line arguments
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
        --two-gpu)
            TWO_GPU=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with separate Encode/Prefill/Decode workers"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  --single-gpu                  Pack all 3 workers on 1 GPU (for small models, e.g. 2B)"
            echo "  --two-gpu                     Pack 3 workers on 2 GPUs (encode+prefill on GPU 0, decode on GPU 1)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
            echo "  $0 --model Qwen/Qwen3-VL-2B-Instruct --single-gpu"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf --two-gpu"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ "$SINGLE_GPU" == "true" && "$TWO_GPU" == "true" ]]; then
    echo "ERROR: --single-gpu and --two-gpu are mutually exclusive" >&2
    exit 2
fi


HTTP_PORT="${DYN_HTTP_PORT:-8000}"
if [[ "$SINGLE_GPU" == "true" ]]; then
    GPU_LABEL="1 GPU"
elif [[ "$TWO_GPU" == "true" ]]; then
    GPU_LABEL="2 GPUs"
else
    GPU_LABEL="3 GPUs"
fi
print_launch_banner --multimodal "Launching Disaggregated Multimodal E/P/D ($GPU_LABEL)" "$MODEL_NAME" "$HTTP_PORT"


# Start frontend (no router mode)
echo "Starting frontend..."
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

# Each worker needs its own system port when tests inject DYN_SYSTEM_PORT{1,2,3}.
unset DYN_SYSTEM_PORT

EXTRA_ARGS=""
PD_EXTRA_ARGS=""
PREFILL_GPU_MEM_ARGS=""
DECODE_GPU_MEM_ARGS=""

# GPU assignments (override via environment variables).
# Modes:
#   --single-gpu : all 3 workers on GPU 0
#   --two-gpu    : encode + prefill on GPU 0, decode on GPU 1
#   default      : encode 0, prefill 1, decode 2 (3 GPUs)
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-0}
    DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-0}
elif [[ "$TWO_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-0}
    DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-1}
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-1}
    DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-2}
fi

# GPU memory utilization for workers.
# NOTE: --kv-cache-memory-bytes (set below for P/D workers) overrides
# --gpu-memory-utilization for KV cache sizing. Per vLLM CacheConfig:
# "kv_cache_memory_bytes (when not-None) ignores gpu_memory_utilization"
# Ref: https://docs.vllm.ai/en/stable/api/vllm/config/cache/
# Therefore _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE has no effect on actual VRAM
# usage when --kv-cache-memory-bytes is set.
if [[ -n "${_PROFILE_PYTEST_VRAM_FRAC_OVERRIDE:-}" ]]; then
    echo "WARNING: _PROFILE_PYTEST_VRAM_FRAC_OVERRIDE is set but has no effect here because" >&2
    echo "  --kv-cache-memory-bytes overrides --gpu-memory-utilization in vLLM." >&2
fi
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.1}
    DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.4}
    DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.4}
elif [[ "$TWO_GPU" == "true" ]]; then
    # encoder + prefill share GPU 0, decode alone on GPU 1
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.1}
    DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.7}
    DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}
else
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
    DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
    DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}
fi

if [[ "$SINGLE_GPU" == "true" || "$TWO_GPU" == "true" ]]; then
    EXTRA_ARGS="--enforce-eager"
    # Default KV cache cap for packed worker layouts.
    #
    # vLLM has a preflight check: KV must hold at least one max-model-len
    # request. For LLaVA-1.5-7b at max-model-len=4096, that's ~2 GiB
    # minimum. 512 MB used to work for older vLLM / smaller max-model-len
    # but vLLM 0.20+ rejects it.
    #
    # The profiler/test framework overrides via _PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES,
    # and gpu_utils.sh builds args.
    : "${_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES:=$((2 * 1024 * 1024 * 1024))}"
    PD_EXTRA_ARGS="--max-model-len 4096 --limit-mm-per-prompt {\"image\":3,\"video\":3,\"audio\":0}"
fi

PD_GPU_MEM_ARGS=$(build_vllm_gpu_mem_args)
if [[ -n "$PD_GPU_MEM_ARGS" ]]; then
    PREFILL_GPU_MEM_ARGS="$PD_GPU_MEM_ARGS"
    DECODE_GPU_MEM_ARGS="$PD_GPU_MEM_ARGS"
else
    PREFILL_GPU_MEM_ARGS="--gpu-memory-utilization $DYN_PREFILL_GPU_MEM"
    DECODE_GPU_MEM_ARGS="--gpu-memory-utilization $DYN_DECODE_GPU_MEM"
fi

VLLM_NIXL_SIDE_CHANNEL_PORT_ENCODE=${VLLM_NIXL_SIDE_CHANNEL_PORT_ENCODE:-20097}
VLLM_NIXL_SIDE_CHANNEL_PORT_PREFILL=${VLLM_NIXL_SIDE_CHANNEL_PORT_PREFILL:-20098}
VLLM_NIXL_SIDE_CHANNEL_PORT_DECODE=${VLLM_NIXL_SIDE_CHANNEL_PORT_DECODE:-20099}
VLLM_ZMQ_PORT_ENCODE=${VLLM_ZMQ_PORT_ENCODE:-20080}
VLLM_ZMQ_PORT_PREFILL=${VLLM_ZMQ_PORT_PREFILL:-20081}
VLLM_ZMQ_PORT_DECODE=${VLLM_ZMQ_PORT_DECODE:-20082}

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
# For LLaVA-1.5-7b the encoder peak is ~13.5 GB regardless of GPU size or fraction
# (verified empirically for the e_pd topology — same load path applies here).
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (--gpu-memory-utilization $DYN_ENCODE_GPU_MEM)..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT1:-8081} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT_ENCODE \
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME --gpu-memory-utilization $DYN_ENCODE_GPU_MEM $EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_ZMQ_PORT_ENCODE}\"}" &

# Start prefill worker (also handles encode routing via --route-to-encoder)
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU (${PREFILL_GPU_MEM_ARGS})..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT2:-8082} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT_PREFILL \
CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU python -m dynamo.vllm --route-to-encoder --disaggregation-mode prefill --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $PREFILL_GPU_MEM_ARGS $EXTRA_ARGS $PD_EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_ZMQ_PORT_PREFILL}\"}" &

# Start decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU (${DECODE_GPU_MEM_ARGS})..."
DYN_SYSTEM_PORT=${DYN_SYSTEM_PORT3:-8083} \
VLLM_NIXL_SIDE_CHANNEL_PORT=$VLLM_NIXL_SIDE_CHANNEL_PORT_DECODE \
CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU python -m dynamo.vllm  --disaggregation-mode decode --enable-multimodal --enable-mm-embeds --model $MODEL_NAME $DECODE_GPU_MEM_ARGS $EXTRA_ARGS $PD_EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:${VLLM_ZMQ_PORT_DECODE}\"}" &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Exit on first worker failure; kill 0 in the EXIT trap tears down the rest
wait_any_exit
