#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"

# --single-gpu: Packs all 3 workers (encode, prefill, decode) onto a single GPU.
# This is intended for functional testing with small models (e.g. 2B) where CI
# only has 1 GPU available. It reduces performance by:
#   - Enabling --enforce-eager (disables torch.compile and CUDA graph capture)
#   - Hardcoding P/D KV cache to 512 MB (skips all memory profiling)
#   - Limiting --max-model-len to 4096 tokens on P/D workers
#   - Limiting P/D workers to image=1,video=0,audio=0 (--limit-mm-per-prompt)
#   - Using lower gpu-memory-utilization fractions to share the GPU
SINGLE_GPU=false

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
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with separate Encode/Prefill/Decode workers"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  --single-gpu                  Pack all 3 workers on 1 GPU (for small models, e.g. 2B)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
            echo "  $0 --model Qwen/Qwen3-VL-2B-Instruct --single-gpu"
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


HTTP_PORT="${DYN_HTTP_PORT:-8000}"
echo "=========================================="
echo "Launching Disaggregated Multimodal E/P/D (3 GPUs)"
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
echo "      \"messages\": [{"
echo "        \"role\": \"user\","
echo "        \"content\": ["
echo "          {\"type\": \"text\", \"text\": \"Describe the image.\"},"
echo "          {\"type\": \"image_url\", \"image_url\": {\"url\": \"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg\"}}"
echo "        ]"
echo "      }],"
echo "      \"max_tokens\": 50"
echo "    }'"
echo ""
echo "=========================================="


# Start frontend (no router mode)
echo "Starting frontend..."
# dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
python -m dynamo.frontend &

EXTRA_ARGS=""
PD_EXTRA_ARGS=""

# GPU assignments (override via environment variables)
DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
DYN_PREFILL_WORKER_GPU=${DYN_PREFILL_WORKER_GPU:-1}
DYN_DECODE_WORKER_GPU=${DYN_DECODE_WORKER_GPU:-2}

# GPU memory utilization for workers
DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
DYN_PREFILL_GPU_MEM=${DYN_PREFILL_GPU_MEM:-0.9}
DYN_DECODE_GPU_MEM=${DYN_DECODE_GPU_MEM:-0.9}

# 512 MB KV cache per P/D worker. Setting --kv-cache-memory-bytes bypasses vLLM's
# memory profiling entirely (both language model and multimodal encoder), which avoids
# OOM during profiling when 3 workers share a GPU. 512 MB covers the
# minimum vLLM requires for max_model_len=4096 on Qwen3-VL-2B.
PD_KV_CACHE_BYTES=$((512 * 1024 * 1024))

if [[ "$SINGLE_GPU" == "true" ]]; then
    EXTRA_ARGS="--enforce-eager"
    PD_EXTRA_ARGS="--max-model-len 4096 --kv-cache-memory-bytes $PD_KV_CACHE_BYTES --limit-mm-per-prompt {\"image\":1,\"video\":0,\"audio\":0}"
fi

# Start encode worker
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU python -m dynamo.vllm --multimodal-encode-worker --enable-multimodal --model $MODEL_NAME --gpu-memory-utilization $DYN_ENCODE_GPU_MEM $EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

# Start prefill worker (also handles encode routing via --route-to-encoder)
echo "Starting prefill worker on GPU $DYN_PREFILL_WORKER_GPU (GPU mem: $DYN_PREFILL_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=$DYN_PREFILL_WORKER_GPU python -m dynamo.vllm --multimodal-worker --route-to-encoder --disaggregation-mode prefill --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --gpu-memory-utilization $DYN_PREFILL_GPU_MEM $EXTRA_ARGS $PD_EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

# Start decode worker
echo "Starting decode worker on GPU $DYN_DECODE_WORKER_GPU (GPU mem: $DYN_DECODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20099 \
CUDA_VISIBLE_DEVICES=$DYN_DECODE_WORKER_GPU python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --enable-mm-embeds --model $MODEL_NAME --gpu-memory-utilization $DYN_DECODE_GPU_MEM $EXTRA_ARGS $PD_EXTRA_ARGS --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20082"}' &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait
