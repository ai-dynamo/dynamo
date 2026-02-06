#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

# Default values
HEAD_NODE=0
MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
EXTRA_ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --head-node)
            HEAD_NODE=1
            shift 1
            ;;
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with separate Prefill/Decode workers for Llama 4"
            echo ""
            echo "Options:"
            echo "  --head-node          Run as head node. Head node will run the HTTP server, processor and prefill worker."
            echo "  --model <model_name> Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help           Show this help message"
            echo ""
            echo "Examples:"
            echo "  # On head node:"
            echo "  $0 --head-node"
            echo ""
            echo "  # On worker node (requires NATS_SERVER and ETCD_ENDPOINTS pointing to head node):"
            echo "  $0"
            echo ""
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

# Use TCP transport to avoid NATS payload limits for multimodal
export DYN_REQUEST_PLANE=tcp

# Configure model-specific args
MODEL_SPECIFIC_ARGS=""
if [[ "$MODEL_NAME" == "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" ]]; then
    MODEL_SPECIFIC_ARGS="--tensor-parallel-size=8 --max-model-len=208960 --gpu-memory-utilization 0.80"
fi

if [[ $HEAD_NODE -eq 1 ]]; then
    # run ingress
    # dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
    python -m dynamo.frontend &

    # run processor (CPU-only to avoid competing for GPU memory with workers)
    CUDA_VISIBLE_DEVICES="" \
    python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME &

    # Prefill worker handles prompt processing and image encoding
    # Uses all 8 GPUs for tensor-parallel
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
    python -m dynamo.vllm \
        --enable-multimodal \
        --model $MODEL_NAME \
        --is-prefill-worker \
        $MODEL_SPECIFIC_ARGS \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' \
        "${EXTRA_ARGS[@]}" &
else
    # run decode worker on non-head node
    # Uses all 8 GPUs for tensor-parallel
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
    python -m dynamo.vllm \
        --enable-multimodal \
        --model $MODEL_NAME \
        $MODEL_SPECIFIC_ARGS \
        --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' \
        "${EXTRA_ARGS[@]}" &
fi

# Wait for all background processes to complete
wait
