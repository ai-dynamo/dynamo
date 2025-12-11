#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -ex

# Default values
HEAD_NODE=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --head-node)
            HEAD_NODE=1
            shift 1
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with separate Prefill/Decode workers for Llama 4"
            echo ""
            echo "Options:"
            echo "  --head-node          Run as head node. Head node will run the HTTP server, processor and prefill worker."
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
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

trap 'echo Cleaning up...; kill 0' EXIT

MODEL_NAME="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

if [[ $HEAD_NODE -eq 1 ]]; then
    # run ingress
    # dynamo.frontend accepts either --http-port flag or DYN_HTTP_PORT env var (defaults to 8000)
    python -m dynamo.frontend &

    # run processor
    python -m dynamo.vllm --multimodal-processor --enable-multimodal --model $MODEL_NAME --mm-prompt-template "<|image|>\n<prompt>" &

    # Llama 4 doesn't support image embedding input, so the prefill worker will also
    # handle image encoding inline.
    # run prefill worker
    VLLM_NIXL_SIDE_CHANNEL_PORT=20097 python -m dynamo.vllm --multimodal-encode-prefill-worker --is-prefill-worker --enable-multimodal --model $MODEL_NAME --tensor-parallel-size=8 --max-model-len=208960 --gpu-memory-utilization 0.80 --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &
else
    # run decode worker on non-head node
    VLLM_NIXL_SIDE_CHANNEL_PORT=20098 python -m dynamo.vllm --multimodal-decode-worker --enable-multimodal --model $MODEL_NAME --tensor-parallel-size=8 --max-model-len=208960 --gpu-memory-utilization 0.80 --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &
fi

# Wait for all background processes to complete
wait
