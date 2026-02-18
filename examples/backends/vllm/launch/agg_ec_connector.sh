#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="llava-hf/llava-1.5-7b-hf"
EC_CONNECTOR_BACKEND="DynamoEcConnector"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Aggregated multimodal serving with ECConnector (ec_both mode)"
            echo ""
            echo "This script launches:"
            echo "  - Frontend server"
            echo "  - Aggregated multimodal worker (ec_both: produces and consumes encoder cache)"
            echo ""
            echo "Options:"
            echo "  --model <model_name>  Specify the VLM model to use (default: $MODEL_NAME)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
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

echo "=================================================="
echo "Aggregated Multimodal Serving (ECConnector ec_both)"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "ECConnector Backend: $EC_CONNECTOR_BACKEND"
echo "=================================================="

# GPU assignment (override via environment variable)
DYN_WORKER_GPU=${DYN_WORKER_GPU:-0}

# GPU memory utilization
DYN_GPU_MEM=${DYN_GPU_MEM:-0.85}

# Start frontend
echo "Starting frontend..."
python -m dynamo.frontend &

# Start aggregated multimodal worker (ec_both: produces and consumes encoder cache)
echo "Starting aggregated multimodal worker (ec_both) on GPU $DYN_WORKER_GPU (mem: $DYN_GPU_MEM)..."
CUDA_VISIBLE_DEVICES=$DYN_WORKER_GPU python -m dynamo.vllm \
    --multimodal-worker \
    --enable-multimodal \
    --model $MODEL_NAME \
    --enable-mm-embeds \
    --connector none \
    --enforce-eager \
    --gpu-memory-utilization $DYN_GPU_MEM \
    --ec-transfer-config "{\"ec_connector\":\"$EC_CONNECTOR_BACKEND\",\"ec_role\":\"ec_both\"}" &

# Wait for all background processes to complete
wait
