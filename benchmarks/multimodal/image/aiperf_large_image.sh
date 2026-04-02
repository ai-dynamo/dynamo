#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CONCURRENCY=1

IMG_URL="https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg"

# Create a JSONL file with 11 identical large image URLs
# NOTE: any kind of caching can significantly affect the benchmark results,
# should make sure what you are doing.
echo '{"images": ["'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'", "'"$IMG_URL"'"]}' \
    > data_large.jsonl
echo "This benchmark uses duplicate image urls, so any kind of caching can significantly affect the benchmark results, please make sure the caching setting is properly configured for your experiment."

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --concurrency)
            CONCURRENCY=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model <model_name> Specify the model to use (default: $MODEL_NAME)"
            echo "  --concurrency <level> Specify the concurrency level to use (default: $CONCURRENCY)"
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

aiperf profile -m $MODEL_NAME --endpoint-type chat \
    --synthetic-input-tokens-mean 1 --synthetic-input-tokens-stddev 0 \
    --streaming --request-count 20 --warmup-request-count 2 \
    --concurrency $CONCURRENCY --osl 1 \
    --input-file data_large.jsonl \
    --custom-dataset-type single_turn --ui none \
    --no-server-metrics
