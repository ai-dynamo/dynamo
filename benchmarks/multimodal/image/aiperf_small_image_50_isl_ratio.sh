#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CONCURRENCY=1

# 500 * 333 pixels image
IMG_URL="http://images.cocodataset.org/test2017/000000000183.jpg"

# Create a JSONL file with 30 identical small image URLs
# NOTE: any kind of caching can significantly affect the benchmark results,
# should make sure what you are doing.
# ~ 11 tokens
DUMMY_PROMPT="This is a prompt to describe the image content briefly."
for i in {1..1500}; do
    DUMMY_PROMPT+=" This is a prompt to describe the image content briefly."
done
echo '{"texts": ["'"$DUMMY_PROMPT"'"], "images": ["'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'","'"$IMG_URL"'"]}' \
    > data_small.jsonl
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
    --streaming --request-count 100 --warmup-request-count 5 \
    --concurrency $CONCURRENCY --osl 1 \
    --input-file data_small.jsonl \
    --custom-dataset-type single_turn --ui none \
    --no-server-metrics
