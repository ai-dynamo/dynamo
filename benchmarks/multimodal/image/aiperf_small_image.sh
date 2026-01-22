#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"
CONCURRENCY=1

python local_media_server.py \
    --image test.jpg:http://images.cocodataset.org/test2017/000000155781.jpg &

# Wait for the server to start
for i in {1..10}; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" localhost:8233/test.jpg)
    if [[ "$HTTP_CODE" -eq 200 ]]; then
        echo "Server is responding with HTTP 200."
        break
    else
        echo "Server did not respond with HTTP 200. Response code: $HTTP_CODE. Retrying in 1 second..."
        sleep 1
    fi
    if [[ $i -eq 10 ]]; then
        echo "Server did not respond with HTTP 200 after 10 attempts. Exiting."
        exit 1
    fi
done

# Create a JSONL file with 12 identical small image URLs
# NOTE: any kind of caching can significantly affect the benchmark results,
# should make sure what you are doing.
echo '{"images": ["http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg","http://localhost:8233/test.jpg"]}' \
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
    --synthetic-input-tokens-mean 1 --synthetic-input-tokens-stddev 0 \
    --streaming --request-count 100 --warmup-request-count 2 \
    --concurrency $CONCURRENCY --osl 1 \
    --input-file data_small.jsonl \
    --custom-dataset-type single_turn --ui none \
    --no-server-metrics
