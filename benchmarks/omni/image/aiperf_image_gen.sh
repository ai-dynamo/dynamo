#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generic aiperf benchmark for vLLM-Omni text-to-image generation.
# Assumes the server (Dynamo or native vllm-omni) is already running.
#
# Usage:
#   bash aiperf_image_gen.sh [OPTIONS]
#
# Options:
#   --model <model>           Model to benchmark (default: black-forest-labs/FLUX.2-klein-4B)
#   --url <url>               Server URL (default: http://localhost:8000)
#   --concurrency <n>         Number of concurrent requests (default: 1)
#   --request-count <n>       Total requests to send (default: 10)
#   --warmup-count <n>        Warmup requests before measurement (default: 2)
#   --image-size <WxH>        Generated image size (default: 1024x1024)
#   --response-format <fmt>   Response format: url or b64_json (default: url)
#   --prompt-tokens-mean <n>  Mean synthetic prompt length in tokens (default: 50)
#   --prompt-tokens-stddev <n> Stddev of synthetic prompt length (default: 10)
#   -h, --help                Show this help message
#
# Examples:
#   bash aiperf_image_gen.sh
#   bash aiperf_image_gen.sh --model zai-org/GLM-Image --concurrency 4
#   bash aiperf_image_gen.sh --model Qwen/Qwen-Image --image-size 512x512 --request-count 20

MODEL="black-forest-labs/FLUX.2-klein-4B"
URL="http://localhost:8000"
CONCURRENCY=1
REQUEST_COUNT=10
WARMUP_COUNT=2
IMAGE_SIZE="1024x1024"
RESPONSE_FORMAT="url"
PROMPT_TOKENS_MEAN=50
PROMPT_TOKENS_STDDEV=10
ARTIFACT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)                MODEL=$2;               shift 2 ;;
        --url)                  URL=$2;                 shift 2 ;;
        --concurrency)          CONCURRENCY=$2;         shift 2 ;;
        --request-count)        REQUEST_COUNT=$2;       shift 2 ;;
        --warmup-count)         WARMUP_COUNT=$2;        shift 2 ;;
        --image-size)           IMAGE_SIZE=$2;          shift 2 ;;
        --response-format)      RESPONSE_FORMAT=$2;     shift 2 ;;
        --prompt-tokens-mean)   PROMPT_TOKENS_MEAN=$2;  shift 2 ;;
        --prompt-tokens-stddev) PROMPT_TOKENS_STDDEV=$2; shift 2 ;;
        --artifact-dir)         ARTIFACT_DIR=$2;        shift 2 ;;
        -h|--help)
            sed -n '/^# Usage/,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

AIPERF_ARGS=(
    aiperf profile
    --model "$MODEL"
    --tokenizer gpt2
    --url "$URL"
    --endpoint-type image-generation
    --synthetic-input-tokens-mean "$PROMPT_TOKENS_MEAN"
    --synthetic-input-tokens-stddev "$PROMPT_TOKENS_STDDEV"
    --extra-inputs "size:${IMAGE_SIZE}"
    --extra-inputs "response_format:${RESPONSE_FORMAT}"
    --concurrency "$CONCURRENCY"
    --request-count "$REQUEST_COUNT"
    --warmup-request-count "$WARMUP_COUNT"
    --ui none
    --no-server-metrics
)

if [[ -n "$ARTIFACT_DIR" ]]; then
    AIPERF_ARGS+=(--artifact-dir "$ARTIFACT_DIR")
fi

"${AIPERF_ARGS[@]}"
