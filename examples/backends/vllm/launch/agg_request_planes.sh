#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Parse command-line arguments for request plane mode
REQUEST_PLANE="tcp"  # Default to TCP

while [[ $# -gt 0 ]]; do
    case $1 in
        --tcp)
            REQUEST_PLANE="tcp"
            shift
            ;;
        --http)
            REQUEST_PLANE="http"
            shift
            ;;
        --nats)
            REQUEST_PLANE="nats"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--tcp|--http|--nats]"
            echo "  --tcp   Use TCP request plane (default)"
            echo "  --http  Use HTTP/2 request plane"
            echo "  --nats  Use NATS request plane"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set the request plane mode
export DYN_REQUEST_PLANE=$REQUEST_PLANE
echo "Using request plane mode: $REQUEST_PLANE"

# Frontend
python -m dynamo.frontend --http-port=8000 &

DYN_SYSTEM_PORT=8081 \
DYN_HEALTH_CHECK_ENABLED=true \
    python -m dynamo.vllm --model Qwen/Qwen3-0.6B --enforce-eager --connector none
