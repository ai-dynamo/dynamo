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
            echo "Options:"
            echo "  --head-node          Run as head node. Head node will run the HTTP server, processor and prefill worker."
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

trap 'echo Cleaning up...; kill 0' EXIT

if [[ $HEAD_NODE -eq 1 ]]; then
    # run ingress
    dynamo run in=http out=dyn &

    # run processor
    python3 components/processor.py --model llava-hf/llava-1.5-7b-hf --prompt-template "USER: <image>\n<prompt> ASSISTANT:" &
    # LLama 4 doesn't support image embedding input, so the prefill worker will also
    # handle image encoding.
    # run EP/D workers
    python3 components/worker.py --model llava-hf/llava-1.5-7b-hf --worker-type encode_prefill --enable-disagg &
else
    # run decode worker on non-head node
    python3 components/worker.py --model llava-hf/llava-1.5-7b-hf --worker-type decode --enable-disagg
fi