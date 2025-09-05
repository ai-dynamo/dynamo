#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get port from first argument, default to 8080 if not provided
PORT=${1:-8080}

curl -X POST http://localhost:${PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Answer in 5 words."},
        {"role": "user", "content": "What is 2+2?"}
    ],
    "stream": true,
    "max_tokens": 10,
    "ignore_eos": true
    }'