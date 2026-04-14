#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This is a simple manual load-test helper for planner validation.
# To validate:
# 1. Run a 1P1D disaggregated deployment.
# 2. Start planner with the desired config.
# 3. Run ./load_test.sh <num_requests>.
# Expected behavior is scale up and then back down after the burst.

if [ $# -ne 1 ]; then
    echo "Usage: $0 <number_of_executions>"
    exit 1
fi

executions=$1

echo "Starting $executions non-blocking executions..."

for (( i=1; i<=$executions; i++ )); do
    curl localhost:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "messages": [
                {
                    "role": "user",
                    "content": "Generate a long response to produce sustained planner load."
                }
            ],
            "stream": true,
            "max_tokens": 500
        }' > /dev/null 2>&1 &
done

echo "All $executions executions have been launched!"
