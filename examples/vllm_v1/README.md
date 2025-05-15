<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# vLLM Deployment Examples

This directory contains examples for deploying vLLM models in both aggregated and disaggregated configurations.

## Prerequisites

1. Install vLLM:
```bash
# Note: Currently requires installation from main branch
# From vLLM 0.8.6 onwards, you can install directly from wheel
git clone https://github.com/vllm-project/vllm.git
VLLM_USE_PRECOMPILED=1 uv pip install --editable ./vllm/
```

2. Start required services:
```bash
docker compose -f deploy/metrics/docker-compose.yml up -d
```

## Running the Server

### Aggregated Deployment
```bash
cd examples/vllm_v1
dynamo serve graphs.agg:Frontend -f configs/agg.yaml
```

### Disaggregated Deployment
```bash
cd examples/vllm_v1
dynamo serve graphs.disagg:Frontend -f configs/disagg.yaml
```

## Testing the API

Send a test request using curl:
```bash
curl localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "prompt": "In the heart of Eldoria...",
    "stream": false,
    "max_tokens": 30
  }'
```

For more detailed explenations, refer to the main [LLM examples README](../llm/README.md).

