<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Speculative Decoding

Speculative decoding is an optimization technique that uses a smaller "draft" model to predict multiple tokens, which are then verified by the main model in parallel. This can significantly reduce latency for autoregressive generation.

## Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| vLLM | ✅ | Eagle3 draft model support |
| SGLang | 🚧 | Not yet documented |
| TensorRT-LLM | 🚧 | Not yet documented |

## Overview

Speculative decoding works by:

1. **Draft phase**: A smaller, faster model generates candidate tokens
2. **Verify phase**: The main model verifies these candidates in a single forward pass
3. **Accept/reject**: Tokens are accepted if they match what the main model would have generated

This approach trades off additional compute for lower latency, as multiple tokens can be generated per forward pass of the main model.

## Quick Start (vLLM + Eagle3)

This guide walks through deploying **Meta-Llama-3.1-8B-Instruct** with **Eagle3** speculative decoding on a single GPU with at least 16GB VRAM.

### Prerequisites

1. Start infrastructure services:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

2. Build and run the vLLM container:

```bash
./container/build.sh --framework VLLM
./container/run.sh -it --framework VLLM --mount-workspace
```

3. Set up Hugging Face access (Meta-Llama-3.1-8B-Instruct is gated):

```bash
export HUGGING_FACE_HUB_TOKEN="your_token_here"
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
```

### Run Speculative Decoding

```bash
cd examples/backends/vllm
bash launch/agg_spec_decoding.sh
```

### Test the Deployment

```bash
curl http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
     "messages": [
       {"role": "user", "content": "Write a poem about why Sakura trees are beautiful."}
     ],
     "max_tokens": 250
   }'
```

## Backend-Specific Guides

| Backend | Guide |
|---------|-------|
| vLLM | [speculative_decoding_vllm.md](./speculative_decoding_vllm.md) |

## See Also

- [vLLM Backend](../../backends/vllm/README.md) - Full vLLM deployment guide
- [Disaggregated Serving](../../design_docs/disagg_serving.md) - Alternative optimization approach
- [Meta-Llama-3.1-8B-Instruct on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
