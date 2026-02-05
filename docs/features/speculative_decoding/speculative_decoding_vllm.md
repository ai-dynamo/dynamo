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

# Speculative Decoding with vLLM

Using Speculative Decoding with the vLLM backend.

> **See also**: [Speculative Decoding Overview](./README.md) for cross-backend documentation.

## Prerequisites

- vLLM container with Eagle3 support
- GPU with at least 16GB VRAM
- Hugging Face access token (for gated models)

## Quick Start: Meta-Llama-3.1-8B-Instruct + Eagle3

This guide walks through deploying **Meta-Llama-3.1-8B-Instruct** with **Eagle3** speculative decoding on a single node.

### Step 1: Set Up Your Docker Environment

First, initialize a Docker container using the vLLM backend. See the [vLLM Quickstart Guide](../../backends/vllm/README.md#vllm-quick-start) for details.

```bash
# Launch infrastructure services
docker compose -f deploy/docker-compose.yml up -d

# Build the container
./container/build.sh --framework VLLM

# Run the container
./container/run.sh -it --framework VLLM --mount-workspace
```

### Step 2: Get Access to the Llama-3 Model

The **Meta-Llama-3.1-8B-Instruct** model is gated. Request access on Hugging Face:
[Meta-Llama-3.1-8B-Instruct repository](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

Approval time varies depending on Hugging Face review traffic.

Once approved, set your access token inside the container:

```bash
export HUGGING_FACE_HUB_TOKEN="insert_your_token_here"
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
```

### Step 3: Run Aggregated Speculative Decoding

```bash
# Requires only one GPU
cd examples/backends/vllm
bash launch/agg_spec_decoding.sh
```

Once the weights finish downloading, the server will be ready for inference requests.

### Step 4: Test the Deployment

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

### Example Output

```json
{
  "id": "cmpl-3e87ea5c-010e-4dd2-bcc4-3298ebd845a8",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "In cherry blossom's gentle breeze ... A delicate balance of life and death, as petals fade, and new life breathes."
      },
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "usage": {
    "prompt_tokens": 16,
    "completion_tokens": 250,
    "total_tokens": 266
  }
}
```

## Configuration

Speculative decoding in vLLM uses Eagle3 as the draft model. The launch script configures:

- Target model: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Draft model: Eagle3 variant
- Aggregated serving mode

See `examples/backends/vllm/launch/agg_spec_decoding.sh` for the full configuration.

## Limitations

- Currently only supports Eagle3 as the draft model
- Requires compatible model architectures between target and draft

## See Also

| Document | Path |
|----------|------|
| Speculative Decoding Overview | [README.md](./README.md) |
| vLLM Backend Guide | [vLLM README](../../backends/vllm/README.md) |
| Meta-Llama-3.1-8B-Instruct | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
