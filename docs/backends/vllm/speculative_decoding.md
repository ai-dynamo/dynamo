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

> **Note**: This content has moved to [Speculative Decoding with vLLM](../../features/speculative_decoding/speculative_decoding_vllm.md).
> See [Speculative Decoding Overview](../../features/speculative_decoding/README.md) for cross-backend documentation.
> This file will be removed in a future release.

# Running **Meta-Llama-3.1-8B-Instruct** with Speculative Decoding (Eagle3)

This guide walks through how to deploy **Meta-Llama-3.1-8B-Instruct** using **aggregated speculative decoding** with **Eagle3** on a single node.
Since the model is only **8B parameters**, you can run it on **any GPU with at least 16GB VRAM**.



## Step 1: Set Up Your Docker Environment

First, we’ll initialize a Docker container using the VLLM backend.
You can refer to the [VLLM Quickstart Guide](./README.md#vllm-quick-start) — or follow the full steps below.

### 1. Launch Docker Compose

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### 2. Build the Container

```bash
./container/build.sh --framework VLLM
```

### 3. Run the Container

```bash
./container/run.sh -it --framework VLLM --mount-workspace
```



## Step 2: Get Access to the Llama-3 Model

The **Meta-Llama-3.1-8B-Instruct** model is gated, so you’ll need to request access on Hugging Face.
Go to the official [Meta-Llama-3.1-8B-Instruct repository](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and fill out the access form.
Approval usually takes around **5 minutes**.

Once you have access, generate a **Hugging Face access token** with permission for gated repositories, then set it inside your container:

```bash
export HUGGING_FACE_HUB_TOKEN="insert_your_token_here"
export HF_TOKEN=$HUGGING_FACE_HUB_TOKEN
```



## Step 3: Run Aggregated Speculative Decoding

Now that your environment is ready, start the aggregated server with **speculative decoding**.

```bash
# Requires only one GPU
cd examples/backends/vllm
bash launch/agg_spec_decoding.sh
```

Once the weights finish downloading and serving begins, you’ll be ready to send inference requests to your model.




## Step 4: Example Request

To verify your setup, try sending a simple prompt to your model:

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
      "text": "In cherry blossom’s gentle breeze ... A delicate balance of life and death, as petals fade, and new life breathes.",
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



## Additional Resources

* [VLLM Quickstart](./README.md#vllm-quick-start)
* [Meta-Llama-3.1-8B-Instruct on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)