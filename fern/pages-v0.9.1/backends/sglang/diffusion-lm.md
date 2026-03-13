---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Running Diffusion LMs with SGLang

Diffusion Language Models (Diffusion LMs) are a class of generative models that use diffusion processes for text generation. This guide shows how to deploy diffusion models like LLaDA2.0 using SGLang as the backend with Dynamo. Diffusion LMs work differently from autoregressive models - they iteratively refine generated text through a diffusion process.

## Launch the Deployment

### Using the Launch Script (Recommended)

The easiest way to start the diffusion LM service is using the provided launch script:

```bash
bash examples/backends/sglang/launch/diffusion_llada.sh
```

### Manual Launch Steps

If you prefer to launch components manually:

**Start frontend**
```bash
python -m dynamo.frontend --http-port 8001 &
```

**Run diffusion worker**
```bash
export CUDA_VISIBLE_DEVICES=0,1
python -m dynamo.sglang \
  --model-path inclusionAI/LLaDA2.0-mini-preview \
  --tp-size 2 \
  --skip-tokenizer-init \
  --trust-remote-code \
  --endpoint dyn://dynamo.backend.generate \
  --enable-metrics \
  --disable-cuda-graph \
  --disable-overlap-schedule \
  --attention-backend triton \
  --dllm-algorithm LowConfidence
```

## Diffusion Algorithms

The diffusion worker uses the **LowConfidence** algorithm for the iterative refinement process. This algorithm refines tokens with low confidence scores, progressively replacing masked tokens with the model's predictions until confidence thresholds are met.

For more details on diffusion algorithms and configuration options, refer to the [SGLang Diffusion Language Models documentation](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/diffusion_language_models.md).


## Testing the Deployment

Once deployed, you can test the service using curl:

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini-preview",
    "messages": [
      {
        "role": "user",
        "content": "Hello! How are you?"
      }
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

Or use the completions endpoint:

```bash
curl -X POST http://localhost:8001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini-preview",
    "prompt": "Once upon a time",
    "max_tokens": 256
  }'
```