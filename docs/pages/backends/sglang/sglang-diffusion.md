---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Diffusion
---

Dynamo SGLang supports three types of diffusion-based generation: **LLM diffusion** (text generation via iterative refinement), **image diffusion** (text-to-image), and **video generation** (text-to-video). Each uses a different worker flag and handler, but all integrate with SGLang's `DiffGenerator`.

## Overview

| Type | Worker Flag | API Endpoint |
|------|------------|--------------|
| LLM Diffusion | `--dllm-algorithm <algo>` | `/v1/chat/completions`, `/v1/completions` |
| Image Diffusion | `--image-diffusion-worker` | `/v1/images/generations` |
| Video Generation | `--video-generation-worker` | `/v1/videos` |

<Note>
If you see a CuDNN version mismatch error on startup (`cuDNN frontend 1.8.1 requires cuDNN lib >= 9.5.0`), set `SGLANG_DISABLE_CUDNN_CHECK=1` before launching. This is common when PyTorch ships a CuDNN version older than what SGLang requires for Conv3d operations.
</Note>

## LLM Diffusion

Diffusion Language Models generate text through iterative refinement rather than autoregressive token-by-token generation. The model starts with masked tokens and progressively replaces them with predictions, refining low-confidence tokens each step.

LLM diffusion is auto-detected: when `--dllm-algorithm` is set, the worker automatically uses `DiffusionWorkerHandler` without needing a separate flag. For more details on diffusion algorithms, see the [SGLang Diffusion Language Models documentation](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/text_generation/diffusion_language_models.md).

### Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/diffusion_llada.sh
```

See the [launch script](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/launch/diffusion_llada.sh) for configuration options.

### Test

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-mini-preview",
    "messages": [{"role": "user", "content": "Hello! How are you?"}],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

## Image Diffusion

Image diffusion workers generate images from text prompts using SGLang's `DiffGenerator`. Generated images are returned as either URLs (when using `--media-output-fs-url` for storage) or base64 data, in an OpenAI-compatible response format.

### Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/image_diffusion.sh
```

Supports local storage (`--fs-url file:///tmp/images`) and S3 (`--fs-url s3://bucket`). Pass `--http-url` to set the base URL for serving stored images. See the launch script for all configuration options.

### Test

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "prompt": "A sunset over the ocean",
    "size": "1024x1024",
    "response_format": "url",
    "nvext": {
      "num_inference_steps": 15
    }
  }'
```

## Video Generation

Video generation workers produce videos from text or image prompts using SGLang's `DiffGenerator` with frame-to-video encoding. Supports text-to-video (T2V) and image-to-video (I2V) workflows.

### Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/text-to-video-diffusion.sh
```

Use `--wan-size 1b` (default, 1 GPU) or `--wan-size 14b` (2 GPUs). See the launch script for all configuration options.

### Test

```bash
curl http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A curious raccoon exploring a garden",
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "seconds": 2,
    "size": "832x480",
    "response_format": "url",
    "nvext": {
      "fps": 8,
      "num_frames": 17,
      "num_inference_steps": 50
    }
  }'
```

## See Also

- **[Examples](sglang-examples.md)**: Launch scripts for all deployment patterns
- **[Reference Guide](sglang-reference-guide.md)**: Worker types and argument reference
- **[SGLang Diffusion LMs (upstream)](https://github.com/sgl-project/sglang/blob/main/docs/supported_models/text_generation/diffusion_language_models.md)**: SGLang diffusion documentation
