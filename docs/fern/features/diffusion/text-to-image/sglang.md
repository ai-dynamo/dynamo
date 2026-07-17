---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Image with SGLang
subtitle: Generate images from text prompts with SGLang's image diffusion worker via /v1/images/generations.
sidebar-title: SGLang
---

Image diffusion workers generate images from text prompts using SGLang's `DiffGenerator`, with the `--image-diffusion-worker` flag. Generated images are returned as either URLs (when using `--media-output-fs-url` for storage) or base64 data, in an OpenAI-compatible response format. See the [Diffusion Overview](../README.md) for backend setup.

## Launch

```bash
cd $DYNAMO_HOME/examples/backends/sglang
./launch/image_diffusion.sh
```

Supports local storage (`--fs-url file:///tmp/images`) and S3 (`--fs-url s3://bucket`). Pass `--http-url` to set the base URL for serving stored images. See the launch script for all configuration options.

## Generate an Image

```bash
curl http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "black-forest-labs/FLUX.1-dev",
    "prompt": "A cat sitting on a windowsill",
    "size": "1024x1024",
    "response_format": "url",
    "nvext": {
      "num_inference_steps": 15
    }
  }'
```

<Note>
This happens with diffusers models (FLUX.1-dev, Wan2.1, etc.) that use `model_index.json` instead of `config.json`. Ensure you are using the `--image-diffusion-worker` flag rather than the standard LLM worker mode. These flags use a registration path that does not require `config.json`.
</Note>

## See Also

- [Text-to-Image with vLLM-Omni](vllm-omni.md)
- [Text-to-Image with TensorRT-LLM](trtllm.md)
- [SGLang Examples](../../../backends/sglang/sglang-examples.md)
- [SGLang Configuration reference](../../../backends/sglang/sglang-config-reference.mdx)
