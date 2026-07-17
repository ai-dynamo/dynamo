---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Image with TensorRT-LLM (Experimental)
subtitle: Experimental text-to-image diffusion in TensorRT-LLM using the visual_gen module and Diffusers pipelines.
sidebar-title: TensorRT-LLM
---

TensorRT-LLM supports **experimental** text-to-image generation through the `--modality image_diffusion` flag. See the [Diffusion Overview](../README.md) for requirements and installation.

## Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `FluxPipeline` | FLUX Text-to-Image | `black-forest-labs/FLUX.1-dev` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

## Launch

```bash
python -m dynamo.trtllm \
  --modality image_diffusion \
  --model-path black-forest-labs/FLUX.1-dev \
  --media-output-fs-url file:///tmp/dynamo_media
```

## Generate an Image

Image generation uses the `/v1/images/generations` endpoint:

```bash
curl -X POST http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "model": "black-forest-labs/FLUX.1-dev",
    "size": "256x256"
  }'
```

## Configuration

For the full flag surface (quantization, TeaCache, torch.compile, attention backend, and request defaults), see the [TensorRT-LLM Configuration reference](../../../backends/trtllm/trtllm-config-reference.mdx#diffusion-experimental).

## Limitations

- Diffusion is experimental and not recommended for production use.
- Requires a GPU with sufficient VRAM for the diffusion model.

## See Also

- [Text-to-Image with vLLM-Omni](vllm-omni.md)
- [Text-to-Image with SGLang](sglang.md)
- [Text-to-Video with TensorRT-LLM](../text-to-video/trtllm.md)
