---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Image-to-Video with vLLM-Omni
subtitle: Animate a source image with vLLM-Omni via the /v1/videos endpoint and the input_reference field.
sidebar-title: vLLM-Omni
---

Image-to-video (I2V) uses the same `/v1/videos` endpoint as [text-to-video](../text-to-video/vllm-omni.md), with an additional `input_reference` field that provides the source image. Run a vLLM-Omni worker with `--output-modalities video`. See the [Diffusion Overview](../README.md) for installation and shared configuration.

## Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Default model |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Dual-expert MoE ([model card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)) |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_i2v.sh --model Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

## Launch

Launch with the provided script using `Wan-AI/Wan2.2-TI2V-5B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_i2v.sh
```

## Generate a Video from an Image

```bash
curl -s http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "prompt": "A bear playing with yarn, smooth motion",
    "input_reference": "https://example.com/bear.png",
    "size": "832x480",
    "response_format": "url",
    "nvext": {
      "num_inference_steps": 40,
      "num_frames": 33,
      "guidance_scale": 1.0,
      "boundary_ratio": 0.875,
      "guidance_scale_2": 1.0,
      "seed": 42
    }
  }'
```

The `input_reference` field accepts:
- **HTTP/HTTPS URL**: `"https://example.com/image.png"`
- **Base64 data URI**: `"data:image/png;base64,iVBORw0KGgo..."`
- **Local file path**: `"/path/to/image.png"` or `"file:///path/to/image.png"`

## Request Parameters (`nvext`)

I2V accepts all [text-to-video `nvext` fields](../text-to-video/vllm-omni.md#request-parameters-nvext), plus two fields that control the dual-expert MoE denoising schedule in Wan2.x models:

<ParamField path="nvext.boundary_ratio" type="float" default="0.875">
  MoE expert switching boundary (I2V).
</ParamField>
<ParamField path="nvext.guidance_scale_2" type="float" default="1.0">
  CFG scale for the low-noise expert (I2V).
</ParamField>

## See Also

- [Text-to-Video with vLLM-Omni](../text-to-video/vllm-omni.md)
- [Image-to-Video with SGLang](sglang.md)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)
