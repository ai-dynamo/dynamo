---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Image with vLLM-Omni
subtitle: Generate images from text prompts with vLLM-Omni via /v1/images/generations and /v1/chat/completions.
sidebar-title: vLLM-Omni
---

Text-to-image generation runs a vLLM-Omni worker with `--output-modalities image`. See the [Diffusion Overview](../README.md) for installation and shared configuration.

## Tested Models

| Model | Notes |
|---|---|
| `Qwen/Qwen-Image` | Default model |
| `AIDC-AI/Ovis-Image-7B` | |
| `zai-org/GLM-Image` | 2-stage; see [Disaggregated Serving](../../../design-docs/vllm-omni-disaggregated.md) |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_image.sh --model AIDC-AI/Ovis-Image-7B
```

## Launch

Launch using the provided script with `Qwen/Qwen-Image`:

```bash
bash examples/backends/vllm/launch/agg_omni_image.sh
```

## Generate an Image

Image generation is available on two endpoints: `/v1/chat/completions` returns base64-encoded images inline in `choices[].delta.content[].image_url`, while `/v1/images/generations` returns a URL or base64 depending on `response_format`.

<Tabs>
<Tab title="/v1/images/generations">

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "prompt": "A cat sitting on a windowsill",
    "size": "1024x1024",
    "response_format": "url"
  }'
```

</Tab>
<Tab title="/v1/chat/completions">

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen-Image",
    "messages": [{"role": "user", "content": "A cat sitting on a windowsill"}],
    "stream": false
  }'
```

</Tab>
</Tabs>

## See Also

- [Text-to-Image with SGLang](sglang.md)
- [Text-to-Image with TensorRT-LLM](trtllm.md)
- [Disaggregated Serving](../../../design-docs/vllm-omni-disaggregated.md) — GLM-Image (2-stage text-to-image)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)
