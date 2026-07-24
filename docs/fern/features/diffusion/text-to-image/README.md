---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Image
subtitle: Generate images from text prompts with vLLM-Omni, SGLang, or TensorRT-LLM
---

Choose a backend for text-to-image generation. See the [Diffusion Overview](../README.md) for installation and shared configuration.

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Text-to-image generation runs a vLLM-Omni worker with `--output-modalities image`.

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

- [Text-to-Image with SGLang](README.md#sglang)
- [Text-to-Image with TensorRT-LLM](README.md#tensorrt-llm)
- [Disaggregated Serving](../../../design-docs/vllm-omni-disaggregated.md) — GLM-Image (2-stage text-to-image)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)

</Tab>
<Tab title="SGLang">

<Anchor id="sglang" />

Image diffusion workers generate images from text prompts using SGLang's `DiffGenerator`, with the `--image-diffusion-worker` flag. Generated images are returned as either URLs (when using `--media-output-fs-url` for storage) or base64 data, in an OpenAI-compatible response format.

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

- [Text-to-Image with vLLM-Omni](README.md#vllm-omni)
- [Text-to-Image with TensorRT-LLM](README.md#tensorrt-llm)
- [SGLang Examples](../../../backends/sglang/sglang-examples.mdx)
- [SGLang Configuration reference](../../../backends/sglang/sglang-config-reference.mdx)

</Tab>
<Tab title="TensorRT-LLM">

<Anchor id="tensorrt-llm" />

TensorRT-LLM supports **experimental** text-to-image generation through the `--modality image_diffusion` flag.

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

- [Text-to-Image with vLLM-Omni](README.md#vllm-omni)
- [Text-to-Image with SGLang](README.md#sglang)
- [Text-to-Video with TensorRT-LLM](../text-to-video/README.md#tensorrt-llm)

</Tab>
</Tabs>
