---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Image
subtitle: Generate images from text prompts with vLLM-Omni, SGLang, or TensorRT-LLM
---

Choose a backend for text-to-image generation. See the [Diffusion Overview](../overview.md) for installation and shared configuration.

For image edits and the unsupported `/v1/images/variations` route, see
[Image API Compatibility](../../../reference/components/frontend-configuration.mdx#image-api-compatibility).

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Text-to-image generation runs a vLLM-Omni worker with `--output-modalities image`.

## Tested Models

| Model | Notes |
|---|---|
| `Qwen/Qwen-Image` | Default model |
| `AIDC-AI/Ovis-Image-7B` | |
| `black-forest-labs/FLUX.2-klein-4B` | Text-to-image and reference-image editing; see [Create a Variation with Image Edits](#create-a-variation-with-image-edits) |
| `zai-org/GLM-Image` | 2-stage; see [Disaggregated Serving](../../../developer-guide/knowledge-base/modular-components/backends/vllm/vllm-omni-disaggregated-serving.md) |

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

For `/v1/images/generations` and `/v1/images/edits`, Dynamo's vLLM-Omni image handler accepts `output_format` values `png` (default), `jpeg`, and `webp`. It reports the encoded format in `output_format`; use `jpeg` for JPEG output. It reports `size` when every output has the same valid dimensions and returns `null` when dimensions are mixed or unavailable. This handler accepts `response_format` values `url` and `b64_json`; if omitted, the response uses base64. JPEG output composites transparent pixels onto white. This handler does not apply `quality` or `background`; requests that set either field are rejected with HTTP 400 rather than echoing values that were not used.

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

The response reports image accounting under `nvext.usage` (images returned, reference images
consumed, and the step count; vLLM-Omni reports the count the request asked for), while the OpenAI
`usage` field stays `null` because diffusion models report no token counts. See
[Image Accounting](../../../developer-guide/additional-resources/nvidia-request-extensions-nvext.md#image-accounting).

## Create a Variation with Image Edits

To generate a new image guided by an existing image, send a reference image and a prompt
to `/v1/images/edits`. Use a model that supports image conditioning; text-to-image support
alone does not establish editing support. This example uses `black-forest-labs/FLUX.2-klein-4B`
with vLLM 0.31.0 and vLLM-Omni 0.31.0rc1.

> [!IMPORTANT]
> Use a Dynamo build containing the [image-reference forwarding fix](https://github.com/ai-dynamo/dynamo/pull/15390).
> Dynamo 1.5.1 ignores edit references. See [Image API Compatibility](../../../reference/components/frontend-configuration.mdx#image-api-compatibility)
> for the JSON request contract and the remaining multipart-upload limitation.

Launch the reference-capable model from the repository root:

```bash
bash examples/backends/vllm/launch/agg_omni_image.sh \
  --model black-forest-labs/FLUX.2-klein-4B \
  --enforce-eager
```

In a second terminal, encode a local PNG named `reference.png` into a JSON request. The
file is read on the client; the server receives a base64 data URL in `input_reference`.

```bash
python3 - <<'PY' > edit-request.json
import base64
import json
from pathlib import Path

reference = base64.b64encode(Path("reference.png").read_bytes()).decode("ascii")
print(json.dumps({
    "model": "black-forest-labs/FLUX.2-klein-4B",
    "prompt": "Create a watercolor variation of this image, preserving its main subject.",
    "input_reference": f"data:image/png;base64,{reference}",
    "size": "512x512",
    "n": 1,
    "response_format": "b64_json",
    "nvext": {"num_inference_steps": 4, "guidance_scale": 1.0, "seed": 42},
}))
PY

curl --fail-with-body -sS http://localhost:8000/v1/images/edits \
  -H 'Content-Type: application/json' \
  --data-binary @edit-request.json > edit-response.json
```

Decode the returned PNG:

```bash
python3 - <<'PY'
import base64
import json
from pathlib import Path

response = json.loads(Path("edit-response.json").read_text())
Path("edited.png").write_bytes(base64.b64decode(response["data"][0]["b64_json"]))
PY
```

To use a hosted reference instead, set `input_reference` to an HTTP(S) image URL that the
worker can fetch. Do not use `curl -F` or the native vLLM-Omni `image` field through the
Dynamo frontend; this endpoint currently requires the JSON `input_reference` contract.

## See Also

- [Cosmos3 Nano example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/omni/cosmos3/README.md) — experimental image and video serving, with explicit guardrail configuration
- [Text-to-Image with SGLang](text-to-image.md#sglang)
- [Text-to-Image with TensorRT-LLM](text-to-image.md#tensorrt-llm)
- [Disaggregated Serving](../../../developer-guide/knowledge-base/modular-components/backends/vllm/vllm-omni-disaggregated-serving.md) — GLM-Image (2-stage text-to-image)
- [vLLM-Omni Configuration reference](../../../reference/backends/vllm-omni-configuration.mdx)

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

> [!NOTE]
> This happens with diffusers models (FLUX.1-dev, Wan2.1, etc.) that use `model_index.json` instead of `config.json`. Ensure you are using the `--image-diffusion-worker` flag rather than the standard LLM worker mode. These flags use a registration path that does not require `config.json`.

## See Also

- [Text-to-Image with vLLM-Omni](text-to-image.md#vllm-omni)
- [Text-to-Image with TensorRT-LLM](text-to-image.md#tensorrt-llm)
- [SGLang Examples](../../../recipes/cli-templates/sglang.mdx)
- [SGLang Configuration reference](../../../reference/backends/sglang-configuration.mdx)

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

For the full flag surface (quantization, TeaCache, torch.compile, attention backend, and request defaults), see the [TensorRT-LLM Configuration reference](../../../reference/backends/tensorrt-llm-configuration.mdx#diffusion-experimental).

## Limitations

- Diffusion is experimental and not recommended for production use.
- Requires a GPU with sufficient VRAM for the diffusion model.

## See Also

- [Text-to-Image with vLLM-Omni](text-to-image.md#vllm-omni)
- [Text-to-Image with SGLang](text-to-image.md#sglang)
- [Text-to-Video with TensorRT-LLM](text-to-video.md#tensorrt-llm)

</Tab>
</Tabs>
