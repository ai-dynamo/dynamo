---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Image to Video (I2V)
subtitle: Animate source images with vLLM-Omni or SGLang
---

Image-to-video (I2V) animates a source image from a text prompt through the
same `/v1/videos` endpoint as [Text to Video (T2V)](text-to-video.md), with an extra
`input_reference` field that points at the source image.

> [!NOTE]
> Before you begin, set up vLLM-Omni or SGLang as described in
> [Text to Video (T2V) › Set up the backend](text-to-video.md#set-up-the-backend).
> TensorRT-LLM does not support image-to-video.

## Generate Videos from Images

<Tabs>
<Tab title="vLLM-Omni">

<Anchor id="vllm-omni" />

Image-to-video (I2V) uses the same `/v1/videos` endpoint as [text-to-video](text-to-video.md#vllm-omni), with an additional `input_reference` field that provides the source image. Run a vLLM-Omni worker with `--output-modalities video`.

### Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | Default model |
| `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | Dual-expert MoE ([model card](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)) |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_i2v.sh --model Wan-AI/Wan2.2-I2V-A14B-Diffusers
```

### Launch

Launch with the provided script using `Wan-AI/Wan2.2-TI2V-5B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_i2v.sh
```

### Generate a Video from an Image

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

### Request Parameters (`nvext`)

I2V accepts all [text-to-video `nvext` fields](text-to-video.md#request-parameters-nvext), plus two fields that control the dual-expert MoE denoising schedule in Wan2.x models:

<ParamField path="nvext.boundary_ratio" type="float" default="0.875">
  MoE expert switching boundary (I2V).
</ParamField>
<ParamField path="nvext.guidance_scale_2" type="float" default="1.0">
  CFG scale for the low-noise expert (I2V).
</ParamField>

### See Also

- [Text-to-Video with vLLM-Omni](text-to-video.md#vllm-omni)
- [Image-to-Video with SGLang](image-to-video.md#sglang)
- [vLLM-Omni Configuration reference](../../reference/backends/vllm-omni-configuration.mdx)

</Tab>
<Tab title="SGLang">

<Anchor id="sglang" />

SGLang's video generation worker (`--video-generation-worker`) supports image-to-video (I2V) in addition to [text-to-video](text-to-video.md#sglang), using SGLang's `DiffGenerator` with frame-to-video encoding.

### Launch

The `text-to-video-diffusion.sh` launcher loads text-to-video checkpoints only
(`Wan-AI/Wan2.1-T2V-*`), which cannot take a source image. For image-to-video,
start the worker directly with an image-to-video checkpoint:

```bash
mkdir -p /tmp/dynamo_inputs
python3 -m dynamo.frontend --http-port 8000 &
DYN_MM_LOCAL_PATH=/tmp/dynamo_inputs \
python3 -m dynamo.sglang \
  --model-path Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --served-model-name Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
  --video-generation-worker \
  --tp 2 \
  --media-output-fs-url file:///tmp/dynamo_media \
  --trust-remote-code \
  --skip-tokenizer-init
```

The 14B checkpoint needs two GPUs here, the same `--tp 2` that `text-to-video-diffusion.sh` uses for the 14B Wan model.
`Wan-AI/Wan2.1-I2V-14B-480P-Diffusers` is an image-to-video checkpoint in
SGLang's [diffusion model list](https://docs.sglang.io/diffusion/compatibility_matrix.html).
To use another checkpoint, pick one that your installed SGLang version supports
and use the same ID in the launch command and the request.

### Generate a Video from an Image

Pass the source image in `input_reference`, as an `https://` URL or as a path inside the
`DYN_MM_LOCAL_PATH` directory of the SGLang worker. The SGLang worker rejects a `data:` URI.
It fetches a URL under the same
[media URL policy](../image-understanding/overview.md#media-url-policy) as image inputs.
This example copies a local `cat.png` into the directory that the launch command allows:

```bash
cp cat.png /tmp/dynamo_inputs/cat.png
curl http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    "prompt": "The cat slowly turns its head and blinks",
    "input_reference": "/tmp/dynamo_inputs/cat.png",
    "seconds": 2,
    "size": "832x480",
    "response_format": "url"
  }'
```

### See Also

- [Text-to-Video with SGLang](text-to-video.md#sglang)
- [Image-to-Video with vLLM-Omni](image-to-video.md#vllm-omni)
- [SGLang Examples](../../recipes/cli-templates/sglang.mdx)

</Tab>
</Tabs>
