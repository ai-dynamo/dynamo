---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Video with vLLM-Omni
subtitle: Generate video from text prompts with vLLM-Omni via the /v1/videos endpoint.
sidebar-title: vLLM-Omni
---

Text-to-video generation runs a vLLM-Omni worker with `--output-modalities video`. See the [Diffusion Overview](../README.md) for installation and shared configuration.

## Tested Models

| Model | Notes |
|---|---|
| `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | Default model (1 GPU) |
| `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | |

To run a non-default model, pass `--model` to the launch script:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh --model Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

## Launch

Launch using the provided script with `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`:

```bash
bash examples/backends/vllm/launch/agg_omni_video.sh
```

## Generate a Video

Generate a video via `/v1/videos`:

```bash
curl -s http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "prompt": "A drone flyover of a mountain landscape",
    "seconds": 2,
    "size": "832x480",
    "response_format": "url"
  }'
```

The response returns a video URL or base64 data depending on `response_format` (e.g. `{"object": "video", "status": "completed", "data": [{"url": "file:///tmp/dynamo_media/videos/req-abc123.mp4"}]}`).

## Request Parameters (`nvext`)

The `/v1/videos` endpoint also accepts NVIDIA extensions via the `nvext` field for fine-grained control:

<ParamField path="nvext.fps" type="int" default="24">
  Frames per second.
</ParamField>
<ParamField path="nvext.num_frames" type="int">
  Number of frames (overrides `fps * seconds`).
</ParamField>
<ParamField path="nvext.negative_prompt" type="string">
  Negative prompt for guidance.
</ParamField>
<ParamField path="nvext.num_inference_steps" type="int" default="50">
  Number of denoising steps.
</ParamField>
<ParamField path="nvext.guidance_scale" type="float" default="5.0">
  CFG guidance scale.
</ParamField>
<ParamField path="nvext.seed" type="int">
  Random seed for reproducibility.
</ParamField>

<Note>
The `nvext.boundary_ratio` and `nvext.guidance_scale_2` fields apply to the dual-expert MoE schedule used in image-to-video. See [Image-to-Video with vLLM-Omni](../image-to-video/vllm-omni.md).
</Note>

## See Also

- [Image-to-Video with vLLM-Omni](../image-to-video/vllm-omni.md) — animate a source image with the same `/v1/videos` endpoint
- [Text-to-Video with SGLang](sglang.md)
- [Text-to-Video with TensorRT-LLM](trtllm.md)
- [Text-to-Video with FastVideo](fastvideo.md)
- [vLLM-Omni Configuration reference](../../../backends/vllm/vllm-omni-config-reference.mdx)
