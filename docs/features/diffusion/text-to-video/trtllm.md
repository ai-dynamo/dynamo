---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Text-to-Video with TensorRT-LLM (Experimental)
subtitle: Experimental text-to-video diffusion in TensorRT-LLM using the visual_gen module and Diffusers pipelines.
sidebar-title: TensorRT-LLM
---

TensorRT-LLM supports **experimental** text-to-video generation through the `--modality video_diffusion` flag. See the [Diffusion Overview](../README.md) for requirements and installation (including the ffmpeg/imageio setup needed for MP4 encoding).

## Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

## Launch

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --media-output-fs-url file:///tmp/dynamo_media
```

## Generate a Video

Video generation uses the `/v1/videos` endpoint:

```bash
curl -X POST http://localhost:8000/v1/videos \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing piano",
    "model": "wan_t2v",
    "seconds": 4,
    "size": "832x480",
    "nvext": {
      "fps": 24
    }
  }'
```

## Configuration

For the full flag surface (quantization, TeaCache, torch.compile, attention backend, and request defaults), see the [TensorRT-LLM Configuration reference](../../../backends/trtllm/trtllm-config-reference.mdx#diffusion-experimental).

## Limitations

- Diffusion is experimental and not recommended for production use.
- Only text-to-video and text-to-image are supported in this release (image-to-video planned).
- Requires a GPU with sufficient VRAM for the diffusion model.

## See Also

- [Text-to-Video with vLLM-Omni](vllm-omni.md)
- [Text-to-Video with SGLang](sglang.md)
- [Text-to-Video with FastVideo](fastvideo.md)
- [Text-to-Image with TensorRT-LLM](../text-to-image/trtllm.md)
