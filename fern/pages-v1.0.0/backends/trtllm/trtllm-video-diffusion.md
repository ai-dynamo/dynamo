---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Diffusion Support (Experimental)
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

---

Dynamo supports video generation using diffusion models through the `--modality video_diffusion` flag.

## Requirements

- **TensorRT-LLM with visual_gen**: The `visual_gen` module is part of TensorRT-LLM (`tensorrt_llm._torch.visual_gen`). Install TensorRT-LLM following the [official instructions](https://github.com/NVIDIA/TensorRT-LLM#installation).
- **imageio with ffmpeg**: Required for encoding generated frames to MP4 video:
  ```bash
  pip install imageio[ffmpeg]
  ```
- **dynamo-runtime with video API**: The Dynamo runtime must include `ModelType.Videos` support. Ensure you're using a compatible version.

## Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |

The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

## Quick Start

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --media-output-fs-url file:///tmp/dynamo_media
```

## API Endpoint

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

## Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--media-output-fs-url` | Filesystem URL for storing generated media | `file:///tmp/dynamo_media` |
| `--default-height` | Default video height | `480` |
| `--default-width` | Default video width | `832` |
| `--default-num-frames` | Default frame count | `81` |
| `--enable-teacache` | Enable TeaCache optimization | `False` |
| `--disable-torch-compile` | Disable torch.compile | `False` |

## Limitations

- Video diffusion is experimental and not recommended for production use
- Only text-to-video is supported in this release (image-to-video planned)
- Requires GPU with sufficient VRAM for the diffusion model
