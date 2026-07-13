---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Video Diffusion Support (Experimental)
subtitle: Experimental text-to-video and text-to-image diffusion in TensorRT-LLM using the visual_gen module and Diffusers pipelines.
---

For general TensorRT-LLM features and configuration, see the [Reference Guide](trtllm-reference-guide.md).

---

Dynamo supports video generation using diffusion models through the `--modality video_diffusion` flag and
image generation through `--modality image_diffusion` flag.

## Requirements

- **TensorRT-LLM with visual_gen**: The `visual_gen` module is part of TensorRT-LLM (`tensorrt_llm._torch.visual_gen`). Install TensorRT-LLM following the [official instructions](https://github.com/NVIDIA/TensorRT-LLM#installation).
- **dynamo-runtime with multimodal API**: The Dynamo runtime must include `ModelType.Videos` or `ModelType.Images` support. Ensure you're using a compatible version.
- **VIDEO diffusion: imageio with ffmpeg**: Required for encoding generated frames to MP4 video. The Dynamo TRT-LLM runtime container ships an LGPL-only ffmpeg CLI built with the NVIDIA NVENC H.264 encoder (`h264_nvenc`) and `libvpx_vp9` for WebM, and points `imageio` at it via `IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg` — the GPL-encumbered ffmpeg binary normally shipped inside the `imageio-ffmpeg` PyPI wheel is **not** installed. If you're running outside the container, install the Python wrapper without the bundled binary and point it at your own ffmpeg:
  ```bash
  pip install --no-binary imageio-ffmpeg "imageio[ffmpeg]"
  export IMAGEIO_FFMPEG_EXE=/path/to/your/ffmpeg
  ```

- **NVENC-capable GPU for video output**: The TRT-LLM `/v1/videos` endpoint currently
  supports only MP4 output and always encodes it with the NVENC H.264 hardware encoder
  (`h264_nvenc`). An NVENC-capable NVIDIA GPU is mandatory. There is no software H.264
  fallback, and the WebM/VP9 path (`libvpx-vp9`) is not exposed by the TRT-LLM video API.
  GPUs without NVENC, including A100, H100, HGX B200, and GB200, cannot produce video
  output. Examples of NVENC-capable data center GPUs include L4, L40, L40S, A10, A16,
  A2, and T4. See the
  [NVIDIA Video Encode and Decode Support Matrix](https://developer.nvidia.com/video-encode-decode-support-matrix).
  NVENC capability alone does not guarantee sufficient VRAM or full model compatibility.

## Supported Models

| Diffusers Pipeline | Description | Example Model |
|--------------------|-------------|---------------|
| `WanPipeline` | Wan 2.1/2.2 Text-to-Video | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` |
| `FluxPipeline` | FLUX Text-to-Image | `black-forest-labs/FLUX.1-dev` |


The pipeline type is **auto-detected** from the model's `model_index.json` — no `--model-type` flag is needed.

## Quick Start

### Video Diffusion

#### Launch worker

```bash
python -m dynamo.trtllm \
  --modality video_diffusion \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --media-output-fs-url file:///tmp/dynamo_media
```

#### API Endpoint

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

### Image Diffusion

#### Launch worker

```bash
python -m dynamo.trtllm \
  --modality image_diffusion \
  --model-path black-forest-labs/FLUX.1-dev \
  --media-output-fs-url file:///tmp/dynamo_media
```

#### API Endpoint

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

## Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--media-output-fs-url` | Filesystem URL for storing generated media | `file:///tmp/dynamo_media` |
| `--default-height` | Default image/video height | `480` |
| `--default-width` | Default image/video width | `832` |
| `--default-num-frames` | Default frame count | `81` |
| `--default-num-images-per-prompt` | Default number of images per prompt | `1` |
| `--enable-teacache` | Enable TeaCache optimization | `False` |
| `--disable-torch-compile` | Disable torch.compile | `False` |

## Limitations

- Diffusion is experimental and not recommended for production use
- Only text-to-video and text-to-image is supported in this release (image-to-video planned)
- Requires GPU with sufficient VRAM for the diffusion model
- MP4 video output requires an NVENC-capable GPU; GPUs without NVENC are unsupported for
  TRT-LLM video output
