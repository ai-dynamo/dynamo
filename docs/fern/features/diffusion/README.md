---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Diffusion
subtitle: Deploy diffusion models for text-to-image, text-to-video, image-to-video, and text-to-audio in Dynamo across the vLLM-Omni, SGLang, TensorRT-LLM, and FastVideo backends.
sidebar-title: Overview
---

## Overview

Dynamo serves diffusion models for **text-to-image**, **text-to-video**, **image-to-video**, **text-to-audio (TTS)**, and **text-to-text (LLM diffusion)** generation. All of these run through the same Dynamo pipeline infrastructure used for LLM inference — frontend routing, scaling, and observability — behind OpenAI-compatible endpoints (`/v1/images/generations`, `/v1/videos`, `/v1/audio/speech`). These use cases are provided across four backends: **vLLM-Omni**, **SGLang**, **TensorRT-LLM**, and **FastVideo**. The docs are organized by use case — pick a modality from the sidebar, then choose a backend on that page.

<Note>
Diffusion is currently supported only through the Dynamo CLI (local launch scripts and `python -m dynamo.*` entrypoints). Dynamo does not yet ship prebuilt Kubernetes recipes for the built-in diffusion backends. The one exception is [FastVideo](text-to-video/fastvideo.md), which provides a Kubernetes deployment path.
</Note>

## Providers

**vLLM-Omni** is the broadest provider, covering every media modality (text-to-image, text-to-video, image-to-video, and text-to-audio) through a single `python -m dynamo.vllm.omni` entrypoint, and the only one supporting TTS and [disaggregated multi-stage serving](../../design-docs/vllm-omni-disaggregated.md). Reach for it when you want one backend for mixed media workloads or need audio. Limitations: image input is only accepted for image-to-video, each worker serves a single output modality at a time, KV cache events are not published, and audio streaming / voice cloning are not yet supported.

**SGLang** is the only provider for text-to-text (LLM diffusion) and also covers text-to-image, text-to-video, and image-to-video via its `DiffGenerator`. Choose it when you want diffusion alongside an existing SGLang LLM deployment, or specifically need diffusion language models. Limitation: no text-to-audio, and Conv3d models may need `SGLANG_DISABLE_CUDNN_CHECK=1` on a CuDNN version mismatch.

**TensorRT-LLM** offers **experimental** text-to-image and text-to-video on NVIDIA-optimized kernels. Use it when you're already on TRT-LLM and want the performance path, and can tolerate experimental status. Limitations: experimental and not recommended for production, no image-to-video (planned) or audio, and video output requires an NVENC-capable NVIDIA GPU for MP4 encoding.

**FastVideo** is a custom text-to-video-only worker built around a distilled LTX-2 model (5-step inference) and is the **only provider with a Kubernetes deployment path**. Choose it for fast, production-oriented text-to-video on Kubernetes. Limitation: text-to-video only, and it runs from a purpose-built runtime image that can take 20–40+ minutes to build the first time.

## Support Matrix

| Modality | vLLM-Omni | SGLang | TRT-LLM | FastVideo |
|----------|-----------|--------|---------|-----------|
| Text-to-Image | ✅ | ✅ | ✅ | ❌ |
| Text-to-Video | ✅ | ✅ | ✅ | ✅ |
| Image-to-Video | ✅ | ✅ | ❌ | ❌ |
| Text-to-Audio (TTS) | ✅ | ❌ | ❌ | ❌ |
| Text-to-Text (LLM diffusion) | ❌ | ✅ | ❌ | ❌ |

**Status:** ✅ Supported | ❌ Not supported

## Installation

Each backend has different installation steps. Choose the tab for the backend you plan to use.

<Tabs>
<Tab title="vLLM-Omni">

The [vLLM-Omni](https://github.com/vllm-project/vllm-omni) backend exposes text-to-image, text-to-video, image-to-video, and text-to-audio (TTS) through a single `python -m dynamo.vllm.omni` entrypoint.

### Prerequisites

- Familiarity with deploying Dynamo with vLLM, as described in the [vLLM backend guide](../../backends/vllm/README.md).
- An `amd64` host. vLLM-Omni is currently only installed on `amd64` builds; on `arm64` the container build skips the install and vLLM-Omni features are unavailable.

### Installation

Dynamo container images include vLLM-Omni pre-installed. If you are using `pip install ai-dynamo[vllm]`, vLLM-Omni is **not** included automatically because the matching release is not yet available on PyPI. Install it separately from source, pinning the vLLM-Omni release that matches your installed vLLM version (see the [vLLM-Omni releases](https://github.com/vllm-project/vllm-omni/releases) page):

```bash
pip install git+https://github.com/vllm-project/vllm-omni.git@<version>
```

<Note>
The `agg_omni_*.sh` and `disagg_omni_glm_image.sh` scripts throughout the vLLM-Omni pages are **single-node local launchers**: each starts `python -m dynamo.frontend` plus one or more `python -m dynamo.vllm.omni` workers directly on the host, placing stages with `CUDA_VISIBLE_DEVICES`. They are for local development and testing on a machine that has the required GPUs — they are **not** Kubernetes manifests and do not create a `DynamoGraphDeployment`. Dynamo does not currently ship prebuilt omni Kubernetes recipes. To run omni on Kubernetes, build a deployment around the same `python -m dynamo.vllm.omni` entrypoint and the flags documented in the [vLLM-Omni Configuration reference](../../backends/vllm/vllm-omni-config-reference.mdx).
</Note>

**Storage:** Generated media is stored via [fsspec](https://filesystem-spec.readthedocs.io/), which supports local filesystems, S3, GCS, and Azure Blob. By default, media is written to `file:///tmp/dynamo_media`. To use cloud storage, pass `--media-output-fs-url` (e.g. `s3://my-bucket/media`) and optionally `--media-output-http-url` to rewrite response URLs as `{base-url}/{storage-path}`. For S3, set the standard AWS environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) or use IAM roles; see the [fsspec S3 docs](https://s3fs.readthedocs.io/en/latest/#credentials).

**Stage config:** Omni pipelines are configured via YAML stage configs. vLLM-Omni ships built-in stage configs for supported models, so no `--stage-configs-path` is needed unless you want to override the defaults. See the [vLLM-Omni Stage Configs documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/).

For the full flag surface, see the [vLLM-Omni Configuration reference](../../backends/vllm/vllm-omni-config-reference.mdx).

</Tab>
<Tab title="SGLang">

Dynamo SGLang supports **LLM diffusion** (text generation via iterative refinement), **image diffusion** (text-to-image), and **video generation** (text-to-video and image-to-video). Each uses a different worker flag and handler, but all integrate with SGLang's `DiffGenerator`.

| Type             | Worker Flag                 | API Endpoint                              |
| ---------------- | --------------------------- | ----------------------------------------- |
| LLM Diffusion    | `--dllm-algorithm <algo>`   | `/v1/chat/completions`, `/v1/completions` |
| Image Diffusion  | `--image-diffusion-worker`  | `/v1/images/generations`                  |
| Video Generation | `--video-generation-worker` | `/v1/videos`                              |

### Prerequisites

- A working [SGLang backend setup](../../backends/sglang/README.md).

### Installation

Diffusion support ships with the standard SGLang backend — no separate install is required. Image and video workers write generated media to local storage (`--fs-url file:///tmp/images`) or S3 (`--fs-url s3://bucket`); pass `--http-url` to set the base URL for serving stored media.

<Note>
If you see a CuDNN version mismatch error on startup (`cuDNN frontend 1.8.1 requires cuDNN lib >= 9.5.0`), set `SGLANG_DISABLE_CUDNN_CHECK=1` before launching. This is common when PyTorch ships a CuDNN version older than what SGLang requires for Conv3d operations.
</Note>

For SGLang configuration flags, see the [SGLang Configuration reference](../../backends/sglang/sglang-config-reference.mdx).

</Tab>
<Tab title="TensorRT-LLM">

TensorRT-LLM supports **experimental** text-to-image and text-to-video generation through the `visual_gen` module and Diffusers pipelines, selected via `--modality image_diffusion` or `--modality video_diffusion`. The pipeline type is auto-detected from the model's `model_index.json`.

### Prerequisites

- **TensorRT-LLM with visual_gen**: The `visual_gen` module is part of TensorRT-LLM (`tensorrt_llm._torch.visual_gen`). Install TensorRT-LLM following the [official instructions](https://github.com/NVIDIA/TensorRT-LLM#installation).
- **dynamo-runtime with multimodal API**: The Dynamo runtime must include `ModelType.Videos` or `ModelType.Images` support.
- **A GPU with sufficient VRAM** for the diffusion model. Video (MP4) output additionally requires an NVENC-capable NVIDIA GPU at runtime.

### Installation

For text-to-image, no extra setup is needed beyond TensorRT-LLM with `visual_gen`.

For **text-to-video**, `imageio` with ffmpeg is required to encode generated frames to MP4. The Dynamo TRT-LLM runtime container ships an LGPL-only ffmpeg CLI built with the NVIDIA NVENC H.264 encoder (`h264_nvenc`) and `libvpx_vp9` for WebM, and points `imageio` at it via `IMAGEIO_FFMPEG_EXE=/usr/local/bin/ffmpeg`. If you're running outside the container, install the Python wrapper without the bundled binary and point it at your own ffmpeg:

```bash
pip install --no-binary imageio-ffmpeg "imageio[ffmpeg]"
export IMAGEIO_FFMPEG_EXE=/path/to/your/ffmpeg
```

For configuration flags, see the [TensorRT-LLM Configuration reference](../../backends/trtllm/trtllm-config-reference.mdx).

</Tab>
<Tab title="FastVideo">

[FastVideo](https://github.com/hao-ai-lab/FastVideo) is a custom text-to-video worker (not a built-in backend) that serves the `/v1/videos` endpoint. The default model `FastVideo/LTX2-Distilled-Diffusers` is a distilled LTX-2 Diffusion Transformer that reduces inference from 50+ steps to just 5.

### Prerequisites

- A GPU-enabled environment (Kubernetes cluster or a host with an NVIDIA GPU runtime).
- A container registry you can push the FastVideo runtime image to.

### Installation

FastVideo runs from a purpose-built runtime image compiled from the [`examples/diffusers/Dockerfile`](https://github.com/ai-dynamo/dynamo/tree/main/examples/diffusers/Dockerfile) (base `nvidia/cuda:13.1.1-devel-ubuntu24.04`, FastVideo from GitHub, Dynamo from `release/1.0.0`, and a compiled flash-attention fork). The first build can take 20–40+ minutes. Kubernetes is the recommended deployment path. See the [FastVideo page](text-to-video/fastvideo.md) for full build, deploy, and configuration instructions.

</Tab>
</Tabs>

## See Also

- [Text-to-Image](text-to-image/vllm-omni.md)
- [Text-to-Video](text-to-video/vllm-omni.md)
- [Image-to-Video](image-to-video/vllm-omni.md)
- [Text-to-Audio](text-to-audio/vllm-omni.md)
- [vLLM-Omni Disaggregated Serving (design)](../../design-docs/vllm-omni-disaggregated.md)
