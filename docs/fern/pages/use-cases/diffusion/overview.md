---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Diffusion
subtitle: Generate images and video with diffusion models
sidebar-title: Overview
---

Dynamo serves diffusion models that generate media from a prompt. This section
is the output side of multimodal serving. To answer questions *about* images or
video instead, see [Image Understanding](../image-understanding/overview.md) and
[Video Understanding](../video-understanding/overview.md). For generated speech,
see [Text to Speech (TTS)](../speech/text-to-speech.md).

## Guides

| You want to… | Endpoint | Backends | Start here |
| --- | --- | --- | --- |
| Generate images from text | `/v1/images/generations` | vLLM-Omni, SGLang, TensorRT-LLM (experimental) | [Image Generation](image-generation.md) |
| Generate video from text, with or without audio | `/v1/videos` | vLLM-Omni, SGLang, TensorRT-LLM (experimental) | [Text to Video (T2V)](text-to-video.md) |
| Animate a source image | `/v1/videos` | vLLM-Omni, SGLang | [Image to Video (I2V)](image-to-video.md) |
| Serve video generation with a custom FastVideo worker | `/v1/videos` | FastVideo | [FastVideo](fastvideo.md) |

Each guide starts with a "Set up the backend" section. The vLLM-Omni, SGLang, and
TensorRT-LLM install steps live once in `use-cases/_snippets/` and are shared by
the guides.

> [!NOTE]
> Diffusion backends run from Dynamo CLI launch scripts and `python -m dynamo.*`
> entrypoints. Dynamo does not yet ship Kubernetes recipes for them, except for
> [FastVideo](fastvideo.md), which has its own Kubernetes deployment path.
