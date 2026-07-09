---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Diffusion
subtitle: Deploy diffusion models for text-to-image, text-to-video, and more in Dynamo
---

## Overview

Dynamo supports serving diffusion models across multiple backends, enabling generation of images and video from text prompts. Backends expose diffusion capabilities through the same Dynamo pipeline infrastructure used for LLM inference, including frontend routing, scaling, and observability.

## Support Matrix

| Modality | vLLM-Omni | SGLang | TRT-LLM |
|----------|-----------|--------|---------|
| Text-to-Text | ❌ | ✅ | ❌ |
| Text-to-Image | ✅ | ✅ | ✅ |
| Text-to-Video | ✅ | ✅ | ✅ (NVENC required) |
| Image-to-Video | ✅ | ❌ | ❌ |

**Status:** ✅ Supported | ❌ Not supported

TRT-LLM video output currently supports MP4 only and requires an NVENC-capable GPU. GPUs
without NVENC are not supported for TRT-LLM video output.

## Backend Documentation

For deployment guides, configuration, and examples for each backend:

- **[vLLM-Omni](../../backends/vllm/vllm-omni.md)**
- **[SGLang Diffusion](../../backends/sglang/sglang-diffusion.md)**
- **[TRT-LLM Diffusion](../../backends/trtllm/trtllm-diffusion.md)**
- **[FastVideo (custom worker)](fastvideo.md)**
