---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Supported Media Codecs
subtitle: Which image and video formats Dynamo decodes out of the box, and how to enable others
---

Dynamo decodes multimodal inputs (image and video) before handing them to a backend. The runtime images ship a deliberately minimal media stack, so only a subset of codecs is available by default. This page lists what is supported and how to enable additional codecs when you need them.

## Supported formats

| Modality | Decode path | Codecs available in the shipped images |
|----------|-------------|----------------------------------------|
| Image | Rust `image` crate | Common still-image formats (PNG, JPEG, WebP, GIF, BMP) |
| Video — frontend decode | In-tree FFmpeg (`dynamo-llm/media-ffmpeg`) | **VP8, VP9** (in MP4 / WebM / MKV) |
| Video — vLLM backend decode (EPD) | OpenCV (`cv2`) — not installed by default | Whatever the installed OpenCV build provides |
| Video — SGLang backend decode (EPD) | `decord` — not installed by default | Whatever the installed decord build provides |
| Audio | — | Not decoded yet |

<Note>
The shipped images build FFmpeg with a **VP8/VP9-only** decoder set and omit the OpenCV and decord backend decoders. **H.264, H.265, and AAC are not available out of the box.** Video encoding (for example, text-to-video output) uses royalty-free VP9.
</Note>

## What a user sees with an unsupported codec

When an input uses a codec that is not built in, Dynamo returns an actionable error rather than a raw decoder failure — for example:

- **Frontend video decode:** *"This deployment decodes only VP8/VP9 video — H.264, H.265, and other codecs are not built into the in-tree FFmpeg. Re-encode the input to VP9 … or run a deployment image built with the required decoder."*
- **vLLM backend (EPD):** *"the backend decoder 'cv2' is not installed … install it with `pip install opencv-python-headless`, or send VP8/VP9 …"*
- **SGLang backend (EPD):** *"the SGLang video path requires the 'decord' decoder … install it with `pip install decord2`, or send images instead of video."*

## Enabling a codec

### Re-encode to VP9 (recommended)

The simplest path is to convert the input to VP9, which the frontend decodes without any extra install:

```bash
ffmpeg -i input.mp4 -c:v libvpx-vp9 -an output.webm
```

### Install a backend decoder (EPD paths)

The vLLM and SGLang encode-prefill-decode (EPD) paths decode video in the backend worker — vLLM via OpenCV, SGLang via decord. These decoders are omitted from the shipped images but can be installed into your own image or at deploy time:

```bash
# vLLM backend worker
pip install opencv-python-headless

# SGLang backend worker
pip install decord2   # provides the `decord` module
```

<Warning>
Installing an additional media decoder brings its bundled codecs into your deployment. You are responsible for any third-party codec licensing obligations that apply to your distribution or use of those codecs.
</Warning>

## See also

- [vLLM Multimodal](multimodal-vllm.md)
- [SGLang Multimodal](multimodal-sglang.md)
- [TensorRT-LLM Multimodal](multimodal-trtllm.md)
- [Encoder Disaggregation](encoder-disaggregation.md)
