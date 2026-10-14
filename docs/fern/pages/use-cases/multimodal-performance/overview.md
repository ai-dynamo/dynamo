---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal Performance
subtitle: Speed up image and video workloads with frontend decoding, caching, routing, and encoder disaggregation
---

These features improve latency and throughput for image and video input. Get a
first request working with [Image Understanding](../image/image-understanding.md)
or [Video Understanding](../video/video-understanding.md) before you add them.
None of them apply to audio yet.

## Which Feature to Use

Use the features together or separately, depending on where your workload spends
its time:

| If your workload… | Use | Benefit |
| --- | --- | --- |
| Spends significant time fetching and decoding media | [Frontend Media Decoding](frontend-media-decoding.md) | Moves fetching and decompression off the worker's critical path |
| Repeats the same images or clips across requests | [Embedding Cache](embedding-cache.md) | Skips re-running the vision encoder for repeated content |
| Repeats media across requests and runs several workers | [Multimodal KV Routing](multimodal-kv-routing.md) | Sends repeated media to the worker that already holds its KV cache |
| Is bottlenecked on vision encoding | [Encoder Disaggregation](encoder-disaggregation.md) | Scales encoders independently of LLM workers |

## Deployment Notation

The pages in this section describe layouts by which stages share a process. A
slash marks a process boundary:

| Notation | Meaning |
| --- | --- |
| EPD | One worker encodes, prefills, and decodes (aggregated) |
| EP/D | The prefill worker also encodes; decode runs separately |
| E/PD | A separate encode worker; one worker prefills and decodes |
| E/P/D | Separate encode, prefill, and decode workers |

## Support Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| [Frontend media decoding](frontend-media-decoding.md) | Image and VP8/VP9 video, aggregated | Image, and VP8/VP9 video on CUDA, aggregated | Image, aggregated |
| [Embedding cache](embedding-cache.md) | Aggregated and E/PD | E/PD | E/PD |
| [Multimodal KV routing](multimodal-kv-routing.md) | Image and video | Image, and video on aggregated CUDA workers | Image only |
| [Encoder disaggregation](encoder-disaggregation.md) | E/PD and E/P/D (images only; video stays on the PD path) | E/PD and E/P/D | E/PD and E/P/D (images and pre-computed embeddings) |

Each feature page lists the supported models, flags, and limitations for its
backends.
