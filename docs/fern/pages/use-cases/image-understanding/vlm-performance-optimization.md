---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: VLM Performance Optimization
subtitle: Speed up image and video requests with routing, decoding, caching, and encoder disaggregation
---

Once you can serve a VLM in Dynamo (see [Image Understanding](overview.md)),
these features cut the time a vision language model (VLM) spends on multiple
stages in the pipeline: fetching media, finding the right worker, running the
vision encoder, and prefill. Depending on your workload characteristics, you may
benefit from using all of these features or only some of them.

## Where the Time Goes

A VLM request takes six steps. Each feature below speeds up one of them.

![The six steps of a VLM request: the frontend receives the request and the router picks a worker; the worker loads the image, runs the vision encoder, prefills, and decodes. Multimodal KV routing helps when picking a worker, frontend media decoding and the shared image cache help when loading the image, and the embedding cache and encoder disaggregation help when encoding.](../../../assets/img/vlm-request-path.svg)

## When Each Feature Helps

| Feature | Use it when | Speeds up | Applies to |
| --- | --- | --- | --- |
| [Multimodal KV routing](vlm-performance-optimization/multimodal-kv-routing.md) | The same images or clips repeat across requests and you run several workers | Step 2 and prefill: sends repeats to the worker that already holds their KV cache | Image and video |
| [Frontend media decoding](vlm-performance-optimization/frontend-media-decoding.md) | Fetching and decoding media takes a large share of request latency or worker CPU time | Step 3: moves decoding to the Rust frontend | Image and VP8/VP9 video |
| [Shared image cache](vlm-performance-optimization/shared-image-cache.md) | Several workers download the same image URLs | Step 3: reuses downloaded bytes across workers | Image URLs |
| [Embedding cache](vlm-performance-optimization/embedding-cache.md) | The same images or clips repeat and the vision encoder is busy | Step 4: skips re-encoding repeated content | Image and video |
| [Encoder disaggregation](vlm-performance-optimization/encoder-disaggregation.md) | Vision encoding is the bottleneck, or encoders need different hardware than the LLM | Step 4: scales encoders separately from LLM workers | Image (vLLM video stays on the PD path) |

If your workload is all unique media, the caches and routing give little benefit.
Start with frontend media decoding or encoder disaggregation instead.

For video-specific routing, see
[Exact Video KV Routing](../video-understanding/exact-video-kv-routing.md).
To run your own vision tower, see [Custom Vision Encoders](custom-vision-encoders.md).

## Support Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| [Multimodal KV routing](vlm-performance-optimization/multimodal-kv-routing.md) | Image and video | Image, and video on aggregated CUDA workers | Image only |
| [Frontend media decoding](vlm-performance-optimization/frontend-media-decoding.md) | Image and VP8/VP9 video, aggregated | Image, and VP8/VP9 video on CUDA, aggregated | Image, aggregated |
| [Shared image cache](vlm-performance-optimization/shared-image-cache.md) | Image URLs loaded through Dynamo | Not supported for standard URL loading | Image URLs loaded through Dynamo |
| [Embedding cache](vlm-performance-optimization/embedding-cache.md) | Aggregated and E/PD | E/PD | E/PD |
| [Encoder disaggregation](vlm-performance-optimization/encoder-disaggregation.md) | E/PD and E/P/D (images only; video stays on the PD path) | E/PD and E/P/D | E/PD and E/P/D (images and pre-computed embeddings) |

Each feature page lists the supported models, flags, and limitations for its
backends.
