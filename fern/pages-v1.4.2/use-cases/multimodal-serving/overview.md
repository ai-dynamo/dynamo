---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal Model Serving
subtitle: Deploy multimodal models with image, video, and audio support in Dynamo
---

Dynamo supports multimodal inference across multiple LLM backends, enabling models to process images, video, and audio alongside text.

## Which Feature to Use

Dynamo provides support for improving latency and throughput for multimodal workloads, with image and video inputs, through the following features. Use them together or separately, depending on your workload characteristics:

| Workload | Feature | Benefit |
|----------|---------|---------|
| Workload where significant time is spent preprocessing. | [Parallel media decoding](parallel-media-decoding.md) | Move image fetching and decompression off the worker's critical path. |
| Workload includes repeated multimodal content across requests. | [Embedding cache](embedding-cache.md) | Skip re-encoding repeated multimodal content. |
| Workload includes repeated multimodal content across requests, and multiple backend workers serve multimodal requests. | [Multimodal KV routing](multimodal-kv-routing.md) | Maximize KV cache hit rates for multimodal content. |
| Workload where media encoding is a bottleneck. | [Encoder disaggregation](encoder-disaggregation.md) | Scale encoders independently of LLM workers. |


<Info>
These features currently support image and video inputs only. Support for audio modalities will be added in upcoming releases.
</Info>

## Multimodal Performance Optimization Features

<CardGroup cols={2}>
  <Card title="Parallel Media Decoding" icon="regular image" href="parallel-media-decoding.md">
    Decode image inputs concurrently in the Rust frontend and transfer pixels through NIXL
  </Card>
  <Card title="Embedding Cache" icon="regular database" href="embedding-cache.md">
    Cache vision encoder embeddings to skip re-encoding repeated multimodal content
  </Card>
  <Card title="Multimodal KV Routing" icon="regular arrows-split-up-and-left" href="multimodal-kv-routing.md">
    Include multimodal identity in cache-aware, load-balanced worker selection
  </Card>
  <Card title="Encoder Disaggregation" icon="regular microchip" href="encoder-disaggregation.md">
    Separate vision encoding into a dedicated worker for independent scaling
  </Card>
</CardGroup>

## Example Workflows

Reference implementations for deploying multimodal models for each backend:

- [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md)
- [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md)
- [vLLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md)

To use an author-provided custom vision tower or projector, see [Custom Vision Encoders](../../developer-guide/advanced-customizations/custom-vision-encoders.md).
