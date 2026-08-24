---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parallel Media Decoding
subtitle: Decode image inputs concurrently in the Rust frontend and transfer pixels to inference backends
---

Parallel media decoding moves image fetching, base64 decoding, and image
decompression from the inference backend to the NVIDIA Dynamo Rust frontend.
The frontend decodes images concurrently on a CPU worker pool and transfers
the decoded pixel buffers to the backend through NIXL.

The backend still runs its model-specific multimodal processor and vision
encoder. This feature changes where image input is decoded; it does not skip
vision encoding.

## Support Matrix

| Input modality | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| Image | Agg | Agg | Agg |
| Video | Not supported | Not supported | Not supported |
| Audio | Not supported | Not supported | Not supported |

`Agg` refers to an aggregated worker. The entries in this matrix represent the
supported topologies for frontend decoding.

This matrix describes parallel media decoding, not the overall multimodal
support of each backend. A backend can support video or audio by decoding it on
the worker even when the frontend decoding path does not support that modality.

## When to Use

Use parallel media decoding when image preprocessing consumes a significant
part of request latency or backend CPU time. It is most useful for workloads
with:

- Concurrent requests containing HTTP, HTTPS, or base64-encoded images
- Multiple images in one request
- Backend workers whose request path is constrained by image fetching or
  decompression

Parallel media decoding can also be combined with the [embedding
cache](embedding-cache.md). Frontend decoding reduces image input processing
work, while the embedding cache can skip vision encoding for repeated images.

## How It Works

For each request, the frontend:

1. Fetches the image URL or decodes the base64 data URL.
2. Decodes images concurrently on a CPU worker pool.
3. Registers the decoded pixel buffer with NIXL.
4. Sends the buffer descriptor to the selected backend worker.

The backend reads the decoded pixels through NIXL, then continues with its
normal multimodal processor and vision encoder.

## Enable Parallel Media Decoding

Add `--frontend-decoding` to the backend worker command. Do not add the flag to
`dynamo.frontend`; the backend advertises the decoder configuration when it
registers the model.

| Backend | Worker flags |
| --- | --- |
| [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/reference-guide.md) | `--enable-multimodal --frontend-decoding` |
| [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md) | `--frontend-decoding` |
| [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/reference-guide.md) | `--enable-multimodal --frontend-decoding` |

## Requirements and Limitations

- The published `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.4.1` image installs
  NIXL wheels but does not expose the native NIXL and UCX libraries and plugins
  required by the Rust frontend. It cannot run parallel media decoding as
  shipped. Run the frontend in a backend runtime image or environment where
  NIXL and UCX are configured and available.
