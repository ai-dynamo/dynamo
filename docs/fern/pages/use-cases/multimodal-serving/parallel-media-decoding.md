---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Parallel Media Decoding
subtitle: Decode image inputs concurrently in the Rust frontend and transfer pixels to inference backends
---

Parallel media decoding moves image fetching, base64 decoding, and image
decompression from the inference backend to the NVIDIA Dynamo Rust frontend.
The frontend schedules media items on a CPU worker pool and transfers the
decoded pixel buffers to the backend through NIXL.

The backend still runs its model-specific multimodal processor and vision
encoder. This feature changes where image input is decoded; it does not skip
vision encoding.

## Support Matrix

| Input modality | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| Image | Supported | Supported | Supported |
| Video | Not supported | Not supported | Not supported |
| Audio | Not supported | Not supported | Not supported |

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

Do not expect this feature to improve text-only requests or workloads dominated
by model inference. Measure the image fetch and decode path before enabling it
when preprocessing is not an observed bottleneck.

## How It Works

For each request, the frontend:

1. Fetches the image URL or decodes the base64 data URL.
2. Schedules image decompression on the Rayon CPU worker pool.
3. Registers the decoded pixel buffer with NIXL.
4. Sends the buffer descriptor to the selected backend worker.

The backend reads the decoded pixels through NIXL, then continues with its
normal multimodal processor and vision encoder.

## Enable Parallel Media Decoding

Add `--frontend-decoding` to the backend worker command. Do not add the flag to
`dynamo.frontend`; the backend advertises the decoder configuration when it
registers the model.

| Backend | Worker flags | Environment variable |
| --- | --- | --- |
| [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/reference-guide.md) | `--enable-multimodal --frontend-decoding` | `DYN_VLLM_FRONTEND_DECODING=1` |
| [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md) | `--frontend-decoding` | `DYN_SGL_FRONTEND_DECODING=1` |
| [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/reference-guide.md) | `--enable-multimodal --frontend-decoding` | `DYN_TRTLLM_FRONTEND_DECODING=1` |

The CLI flag takes precedence over the corresponding environment variable.

## Requirements and Limitations

- Run the frontend with NIXL and UCX available. The image built from
  [`container/templates/frontend.Dockerfile`](https://github.com/ai-dynamo/dynamo/blob/main/container/templates/frontend.Dockerfile)
  does not include the required UCX media-transfer support.
- Run the frontend on a node with GPU access. NIXL initialization requires
  `libcuda.so.1`, even though image decompression runs on the CPU.
- To fetch images from trusted internal IP addresses or ports, set
  `DYN_MM_ALLOW_INTERNAL=1` on the backend worker. Direct IP addresses and
  nonstandard ports are blocked by default.
- For vLLM, do not combine `--frontend-decoding` with
  `--custom-encoder-class`. The custom encoder consumes image URLs rather than
  frontend-decoded pixel buffers.
- For SGLang, do not combine `--frontend-decoding` with
  `--disaggregation-mode=encode` or `--dedicated-mm-encoder`. Dedicated encode
  workers require the original image URLs.
