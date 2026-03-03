---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodality
subtitle: Deploy multimodal models with image, video, and audio support in Dynamo
---

Dynamo supports multimodal inference across multiple LLM backends, enabling models to process images, video, and audio alongside text.

<Warning>
**Security Requirement**: Multimodal processing must be explicitly enabled at startup. See the relevant backend documentation for the necessary flags. This prevents unintended processing of multimodal data from untrusted sources.
</Warning>

## Key Features

| Feature | Description |
|---------|-------------|
| **[Embedding Cache](embedding-cache.md)** | CPU-side LRU cache that skips re-encoding repeated images |
| **[Encoder Disaggregation](encoder-disaggregation.md)** | Separate vision encoder worker for independent scaling |
| **[Multimodal KV Routing](multimodal-kv-routing.md)** | MM-aware KV cache routing for optimal worker selection |

## Backend Documentation

Detailed deployment guides, configuration, and examples for each backend:

- **[vLLM Multimodal](multimodal-vllm.md)**
- **[TensorRT-LLM Multimodal](multimodal-trtllm.md)**
- **[SGLang Multimodal](multimodal-sglang.md)**

## Support Matrix

### Backend Capabilities

| Stack | Image | Video | Audio |
|-------|-------|-------|-------|
| **[vLLM](multimodal-vllm.md)** | ✅ | 🧪  | 🧪 |
| **[TRT-LLM](multimodal-trtllm.md)** | ✅ | ❌ | ❌ |
| **[SGLang](multimodal-sglang.md)** | ✅ | ❌ | ❌ |

**Status:** ✅ Supported | 🧪 Experimental | ❌ Not supported

### Input Format Support

| Format | SGLang | TRT-LLM | vLLM |
|--------|--------|---------|------|
| HTTP/HTTPS URL | ✅ | ✅ | ✅ |
| Data URL (Base64) | ❌ | ❌ | ✅ |
| Pre-computed Embeddings (.pt) | ❌ | ✅ | ❌ |

## Example Workflows

Reference implementations for deploying multimodal models:

- [vLLM multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch)
- [TRT-LLM multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/launch)
- [SGLang multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/launch)
- [Experimental multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal/launch) (video, audio)
