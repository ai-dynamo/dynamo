---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Multimodal Model Serving
subtitle: Deploy multimodal models with image, video, and audio support in Dynamo
---

Dynamo supports multimodal inference across multiple LLM backends, enabling models to process images, video, and audio alongside text.

<Warning>
**Security Requirement**: Multimodal processing must be explicitly enabled at startup. See the relevant backend documentation ([vLLM](multimodal-vllm.md), [SGLang](multimodal-sglang.md), [TRT-LLM](multimodal-trtllm.md)) for the necessary flags. This prevents unintended processing of multimodal data from untrusted sources.
</Warning>

```mermaid
---
title: Sample flow for an aggregated VLM serving scenario
---
flowchart TD
    A[Request] --> B{KV cache hit?}
    B -->|Yes| C[Use KV]
    B -->|No| D{Embedding cache hit?}
    D -->|Yes| E[Load embedding]
    D -->|No| F[Run encoder]
    F --> G[save to cache]
    G --> H["PREFILL (image tokens + text tokens → KV cache)"]
    E --> H
    C --> I[DECODE]
    H --> I
    I --> J[Response]
```

## Key Features

Dynamo provides support for improving latency and throughput for vision-and-language workloads through the following features, that can be used together or separately, depending on your workload characteristics:
| Feature | Description |
|---------|-------------|
| **[Embedding Cache](embedding-cache.md)** | CPU-side LRU cache that skips re-encoding repeated images |
| **[Encoder Disaggregation](encoder-disaggregation.md)** | Separate vision encoder worker for independent scaling |
| **[Multimodal KV Routing](multimodal-kv-routing.md)** | MM-aware KV cache routing for optimal worker selection |

## Support Matrix

| Stack | Image | Video | Audio |
|-------|-------|-------|-------|
| **[vLLM](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md)** | ✅ | 🧪  | 🧪 |
| **[TRT-LLM](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-trtllm.md)** | ✅ | ❌ | ❌ |
| **[SGLang](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-sglang.md)** | ✅ | 🧪 | ❌ |

**Status:** ✅ Supported | 🧪 Experimental | ❌ Not supported

## Security: URL Validation

All multimodal loaders route remote fetches through a shared URL policy
(`dynamo.common.multimodal.url_validator`). Only
`https://` and `data:` URLs are allowed by default, private / internal IPs are blocked,
and local file access is disabled. Every HTTP redirect hop is re-validated
against the policy.

Two environment variables loosen the defaults for non-public deployments:

| Variable | Default | Effect |
|----------|---------|--------|
| `DYN_MM_ALLOW_INTERNAL` | `0` | Set to `1` to allow `http://` and private / internal IP targets. Intended for on-prem or local-dev setups where media lives on an internal network. |
| `DYN_MM_LOCAL_PATH` | *(empty)* | Absolute directory prefix. When set, `file://` URIs and bare paths are allowed if they resolve inside this prefix. |

<Warning>
**Never set `DYN_MM_ALLOW_INTERNAL=1` on public-facing deployments.** It opens SSRF paths to cloud metadata endpoints (AWS IMDS, GCE, Azure) and other internal services.
</Warning>

## Example Workflows

Reference implementations for deploying multimodal models:

- [vLLM multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/launch) (image, video)
- [TRT-LLM multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/launch)
- [SGLang multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/launch)

## Backend Documentation

Detailed deployment guides, configuration, and examples for each backend:

- **[vLLM Multimodal](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-vllm.md)**
- **[TensorRT-LLM Multimodal](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-trtllm.md)**
- **[SGLang Multimodal](https://github.com/ai-dynamo/dynamo/blob/main/docs/features/multimodal/multimodal-sglang.md)**
