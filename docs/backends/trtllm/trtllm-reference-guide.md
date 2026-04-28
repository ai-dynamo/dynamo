---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Reference Guide
subtitle: Features, configuration, and operational details for the TensorRT-LLM backend
---

## Building a Custom Container

To build a TensorRT-LLM container from source (e.g., for custom modifications or a different CUDA version), see the [Building a Custom Container](./trtllm-building-custom-container.md) guide.

## KV Cache Transfer

Dynamo with TensorRT-LLM supports two methods for transferring KV cache in disaggregated serving: UCX (default) and NIXL (experimental). For detailed information and configuration instructions for each method, see the [KV Cache Transfer Guide](./trtllm-kv-cache-transfer.md).

## Request Migration

Dynamo supports [request migration](../../fault-tolerance/request-migration.md) to handle worker failures gracefully. When enabled, requests can be automatically migrated to healthy workers if a worker fails mid-generation. See the [Request Migration Architecture](../../fault-tolerance/request-migration.md) documentation for configuration details.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources for other requests.

### Cancellation Support Matrix

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../fault-tolerance/request-cancellation.md) documentation.

## Multimodal Support

Dynamo with the TensorRT-LLM backend supports multimodal models, enabling you to process both text and images (or pre-computed embeddings) in a single request. For detailed setup instructions, example requests, and best practices, see the [TensorRT-LLM Multimodal Guide](../../features/multimodal/multimodal-trtllm.md).

## Diffusion Support (Experimental)

Dynamo supports video and image generation using diffusion models through TensorRT-LLM. For requirements, supported models, API usage, and configuration options, see the [Diffusion Guide](./trtllm-diffusion.md).

## Logits Processing

Logits processors let you modify the next-token logits at every decoding step. Dynamo provides a backend-agnostic interface and an adapter for TensorRT-LLM. For the API, examples, and how to bring your own processor, see the [Logits Processing Guide](./trtllm-logits-processing.md).

## DP Rank Routing (Attention Data Parallelism)

TensorRT-LLM supports attention data parallelism for models like DeepSeek, enabling KV-cache-aware routing to specific DP ranks. For configuration and usage details, see the [DP Rank Routing Guide](./trtllm-dp-rank-routing.md).

## KVBM Integration

Dynamo with TensorRT-LLM currently supports integration with the Dynamo KV Block Manager. This integration can significantly reduce time-to-first-token (TTFT) latency, particularly in usage patterns such as multi-turn conversations and repeated long-context requests.

See the instructions here: [Running KVBM in TensorRT-LLM](../../components/kvbm/kvbm-guide.md#run-kvbm-in-dynamo-with-tensorrt-llm).

## Observability

TensorRT-LLM exposes Prometheus metrics for monitoring inference performance. For detailed metrics reference, collection setup, and Grafana integration, see the [Observability Guide](./trtllm-observability.md).

## Known Issues and Mitigations

For known issues, workarounds, and mitigations, see the [Known Issues and Mitigations](./trtllm-known-issues.md) page.
