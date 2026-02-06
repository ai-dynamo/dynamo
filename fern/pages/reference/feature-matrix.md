---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Dynamo Feature Compatibility Matrices

This document provides a comprehensive compatibility matrix for key Dynamo features across the supported backends.

*Updated for Dynamo v0.9.0*

**Legend:**
*   ✅ : Supported
*   🚧 : Work in Progress / Experimental / Limited

## Quick Comparison

| Feature | vLLM | TensorRT-LLM | SGLang | Source |
| :--- | :---: | :---: | :---: | :--- |
| **Disaggregated Serving** | ✅ | ✅ | ✅ | [Design Doc](../design-docs/disagg-serving.md) |
| **KV-Aware Routing** | ✅ | ✅ | ✅ | [Router Doc](../router/kv-cache-routing.md) |
| **SLA-Based Planner** | ✅ | ✅ | ✅ | [Planner Doc](../planner/planner-intro.md) |
| **KV Block Manager** | ✅ | ✅ | 🚧 | [KVBM Doc](../kvbm/kvbm-intro.md) |
| **Multimodal (Image)** | ✅ | ✅ | ✅ | [Multimodal Doc](../multimodal/index.md) |
| **Multimodal (Video)** | ✅ | | | [Multimodal Doc](../multimodal/index.md) |
| **Multimodal (Audio)** | 🚧 | | | [Multimodal Doc](../multimodal/index.md) |
| **Request Migration** | ✅ | 🚧 | ✅ | [Migration Doc](../fault-tolerance/request-migration.md) |
| **Request Cancellation** | ✅ | ✅ | 🚧 | Backend READMEs |
| **LoRA** | ✅ | | | [K8s Guide](../kubernetes/deployment/dynamomodel-guide.md) |
| **Tool Calling** | ✅ | ✅ | ✅ | [Tool Calling Doc](../agents/tool-calling.md) |
| **Speculative Decoding** | ✅ | ✅ | 🚧 | Backend READMEs |

## 1. vLLM Backend

vLLM offers the broadest feature coverage in Dynamo, with full support for disaggregated serving, KV-aware routing, KV block management, LoRA adapters, and multimodal inference including video and audio.

*Source: [docs/backends/vllm/README.md](../backends/vllm/README.md)*

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | — | | | | | | |
| **Multimodal** | ✅ | <sup>1</sup> | — | ✅ | — | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | |
| **LoRA** | ✅ | ✅<sup>2</sup> | — | ✅ | — | ✅ | ✅ | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | — | ✅ | — |

> **Notes:**
> 1. **Multimodal + KV-Aware Routing**: The KV router uses token-based hashing and does not yet support image/video hashes, so it falls back to random/round-robin routing. ([Source](../router/kv-cache-routing.md))
> 2. **KV-Aware LoRA Routing**: vLLM supports routing requests based on LoRA adapter affinity.
> 3. **Audio Support**: vLLM supports audio models like Qwen2-Audio (experimental). ([Source](../multimodal/vllm.md))
> 4. **Video Support**: vLLM supports video input with frame sampling. ([Source](../multimodal/vllm.md))
> 5. **Speculative Decoding**: Eagle3 support documented. ([Source](../backends/vllm/speculative-decoding.md))

## 2. SGLang Backend

SGLang is optimized for high-throughput serving with fast primitives, providing robust support for disaggregated serving, KV-aware routing, and request migration.

*Source: [docs/backends/sglang/README.md](../backends/sglang/README.md)*

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | 🚧 | 🚧 | 🚧 | — | | | | | | |
| **Multimodal** | ✅<sup>2</sup> | <sup>1</sup> | — | 🚧 | — | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | 🚧 | ✅ | — | | | | |
| **Request Cancellation** | 🚧<sup>3</sup> | ✅ | ✅ | 🚧 | 🚧 | ✅ | — | | | |
| **LoRA** | | | | 🚧 | | | | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | 🚧 | ✅ | ✅ | ✅ | | — | |
| **Speculative Decoding** | 🚧 | 🚧 | — | 🚧 | — | 🚧 | — | | 🚧 | — |

> **Notes:**
> 1. **Multimodal + KV-Aware Routing**: Not supported. ([Source](../router/kv-cache-routing.md))
> 2. **Multimodal Patterns**: Supports **E/PD** and **E/P/D** only (requires separate vision encoder). Does **not** support simple Aggregated (EPD) or Traditional Disagg (EP/D). ([Source](../multimodal/sglang.md))
> 3. **Request Cancellation**: Cancellation during the remote prefill phase is not supported in disaggregated mode. ([Source](../backends/sglang/README.md))
> 4. **Speculative Decoding**: Code hooks exist (`spec_decode_stats` in publisher), but no examples or documentation yet.

## 3. TensorRT-LLM Backend

TensorRT-LLM delivers maximum inference performance and optimization, with full KVBM integration and robust disaggregated serving support.

*Source: [docs/backends/trtllm/README.md](../backends/trtllm/README.md)*

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | — | | | | | | |
| **Multimodal** | ✅<sup>1</sup> | <sup>2</sup> | — | ✅ | — | | | | | |
| **Request Migration** | 🚧<sup>3</sup> | ✅ | ✅ | ✅ | 🚧 | — | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | |
| **LoRA** | | | | | | | | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | | ✅ | — |

> **Notes:**
> 1. **Multimodal Disaggregation**: Fully supports **EP/D** (Traditional) pattern. **E/P/D** (Full Disaggregation) is WIP and currently supports pre-computed embeddings only. ([Source](../multimodal/trtllm.md))
> 2. **Multimodal + KV-Aware Routing**: Not supported. The KV router currently tracks token-based blocks only. ([Source](../router/kv-cache-routing.md))
> 3. **Request Migration**: Supported on **Decode/Aggregated** workers only. **Prefill** workers do not support migration. ([Source](../backends/trtllm/README.md))
> 4. **Speculative Decoding**: Llama 4 + Eagle support documented. ([Source](../backends/trtllm/llama4-plus-eagle.md))
