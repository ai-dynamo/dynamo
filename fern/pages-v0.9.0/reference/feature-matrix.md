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
| **Disaggregated Serving** | ✅ | ✅ | ✅ | [Design Doc][disagg] |
| **KV-Aware Routing** | ✅ | ✅ | ✅ | [Router Doc][kv-routing] |
| **SLA-Based Planner** | ✅ | ✅ | ✅ | [Planner Doc][planner] |
| **KV Block Manager** | ✅ | ✅ | 🚧 | [KVBM Doc][kvbm] |
| **Multimodal (Image)** | ✅ | ✅ | ✅ | [Multimodal Doc][mm] |
| **Multimodal (Video)** | ✅ | | | [Multimodal Doc][mm] |
| **Multimodal (Audio)** | 🚧 | | | [Multimodal Doc][mm] |
| **Request Migration** | 🚧 | 🚧 | ✅ | [Migration Doc][migration] |
| **Request Cancellation** | ✅ | ✅ | 🚧 | Backend READMEs |
| **LoRA** | ✅ | | | [K8s Guide][lora] |
| **Tool Calling** | ✅ | ✅ | ✅ | [Tool Calling Doc][tools] |
| **Speculative Decoding** | ✅ | ✅ | 🚧 | Backend READMEs |

## 1. vLLM Backend

vLLM offers the broadest feature coverage in Dynamo, with full support for disaggregated serving, KV-aware routing, KV block management, LoRA adapters, and multimodal inference including video and audio.

*Source: [docs/backends/vllm/README.md][vllm-readme]*

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
> 1. **Multimodal + KV-Aware Routing**: The KV router uses token-based hashing and does not yet support image/video hashes, so it falls back to random/round-robin routing. ([Source][kv-routing])
> 2. **KV-Aware LoRA Routing**: vLLM supports routing requests based on LoRA adapter affinity.
> 3. **Audio Support**: vLLM supports audio models like Qwen2-Audio (experimental). ([Source][mm-vllm])
> 4. **Video Support**: vLLM supports video input with frame sampling. ([Source][mm-vllm])
> 5. **Speculative Decoding**: Eagle3 support documented. ([Source][vllm-spec])

## 2. SGLang Backend

SGLang is optimized for high-throughput serving with fast primitives, providing robust support for disaggregated serving, KV-aware routing, and request migration.

*Source: [docs/backends/sglang/README.md][sglang-readme]*

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
> 1. **Multimodal + KV-Aware Routing**: Not supported. ([Source][kv-routing])
> 2. **Multimodal Patterns**: Supports **E/PD** and **E/P/D** only (requires separate vision encoder). Does **not** support simple Aggregated (EPD) or Traditional Disagg (EP/D). ([Source][mm-sglang])
> 3. **Request Cancellation**: Cancellation during the remote prefill phase is not supported in disaggregated mode. ([Source][sglang-readme])
> 4. **Speculative Decoding**: Code hooks exist (`spec_decode_stats` in publisher), but no examples or documentation yet.

## 3. TensorRT-LLM Backend

TensorRT-LLM delivers maximum inference performance and optimization, with full KVBM integration and robust disaggregated serving support.

*Source: [docs/backends/trtllm/README.md][trtllm-readme]*

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | — | | | | | | |
| **Multimodal** | ✅<sup>1</sup> | <sup>2</sup> | — | ✅ | — | | | | | |
| **Request Migration** | 🚧<sup>3</sup> | ✅ | ✅ | ✅ | 🚧 | — | | | | |
| **Request Cancellation** | ✅<sup>5</sup> | ✅<sup>5</sup> | ✅<sup>5</sup> | ✅<sup>5</sup> | ✅<sup>5</sup> | ✅<sup>5</sup> | — | | | |
| **LoRA** | | | | | | | | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | | ✅ | — |

> **Notes:**
> 1. **Multimodal Disaggregation**: Fully supports **EP/D** (Traditional) pattern. **E/P/D** (Full Disaggregation) is WIP and currently supports pre-computed embeddings only. ([Source][mm-trtllm])
> 2. **Multimodal + KV-Aware Routing**: Not supported. The KV router currently tracks token-based blocks only. ([Source][kv-routing])
> 3. **Request Migration**: Supported on **Decode/Aggregated** workers only. **Prefill** workers do not support migration. ([Source][trtllm-readme])
> 4. **Speculative Decoding**: Llama 4 + Eagle support documented. ([Source][trtllm-eagle])
> 5. **Request Cancellation**: Due to known issues, the TensorRT-LLM engine is temporarily not notified of request cancellations, meaning allocated resources for cancelled requests are not freed.

{/* Backend READMEs */}
[vllm-readme]: /dynamo/v-0-9-0/components/backends/v-llm
[sglang-readme]: /dynamo/v-0-9-0/components/backends/sg-lang
[trtllm-readme]: /dynamo/v-0-9-0/components/backends/tensor-rt-llm

{/* Design Docs */}
[disagg]: /dynamo/v-0-9-0/design-docs/disaggregated-serving
[kv-routing]: /dynamo/v-0-9-0/user-guides/kv-cache-aware-routing
[planner]: /dynamo/v-0-9-0/components/planner
[kvbm]: /dynamo/v-0-9-0/components/kvbm
[migration]: /dynamo/v-0-9-0/user-guides/fault-tolerance/request-migration
[tools]: /dynamo/v-0-9-0/user-guides/tool-calling

{/* Multimodal */}
[mm]: /dynamo/v-0-9-0/user-guides/multimodality-support
[mm-vllm]: /dynamo/v-0-9-0/user-guides/multimodality-support/v-llm-multimodal
[mm-trtllm]: /dynamo/v-0-9-0/user-guides/multimodality-support/tensor-rt-llm-multimodal
[mm-sglang]: /dynamo/v-0-9-0/user-guides/multimodality-support/sg-lang-multimodal

{/* Feature-specific */}
[lora]: /dynamo/v-0-9-0/kubernetes-deployment/deployment-guide/managing-models-with-dynamo-model
[vllm-spec]: /dynamo/v-0-9-0/additional-resources/speculative-decoding/speculative-decoding-with-v-llm
[trtllm-eagle]: /dynamo/v-0-9-0/additional-resources/tensor-rt-llm-details/llama-4-eagle
