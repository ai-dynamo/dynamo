---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Feature Matrix
---

This document provides a comprehensive compatibility matrix for key Dynamo features across the supported backends.

*Updated for Dynamo v1.2.0*

**Legend:**
*   ✅ : Supported
*   🚧 : Work in Progress / Experimental / Limited

## Quick Comparison

| Feature | SGLang | TensorRT-LLM | vLLM | Source |
| :--- | :---: | :---: | :---: | :--- |
| **Disaggregated Serving** | ✅ | ✅ | ✅ | [Design Doc][disagg] |
| **KV-Aware Routing** | ✅ | ✅ | ✅ | [Router Doc][kv-routing] |
| **SLA-Based Planner** | ✅ | ✅ | ✅ | [Planner Doc][planner] |
| **KV Block Manager** | 🚧 | ✅ | ✅ | [KVBM Doc][kvbm] |
| **Multimodal (Image)** | ✅ | ✅ | ✅ | [Multimodal Doc][mm] |
| **Multimodal (Video)** | ✅ | | ✅ | [Multimodal Doc][mm] |
| **Multimodal (Audio)** | | | 🚧 | [Multimodal Doc][mm] |
| **Request Migration** | ✅ | 🚧 | ✅ | [Migration Doc][migration] |
| **Request Cancellation** | 🚧 | ✅ | ✅ | Backend READMEs |
| **LoRA** | | | ✅ | [K8s Guide][lora] |
| **Tool Calling** | ✅ | ✅ | ✅ | [Tool Calling Doc][tools] |
| **Speculative Decoding** | 🚧 | ✅ | ✅ | Backend READMEs |
| **Dynamo Snapshot** | ✅ | | ✅ | [Snapshot Docs][snapshot] |

## 1. vLLM Backend

vLLM offers the broadest feature coverage in Dynamo, with full support for disaggregated serving, KV-aware routing, KV block management, LoRA adapters, and multimodal inference including video and audio.

*Source: [docs/backends/vllm/README.md][vllm-readme]*

| Feature | Disaggregated Serving | KV-Aware Routing | SLA-Based Planner | KV Block Manager | Multimodal | Request Migration | Request Cancellation | LoRA | Tool Calling | Speculative Decoding |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Disaggregated Serving** | — | | | | | | | | | |
| **KV-Aware Routing** | ✅ | — | | | | | | | | |
| **SLA-Based Planner** | ✅ | ✅ | — | | | | | | | |
| **KV Block Manager** | ✅ | ✅ | ✅ | — | | | | | | |
| **Multimodal** | ✅ | ✅<sup>1</sup> | — | ✅ | — | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | | |
| **Request Cancellation** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | | | |
| **LoRA** | ✅ | ✅<sup>2</sup> | — | ✅ | — | ✅ | ✅ | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | — | ✅ | — |

> **Notes:**
> 1. **Multimodal + KV-Aware Routing**: Image-aware KV routing is supported in the documented vLLM paths. The default Rust frontend path supports model families handled by `llm-multimodal`; the Python chat-processor path delegates to vLLM's multimodal processor. ([Source][mm-kv-routing])
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
> 2. **Multimodal Patterns**: Supports simple Aggregated **EPD**, **E/PD**, and **E/P/D** patterns. Traditional Disagg **EP/D** is not supported. ([Source][mm-sglang])
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
| **Multimodal** | ✅<sup>1</sup> | ✅<sup>2</sup> | — | ✅ | — | | | | | |
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | 🚧 | — | | | | |
| **Request Cancellation** | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | — | | | |
| **LoRA** | | | | | | | | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | | ✅ | — |

> **Notes:**
> 1. **Multimodal Disaggregation**: Supports **EP/D** (Traditional) and **E/P/D** (Full Disaggregation) image flows, including image URLs and pre-computed embeddings. ([Source][mm-trtllm])
> 2. **Multimodal + KV-Aware Routing**: Image-aware KV routing is supported through the dedicated TRT-LLM MM Router Worker. It requires KV event publishing on the TRT-LLM workers. ([Source][mm-kv-routing])
> 3. **Request Cancellation**: Due to known issues, the TensorRT-LLM engine is temporarily not notified of request cancellations, meaning allocated resources for cancelled requests are not freed.

---


{/* Backend READMEs — paths relative to rendered URL /resources/feature-matrix */}
[vllm-readme]: ../backends/v-llm
[sglang-readme]: ../backends/sg-lang
[trtllm-readme]: ../backends/tensor-rt-llm

{/* Design Docs */}
[disagg]: ../design-docs/disaggregated-serving
[kv-routing]: ../user-guides/kv-cache-aware-routing
[planner]: ../components/planner
[kvbm]: ../components/kvbm
[migration]: ../user-guides/fault-tolerance/request-migration
[tools]: ../user-guides/tool-calling

{/* Multimodal */}
[mm]: ../user-guides/multimodal
[mm-vllm]: ../features/multimodal/multimodal-vllm.md
[mm-trtllm]: ../features/multimodal/multimodal-trtllm.md
[mm-sglang]: ../features/multimodal/multimodal-sglang.md
[mm-kv-routing]: ../features/multimodal/multimodal-kv-routing.md

{/* Feature-specific */}
[lora]: ../kubernetes-deployment/deploy-models/managing-models-with-dynamo-model
[vllm-spec]: ../additional-resources/speculative-decoding/speculative-decoding-with-v-llm
[trtllm-eagle]: ../additional-resources/tensor-rt-llm-details/llama-4-eagle

{/* Dynamo Snapshot */}
[snapshot]: ../kubernetes-deployment/advanced-platform/snapshot
