---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Feature Matrix
---

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
| **Request Migration** | ✅ | 🚧 | ✅ | [Migration Doc][migration] |
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
| **Request Migration** | ✅ | ✅ | ✅ | ✅ | 🚧 | — | | | | |
| **Request Cancellation** | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | ✅<sup>3</sup> | — | | | |
| **LoRA** | | | | | | | | — | | |
| **Tool Calling** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | | — | |
| **Speculative Decoding** | ✅ | ✅ | — | ✅ | — | ✅ | ✅ | | ✅ | — |

> **Notes:**
> 1. **Multimodal Disaggregation**: Fully supports **EP/D** (Traditional) pattern. **E/P/D** (Full Disaggregation) is WIP and currently supports pre-computed embeddings only. ([Source][mm-trtllm])
> 2. **Multimodal + KV-Aware Routing**: Not supported. The KV router currently tracks token-based blocks only. ([Source][kv-routing])
> 3. **Request Cancellation**: Due to known issues, the TensorRT-LLM engine is temporarily not notified of request cancellations, meaning allocated resources for cancelled requests are not freed.

---

## Source References

{/* Backend READMEs */}
[vllm-readme]: docs/backends/vllm/README.md
[sglang-readme]: docs/backends/sglang/README.md
[trtllm-readme]: docs/backends/trtllm/README.md

{/* Design Docs */}
[disagg]: docs/design_docs/disagg_serving.md
[kv-routing]: docs/components/router/router_guide.md
[planner]: docs/components/planner/README.md
[kvbm]: docs/components/kvbm/README.md
[migration]: docs/fault_tolerance/request_migration.md
[tools]: docs/agents/tool-calling.md

{/* Multimodal */}
[mm]: docs/features/multimodal/README.md
[mm-vllm]: docs/features/multimodal/multimodal_vllm.md
[mm-trtllm]: docs/features/multimodal/multimodal_trtllm.md
[mm-sglang]: docs/features/multimodal/multimodal_sglang.md

{/* Feature-specific */}
[lora]: docs/kubernetes/deployment/dynamomodel-guide.md
[vllm-spec]: docs/features/speculative_decoding/speculative_decoding_vllm.md
[trtllm-eagle]: docs/backends/trtllm/llama4_plus_eagle.md
