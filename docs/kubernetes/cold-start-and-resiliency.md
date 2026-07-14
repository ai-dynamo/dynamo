<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cold Start Optimizations and Resiliency Support Matrix

Backend status for Dynamo's in-flight features targeting cold-start optimizations and resiliency.

## Overview

Dynamo is building composable primitives across two themes:

- **Cold start**: reducing time-to-serve for initialized LLM workers.
- **Resiliency**: keeping deployments serving through software (and eventually hardware) failures.

This document tracks backend support across three composable projects in this workstream:

- **[GPU Memory Service (GMS)](#gpu-memory-service-gms)**: out-of-process GPU memory manager for zero-copy sharing of weights and KV across worker processes.
- **[Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover)**: pre-initialized "shadow" engines sharing weights and KV cache can be quickly failed over to on software and hardware failures.
- **[Dynamo Snapshot](./snapshot.md)**: CRIU-based checkpoint/restore of initialized workers, cutting cold starts from minutes to seconds.

**Legend:**

- ✅ : Supported
- 🚧 : Work in Progress / Experimental / Limited
- ❌ : Not started

## Support Matrix

| Backend | [GPU Memory Service](#gpu-memory-service-gms) | [Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover) | [Dynamo Snapshot](./snapshot.md) |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | ✅ |
| **SGLang** | ✅ | 🚧 | ✅ |
| **TensorRT-LLM** | 🚧 | 🚧 | ✅ |

See the per-feature sections below for detailed per-backend status.

## Features

### GPU Memory Service (GMS)

- Out-of-process GPU memory manager for zero-copy sharing of weights and KV across workers on the same GPU; foundation for Dynamo Bulwark. [Architecture](../../lib/gpu_memory_service/README.md)
- In Kubernetes, GMS is wired in via [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), configured through the `gpuMemoryService` field on the `DynamoGraphDeployment` CR.

#### Status

| Backend | Managed memory | Multi-node | Upstream integration |
| :--- | :--- | :---: | :--- |
| **vLLM** | weights, KV | ✅ | 🚧 |
| **SGLang** | weights, KV | ✅ | 🚧 |
| **TensorRT-LLM** | weights | 🚧 | 🚧 |

**Notes:**
- vLLM, SGLang, and TensorRT-LLM require upstream changes to inference that are currently being upstreamed.

### Dynamo Bulwark (Shadow Engine Failover)

- Shadow engines share weights (and soon KV) with a primary via [GMS](#gpu-memory-service-gms) and take over within seconds on primary failure using a kernel-mediated flock for leader election.
- In Kubernetes, configured through the `failover` field on the `DynamoGraphDeployment` CR (on top of `gpuMemoryService`).

#### Status

| Backend | Single Node | Multi-node | KV-Cache Reuse | Hardware Fault Tolerance |
| :--- | :---: | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | ❌ | ❌ |
| **SGLang** | 🚧 | 🚧 | ❌ | ❌ |
| **TensorRT-LLM** | 🚧 | 🚧 | ❌ | ❌ |

**Notes:**
- **KV-Cache Reuse**: whether KV cache is remapped across engines on failover (preserving in-flight requests) rather than each shadow starting from a fresh allocation.
- **Hardware Fault Tolerance**: whether shadow engines are placed on disjoint hardware from the primary, so GPU/node failures are recoverable rather than taking out primary and shadow together.

### Dynamo Snapshot

Dynamo Snapshot uses CRIU and CUDA checkpointing capabilities to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from minutes to seconds for large LLMs. This feature is enabled via the DynamoCheckpoint custom resource and is natively supported via the Dynamo Graph Deployments (DGDs).

#### Status

| Backend | Single GPU | Multi-GPU, Single Node | Multinode |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ (highly experimental) | 🚧 |
| **SGLang** | ✅ | 🚧 | 🚧 |
| **TensorRT-LLM** | ✅ | 🚧 | 🚧 |

**Notes:**
- GMS integration is currently disabled by default. CUDA Driver r610 is the minimum required version for Dynamo Snapshot + GMS integration.
- Multi-GPU, Single Node is available in a highly experimental/slightly limited path that uses legacy IPC only for P2P.
