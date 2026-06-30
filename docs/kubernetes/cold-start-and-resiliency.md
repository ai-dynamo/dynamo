---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Cold Start Optimizations and Resiliency Support Matrix
subtitle: Backend support for GMS, Shadow Engine Failover, and Snapshot
---

Backend status for Dynamo's in-flight features targeting cold-start optimizations and resiliency.

## Overview

Dynamo is building composable primitives across two themes:

- **Cold start**: reducing time-to-serve for initialized LLM workers.
- **Resiliency**: keeping deployments serving through software (and eventually hardware) failures.

This document tracks backend support across three composable projects in this workstream:

- **[GPU Memory Service (GMS)](#gpu-memory-service-gms)**: out-of-process GPU memory manager for
  sharing committed model weights and managing backend-specific GPU memory lifecycles.
- **[Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover)**: pre-initialized shadow engines reuse
  resident model weights after same-node software-process failures. KV cache state is not preserved.
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
| **TensorRT-LLM** | 🚧 | 🚧 | 🚧 |

See the per-feature sections below for detailed per-backend status.

## Features

### GPU Memory Service (GMS)

- Out-of-process GPU memory manager for zero-copy import of committed weights and stable virtual
  address (VA) memory lifecycles on the same GPU; foundation for Dynamo Bulwark. See the
  [GMS architecture](https://github.com/ai-dynamo/dynamo/blob/main/lib/gpu_memory_service/README.md).
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

- Shadow engines reuse committed weights through [GMS](#gpu-memory-service-gms) and take over after
  a primary software-process failure using a kernel-mediated flock for leader election. The
  replacement engine allocates fresh KV cache backing.
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

Dynamo Snapshot uses CRIU and NVIDIA's `cuda-checkpoint` utility to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from minutes to seconds for large LLMs. This feature is enabled via the DynamoCheckpoint custom resource and is natively supported via the Dynamo Graph Deployments (DGDs).

#### Status

| Backend | Single GPU | Multi-GPU, Single Node | Multinode |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ (highly experimental) | 🚧 |
| **SGLang** | ✅ | 🚧 | 🚧 |
| **TensorRT-LLM** | 🚧 | 🚧 | 🚧 |

**Notes:**
- GMS integration is currently gated on a pending driver release.
- CRIU performance is only optimal with the following patches folded into criu-dev: [AIO support](https://github.com/checkpoint-restore/criu/pull/3022) and [parallel memfd support](https://github.com/checkpoint-restore/criu/pull/3021)
- Multi-GPU, Single Node is available in a highly experimental/slightly limited path that uses legacy IPC only for P2P.
