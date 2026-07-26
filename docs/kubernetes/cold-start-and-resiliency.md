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
- **[Shadow Engine Failover](#shadow-engine-failover)**: pre-initialized "shadow" engines sharing weights and KV cache can be quickly failed over to on software and hardware failures.
- **[Dynamo Snapshot](./snapshot.md)**: CRIU-based checkpoint/restore of initialized workers, cutting cold starts from minutes to seconds.

**Legend:**

- ✅ : Supported
- 🚧 : Work in Progress / Experimental / Limited
- ❌ : Not started

**How to read this matrix:** ✅ means Dynamo has a supported path for the backend and scope shown in the row or column; it does not mean every deployment topology, workload shape, or integration is generally available. 🚧 covers work in progress, preview, experimental, or narrowly validated paths. Use the per-feature notes and linked guides for setup steps, prerequisites, and limitations.

## Support Matrix

| Backend | [GPU Memory Service](#gpu-memory-service-gms) | [Shadow Engine Failover](#shadow-engine-failover) | [Dynamo Snapshot](./snapshot.md) |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | ✅ |
| **SGLang** | ✅ | 🚧 | ✅ |
| **TensorRT-LLM** | 🚧 | 🚧 | ✅ |

See the per-feature sections below for detailed per-backend status.

## Features

### GPU Memory Service (GMS)

- Out-of-process GPU memory manager for zero-copy sharing of weights and KV across workers on the same GPU; foundation for Shadow Engine Failover. [Architecture](../../lib/gpu_memory_service/README.md)
- In Kubernetes, GMS is wired in via [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), configured through the `gpuMemoryService` field on the `DynamoGraphDeployment` CR.

#### Status

| Backend | Managed memory | Multi-node | Upstream integration |
| :--- | :--- | :---: | :--- |
| **vLLM** | weights, KV | ✅ | 🚧 |
| **SGLang** | weights, KV | ✅ | 🚧 |
| **TensorRT-LLM** | weights | 🚧 | 🚧 |

**Notes:**
- vLLM, SGLang, and TensorRT-LLM require upstream changes to inference that are currently being upstreamed.

### Shadow Engine Failover

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

When used with GMS, weights can be restored in parallel with the CRIU/CUDA checkpoint. Currently, weights are saved/restored via experimental transfer backends implemented via NIXL. [ModelExpress](https://github.com/ai-dynamo/modelexpress) backend integration with GMS is coming soon in a future Dynamo release.

#### Status

| Backend | Single GPU | Multi-GPU, Single Node | Multinode |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ (highly experimental) | 🚧 |
| **SGLang** | ✅ | 🚧 | 🚧 |
| **TensorRT-LLM** | ✅ | 🚧 | 🚧 |

**Notes:**
- Dynamo Snapshot + GMS integration is currently disabled by default and requires both the operator gate `DYN_OPERATOR_ALLOW_GMS_SNAPSHOT=1` and CUDA Driver r610 or later.
- TensorRT-LLM Snapshot is supported only for the experimental single-GPU aggregated text worker path; broader TensorRT-LLM coverage remains work in progress. See [Snapshotting GPU Workers](./snapshot.md) for setup steps, prerequisites, and limitations.
- Multi-GPU, Single Node is available in a highly experimental/slightly limited path that uses legacy IPC only for P2P.
