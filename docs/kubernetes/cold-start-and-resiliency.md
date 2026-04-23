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
- **[Dynamo Snapshot (ChReK)](#dynamo-snapshot-chrek)**: CRIU-based checkpoint/restore of initialized workers, cutting cold starts from minutes to seconds.

**Legend:**

- ✅ : Supported
- 🚧 : Work in Progress / Experimental / Limited

Blank cells indicate "not started" or "not supported".

## Support Matrix

| Backend | [GPU Memory Service](#gpu-memory-service-gms) | [Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover) | [Dynamo Snapshot](#dynamo-snapshot-chrek) |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | 🚧 |
| **SGLang** | ✅ | 🚧 | 🚧 |
| **TensorRT-LLM** | 🚧 | 🚧 | 🚧 |

See the per-feature sections below for detailed per-backend status.

## Features

### GPU Memory Service (GMS)

- Out-of-process GPU memory manager for zero-copy sharing of weights and KV across workers on the same GPU; foundation for Dynamo Bulwark. [Architecture](../../lib/gpu_memory_service/README.md)
- In Kubernetes, GMS is wired in via [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), configured through the `gpuMemoryService` field on the `DynamoGraphDeployment` CR.

#### Status

| Backend | Managed memory | Multi-node | Upstream integration¹ |
| :--- | :--- | :---: | :--- |
| **vLLM** | weights, KV | ✅ | 🚧 minimal patches² |
| **SGLang** | weights, KV | ✅ | 🚧 patches needed³ |
| **TensorRT-LLM** | weights⁴ | 🚧 | 🚧 patches needed⁵ |

**Notes:**

1. **Upstream integration**: whether the backend's integration lives in the upstream framework or still carries out-of-tree patches that need to land upstream.
2. vLLM carries a small set of out-of-tree patches pending upstreaming.
3. SGLang currently requires monkey-patching for GMS; upstreaming is in progress.
4. TensorRT-LLM manages weights through GMS today; KV cache management through GMS is pending.
5. TensorRT-LLM integration requires out-of-tree patches, targeted for upstream once the shadow-engine flow is validated end-to-end.

### Dynamo Bulwark (Shadow Engine Failover)

- Shadow engines share weights (and soon KV) with a primary via [GMS](#gpu-memory-service-gms) and take over within seconds on primary failure using a kernel-mediated flock for leader election.
- In Kubernetes, configured through the `failover` field on the `DynamoGraphDeployment` CR (on top of `gpuMemoryService`).

#### Status

| Backend | Single Node | Multi-node | Upstream Integration¹ | KV-Cache Reuse² | Hardware Fault Tolerance³ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | ✅ | | |
| **SGLang** | 🚧 | 🚧 | 🚧 | | |
| **TensorRT-LLM** | 🚧 | 🚧 | 🚧 | | |

**Notes:**

1. **Upstream Integration**: whether the backend's integration lives in the upstream framework or still carries out-of-tree patches that need to land upstream. Shared definition with the GMS table.
2. **KV-Cache Reuse**: whether KV cache is remapped across engines on failover (preserving in-flight requests) rather than each shadow starting from a fresh allocation.
3. **Hardware Fault Tolerance**: whether shadow engines are placed on disjoint hardware from the primary, so GPU/node failures are recoverable rather than taking out primary and shadow together.

### Dynamo Snapshot (ChReK)

Dynamo Snapshot (internal name: **ChReK**, *Checkpoint Restore in Kubernetes*) uses CRIU and NVIDIA's `cuda-checkpoint` utility to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from roughly a minute to roughly ten seconds for large LLMs.

See [Dynamo Snapshot](snapshot.md) for usage.
