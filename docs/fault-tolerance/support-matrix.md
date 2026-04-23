<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fault Tolerance Support Matrix

Backend support for Dynamo's in-flight fault tolerance features.

This document tracks the status of Dynamo's in-flight fault tolerance features across the supported inference backends.

**Legend:**

- ✅ Supported
- 🚧 In Progress
- ⬜ Not Started

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
- In Kubernetes, GMS is wired in via [Dynamic Resource Allocation](https://kubernetes.io/docs/concepts/scheduling-eviction/dynamic-resource-allocation/), configured through the `gpuMemoryService` field on the `DynamoGraphDeployment` CR. [Operator usage](../kubernetes/gpu-memory-service.md)

#### Status

| Backend | Managed memory | Multi-node | Upstream integration¹ |
| :--- | :--- | :---: | :--- |
| **vLLM** | weights, KV | ✅ | ✅ upstream |
| **SGLang** | weights, KV | ✅ | 🚧 patches needed² |
| **TensorRT-LLM** | weights³ | 🚧 | 🚧 patches needed⁴ |

**Notes:**

1. **Upstream integration**: whether the backend's integration lives in the upstream framework or still carries out-of-tree patches that need to land upstream.
2. SGLang currently requires monkey-patching for GMS; upstreaming is in progress.
3. TensorRT-LLM manages weights through GMS today; KV cache management through GMS is pending.
4. TensorRT-LLM integration requires out-of-tree patches, targeted for upstream once the shadow-engine flow is validated end-to-end.

### Dynamo Bulwark (Shadow Engine Failover)

- Shadow engines share weights (and soon KV) with a primary via [GMS](#gpu-memory-service-gms) and take over within seconds on primary failure using a kernel-mediated flock for leader election. [Architecture](bulwark.md)
- In Kubernetes, configured through the `failover` field on the `DynamoGraphDeployment` CR (on top of `gpuMemoryService`). [Operator usage](../kubernetes/bulwark.md)

#### Status

| Backend | Single Node | Multi-node | Upstream Integration¹ | KV-Cache Reuse² | Hardware Fault Tolerance³ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | ✅ | ⬜ | ⬜ |
| **SGLang** | ⬜ | ⬜ | ⬜ | ⬜ | ⬜ |
| **TensorRT-LLM** | 🚧 | 🚧 | 🚧 | 🚧 | 🚧 |

**Notes:**

1. **Upstream Integration**: whether the backend's integration lives in the upstream framework or still carries out-of-tree patches that need to land upstream. Shared definition with the GMS table.
2. **KV-Cache Reuse**: whether KV cache is remapped across engines on failover (preserving in-flight requests) rather than each shadow starting from a fresh allocation.
3. **Hardware Fault Tolerance**: whether shadow engines are placed on disjoint hardware from the primary, so GPU/node failures are recoverable rather than taking out primary and shadow together.

### Dynamo Snapshot (ChReK)

Dynamo Snapshot (internal name: **ChReK**, *Checkpoint Restore in Kubernetes*) uses CRIU and NVIDIA's `cuda-checkpoint` utility to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from roughly a minute to roughly ten seconds for large LLMs.

See [Dynamo Snapshot](../kubernetes/snapshot.md) for usage.

## Contributing

This matrix is a living document. To update a backend's status:

1. Verify the feature state in the backend (link to a PR, test, or design doc).
2. Update the relevant cell above and add or revise the corresponding footnote.
3. Open a PR and tag the feature owners for review.
