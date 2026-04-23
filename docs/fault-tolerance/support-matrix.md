---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Fault Tolerance Support Matrix
subtitle: Backend support for Dynamo's in-flight fault tolerance features
---

This document tracks the status of Dynamo's in-flight fault tolerance features across the supported inference backends. For request-level fault tolerance (migration, cancellation, graceful shutdown), see the [Fault Tolerance README](README.md).

**Legend:**

- ✅ Supported
- 🚧 Work in Progress / Experimental / Limited
- (blank) Not yet started

## Support Matrix

| Backend | [GPU Memory Service](#gpu-memory-service-gms) | [Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover) | [Dynamo Snapshot](#dynamo-snapshot-chrek) |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | 🚧 <sup>†</sup> |
| **SGLang** | ✅ | 🚧 <sup>2</sup> | 🚧 <sup>†</sup> |
| **TensorRT-LLM** | 🚧 <sup>1</sup> | 🚧 <sup>2</sup> | 🚧 <sup>†</sup> |

### Notes

1. **TensorRT-LLM × GMS:** single-node support works. Multi-node support and KV cache management through GMS are pending.
2. **SGLang / TensorRT-LLM × Bulwark:** shadow-engine failover is in progress; parity with vLLM is the next milestone.

<sup>†</sup> **Dynamo Snapshot column is a stub.** Current row statuses need review by the Dynamo Snapshot owners before this column is authoritative.

## Features

### GPU Memory Service (GMS)

GMS is an out-of-process GPU memory manager that decouples ownership of GPU memory from the processes that use it. This enables:

- **Zero-copy sharing** of GPU memory across multiple processes on the same GPU.
- **Data survival** across process crashes, since memory is owned by the GMS server and not the client worker.
- **Fast model loading** via memory import instead of disk I/O for subsequent workers.

GMS provides PyTorch integration via `CUDAPluggableAllocator` along with pre-built backend integrations. It is the foundation for Dynamo Bulwark today and for upcoming hardware-fault-tolerance layouts.

See [`lib/gpu_memory_service/README.md`](../../lib/gpu_memory_service/README.md) for the full architecture.

### Dynamo Bulwark (Shadow Engine Failover)

Dynamo Bulwark keeps LLM deployments serving through software failures without request disruption. Warm, pre-initialized "shadow" engines share weights (and, in the near term, KV cache) with the primary via GMS + DRA, and take over within seconds when the primary dies, using a kernel-mediated flock for leader election.

Supported topologies:

- **Intra-pod:** active + standby engine containers within a single pod, sharing GPUs and the GMS sidecar.
- **Inter-pod:** rank-dedicated GMS weight servers with N shadow engine pods per rank, for large-scale redundancy and as the basis for hardware-fault recovery.

Configured via the `gpuMemoryService` and `failover` fields on the `DynamoGraphDeployment` CR.

### Dynamo Snapshot (ChReK)

> **Stub.** Owner to expand with full description, supported configurations, and current limitations.

Dynamo Snapshot (internal name: **ChReK**, *Checkpoint Restore in Kubernetes*) uses CRIU and NVIDIA's `cuda-checkpoint` utility to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from roughly a minute to roughly ten seconds for large LLMs.

See [Dynamo Snapshot](../kubernetes/snapshot.md) for usage.

## Contributing

This matrix is a living document. To update a backend's status:

1. Verify the feature state in the backend (link to a PR, test, or design doc).
2. Update the relevant cell above and add or revise the corresponding footnote.
3. Open a PR and tag the feature owners for review.
