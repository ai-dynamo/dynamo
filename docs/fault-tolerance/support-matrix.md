<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fault Tolerance Support Matrix

Backend support for Dynamo's in-flight fault tolerance features.

This document tracks the status of Dynamo's in-flight fault tolerance features across the supported inference backends. For request-level fault tolerance (migration, cancellation, graceful shutdown), see the [Fault Tolerance README](README.md).

**Legend:**

- ✅ Supported
- 🚧 Work in Progress / Experimental / Limited
- (blank) Not yet started

## Support Matrix

| Backend | [GPU Memory Service](#gpu-memory-service-gms) | [Dynamo Bulwark](#dynamo-bulwark-shadow-engine-failover) | [Dynamo Snapshot](#dynamo-snapshot-chrek) |
| :--- | :---: | :---: | :---: |
| **vLLM** | ✅ | ✅ | 🚧 |
| **SGLang** | ✅ | 🚧 [^bulwark-sglang-trtllm] | 🚧 |
| **TensorRT-LLM** | 🚧 [^gms-trtllm] | 🚧 [^bulwark-sglang-trtllm] | 🚧 |

[^gms-trtllm]: TensorRT-LLM supports GMS on single-node today; multi-node support and KV cache management through GMS are pending. See the [GMS detailed matrix](#per-backend-gms-status) below.
[^bulwark-sglang-trtllm]: Shadow-engine failover for SGLang and TensorRT-LLM is in progress; parity with vLLM is the next milestone.

## Features

### GPU Memory Service (GMS)

- Out-of-process GPU memory manager for zero-copy sharing of weights and KV across workers on the same GPU; foundation for Dynamo Bulwark. [Architecture →](../../lib/gpu_memory_service/README.md)
- In Kubernetes, exposed as a DRA-backed sidecar via the `gpuMemoryService` field on the `DynamoGraphDeployment` CR. [Operator usage →](../kubernetes/gpu-memory-service.md)

#### Per-backend GMS status

| Backend | Managed memory | Multi-node | Upstream integration | KV remap/reuse |
| :--- | :--- | :---: | :--- | :--- |
| **vLLM** | weights, KV | ✅ | ✅ upstream | backlog |
| **SGLang** | weights, KV | ✅ | 🚧 patches needed [^sglang-upstream] | backlog |
| **TensorRT-LLM** | weights [^trtllm-kv] | 🚧 | 🚧 patches needed [^trtllm-upstream] | backlog |

**Column definitions:**

- **Managed memory** — which GPU memory regions the engine allocates through GMS today. Weights are the baseline; KV cache is an additional scope.
- **Multi-node** — whether GMS has been validated across nodes (DRA claims spanning multi-node topologies).
- **Upstream integration** — whether the backend's GMS integration lives upstream in the framework or requires out-of-tree patches.
- **KV remap/reuse** — whether KV cache handles are remapped across engines (e.g. shadow takeover preserving in-flight requests) rather than each engine being handed a fresh allocation.

[^sglang-upstream]: SGLang currently requires monkey-patching for GMS; upstreaming is in progress.
[^trtllm-kv]: TensorRT-LLM manages weights through GMS today; KV cache management through GMS is pending.
[^trtllm-upstream]: TensorRT-LLM integration requires out-of-tree patches, targeted for upstream once the shadow-engine flow is validated end-to-end.

### Dynamo Bulwark (Shadow Engine Failover)

Dynamo Bulwark keeps LLM deployments serving through software failures without request disruption. Warm, pre-initialized "shadow" engines share weights (and, in the near term, KV cache) with the primary via GMS + DRA, and take over within seconds when the primary dies, using a kernel-mediated flock for leader election.

Supported topologies:

- **Intra-pod:** active + standby engine containers within a single pod, sharing GPUs and the GMS sidecar.
- **Inter-pod:** rank-dedicated GMS weight servers with N shadow engine pods per rank, for large-scale redundancy and as the basis for hardware-fault recovery.

Configured via the `gpuMemoryService` and `failover` fields on the `DynamoGraphDeployment` CR.

### Dynamo Snapshot (ChReK)

Dynamo Snapshot (internal name: **ChReK**, *Checkpoint Restore in Kubernetes*) uses CRIU and NVIDIA's `cuda-checkpoint` utility to capture a worker's initialized state once (including GPU memory and CUDA contexts) and restore subsequent workers from that checkpoint, reducing cold starts from roughly a minute to roughly ten seconds for large LLMs.

See [Dynamo Snapshot](../kubernetes/snapshot.md) for usage.

## Contributing

This matrix is a living document. To update a backend's status:

1. Verify the feature state in the backend (link to a PR, test, or design doc).
2. Update the relevant cell above and add or revise the corresponding footnote.
3. Open a PR and tag the feature owners for review.
