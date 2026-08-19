---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Shadow Engine Failover
subtitle: Keeps a standby engine attached to GPU-resident model weights so process failures recover without reloading the model.
---

> ⚠️ **Experimental Feature**: Shadow Engine Failover is an opt-in preview
> feature. It depends on GPU Memory Service (GMS), Dynamic Resource Allocation
> (DRA), and backend-specific support. Its API shape and behavior may change,
> and the failover state machine is still settling. Use it only for
> non-production evaluation unless you have validated the exact backend,
> topology, and failure mode in your cluster.

## Overview

Use Shadow Engine Failover when you want a standby engine to take over after an
unknown backend engine or software-process failure while the GPU and node remain
healthy. The goal is to avoid paying a full model weight reload after a
same-node process failure.

Shadow Engine Failover is the Kubernetes workflow. GPU Memory Service is the
enabling mechanism underneath it: GMS owns the GPU-resident model weights, and
the active and standby engines attach to those weights through DRA.

This is separate from [Dynamo Snapshot](snapshot.md). Snapshot captures and
restores a process image with CRIU and `cuda-checkpoint`. Shadow Engine Failover
keeps model weights resident in GPU memory so a standby or replacement engine
can attach after selected process-level failures. They both target recovery
latency, but they solve different problems and are not interchangeable.

## Failure Recovery Flow

The following diagram illustrates same-node process-level recovery:

```text
┌──────────────────────── Same healthy node + GPU ───────────────────────┐
│                                                                        │
│  Before failure                                                        │
│  ┌──────────────┐      attach/use      ┌───────────────────────────┐   │
│  │ Engine A     │ ───────────────────▶ │ GMS-owned model weights   │   │
│  │ active       │                      │ resident in GPU memory    │   │
│  └──────┬───────┘                      └────────────┬──────────────┘   │
│         │                                           ▲                  │
│         │                                           │ attach/use       │
│         │ unknown software/engine failure           │                  │
│         ▼                                           │                  │
│  ┌──────────────┐                            ┌──────┴───────┐          │
│  │ Engine A     │ exits                      │ Engine B     │          │
│  └──────────────┘                            │ shadow       │          │
│                                              └──────┬───────┘          │
│                                                     │ takeover         │
│                                                     ▼                  │
│                                              ┌──────────────┐          │
│                                              │ Engine B     │          │
│                                              │ active       │          │
│                                              └──────────────┘          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**How it works:**

1. The operator creates active and standby engine containers or pods for the
   worker, depending on the selected failover mode.
2. The engines share GPU access through DRA and attach to model weights owned by
   GMS.
3. An unknown software or engine failure terminates the active engine, while the
   GMS process, GPU, and node remain healthy.
4. The standby or replacement engine takes over and attaches to the resident
   GMS-owned weights instead of performing a full weight reload.
5. In-flight requests and KV cache state are not preserved. If the GPU, node, or
   GMS process is lost, the replacement worker must use the normal rescheduling
   and model-load path.

## When to Use It Today

- Use it to evaluate same-node recovery from unknown vLLM engine or
  software-process failures.
- Use it when the cost you are trying to avoid is loading another independent
  copy of model weights into GPU memory.
- Use the GMS-only examples to validate backend weight loading through GMS, not
  as a complete failover workflow.
- Do not use it for hardware failure, GPU loss, node loss, cross-node recovery,
  in-flight request recovery, or KV-cache recovery.
- Do not combine it with Snapshot restore. Snapshot plus GMS is not yet
  available.

## GPU Memory Service

GMS moves ownership of GPU-resident model weights out of the engine process and
into a separate GPU memory service. In the failover workflow, this lets the
active and standby engines share the same weight memory boundary instead of
loading independent copies.

Direct GMS enablement is useful for backend integration testing and
pause/resume-style lifecycle experiments. By itself, it does not configure
active/passive failover; use the `failover` field for the shadow engine flow.

## Backend Support

### GPU Memory Service

| Backend | Managed Memory | Multinode | Upstream Integration |
| :--- | :--- | :---: | :---: |
| **vLLM** | Weights and KV | Supported | Work in progress |
| **SGLang** | Weights and KV | Supported | Work in progress |
| **TensorRT-LLM** | Weights | Work in progress | Work in progress |

The backend integrations rely on inference-engine changes that are still being upstreamed.

### Shadow Engine Failover

| Backend | Single Node | Multinode | KV-Cache Reuse | Hardware Fault Tolerance |
| :--- | :---: | :---: | :---: | :---: |
| **vLLM** | Supported | Supported | Not supported | Not supported |
| **SGLang** | Experimental | Experimental | Not supported | Not supported |
| **TensorRT-LLM** | Experimental | Experimental | Not supported | Not supported |

KV-cache reuse means remapping the existing cache into the standby engine so in-flight requests can
continue after failover. Hardware fault tolerance requires placing the active and standby engines on
disjoint hardware. Neither capability is currently supported.

For the cross-feature backend overview, see [Compatibility](../../../../reference/general/compatibility.mdx).

## Prerequisites

- Kubernetes 1.34 or newer with DRA v1 (`resource.k8s.io/v1`) enabled.
- NVIDIA GPU DRA driver installed.
- A matching DRA `DeviceClass`, defaulting to `gpu.nvidia.com`.
- A supported backend image. The current failover examples are vLLM-focused.
- Backend command-line support for GMS loading, such as `--load-format gms`.
- Enough GPU memory for the GMS processes and active or standby engines sharing
  the device.

## Verifying When the Device Plugin and the DRA Driver Coexist

> [!WARNING]
> On clusters where the NVIDIA device plugin also serves `nvidia.com/gpu`
> workloads, the DRA scheduler cannot see device-plugin allocations and can
> place the GMS claim on a GPU that is already fully occupied. The engine then
> fails at startup with a `Free memory on device ... is less than desired GPU
> memory utilization` error, and recreating the pod allocates the same device.

Most existing clusters run the device plugin, so this situation is easy to hit
when trying GMS out. Since Shadow Engine Failover is still experimental, a
temporary workaround is enough to verify the feature: reserve the occupied GPUs
with a blocker ResourceClaim before deploying the GMS workload, so the
[gms-dra-blocker.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/gms-dra-blocker.yaml)
with the CEL expression and the request `count` adjusted to the occupied device
UUIDs on your node (list them with `nvidia-smi -L`, or map DRA device names to
UUIDs from the ResourceSlices as shown in the manifest comments), then
deploy the GMS example as usual and confirm the allocation with
`kubectl get resourceclaims`. The blocker pod holds the devices in the DRA
ledger without starting any CUDA process, so the occupied GPUs keep serving
their existing workloads. Delete the blocker Pod and ResourceClaim when the
verification is done. The ResourceClaim spec is immutable, so when the set of
occupied GPUs changes, delete both objects and apply the manifest again with
the updated UUIDs.

## Limitations

- It is not a general checkpoint/restore system.
- It is not a hardware fault tolerance mechanism for GPU, node, or rack loss.
- It does not diagnose or fix the backend failure.
- It does not preserve in-flight requests, network sockets, or KV cache state.
- It does not make Snapshot restore supported for GPU memory workloads.
- Snapshot plus GMS is temporarily blocked by admission because of known GPU
  driver restore issues.
- It is not covered by the normal v1beta1 compatibility guarantees while it
  lives under `experimental`.

## API Placement

For `v1alpha1` `DynamoGraphDeployment`, GMS and failover are service-level
fields:

```yaml
gpuMemoryService:
  enabled: true
failover:
  enabled: true
```

For `v1beta1`, preview fields are grouped under `experimental` to make the
stability contract explicit:

```yaml
experimental:
  gpuMemoryService:
    mode: IntraPod
  failover:
    mode: IntraPod
```

See the [API reference](../../../../reference/kubernetes-api/full-api-reference.mdx) for the exact schema supported by your
CRD version.

## Basic Shadow Engine Failover Example

Failover builds on GMS. In intra-pod mode, the operator clones the worker's main
container into active and standby engine containers that share GPUs through DRA
and the GMS sidecar. The standby engine takes over when the active engine fails.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-failover
  annotations:
    nvidia.com/dynamo-kube-discovery-mode: container
spec:
  services:
    VllmWorker:
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "2"
      gpuMemoryService:
        enabled: true
      failover:
        enabled: true
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --tensor-parallel-size
            - "2"
            - --load-format
            - gms
```

See the [vLLM failover example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_failover.yaml)
for the full manifest.

## Basic GMS Example

The worker must request GPUs through the normal Dynamo service resources, enable
`gpuMemoryService`, and run a backend command that can load from GMS.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-gms
spec:
  services:
    VllmWorker:
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
      gpuMemoryService:
        enabled: true
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --load-format
            - gms
```

Working GMS-only examples:

- [vLLM GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_gms.yaml)
- [SGLang GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/agg_gms.yaml)

## Related Documentation

- [Snapshot](snapshot.md)
- [API Reference](../../../../reference/kubernetes-api/full-api-reference.mdx)
- [vLLM failover example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_failover.yaml)
- [vLLM GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_gms.yaml)
- [SGLang GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/agg_gms.yaml)
