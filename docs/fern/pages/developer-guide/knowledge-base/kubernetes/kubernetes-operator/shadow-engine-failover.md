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

## Snapshot-backed Intra-Pod Failover

The experimental vLLM single-GPU profile can combine a DGD-managed automatic
checkpoint with intra-pod failover. Set `numShadows` to `1` or `2`. The
operator creates two or three destination engines, respectively:

Snapshot restore and active/passive failover remain distinct mechanisms. This
profile composes process restoration with failover election.

| `numShadows` | Restored engine targets |
| :---: | :--- |
| `1` | `engine-0`, `engine-1` |
| `2` | `engine-0`, `engine-1`, `engine-2` |

Checkpoint capture always selects the canonical `main` container. The
destination restore annotation selects only the generated engine containers.
GMS containers and user helpers are not restore targets; they cold-start from
the Pod specification the operator constructed.

Failover restore has two independent gates. The Snapshot restore sentinel
completes CRIU process restoration, but it does not wake the vLLM model. Every
restored engine remains application-paused until it acquires the shared
intra-pod file lock. Only the lock owner wakes the model, registers
model-serving discovery, and publishes routes. The lock remains held while
that engine serves. Each target restores and enters election independently;
there is no cohort-readiness barrier across the restored engines.

This combined profile is limited to an aggregated vLLM generation Worker with
the TCP request plane, TP=PP=DP=1, one node and GPU, intra-pod GMS, and an
automatic checkpoint managed by the same DGD. Manual `checkpointRef` and
InterPod or multinode Snapshot-backed failover are not supported. They are
follow-on work.

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

## Limitations

- It is not a general checkpoint/restore system.
- It is not a hardware fault tolerance mechanism for GPU, node, or rack loss.
- It does not diagnose or fix the backend failure.
- It does not preserve in-flight requests, network sockets, or KV cache state.
- Standalone and ordinary prepared IntraPod GMS Snapshot workflows are
  supported. Snapshot-backed active/passive failover is limited to the narrow
  automatic vLLM IntraPod profile described above. InterPod and multinode
  Snapshot-backed failover are follow-on work.
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

See the [API reference](../../../../reference/kubernetes-api/additional-resources/api-reference-k8s.md) for the exact schema supported by your
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
- [API Reference](../../../../reference/kubernetes-api/additional-resources/api-reference-k8s.md)
- [vLLM failover example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_failover.yaml)
- [vLLM GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_gms.yaml)
- [SGLang GMS example](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/sglang/deploy/agg_gms.yaml)
