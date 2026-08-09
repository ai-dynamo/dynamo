---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LWS
subtitle: LeaderWorkerSet integration for multinode Dynamo deployments
---

Dynamo can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/) as the Kubernetes orchestration layer for multinode workloads. LWS is the lightweight path for spanning one Dynamo worker service across multiple nodes; Dynamo pairs it with [Volcano](https://volcano.sh/) for gang scheduling.

Use LWS when you want a simpler multinode orchestrator than Grove, or when your cluster already standardizes on LWS and Volcano. Grove remains the default when both Grove and LWS are available.

## Prerequisites

- Kubernetes cluster with GPU nodes.
- LWS version `0.7.0` or newer.
- Volcano installed for gang scheduling.
- Dynamo Kubernetes Platform installed.

The installation guide includes the exact Helm commands for [LWS and Volcano](../../../../kubernetes/installation/install-dynamo.md#lws--volcano).

## Orchestrator Selection

For multinode deployments, the operator applies this routing precedence:

1. Grove, when its API is available and `nvidia.com/enable-grove` is not `"false"`.
2. Opt-in DisaggregatedSet (DS), when Grove is not selected, the DGD sets `nvidia.com/enable-disaggregatedset: "true"`, and the DS API and requested roles are supported.
3. The standard DynamoComponentDeployment (DCD) pathway, which requires LWS and Volcano for multinode components.

Installing the DS API does not move existing DGDs to DS. DS requires explicit opt-in, and Grove has higher routing priority.

| Cluster state | Operator behavior |
| --- | --- |
| Grove is available and `nvidia.com/enable-grove` is not `"false"` | Uses Grove. |
| Grove is disabled or unavailable and DS is explicitly requested | Uses DS when the DS API and requested roles are supported. |
| Grove and DS are not selected or DS cannot be used | Uses the standard DCD pathway. |
| No selected pathway supports the multinode components | Rejects the deployment. |

To force the LWS path when Grove is also present:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-multinode-deployment
  annotations:
    nvidia.com/enable-grove: "false"
spec:
  # ...
```

## DisaggregatedSet Path

Use DS when one object should own multiple multinode worker roles. Install an LWS release that serves `disaggregatedset.x-k8s.io/v1`, then add `nvidia.com/enable-disaggregatedset: "true"` to the DGD. If Grove is available and enabled, also set `nvidia.com/enable-grove: "false"`.

Dynamo falls back to the standard DCD pathway when the DS request cannot be honored. Multinode components on that fallback require LWS and Volcano.

## Multinode Spec

Set `multinode.nodeCount` on the service that should span nodes. The total GPU count is `multinode.nodeCount` multiplied by the per-node GPU limit:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-multinode
  annotations:
    nvidia.com/enable-grove: "false"
spec:
  services:
    backend:
      multinode:
        nodeCount: 2
      resources:
        limits:
          gpu: "4"
      extraPodSpec:
        mainContainer:
          args:
            - "--tp-size"
            - "8"
```

In this example, Dynamo asks LWS to place the backend across 2 nodes with 4 GPUs per node, for 8 GPUs total. Make sure your backend's tensor parallel or distributed execution flags match that total.

## Backend Behavior

The operator injects backend-specific multinode settings into the generated LeaderWorkerSet:

| Backend | LWS behavior |
| --- | --- |
| vLLM | Uses PyTorch multiprocessing (mp) for multi-node tensor or pipeline parallelism. Data-parallel flags are injected for DP deployments. |
| SGLang | Injects `--dist-init-addr`, `--nnodes`, and per-node `--node-rank`. |
| TensorRT-LLM | Wraps the leader command with `mpirun` and configures worker nodes with SSH. |

For detailed backend-specific behavior and examples, see the [Multinode Deployments](../../../../kubernetes/model-deployment/multinode-deployments.md) guide.
