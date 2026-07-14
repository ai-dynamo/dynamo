---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: LWS
subtitle: LeaderWorkerSet and DisaggregatedSet integration for multinode Dynamo deployments
---

Dynamo can use [LeaderWorkerSet (LWS)](https://lws.sigs.k8s.io/docs/) as the Kubernetes orchestration layer for multinode workloads. LWS is the lightweight path for spanning one Dynamo worker service across multiple nodes; Dynamo pairs it with [Volcano](https://volcano.sh/) for gang scheduling.

Recent LWS releases also serve the `disaggregatedset.x-k8s.io/v1` API. When that API is installed, Dynamo can place eligible multinode worker roles into a single DisaggregatedSet (DS) instead of creating one LeaderWorkerSet per component.

Use LWS when you want a simpler multinode orchestrator than Grove, or when your cluster already standardizes on LWS and Volcano. Grove remains the default when both Grove and LWS are available.

## Prerequisites

- Kubernetes cluster with GPU nodes.
- LWS version `0.7.0` or newer.
- Volcano installed for gang scheduling.
- Dynamo Kubernetes Platform installed.

If you want the DisaggregatedSet pathway, install an LWS release that serves the `disaggregatedset.x-k8s.io/v1` API. DS availability is detected separately from the LWS + Volcano pathway.

The installation guide includes the exact Helm commands for [LWS and Volcano](installation-guide.md#lws--volcano).

## Orchestrator Selection

For multinode deployments, the Dynamo operator selects an orchestrator based on what is installed:

| Cluster state | Operator behavior |
| --- | --- |
| Grove and LWS installed | Uses Grove by default. |
| Grove and LWS installed, DGD has `nvidia.com/enable-grove: "false"` | Uses LWS. |
| Only LWS installed | Uses LWS. |
| Neither Grove nor LWS installed | Rejects multinode deployments. |

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

Use the DisaggregatedSet path when you want one DS object to own multiple multinode worker roles. This is an opt-in path and does not replace Grove.

To request the DS path:

1. Install an LWS release that serves `disaggregatedset.x-k8s.io/v1`.
2. Add `nvidia.com/enable-disaggregatedset: "true"` to the DGD.
3. If Grove is installed in the cluster, also set `nvidia.com/enable-grove: "false"` so the request does not stay on the Grove path.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: qwen3-disaggset
  annotations:
    nvidia.com/enable-grove: "false"
    nvidia.com/enable-disaggregatedset: "true"
spec:
  # ...
```

Dynamo falls back to the standard DCD pathway when the DS request cannot be honored. Multinode components then require the existing LWS + Volcano pathway; if that pathway is unavailable, reconciliation reports that no multinode orchestrator is available. Common fallback cases are:

- The `disaggregatedset.x-k8s.io/v1` API is not installed.
- Fewer than two eligible multinode worker roles are selected.
- A selected component uses `scalingAdapter`.
- Selected roles mix zero replicas with positive replicas.
- More than 10 eligible multinode worker roles are selected.

DS does not depend on Volcano for API detection. The current non-DS LWS path still requires both LWS and Volcano.

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

For the DS path, Dynamo uses the same multinode pod-template rendering logic that the LWS path uses. The difference is the owning resource shape: one DS can hold multiple selected worker roles, while the LWS path creates one LeaderWorkerSet per selected component.

## Backend Behavior

The operator injects backend-specific multinode settings into the generated LeaderWorkerSet:

| Backend | LWS behavior |
| --- | --- |
| vLLM | Uses Ray for multi-node tensor or pipeline parallelism, and injects data-parallel flags for DP deployments. |
| SGLang | Injects `--dist-init-addr`, `--nnodes`, and per-node `--node-rank`. |
| TensorRT-LLM | Wraps the leader command with `mpirun` and configures worker nodes with SSH. |

For detailed backend-specific behavior and examples, see the [Multinode Deployments](deployment/multinode-deployment.md) guide.
