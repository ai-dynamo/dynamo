---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: GMS with Device Plugin and DRA
subtitle: Verify GPU Memory Service shadow engine failover on clusters where the NVIDIA device plugin and the DRA driver allocate the same GPUs.
---

## Prerequisites

This page applies to a Kubernetes cluster that already runs the NVIDIA DRA
driver, so `resource.k8s.io/v1` is served (Kubernetes 1.34.2 or later, per the
GPU Operator DRA documentation), and to a GMS deployment from one of the vLLM
examples:
[agg_gms.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/agg_gms.yaml)
for a single node or
[gms-failover.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/gms-failover.yaml)
for multinode failover. The procedure was verified with Kubernetes v1.35.1,
NVIDIA DRA driver 0.4.1, and Dynamo 1.4.2. You no longer need it once the DRA
nodes run without the device plugin, or once the operator adds a device
selector to the GMS ResourceClaimTemplate.

## Verification-only workaround

> [!WARNING]
> NVIDIA does not support running the device plugin and the DRA driver on the
> same node. Neither allocator sees the other's allocations, so the same GPU
> can be handed to two pods. See the
> [GPU Operator DRA documentation](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/dra-intro-install.html):
> a cluster uses either `GPUCluster` for DRA or `ClusterPolicy` for the device
> plugin, not both.

The supported configuration is to run the DRA driver on nodes where the device
plugin is disabled. If you cannot separate the nodes yet and only want to verify
Shadow Engine Failover, a blocker ResourceClaim works as a verification-only
stopgap: on a node with both allocators, the DRA scheduler can place the GMS
claim on a GPU the device plugin already handed out, and the engine fails at
startup with `Free memory on device ... is less than desired GPU memory
utilization`. Reserving those GPUs with the blocker before deploying GMS steers
the claim to free devices. It does not make the node layout supported, and it
protects one direction only: the device plugin keeps advertising every GPU on
the node, so hold off new `nvidia.com/gpu` workloads on that node while the
GMS claim is live, or a new pod can land on the GPU DRA just gave to GMS. Apply
[gms-dra-blocker.yaml](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/deploy/gms-dra-blocker.yaml)
with the CEL expression and the request `count` adjusted to the occupied device
UUIDs on your node (list them with `nvidia-smi -L`, or map DRA device names to
UUIDs from the ResourceSlices as shown in the manifest comments), then
deploy the GMS example as usual and confirm which GPU each claim received:

```bash
kubectl get resourceclaims -n <namespace> -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{range .status.allocation.devices.results[*]}{.device}{" "}{end}{"\n"}{end}'
```

The GMS claims must not list any GPU you put in the blocker. The blocker pod
holds the devices in the DRA ledger without starting any CUDA process, so the
occupied GPUs keep serving their existing workloads. Delete the blocker Pod and ResourceClaim when the
verification is done. The ResourceClaim spec is immutable, so when the set of
occupied GPUs changes, delete both objects and apply the manifest again with
the updated UUIDs.
