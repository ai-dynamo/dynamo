---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: AKS Spot VMs
---

# Running Dynamo on AKS Spot VMs

[Azure Spot VMs](https://azure.microsoft.com/en-us/products/virtual-machines/spot) offer significant cost savings for GPU workloads but can be evicted by Azure at any time. This guide covers the configuration required to schedule Dynamo on Spot VM node pools.

## How AKS Taints Spot Nodes

When a node pool uses Spot VMs, AKS automatically applies the following taint to all nodes in that pool:

```yaml
kubernetes.azure.com/scalesetpriority=spot:NoSchedule
```

This prevents standard workloads from landing on Spot nodes by default. Any pod that should run on a Spot node must explicitly tolerate this taint.

## Required Toleration

Add the following toleration to any workload that should run on Spot nodes:

```yaml
tolerations:
  - key: kubernetes.azure.com/scalesetpriority
    operator: Equal
    value: spot
    effect: NoSchedule
```

## Deploying Dynamo on Spot Nodes

The Dynamo platform Helm chart includes a pre-built values file for Spot VM deployments — [`examples/deployments/AKS/values-aks-spot.yaml`](https://github.com/ai-dynamo/dynamo/blob/main/examples/deployments/AKS/values-aks-spot.yaml) — which adds the required toleration to all Dynamo components:

- Dynamo operator controller manager
- Webhook CA inject and cert generation jobs
- etcd
- NATS
- MPI SSH key generation job
- Other core Dynamo platform pods

Install Dynamo with the Spot values file:

```bash
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  --create-namespace \
  -f ./values-aks-spot.yaml
```

To upgrade an existing installation:

```bash
helm upgrade dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz \
  --namespace dynamo-system \
  -f ./values-aks-spot.yaml
```

## Creating a Spot GPU Node Pool

Add a Spot GPU node pool to an existing AKS cluster:

```bash
az aks nodepool add \
  --resource-group <RESOURCE_GROUP> \
  --cluster-name <CLUSTER_NAME> \
  --name spotgpunp \
  --node-count 2 \
  --node-vm-size Standard_NC24ads_A100_v4 \
  --priority Spot \
  --eviction-policy Delete \
  --spot-max-price -1 \
  --skip-gpu-driver-install
```

`--spot-max-price -1` means pay up to the on-demand price (recommended). `--eviction-policy Delete` removes evicted nodes from the pool; use `Deallocate` if you want to preserve node state across evictions.

## See Also

- [Azure Spot VMs overview](https://learn.microsoft.com/en-us/azure/virtual-machines/spot-vms)
- [Use Spot VMs in AKS](https://learn.microsoft.com/en-us/azure/aks/spot-node-pool)
