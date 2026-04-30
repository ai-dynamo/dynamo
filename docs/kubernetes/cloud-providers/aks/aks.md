---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Azure Kubernetes Service (AKS)
---

# Dynamo on AKS

This guide covers setting up an AKS cluster with GPU nodes and deploying Dynamo.

## Prerequisites

- An active Azure subscription with sufficient GPU VM quota
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) (`az`) installed and logged in
- [kubectl](https://kubernetes.io/docs/tasks/tools/) installed
- [Helm](https://helm.sh/docs/intro/install/) v3.0+ installed

## Step 1: Create a Resource Group and Cluster

```bash
az group create \
  --name <RESOURCE_GROUP> \
  --location <REGION>
```

```bash
az aks create \
  --resource-group <RESOURCE_GROUP> \
  --name <CLUSTER_NAME> \
  --node-count 1 \
  --generate-ssh-keys
```

Then get credentials:

```bash
az aks get-credentials \
  --resource-group <RESOURCE_GROUP> \
  --name <CLUSTER_NAME>
```

## Step 2: Add a GPU Node Pool

Add a GPU-enabled node pool with driver installation skipped. The `--skip-gpu-driver-install` flag prevents AKS from managing GPU drivers — the NVIDIA GPU Operator (Step 3) will handle that instead.

```bash
az aks nodepool add \
  --resource-group <RESOURCE_GROUP> \
  --cluster-name <CLUSTER_NAME> \
  --name gpunp \
  --node-count 2 \
  --node-vm-size Standard_NC24ads_A100_v4 \
  --skip-gpu-driver-install
```

For RDMA-capable workloads (disaggregated inference), use ND-series VMs such as `Standard_ND96asr_v4` or `Standard_ND96isr_H100_v5`. See the [RDMA / InfiniBand guide](rdma-infiniband.md) for the additional setup required on those nodes.

For a full list of GPU VM sizes, see [GPU-optimized VM sizes](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes-gpu).

## Step 3: Install the NVIDIA GPU Operator

The GPU Operator manages NVIDIA drivers, container toolkit, device plugin, and monitoring on GPU nodes.

```bash
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update
```

```bash
helm install gpu-operator nvidia/gpu-operator \
  --namespace gpu-operator --create-namespace
```

Verify the pods are running:

```bash
kubectl get pods -n gpu-operator
```

Expected output (abbreviated):

```text
NAMESPACE      NAME                                       READY   STATUS      RESTARTS   AGE
gpu-operator   gpu-feature-discovery-xxxxx                1/1     Running     0          2m
gpu-operator   gpu-operator-xxxxx                         1/1     Running     0          2m
gpu-operator   nvidia-container-toolkit-daemonset-xxxxx   1/1     Running     0          2m
gpu-operator   nvidia-cuda-validator-xxxxx                0/1     Completed   0          1m
gpu-operator   nvidia-device-plugin-daemonset-xxxxx       1/1     Running     0          2m
gpu-operator   nvidia-driver-daemonset-xxxxx              1/1     Running     0          2m
```

> [!NOTE]
> If you need RDMA / InfiniBand for disaggregated inference, **do not install the GPU Operator yet** — the RDMA setup requires different Helm values. See [RDMA / InfiniBand](rdma-infiniband.md) for the full setup, which includes the correct GPU Operator install command.

## Step 4: Install Dynamo

Follow the [Installation Guide](../../installation-guide.md) to install the Dynamo Platform and deploy your first model.

## Additional Guides

### [RDMA / InfiniBand](rdma-infiniband.md)

Required for disaggregated inference in production. Without RDMA, KV cache transfers between prefill and decode workers fall back to TCP with severe latency degradation (~98s TTFT vs ~200–500ms with RDMA). ND-series VMs (e.g., `Standard_ND96asr_v4`, `Standard_ND96isr_H100_v5`) include Mellanox ConnectX InfiniBand NICs but require additional setup beyond the GPU Operator: the NVIDIA Network Operator, a NicClusterPolicy for MOFED drivers, an `ib-node-config` DaemonSet to configure kernel modules and memlock limits, and an RDMA Shared Device Plugin to expose the NICs to pods.

### [Storage for Model Caching](storage.md)

Prevents each pod from independently downloading model weights on startup. Without shared storage, large models take hours to load per pod and will hit HuggingFace rate limits at scale. Covers Azure Managed Lustre, Azure Files, Azure Disk, and Local CSI options with per-cache-type recommendations (model cache, compilation cache, performance cache).

### [Azure Lustre CSI Driver](azure-lustre-csi.md)

The recommended storage for large multi-node models requiring high-throughput shared access. Azure Managed Lustre is not installed by default — this guide covers installing and configuring the Lustre CSI driver before you can use it as a PVC storage class.

### [Spot VMs](spot-vms.md)

Significantly reduces GPU compute costs by running on preemptible Spot VM node pools. AKS automatically taints Spot nodes with `kubernetes.azure.com/scalesetpriority=spot:NoSchedule`, so Dynamo components need explicit tolerations. The Dynamo Helm chart includes a pre-built `values-aks-spot.yaml` that handles this.

## Clean Up Resources

```bash
# Delete all Dynamo Graph Deployments
kubectl delete dynamographdeployments.nvidia.com --all --all-namespaces

# Uninstall Dynamo Platform
export NAMESPACE="dynamo-system"
helm uninstall dynamo-platform -n $NAMESPACE

# If running Dynamo < 1.0 with a separate CRDs chart:
# helm uninstall dynamo-crds -n $NAMESPACE
```

If you want to delete the GPU Operator, follow the [Uninstalling the NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/uninstall.html) guide.

If you want to delete the entire AKS cluster, follow the [Delete an AKS cluster](https://learn.microsoft.com/en-us/azure/aks/delete-cluster) guide.
