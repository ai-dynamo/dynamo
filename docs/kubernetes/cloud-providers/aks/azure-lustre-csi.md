---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Azure Lustre CSI Driver for AKS
---

# Azure Lustre CSI Driver for AKS

This guide covers installing and configuring the [Azure Lustre CSI driver](https://github.com/kubernetes-sigs/azurelustre-csi-driver) on an AKS cluster so that Dynamo workloads can use Azure Managed Lustre (AMLFS) filesystems for high-performance model storage.

## Prerequisites

**AKS cluster requirements**
- Kubernetes 1.21 or later
- Node pools must use the **Ubuntu** OS SKU — Windows and Azure Linux (CBL Mariner) nodes are not supported
- AKS is the only supported Kubernetes distribution (self-managed clusters are not supported)

**Tools**
- Azure CLI (`az`)
- `kubectl`

**Network connectivity**

AKS and your AMLFS filesystem must have network reachability. Two supported topologies:

- **VNet peering**: Deploy AKS in its own VNet and peer it with the AMLFS VNet. The AKS infrastructure VNet lives in the auto-created resource group `MC_<aks-rg>_<aks-name>_<region>`.
- **Shared VNet**: Use AKS's "Bring your own VNet" feature and deploy AKS in a dedicated subnet inside the AMLFS VNet. Do not use the same subnet as AMLFS.

> [!WARNING]
> Do not place AKS nodes and the AMLFS filesystem in the same subnet, even when sharing a VNet.

## Step 1: Connect to your AKS cluster

```bash
az login

az aks get-credentials \
  --subscription <SUBSCRIPTION_ID> \
  --resource-group <AKS_RESOURCE_GROUP> \
  --name <AKS_CLUSTER_NAME>

kubectl config current-context
```

## Step 2: Install the CSI driver

There is no Helm chart. Install via the provided shell script:

```bash
# Install latest version
curl -skSL https://raw.githubusercontent.com/kubernetes-sigs/azurelustre-csi-driver/main/deploy/install-driver.sh | bash -s main

# Or install a specific version
curl -skSL https://raw.githubusercontent.com/kubernetes-sigs/azurelustre-csi-driver/main/deploy/install-driver.sh | bash -s v0.3.1
```

The script deploys the CSI controller (2-replica Deployment) and node plugin (DaemonSet) into `kube-system`, and waits for them to become ready.

**Verify the installation:**

```bash
# Controller pods — expect 2/2 or 3/3 Running
kubectl get -n kube-system pod -l app=csi-azurelustre-controller

# Node plugin pods — expect 3/3 Running on each node
kubectl get -n kube-system pod -l app=csi-azurelustre-node -o wide
```

## Step 3: Configure storage

There are two provisioning modes depending on whether your AMLFS filesystem already exists.

### Option A: Static provisioning (existing AMLFS filesystem)

Use this when you want to bring your own Azure Managed Lustre filesystem. If you don't have one yet, create it first, then configure the CSI driver to use it.

#### Create an Azure Managed Lustre filesystem

**1. Register the resource provider (first time only):**

```bash
az provider register --namespace Microsoft.StorageCache
# Wait until state is "Registered"
az provider show --namespace Microsoft.StorageCache --query "registrationState"
```

**2. Validate your subnet before creating the filesystem:**

The subnet must be dedicated to AMLFS (do not share with AKS nodes or other resources) and sized to hold the filesystem. Check requirements first:

```bash
# Get the required subnet size for your planned SKU and capacity
az amlfs get-subnets-size \
  --sku AMLFS-Durable-Premium-250 \
  --storage-capacity 16

# Validate that your subnet meets the requirements
az amlfs check-amlfs-subnet \
  --filesystem-subnet /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RG>/providers/Microsoft.Network/virtualNetworks/<VNET>/subnets/<SUBNET> \
  --sku AMLFS-Durable-Premium-250 \
  --location <REGION> \
  --storage-capacity 16
```

**3. Create a dedicated subnet for AMLFS:**

AMLFS requires its own subnet — it cannot share the subnet used by AKS nodes. Create a new subnet in the AKS VNet (or in a peered VNet):

```bash
# Get the node resource group and check for a custom VNet subnet
az aks show \
  --name <AKS_CLUSTER_NAME> \
  --resource-group <AKS_RESOURCE_GROUP> \
  --query "{vnet: agentPoolProfiles[0].vnetSubnetId, nodeRG: nodeResourceGroup}"
```

If `vnet` is non-null, your cluster uses Azure CNI with a custom VNet — use that VNet name and resource group below.

If `vnet` is `null`, AKS manages its own VNet in the node resource group. Find it:

```bash
az network vnet list \
  --resource-group <NODE_RESOURCE_GROUP> \
  --query "[].{name:name, addressPrefixes:addressSpace.addressPrefixes}"
```

List existing subnets to find a free CIDR range:

```bash
az network vnet subnet list \
  --resource-group <VNET_RESOURCE_GROUP> \
  --vnet-name <AKS_VNET_NAME> \
  --query "[].{name:name, prefix:addressPrefix}"
```

Pick a non-overlapping CIDR within the VNet's address space. The `filesystemSubnetSize` value from `get-subnets-size` is the number of IPs required. Azure also reserves 5 IPs per subnet, so add those when sizing the prefix (e.g., `filesystemSubnetSize: 8` → 13 IPs needed → use `/28` for 16 addresses or more).

Then create the dedicated AMLFS subnet:

```bash
az network vnet subnet create \
  --name amlfs-subnet \
  --resource-group <VNET_RESOURCE_GROUP> \
  --vnet-name <AKS_VNET_NAME> \
  --address-prefix <CIDR>
```

Use the full subnet resource ID in the next step:
`/subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<VNET_RESOURCE_GROUP>/providers/Microsoft.Network/virtualNetworks/<AKS_VNET_NAME>/subnets/amlfs-subnet`

**4. Create the filesystem:**

```bash
az amlfs create \
  --name <AMLFS_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --location <REGION> \
  --sku AMLFS-Durable-Premium-250 \
  --storage-capacity 16 \
  --zones "[1]" \
  --filesystem-subnet /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/<RG>/providers/Microsoft.Network/virtualNetworks/<VNET>/subnets/<SUBNET> \
  --maintenance-window "{dayOfWeek:Sunday,timeOfDayUtc:'22:00'}"
```

This takes **10–20 minutes**. Use `--no-wait` to return immediately and poll with `az amlfs show`.

**Available SKUs:**

| SKU | Min size | Throughput |
|-----|----------|------------|
| `AMLFS-Durable-Premium-40` | 48 TiB | 40 MB/s per TiB |
| `AMLFS-Durable-Premium-125` | 16 TiB | 125 MB/s per TiB |
| `AMLFS-Durable-Premium-250` | 8 TiB | 250 MB/s per TiB |
| `AMLFS-Durable-Premium-500` | 4 TiB | 500 MB/s per TiB |

**5. Get the MGS IP address:**

```bash
az amlfs show \
  --name <AMLFS_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --query "{mgsAddress: clientInfo.mgsAddress, mountCommand: clientInfo.mountCommand}"
```

Use the `mgsAddress` value in the StorageClass below. Alternatively, find it in the Azure portal under your filesystem's **Client connection** pane.

**StorageClass:**

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azurelustre-static
provisioner: azurelustre.csi.azure.com
parameters:
  mgs-ip-address: <MGS_IP_ADDRESS>  # From portal > Client connection
reclaimPolicy: Retain
volumeBindingMode: Immediate
mountOptions:
  - noatime
  - flock
```

**PersistentVolumeClaim:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-lustre
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: <AMLFS_STORAGE_CAPACITY>  # Match your filesystem size, e.g. 16Ti
  storageClassName: azurelustre-static
```

```bash
kubectl apply -f storageclass.yaml
kubectl apply -f pvc.yaml
```

### Option B: Dynamic provisioning (auto-create AMLFS filesystem)

Requires driver v0.3.0 or later. The driver creates an AMLFS cluster automatically when the PVC is created — this takes **10+ minutes**.

**Additional IAM permissions required** on the kubelet managed identity (grant before creating the PVC):

```
Microsoft.StorageCache/amlFilesystems/read
Microsoft.StorageCache/amlFilesystems/write
Microsoft.StorageCache/amlFilesystems/delete
Microsoft.StorageCache/checkAmlFSSubnets/action
Microsoft.StorageCache/getRequiredAmlFSSubnetsSize/*
Microsoft.Network/virtualNetworks/subnets/read
Microsoft.Network/virtualNetworks/subnets/join/action
Microsoft.ManagedIdentity/userAssignedIdentities/assign/action
```

Alternatively assign the broader roles: **Reader** at subscription scope, **Contributor** at the target resource group, and **Network Contributor** at the VNet scope.

**Available SKUs:**

| SKU | Throughput |
|-----|------------|
| `AMLFS-Durable-Premium-40` | 40 MB/s per TiB |
| `AMLFS-Durable-Premium-125` | 125 MB/s per TiB (min 48 TiB) |
| `AMLFS-Durable-Premium-250` | 250 MB/s per TiB |
| `AMLFS-Durable-Premium-500` | 500 MB/s per TiB |

**StorageClass:**

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azurelustre-dynamic
provisioner: azurelustre.csi.azure.com
parameters:
  sku-name: "AMLFS-Durable-Premium-125"
  zone: "1"                          # Availability zone: "1", "2", or "3"
  maintenance-day-of-week: "Sunday"
  maintenance-time-of-day-utc: "22:00"
  # Optional overrides (defaults to AKS cluster values):
  # location: "eastus"
  # resource-group-name: "my-rg"
  # vnet-name: "my-vnet"
  # subnet-name: "my-subnet"
reclaimPolicy: Delete   # WARNING: deletes the AMLFS cluster when PVC is deleted — use Retain in production
volumeBindingMode: Immediate
mountOptions:
  - noatime
  - flock
```

**PersistentVolumeClaim:**

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-lustre-dynamic
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 48Ti   # Minimum for AMLFS-Durable-Premium-125
  storageClassName: azurelustre-dynamic
```

```bash
kubectl apply -f storageclass-dynamic.yaml
kubectl apply -f pvc-dynamic.yaml

# Monitor provisioning (takes 10+ minutes)
kubectl describe pvc pvc-lustre-dynamic
```

## Troubleshooting

**Pod stuck in `ContainerCreating`**

```bash
kubectl describe pod <pod-name>
# Look for volume mount errors in Events

kubectl logs -n kube-system -l app=csi-azurelustre-node -c azurelustre --tail=50
```

**PVC stuck in `Pending` (dynamic provisioning)**

```bash
kubectl describe pvc <pvc-name>
# Check Events for authorization errors — kubelet identity may lack IAM permissions
```

**Node cannot mount** — verify Ubuntu OS SKU:

```bash
kubectl get nodes -o custom-columns="NAME:.metadata.name,OS:.status.nodeInfo.osImage"
```

## See also

- [Azure Managed Lustre CSI Driver — GitHub](https://github.com/kubernetes-sigs/azurelustre-csi-driver)
- [Use Azure Managed Lustre with AKS — Microsoft Learn](https://learn.microsoft.com/en-us/azure/azure-managed-lustre/use-csi-driver-kubernetes)
