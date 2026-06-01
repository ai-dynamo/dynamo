---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Storage for Model Caching on AKS
---

# Storage for Model Caching on AKS

For implementing tiered storage on AKS, you can take advantage of the different storage options available in Azure. This guide covers choosing the right storage for each Dynamo cache type and configuring PVCs.

## Available Storage Options

| Storage Option | Performance | Best For |
|----------------|-------------|----------|
| Local CSI (Ephemeral Disk) | Very high | Fast model caching, warm restarts |
| [Azure Managed Lustre](azure-lustre-csi.md) | Extremely high | Large multi-node models, shared cache |
| [Azure Disk (Managed Disk)](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-disk#create-azure-disk-pvs-using-built-in-storage-classes) | High | Persistent single-writer model cache |
| [Azure Files](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-files#use-a-persistent-volume-for-storage) | Medium | Shared small/medium models |
| [Azure Blob (via Fuse or init)](https://learn.microsoft.com/en-us/azure/aks/azure-csi-driver-volume-provisioning?tabs=dynamic-volume-blob%2Cnfs%2Ckubernetes-secret%2Cnfs-3%2Cgeneral%2Cgeneral2%2Cdynamic-volume-disk%2Cgeneral-disk%2Cdynamic-volume-files%2Cgeneral-files%2Cgeneral-files2%2Cdynamic-volume-files-mid%2Coptimize%2Csmb-share&pivots=csi-blob#create-a-pvc-using-built-in-storage-class) | Low-Medium | Cold model storage, bootstrap downloads |

> [!NOTE]
> Azure Managed Lustre and Local CSI (ephemeral disk) are not installed by default in AKS and require additional setup before use. Azure Disk, Azure Files, and Azure Blob CSI drivers are available out of the box. See the [Azure Lustre CSI Driver](azure-lustre-csi.md) guide for Lustre setup, or the [AKS CSI storage options documentation](https://learn.microsoft.com/azure/aks/csi-storage-drivers) for a full overview of built-in drivers.

For Azure Managed Lustre setup, see the [Azure Lustre CSI Driver](azure-lustre-csi.md) guide.

## Recommendations by Cache Type

- **Model Cache** — raw model artifacts, configuration files, tokenizers, etc.
  - Persistence: Required to avoid repeated downloads and reduce cold-start latency.
  - Recommended storage: Azure Managed Lustre (shared, high throughput) or Azure Disk (single-replica, persistent).

- **Compilation Cache** — backend-specific compiled artifacts (e.g., TensorRT engines).
  - Persistence: Optional.
  - Recommended storage: Local CSI (fast, node-local) or Azure Disk (persistent when GPU configuration is fixed).

- **Performance Cache** — runtime tuning and profiling data.
  - Persistence: Not required.
  - Recommended storage: Local CSI (or other ephemeral storage).

## Check Available Storage Classes

List the storage classes available in your AKS cluster:

```bash
kubectl get storageclass

NAME                           PROVISIONER                 RECLAIMPOLICY
azureblob-csi                  blob.csi.azure.com          Delete
azurefile                      file.csi.azure.com          Delete
azurefile-csi                  file.csi.azure.com          Delete
azurefile-csi-premium          file.csi.azure.com          Delete
azurefile-premium              file.csi.azure.com          Delete
default                        disk.csi.azure.com          Delete
managed                        disk.csi.azure.com          Delete
managed-csi                    disk.csi.azure.com          Delete
managed-csi-premium            disk.csi.azure.com          Delete
managed-premium                disk.csi.azure.com          Delete
sc.azurelustre.csi.azure.com   azurelustre.csi.azure.com   Retain
```

## Example PVC Configuration

In the `cache.yaml` in the different [recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes), you can set the `storageClassName` to a storage option available in your AKS cluster:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 100Gi
  storageClassName: "sc.azurelustre.csi.azure.com"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: compilation-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: "azurefile-csi"
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: perf-cache
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: "local-ephemeral"
```

## See Also

- [Azure Lustre CSI Driver](azure-lustre-csi.md) — Full setup guide for Azure Managed Lustre
- [Model Caching](../../model-caching.md) — Full walkthrough for setting up model caching with Dynamo, including download Jobs and mount configuration
- [AKS CSI Storage Drivers](https://learn.microsoft.com/azure/aks/csi-storage-drivers) — Microsoft documentation for all built-in CSI drivers
