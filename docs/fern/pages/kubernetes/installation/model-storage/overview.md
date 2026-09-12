---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Model Storage Overview
subtitle: Choose Kubernetes storage for model weights, compilation artifacts, and shared caches
---

Dynamo deployments often need shared storage before the first model pod starts. Use this page to choose the right storage path for model weights, backend compilation artifacts, and cache data, then follow the provider-specific setup guide for your cluster.

## What Needs Storage

| Data | Why it matters | Typical requirement |
|---|---|---|
| Model weights and tokenizer files | Avoid repeated downloads and give all workers the same model artifacts. | Persistent storage with enough throughput for cold starts and scale-out. |
| Backend compilation artifacts | Reuse compiled engines or kernels when the same model and GPU shape starts again. | Persistent or fast node-local storage, depending on backend and deployment pattern. |
| Runtime cache and scratch data | Keep temporary data close to workers during serving. | Fast ephemeral or node-local storage. |

Storage is especially important for large models and multi-node deployments. A slow or single-node model store can dominate cold-start time even when the serving stack is configured correctly.

## Choose a Storage Option

| Environment | Start here | Best fit |
|---|---|---|
| AKS | [Storage for Model Caching on AKS](aks-storage.md) | Compare Azure Disk, Azure Files, Azure Blob, local CSI, and Azure Managed Lustre. |
| AKS with high-throughput shared storage | [Azure Lustre CSI Driver](azure-lustre-csi-driver.mdx) | Configure Azure Managed Lustre for large shared model caches. |
| EKS | [Amazon EFS Setup for EKS](efs.mdx) | Share model weights and compilation cache across EKS nodes. |
| Any Kubernetes cluster | [Shared Storage for Model Caching](../install-dynamo.md#shared-storage-for-model-caching) | Wire a cache PVC into Dynamo model loading and deployment templates. |

## Selection Guidelines

- Use high-throughput shared storage for large models that start on multiple nodes or replicas.
- Use persistent storage for model weights when repeated downloads would slow cold starts or stress object storage.
- Use fast ephemeral or node-local storage for scratch data that does not need to survive pod replacement.
- Confirm the storage class supports the access mode your deployment needs, such as `ReadWriteMany` for shared model caches.
- Pin model downloads to an immutable commit rather than a mutable branch or tag.
  Restrict write access to shared model storage and mount reviewed snapshots
  read-only in serving pods where possible.
- Validate throughput during a scale-out test, not only with a single pod.

The snapshot path records the resolved model commit, but its contents remain
trustworthy only while untrusted principals cannot replace files in the backing
storage. See [Secure Model and Image Provenance](../../model-deployment/introduction.mdx#secure-model-and-image-provenance).

## Next Steps

1. Pick the storage guide that matches your cloud provider.
2. Create or verify the storage class and persistent volume claims.
3. Follow [Shared Storage for Model Caching](../install-dynamo.md#shared-storage-for-model-caching) to connect the storage to a Dynamo deployment.
