<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Dynamo on Amazon EKS

Supported manifests and cluster templates for the EKS deployment guide.

**Full guide:** [docs/kubernetes/cloud-providers/eks/eks.md](../../../docs/fern/pages/kubernetes/installation/managed-kubernetes/eks/eks-setup.mdx)

**Related guides:**

- [Amazon EFS setup](../../../docs/fern/pages/kubernetes/installation/model-storage/efs.mdx)
- [Elastic Fabric Adapter (EFA)](../../../docs/fern/pages/kubernetes/installation/rdma-setup/efa-on-aws.mdx)

## Contents

| Path | Description |
|------|-------------|
| `templates/eksctl.yaml` | eksctl cluster config for EKS Auto Mode |
| `automode-np-gpu.yaml` | GPU NodePool for EKS Auto Mode |
| `manifests/vllm/` | vLLM `v1beta1` DGD manifests |
| `manifests/model-download/` | Kustomize overlay for model-download Jobs |

## Working Directory

Commands in the guide that reference `templates/`, `manifests/`, or `automode-np-gpu.yaml` assume you are in this directory:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo/examples/deployments/EKS
```
