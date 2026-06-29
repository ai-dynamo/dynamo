<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Dynamo on Amazon EKS

Supported manifests and cluster templates for the EKS deployment guide.

**Full guide:** [docs/kubernetes/cloud-providers/eks/eks.md](../../../docs/kubernetes/cloud-providers/eks/eks.md)

**Related guides:**

- [Amazon EFS setup](../../../docs/kubernetes/cloud-providers/eks/efs.md)
- [Elastic Fabric Adapter (EFA)](../../../docs/kubernetes/cloud-providers/eks/efa.md)

## Contents

| Path | Description |
|------|-------------|
| `templates/eksctl.yaml` | eksctl cluster config for EKS Auto Mode |
| `automode-np-gpu.yaml` | GPU NodePool for EKS Auto Mode |
| `manifests/vllm/` | vLLM DGD manifests (v1alpha1 and `v1beta1/`) |
| `manifests/model-download/` | Kustomize overlay for model-download Jobs |

## Working Directory

Commands in the guide that reference `templates/`, `manifests/`, or `automode-np-gpu.yaml` assume you are in this directory:

```bash
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo/examples/deployments/EKS
```
