<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3-32B vLLM Cloud Provider Overlays

This directory adapts the provider-specific examples from
[`ai-dynamo/dynamo#10202`](https://github.com/ai-dynamo/dynamo/pull/10202) into
the Kustomize recipe layout.

The base keeps the common Qwen3-32B 1P1D vLLM deployment in one
`DynamoGraphDeployment`. Provider overlays patch only the pieces that vary by
fabric: RDMA resources, annotations, host mounts, image selection, and runtime
environment.

The vLLM command line intentionally reads the noisy provider-specific flag
values from ConfigMaps:

- `KV_TRANSFER_CONFIG`
- `GPU_MEMORY_UTILIZATION`
- `HF_HOME`
- transport-specific environment variables such as `UCX_NET_DEVICES` or
  `DYN_KVBM_NIXL_BACKEND_LIBFABRIC`

This avoids replacing the full `args` list in each overlay. The rendered
`deploy-*.yaml` files are checked in for GitHub review and direct `kubectl
apply`.

| Rendered manifest | Provider fabric | Overlay |
|-------------------|-----------------|---------|
| `deploy-aks-ib.yaml` | Azure AKS InfiniBand | `kustomize/overlays/aks-ib/` |
| `deploy-aws-efa.yaml` | AWS EFA + libfabric | `kustomize/overlays/aws-efa/` |
| `deploy-gke-roce.yaml` | GKE RoCE | `kustomize/overlays/gke-roce/` |
| `deploy-nebius-ib.yaml` | Nebius InfiniBand | `kustomize/overlays/nebius-ib/` |
| `deploy-nscale-ib.yaml` | Nscale InfiniBand | `kustomize/overlays/nscale-ib/` |
