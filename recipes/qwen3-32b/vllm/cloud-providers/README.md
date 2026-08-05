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

Shared provider Components apply to the backend-neutral `PrefillWorker` and
`DecodeWorker` service keys. Model-specific images, mounts, and command
configuration remain in local Components. The EFA leaf Component includes its
AWS and libfabric parents and names the per-worker EFA request explicitly.

The vLLM command line reads provider-specific values from environment variables
so overlays can patch individual values without replacing the shared argument
list:

- `KV_TRANSFER_CONFIG`
- `GPU_MEMORY_UTILIZATION`
- `HF_HOME`
- transport-specific environment variables such as `UCX_NET_DEVICES` or
  `DYN_KVBM_NIXL_BACKEND_LIBFABRIC`

This avoids replacing the full `args` list in each overlay.

## Applying and maintaining variants

Cluster users select a checked-in `deploy-*.yaml` manifest below and apply it
directly. Those manifests are the stable, reviewable deployment interface; no
Kustomize command is required to consume this recipe.

| Rendered manifest | Provider fabric | Overlay |
|-------------------|-----------------|---------|
| `deploy-aks-ib.yaml` | Azure AKS InfiniBand | `kustomize/overlays/aks-ib/` |
| `deploy-aws-efa-p16d16.yaml` | AWS EFA + libfabric, 16 EFA per worker | `kustomize/overlays/aws-efa-p16d16/` |
| `deploy-gke-roce.yaml` | GKE RoCE | `kustomize/overlays/gke-roce/` |
| `deploy-nebius-ib.yaml` | Nebius InfiniBand | `kustomize/overlays/nebius-ib/` |
| `deploy-nscale-ib.yaml` | Nscale InfiniBand | `kustomize/overlays/nscale-ib/` |

For recipe contributors, the source of truth is
[`.kustomize-matrix.yaml`](.kustomize-matrix.yaml), `kustomize/base/`, the
recipe-local Components, plus the referenced shared
Components under `recipes/kustomize/components/`. The public overlay
`kustomization.yaml` files and `deploy-*.yaml` files are generated, committed
for review, and must not be hand-edited. Regenerate them with:

```bash
scripts/kustomize-matrix.py unfold .kustomize-matrix.yaml
scripts/kustomize-matrix.py render .kustomize-matrix.yaml
```
