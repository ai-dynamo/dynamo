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

Kustomize is both the authoring model and documentation for these variants: the
base and Components explain the provider settings, and each public overlay
documents the concrete selection. Cluster users can apply the fully materialized
`deploy-*.yaml` files below directly, or apply a public overlay with Kustomize.
Neither path requires regeneration.

| Rendered manifest | Provider fabric | Overlay |
|-------------------|-----------------|---------|
| `deploy-aks-ib.yaml` | Azure AKS InfiniBand | `kustomize/overlays/aks-ib/` |
| `deploy-aws-p5.48xlarge.yaml` | AWS EFA + libfabric on `p5.48xlarge`, derived from each worker's GPU count | `kustomize/overlays/aws-p5.48xlarge/` |
| `deploy-gke-roce.yaml` | GKE RoCE | `kustomize/overlays/gke-roce/` |
| `deploy-nebius-ib.yaml` | Nebius InfiniBand | `kustomize/overlays/nebius-ib/` |
| `deploy-nscale-ib.yaml` | Nscale InfiniBand | `kustomize/overlays/nscale-ib/` |

For example, apply the GKE RoCE composition directly from its checked-in
Kustomization:

```bash
kubectl apply -k kustomize/overlays/gke-roce -n ${NAMESPACE}
```

To make a local, uncommitted composition, create your own `kustomization.yaml` in
the repository checkout or use `compose` from the repository root:

```bash
scripts/kustomize-matrix.py compose \
  recipes/qwen3-32b/vllm/cloud-providers/kustomize/base \
  recipes/kustomize/components/aks-ib \
  | kubectl apply -f - -n ${NAMESPACE}
```

For recipe contributors, the source of truth is
[`.kustomize-matrix.yaml`](.kustomize-matrix.yaml), `kustomize/base/`, the
recipe-local Components, shared template sources, plus the referenced shared
Components under `recipes/kustomize/components/`. Only contributors update
committed derived artifacts: public overlay `kustomization.yaml` files,
generated Components under their local `components/` directories, and
`deploy-*.yaml` files are generated, committed for review, and must not be
hand-edited. Regenerate them with:

```bash
scripts/kustomize-matrix.py unfold recipes/qwen3-32b/vllm/cloud-providers/.kustomize-matrix.yaml
scripts/kustomize-matrix.py render recipes/qwen3-32b/vllm/cloud-providers/.kustomize-matrix.yaml
```
