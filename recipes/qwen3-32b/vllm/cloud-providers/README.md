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
| `deploy-aws-p5.48xlarge.yaml` | AWS EFA + libfabric on `p5.48xlarge`, 16 EFA per worker | `kustomize/overlays/aws-p5.48xlarge/` |
| `deploy-gke-a4-dranet.yaml` | GKE A4 managed DRANET with four RDMA NICs per worker | `kustomize/overlays/gke-a4-dranet/` |
| `deploy-gke-roce.yaml` | GKE RoCE | `kustomize/overlays/gke-roce/` |
| `deploy-nebius-ib.yaml` | Nebius InfiniBand | `kustomize/overlays/nebius-ib/` |
| `deploy-nscale-ib.yaml` | Nscale InfiniBand | `kustomize/overlays/nscale-ib/` |

The `gke-a4-dranet` variant targets an `a4-highgpu-8g` node. It requires GKE
Standard 1.34.1-gke.1829001 or later, Dataplane V2, and a node pool with
[GKE managed Dynamic Resource Allocation for Networking
(DRANET)](https://cloud.google.com/kubernetes-engine/docs/how-to/allocate-network-resources-dra)
enabled. Each worker Pod creates a claim for four of the node's eight
`mrdma.google.com` RDMA NICs. The Decode worker has required Pod affinity with
the Prefill worker, placing the 4-GPU workers on the same A4 node. Its UCX
transport allowlist omits TCP and CUDA IPC so a missing RDMA path fails instead
of silently using a slower transport.

Verify the DeviceClass before applying the variant:

```bash
kubectl get deviceclass mrdma.google.com
```

Apply the GKE A4 DRANET composition directly from its checked-in
Kustomization:

```bash
kubectl apply -k kustomize/overlays/gke-a4-dranet -n ${NAMESPACE}
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
recipe-local Components, plus the referenced shared Components under
`recipes/kustomize/components/`. Only contributors update committed derived
artifacts: public overlay `kustomization.yaml` files and `deploy-*.yaml` files
are generated, committed for review, and must not be hand-edited. Regenerate
them with:

```bash
scripts/kustomize-matrix.py unfold .kustomize-matrix.yaml
scripts/kustomize-matrix.py render .kustomize-matrix.yaml
```
