<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 Disaggregated Serving

This variant serves A.X-K2-NVFP4 through the Dynamo KV-aware router using
12 B200 GPUs: two TP4 prefill replicas and one TP4 decode replica.

| Role | Replicas | TP per replica | Async scheduling | Speculation |
| --- | ---: | ---: | --- | --- |
| Prefill | 2 | 4 | Disabled (`--no-async-scheduling`) | EAGLE3, 3 tokens |
| Decode | 1 | 4 | Enabled (`--async-scheduling`) | EAGLE3, 3 tokens |

Both roles use DP1, ordinary FP8 KV cache, `FLASHINFER_MLA_SPARSE`, prefix
caching, and the same pinned target and EAGLE3 revisions. Expert parallelism
is disabled. FlashInfer autotuning is disabled on both roles with
`--kernel-config '{"enable_flashinfer_autotune": false}'`.
EAGLE3 uses real acceptance on both roles. The frontend consumes
prefill KV events with a 64-token block size. NIXL transfers KV state using
CUDA buffers and UCX; each worker requests one `rdma/shared_ib` resource.

## Deploy

The namespace must contain the `model-cache` PVC with both pinned
model snapshots and a `runtime-imagepullsecret` that can pull the pinned runtime.
At least three groups of four B200 GPUs and 400 GiB of host memory per worker
must be schedulable. Set `CONTEXT` and `NAMESPACE` to your cluster context and
namespace, then run from this directory:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f deploy-generic.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=axk2-disagg-b200-chat \
  --timeout=7200s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  service/axk2-disagg-b200-chat-frontend 8000:8000
```

Call `/v1/models` and `/v1/chat/completions` through the forwarded port with
model `skt/A.X-K2-NVFP4`. Confirm both prefill replicas receive requests and
the decode replica generates tokens before benchmarking. The deployment and
ConfigMap have distinct names so this variant can coexist with the aggregate recipe.

## Edit and render

Edit `kustomize/base/deploy.yaml`; `deploy-generic.yaml` is generated. From
the repository root, regenerate it with:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```

The same configuration can be applied through the generated overlay:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -k kustomize/overlays/generic
```
