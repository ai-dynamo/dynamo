<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 NVFP4

See the [Motif-3 NVFP4 recipe documentation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx) for deployment, smoke-test, benchmarking, and configuration guidance.

## Nscale B200

The scheduling component selects amd64 nodes and schedules the worker on
`g.192.b200.8` instances with `nvidia.com/gpu.product=NVIDIA-B200`. It tolerates
the `nvidia.com/gpu=true:NoSchedule` taint and requests two GPUs for TP2 serving.
The frontend can run on amd64 CPU nodes. This aggregated deployment uses two
GPUs on one node and does not request RDMA devices.

Use a namespace with a populated
`shared-model-cache` PVC. Reuse that PVC when it already exists. For a new
cache on Nscale, set `storageClassName: vast` in
[`model-cache/model-cache.yaml`](model-cache/model-cache.yaml), then run the
model-download Job before deploying. The frontend and worker run with
Hugging Face offline mode enabled.

From the repository root, apply the Nscale configuration:

```bash
kubectl apply -k recipes/motif-3/vllm/agg-b200-chat/kustomize -n "${NAMESPACE}"
```

To render a complete manifest for sharing:

```bash
kubectl kustomize recipes/motif-3/vllm/agg-b200-chat/kustomize > /tmp/motif-3-nscale-b200.yaml
```

## Configuration Layout

```text
vllm/agg-b200-chat/
├── base/
│   ├── deploy.yaml
│   └── kustomization.yaml
└── kustomize/
    ├── kustomization.yaml
    └── components/
        └── scheduling/
            └── agg/
                ├── kustomization.yaml
                └── patch-dgd.yaml
```

Edit `base/deploy.yaml` for model settings, images, and GPU requests. Edit
`kustomize/components/scheduling/agg/patch-dgd.yaml` for cluster scheduling.
The top-level `kustomize/kustomization.yaml` selects the base and components.
Additional serving modes can add a `disagg/` component alongside `agg/`.

The top-level Kustomization fetches the shared Dynamo OpenAPI schema from GitHub
at a pinned commit on `release/1.5.0`. That schema lets Kustomize merge Dynamo
components and container lists by name. Share the entire `agg-b200-chat/`
directory to share the editable configuration; rendering requires Git and
network access to GitHub. The rendered manifest is self-contained.
