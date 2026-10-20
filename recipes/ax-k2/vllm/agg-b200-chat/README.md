<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 Aggregated Serving

This variant uses two TP4 aggregate workers on eight B200 GPUs. Follow the
[A.X-K2 deployment guide](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/ax-k2.mdx#deploy)
to prepare the model cache and credentials.

## Deploy

Set `CONTEXT` and `NAMESPACE` to your cluster context and namespace. From the
repository root, apply the generated manifest:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f recipes/ax-k2/vllm/agg-b200-chat/deploy-generic.yaml
```

The same configuration can be applied through the generated overlay:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -k recipes/ax-k2/vllm/agg-b200-chat/kustomize/overlays/generic
```

## Edit and render

Edit `kustomize/base/deploy.yaml`; `deploy-generic.yaml` is generated. From
the repository root, regenerate the overlay and manifest:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/ax-k2/vllm/agg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/ax-k2/vllm/agg-b200-chat/.kustomize-matrix.yaml
```
