<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

## Motif-3 disaggregated chat

**Experimental.** 3P1D with KV-aware routing on eight NVIDIA B200 GPUs:
three TP2 prefill workers and one TP2 decode worker, with NIXL KV transfer
and real MTP2 verification by default.

Follow the [Fern recipe documentation](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/recipes/model-recipes/motif-3.mdx)
for image access, model-cache setup, deployment, smoke tests, and benchmarking.
After completing the prerequisites, apply the generic manifest from the repository root:

```bash
kubectl apply -f recipes/motif-3/vllm/disagg-b200-chat/deploy-generic.yaml -n "${NAMESPACE}"
```

Edit [kustomize/base](kustomize/base), then regenerate the manifest:

```bash
python3 scripts/kustomize-matrix.py unfold recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
python3 scripts/kustomize-matrix.py render recipes/motif-3/vllm/disagg-b200-chat/.kustomize-matrix.yaml
```
