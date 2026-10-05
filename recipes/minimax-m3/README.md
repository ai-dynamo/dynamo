<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3

These recipes serve `nvidia/MiniMax-M3-NVFP4` with vLLM on NVIDIA GB200 GPUs.
Both profiles use tensor parallelism (TP) 4, FP8 KV cache, EAGLE3 speculative
decoding, KV-aware routing, and a one-million-token context limit.

## Deployment Profiles

| Profile | Topology | GPUs | KV Storage |
| --- | --- | ---: | --- |
| `agg-gb200-agentic` | 3 aggregated TP4 replicas | 12x GB200 | GPU plus 400 GB CPU offload per replica |
| `disagg-gb200-agentic` | 3 prefill and 3 decode TP4 replicas | 24x GB200 | GPU with NIXL transfer between prefill and decode |

Both profiles use the `bea41d1a13` runtime image with the required
fixes built in, real EAGLE3 verification, and multimodal input enabled. No
runtime patches are applied. Each ConfigMap also contains a benchmark-only
synthetic-acceptance option; that option is not the production default.
See the [AGG setup](vllm/agg-gb200-agentic/README.md) and
[DISAGG setup](vllm/disagg-gb200-agentic/README.md) for portable Kustomize
rendering, private cluster bindings, and validation scope.

## Prepare the Model Cache

From the Dynamo repository root, select a namespace and create the download
Job's required Hugging Face token Secret. The token must have access to
`nvidia/MiniMax-M3-NVFP4` and `Inferact/MiniMax-M3-EAGLE3-GQA`:

```bash
export NAMESPACE=your-namespace
kubectl create namespace "$NAMESPACE"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "$NAMESPACE"
```

If the namespace and Secret already exist, use them instead. Both deployment
profiles require a populated `shared-model-cache` PVC in that namespace unless
the cluster binding coordinates a different claim for download and serving.

Set `storageClassName` in `model-cache/model-cache.yaml`, then create the cache
with a ReadWriteMany storage class and download the pinned model and EAGLE3
checkpoints:

```bash
kubectl apply -f recipes/minimax-m3/model-cache/model-cache.yaml -n "$NAMESPACE"
kubectl apply -f recipes/minimax-m3/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl wait --for=condition=complete job/minimax-m3-model-download \
  -n "$NAMESPACE" --timeout=14400s
```

## Deploy

Prepare and validate the private cluster Kustomization described
in the [AGG setup](vllm/agg-gb200-agentic/README.md) or
[DISAGG setup](vllm/disagg-gb200-agentic/README.md). The cluster requires the
ComputeDomain controller and DRA driver. Configure namespace-local registry
pull credentials and GB200 placement in the private composition.

With standalone Kustomize v5.8.1, set `CLUSTER_KUSTOMIZATION` to that filled
private directory, then render and apply it:

```bash
kustomize build --load-restrictor LoadRestrictionsNone "$CLUSTER_KUSTOMIZATION" | \
  kubectl apply --dry-run=server -f - -n "$NAMESPACE"
kustomize build --load-restrictor LoadRestrictionsNone "$CLUSTER_KUSTOMIZATION" | \
  kubectl apply -f - -n "$NAMESPACE"
```

Use `LoadRestrictionsNone` only with reviewed paths. The pinned runtime image
is in a private registry. Use an accessible image containing
the required MiniMax M3 support and dependencies if private-registry access is
unavailable, and qualify that image separately.
