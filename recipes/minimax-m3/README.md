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
| `agg-gb200-agentic` | 2 aggregated TP4 replicas | 8x GB200 | GPU plus 400 GB CPU offload per replica |
| `disagg-gb200-agentic` | 3 prefill and 3 decode TP4 replicas | 24x GB200 | GPU with NIXL transfer between prefill and decode |

The aggregated profile applies two vLLM scheduler patches from a ConfigMap
before starting each worker. Its init container verifies the patched file hash
and fails startup if the runtime image no longer matches the expected source.

## Prepare the Model Cache

Set `storageClassName` in `model-cache/model-cache.yaml`, then create the cache
and download the pinned model and EAGLE3 checkpoints:

```bash
kubectl apply -f recipes/minimax-m3/model-cache/model-cache.yaml -n "$NAMESPACE"
kubectl apply -f recipes/minimax-m3/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl wait --for=condition=complete job/minimax-m3-model-download \
  -n "$NAMESPACE" --timeout=14400s
```

## Deploy

Apply one profile:

```bash
kubectl apply \
  -f recipes/minimax-m3/vllm/agg-gb200-agentic/deploy.yaml \
  -n "$NAMESPACE"
```

```bash
kubectl apply \
  -f recipes/minimax-m3/vllm/disagg-gb200-agentic/deploy.yaml \
  -n "$NAMESPACE"
```

These manifests currently contain cluster-specific scheduling, registry,
networking, namespace, and ComputeDomain values. Replace those values for the
target cluster before applying either manifest. The referenced runtime image is
not publicly accessible; substitute an image that contains the required
MiniMax M3 support and dependencies.
