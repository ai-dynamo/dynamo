<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2-NVFP4 Recipes

Recipes for [A.X-K2-NVFP4](https://huggingface.co/skt/A.X-K2-NVFP4).

## Configurations

NVIDIA Dynamo 1.4.1 + vLLM 0.26.0 deployment profiles for B200:

| | B200 aggregated chat | B200 disaggregated chat (2P1D) |
| --- | --- | --- |
| **GPU** (per worker) | 4x B200 | 4x B200 |
| **Workers** | 2 aggregate | 2 prefill + 1 decode |
| **Total GPUs** | 8 | 12 |
| **Precision** | NVFP4 + FP8 KV | NVFP4 + FP8 KV |
| **Parallelism** | TP4, DP1, EP disabled | TP4, DP1, EP disabled |
| **Routing** | KV-aware | KV-aware |
| **Speculative decoding** | EAGLE3, 3 tokens | EAGLE3, 3 tokens on both roles |
| **Context length** | 262,144 | 262,144 |
| **Async scheduling** | Enabled | Prefill disabled, decode enabled |
| **KV transfer** | N/A | NIXL/UCX over InfiniBand |

## Supported Features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed**: see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **B200 GPUs**: eight for aggregate; twelve for 2P1D. Disaggregated workers require the `rdma/shared_ib` device resource.
3. **Hugging Face token** with access to `skt/A.X-K2-NVFP4` and `skt/A.X-K2-EAGLE3`.
4. **NGC credentials** with access to the experimental `nvcr.io/nvstaging/nim/ax-k2-nvfp4` runtime image.

## Quick Start

Run the commands from `recipes/ax-k2-nvfp4`.

### 1. Create namespace and secrets

```bash
export NAMESPACE=your-namespace
kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "${NAMESPACE}"
kubectl create secret docker-registry nvcr-imagepullsecret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="your-ngc-api-key" \
  -n "${NAMESPACE}"
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n "${NAMESPACE}"
```

### 3. Download the models

Download the target and EAGLE3 draft checkpoints:

```bash
kubectl apply -f model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/axk2-model-download \
  -n "${NAMESPACE}" --timeout=14400s
```

### 4. Deploy the DGD

Deploy the aggregated profile:

```bash
kubectl apply -f vllm/agg-b200-chat/deploy.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=axk2-agg-b200-chat \
  -n "${NAMESPACE}" --timeout=7200s
```

For 2P1D, follow the
[2P1D deployment instructions](vllm/disagg-b200-chat-2p1d/README.md).
The disaggregated profile requires a `shared-model-cache` PVC containing
both model snapshots in the target namespace.

### 5. Benchmark

See [perf/README.md](perf/README.md) for the AIPerf benchmark workflow:
8K/1K chat, 70% KV reuse, concurrency 32 against the aggregated profile.
