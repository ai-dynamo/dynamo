<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 Recipes

Serve [A.X-K2](https://huggingface.co/skt/A.X-K2-NVFP4) with NVIDIA Dynamo and
vLLM on B200 GPUs. These profiles use NVFP4 weights, FP8 KV cache, sparse MLA
attention, KV-aware routing, and EAGLE3 speculative decoding.

The [Fern recipe page](../../docs/fern/pages/recipes/model-recipes/ax-k2.mdx)
provides an interactive topology picker, deployment commands, and a smoke test.

## Configurations

| | [B200 aggregated chat](vllm/agg-b200-chat/deploy.yaml) | [B200 disaggregated chat](vllm/disagg-b200-chat/) |
| --- | --- | --- |
| Workers | 2 aggregate | 2 prefill + 1 decode |
| Total GPUs | 8x B200 | 12x B200 |
| Parallelism | TP4 per worker, DP1, expert parallelism disabled | Same on both roles |
| Precision | NVFP4 weights, FP8 KV | NVFP4 weights, FP8 KV |
| Attention | `FLASHINFER_MLA_SPARSE` | `FLASHINFER_MLA_SPARSE` |
| FlashInfer autotuning | Disabled | Disabled on prefill and decode |
| Speculative decoding | EAGLE3, 3 tokens | EAGLE3, 3 tokens on both roles |
| Async scheduling | Enabled | Prefill disabled, decode enabled |
| Context length | 262,144 tokens | 262,144 tokens |
| Routing | KV-aware | KV-aware |
| KV transfer | N/A | NIXL/UCX over InfiniBand |

Both profiles pin this runtime image for the frontend and workers:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo:1.4.1-ci-30282ab72cc420f8bd7d0d509cb04a1637b80754-vllm-runtime
```

## Supported Features

- Text generation and reasoning (`deepseek_v3` parser)
- Tool calling (`hermes` parser)
- Prefix caching and KV-aware routing
- Disaggregated prefill/decode serving

## Prerequisites

- A Kubernetes cluster with the [Dynamo platform](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx) installed.
- Eight B200 GPUs for aggregated serving or twelve for disaggregated serving,
  with four GPUs and 400 GiB of host memory available per worker.
- For disaggregation, InfiniBand and the `rdma/shared_ib` device resource.
- A ReadWriteMany model cache and a Hugging Face token with access to
  `skt/A.X-K2-NVFP4` and `skt/A.X-K2-EAGLE3`.
- Access to the pinned runtime in `dynamoci.azurecr.io`.

## Quick Start

Run from `recipes/ax-k2`. Set `CONTEXT`, `NAMESPACE`, `HF_TOKEN`,
`REGISTRY_USERNAME`, and `REGISTRY_PASSWORD` for your cluster and credentials.
The `runtime-imagepullsecret` must authenticate to `dynamoci.azurecr.io`.

```bash
kubectl --context "${CONTEXT}" create namespace "${NAMESPACE}"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create secret docker-registry runtime-imagepullsecret \
  --docker-server=dynamoci.azurecr.io \
  --docker-username="${REGISTRY_USERNAME}" \
  --docker-password="${REGISTRY_PASSWORD}"
```

Set `storageClassName` in `model-cache/model-cache.yaml` to your cluster's
ReadWriteMany storage class, then download both pinned checkpoints:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f model-cache/model-cache.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f model-cache/model-download.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Complete \
  job/axk2-model-download --timeout=14400s
```

Both profiles mount the `model-cache` PVC. If your namespace already has a
populated cache under another name, change `claimName` in the download job and
selected deployment source to that PVC and skip creating a new claim.

Choose one deployment:

```bash
# Aggregated: 8 GPUs.
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f vllm/agg-b200-chat/deploy.yaml

# Disaggregated: 12 GPUs.
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f vllm/disagg-b200-chat/deploy-generic.yaml
```

See the [Fern recipe page](../../docs/fern/pages/recipes/model-recipes/ax-k2.mdx)
for readiness checks and a Chat Completions request. See
[benchmark instructions](perf/README.md) for the 8K/1K chat trace, and
[disaggregated source instructions](vllm/disagg-b200-chat/README.md) for Kustomize edits.

## Preview the Documentation

From the repository root, run `bash docs/fern/scripts/preview.sh`.
See [local and remote preview instructions](../../docs/fern/pages/community/contributing/documentation/building-and-publishing.md#running-locally)
for opening the Fern site on a Mac through an SSH tunnel.
