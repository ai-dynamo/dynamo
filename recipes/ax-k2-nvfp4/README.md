<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2-NVFP4 recipe

This recipe serves [`skt/A.X-K2-NVFP4`](https://huggingface.co/skt/A.X-K2-NVFP4)
with Dynamo and vLLM on B200 GPUs. It is an aggregated deployment with two
independent TP4 workers behind the Dynamo KV-aware router.

## Configuration

| Setting | Value |
| --- | --- |
| GPU | 8x B200 total; 4x B200 per worker |
| Topology | Aggregated, two worker replicas |
| Parallelism | TP4, DP1, expert parallel enabled (effective EP4) |
| Weight precision | NVFP4 |
| KV-cache precision | `auto` (BF16 for this checkpoint/backend) |
| Attention | `FLASHINFER_MLA_SPARSE` |
| Routing | KV-aware, vLLM prefix-cache events, 64-token blocks |
| Context | 262,144 tokens |
| Reasoning parser | `deepseek_v3` |
| Tool-call parser | `hermes` |

Two workers are intentional: one TP4 worker can prove that KV events are
published, but the router needs at least two candidates to exercise KV-aware
placement.

The deployment uses the revision-pinned A.X-K2 development image built from
Dynamo 1.4.1 and vLLM 0.26.0. The image contains the A.X-K2 model port, DSpark
support, and the native FP8 DS-MLA scale-writer fix. This recipe keeps
`--kv-cache-dtype auto`, so the native FP8 DS-MLA fix is available but is not
active in this BF16-KV routing test.

## Prerequisites

1. Install the Dynamo Kubernetes Platform by following the
   [Kubernetes deployment guide](../../docs/fern/kubernetes/quickstart.mdx).
2. Use a cluster with at least eight schedulable B200 GPUs.
3. Create a Hugging Face secret named `hf-token-secret` with an `HF_TOKEN`
   key in the target namespace.
4. Create an image-pull secret named `nvcr-imagepullsecret` that can read the
   branch-specific `nvcr.io/nvstaging/nim` image.

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

## Deploy

Edit `model-cache/model-cache.yaml` and select a ReadWriteMany storage class,
then create the cache and download the revision-pinned model snapshot:

```bash
kubectl apply -f model-cache/model-cache.yaml -n "${NAMESPACE}"
kubectl apply -f model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/axk2-model-download \
  -n "${NAMESPACE}" --timeout=14400s
```

Deploy the aggregate recipe:

```bash
kubectl apply -f vllm/agg-b200-chat/deploy.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Ready pod \
  -l nvidia.com/dynamo-graph-deployment-name=axk2-agg-b200-chat \
  -n "${NAMESPACE}" --timeout=7200s
```

The frontend service is `axk2-agg-b200-chat-frontend:8000`. Run a smoke test
from inside the cluster or port-forward that service before calling
`/v1/chat/completions` with served model name `skt/A.X-K2-NVFP4`.

## Benchmark

The benchmark recipe replays the 8K/1K, 70%-KV-reuse Mooncake no-schedule
trace at concurrency 32 with AIPerf 0.12.0. See
[`perf/README.md`](perf/README.md).

## Implementation notes

- Frontend and worker block sizes must match. Both are explicitly set to 64;
  the sparse MLA backend cannot use vLLM's usual 16-token default.
- `--enable-prefix-caching` and an explicit KV-events configuration are both
  required. Enabling only one does not give the router authoritative worker
  cache state.
- Async scheduling and CUDA graphs remain enabled. `--enforce-eager` and
  `--no-async-scheduling` are intentionally absent.
- Each TP4 worker requests 400 GiB of host memory, matching the proven A.X-K2
  TP4 cluster deployment while allowing both replicas to schedule.
- This branch-specific image is experimental and hosted under `nvstaging`.
  Replace it with the corresponding released runtime image when the A.X-K2
  patches land in a Dynamo release.
