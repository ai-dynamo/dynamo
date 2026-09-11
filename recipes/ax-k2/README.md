<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A.X-K2 Recipes

Recipes for [A.X-K2](https://huggingface.co/skt/A.X-K2-NVFP4) on NVIDIA Dynamo +
vLLM, targeting B200 GPUs. Both chat profiles use NVFP4 weights, FP8 KV cache,
sparse MLA attention, KV-aware routing, and EAGLE3 speculative decoding.

Each worker uses TP4 within a node. The aggregated profile routes requests to
two workers; the disaggregated profile uses two prefill workers and one decode
worker, with NIXL/UCX transferring KV state over InfiniBand.

## Configurations

Dynamo + vLLM deployment profiles for the B200 chat workload:

| | [B200 aggregated chat](vllm/agg-b200-chat/deploy.yaml) | [B200 disaggregated chat](vllm/disagg-b200-chat/deploy-generic.yaml) |
| --- | --- | --- |
| **GPU** (per worker) | 4x B200 | 4x B200 on both roles |
| **Replicas** | 2 aggregate workers; 8 GPUs total | 2 prefill + 1 decode; 12 GPUs total |
| **Mode** | Aggregated | Prefill/decode disaggregated |
| **Framework** | vLLM | vLLM |
| **Precision** | NVFP4 weights, FP8 KV | NVFP4 weights, FP8 KV |
| **Parallelism** | TP4, DP1, expert parallelism disabled | Same on both roles |
| **Attention backend** | `FLASHINFER_MLA_SPARSE` | Same on both roles |
| **FlashInfer autotuning** | Disabled | Disabled on prefill and decode |
| **Speculative decoding** | EAGLE3, 3 tokens | EAGLE3, 3 tokens on both roles |
| **Async scheduling** | Enabled | Prefill disabled, decode enabled |
| **Routing** | KV-aware | KV-aware |
| **Prefix caching** | Enabled | Enabled on both roles |
| **KV transfer** | N/A | NIXL/UCX over InfiniBand |
| **Context length** | 262,144 tokens | 262,144 tokens |

## Supported features

- Modalities: Text
- Reasoning (`deepseek_v3` parser)
- Tool calling (`hermes` parser)
- KV-aware routing and prefix caching
- Disaggregated serving

## Prerequisites

1. **Dynamo Platform installed**: see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **B200 GPUs**: eight for aggregated serving or twelve for disaggregated
   serving, with four GPUs and 400 GiB of host memory available per worker.
3. **InfiniBand** for disaggregation, with the `rdma/shared_ib` device resource
   exposed to Kubernetes.
4. **Hugging Face token** with access to `skt/A.X-K2-NVFP4` and
   `skt/A.X-K2-EAGLE3`. See [Download the models](#3-download-the-models).

## Cluster assumptions

Review these values before deploying on your cluster:

| Assumption | Where | Shipped value |
| --- | --- | --- |
| GPU product label | Worker `nodeAffinity` | `NVIDIA-B200` |
| GPU pool taint | Worker `tolerations` | `nvidia.com/gpu=true:NoSchedule` |
| Resources per worker | Container requests | 4 GPUs, 400 GiB host memory |
| InfiniBand device | Disaggregated worker requests and limits | `rdma/shared_ib: 1` |
| Shared model storage | PVC and volume mounts | `model-cache`, 600 GiB, ReadWriteMany |
| Storage class | `model-cache/model-cache.yaml` | `your-storage-class-name` placeholder |

### Reading the values off your cluster

Set `CONTEXT` to your Kubernetes context:

```bash
kubectl --context "${CONTEXT}" get nodes \
  -o custom-columns='NODE:.metadata.name,PRODUCT:.metadata.labels.nvidia\.com/gpu\.product,GPUS:.status.allocatable.nvidia\.com/gpu,IB:.status.allocatable.rdma/shared_ib'
kubectl --context "${CONTEXT}" get storageclass
```

Update the product label and tolerations if your GPU pool differs. For the
disaggregated profile, edit the Kustomize source and regenerate the manifest
as described in [Configuration notes](#configuration-notes).

## Quick Start

Run from `recipes/ax-k2`. Set `CONTEXT`, `NAMESPACE`, `HF_TOKEN`,
`REGISTRY_USERNAME`, and `REGISTRY_PASSWORD` for your cluster and credentials.

### 1. Create namespace and secrets

```bash
kubectl --context "${CONTEXT}" create namespace "${NAMESPACE}"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" create secret docker-registry runtime-imagepullsecret \
  --docker-server=dynamoci.azurecr.io \
  --docker-username="${REGISTRY_USERNAME}" \
  --docker-password="${REGISTRY_PASSWORD}"
```

### 2. Create storage

> [!NOTE]
> Set `storageClassName` in `model-cache/model-cache.yaml` to a ReadWriteMany
> storage class on your cluster before applying it.

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f model-cache/model-cache.yaml
```

Both profiles and the download job mount `model-cache`. To use an existing
populated cache, change `claimName` in the job and selected deployment source
to that PVC and skip creating a new claim.

### 3. Download the models

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f model-cache/model-download.yaml
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Complete \
  job/axk2-model-download --timeout=14400s
```

The job sets `HF_HOME=/model-cache` and downloads the pinned target and EAGLE3
draft checkpoints into the PVC's Hugging Face cache. Both worker roles use
that cache with `HF_HUB_OFFLINE=1`.

### 4. Deploy the DGD

For aggregated serving:

```bash
export DGD=axk2-agg-b200-chat
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f vllm/agg-b200-chat/deploy.yaml
```

For disaggregated serving:

```bash
export DGD=axk2-disagg-b200-chat
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply -f vllm/disagg-b200-chat/deploy-generic.yaml
```

Wait for the selected deployment:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready \
  "dynamographdeployment/${DGD}" --timeout=7200s
```

### 5. Smoke test

Forward the frontend port, keeping this terminal open:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  "service/${DGD}-frontend" 8000:8000
```

In another terminal:

```bash
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"skt/A.X-K2-NVFP4","messages":[{"role":"user","content":"What is 2 + 2?"}],"temperature":0,"max_tokens":4096,"stream":false}'
```

Model discovery should list `skt/A.X-K2-NVFP4`. The completion returns a
`choices` array, with reasoning in `message.reasoning_content` and the final
answer in `message.content`.

## Performance Results

Benchmarking uses synthetic EAGLE3 acceptance length 2.12.

| Workload | Framework | Recipe | SKU | Concurrency | System output tok/s/GPU | User output tok/s (mean) | TTFT P50 (seconds) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Chat (15% subset) | vLLM | Aggregated (2 workers) | B200 | 16 | 76.49 | 55.83 | 0.525 |
| Chat (15% subset) | vLLM | Disaggregated (2P1D) | B200 | 16 | 61.97 | 83 | 3.879 |

GPU throughput is output-token throughput divided by all serving GPUs:
eight for aggregated serving and twelve for disaggregated serving.
Per-user throughput is the mean request output-token rate. TTFT means time
to first token; the table reports its median in seconds. The 15% subset
of the 8K/1K Mooncake chat trace retains 70% KV reuse.

The aggregate run completed all 1,765 requests; the disaggregated run completed 1,755 of 1,765, with 10 failed
requests. Warmup differed: 20 requests for aggregate and 32 for disaggregate.

See [benchmark instructions](perf/README.md) for the AIPerf workflow. Its
checked-in job runs the full eligible trace at C=32; the results above are
from the short trace at C=16.

## Configuration notes

- Both profiles default to real EAGLE3 acceptance. The benchmark-only synthetic
  acceptance setting does not apply to accuracy evaluation.
- Edit `vllm/disagg-b200-chat/kustomize/base/deploy.yaml` to change the
  disaggregated profile, then regenerate as described in its
  [README](vllm/disagg-b200-chat/README.md).
- The [Fern recipe page](../../docs/fern/pages/recipes/model-recipes/ax-k2.mdx)
  includes an interactive topology picker. Run `bash docs/fern/scripts/preview.sh`
  from the repository root to preview it; see
  [Mac SSH tunnel instructions](../../docs/fern/pages/community/contributing/documentation/building-and-publishing.md#running-locally)
  for remote access.
