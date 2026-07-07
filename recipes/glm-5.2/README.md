<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 Recipes

Recipes for
[`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4).

## Configurations

Dynamo + SGLang deployment profiles for the B200 agentic workload:

| | B200 aggregate agentic | B200 disaggregated agentic |
| --- | --- | --- |
| **GPU** (per worker) | 4x B200 | 4x B200 prefill + 8x B200 decode |
| **Mode** | Aggregated | Prefill/decode disaggregated |
| **Framework** | SGLang preview runtime | SGLang preview runtime |
| **Precision** | NVFP4 + FP8 KV | NVFP4 + FP8 KV |
| **Parallelism** | TP4 / DP4 | Prefill TP4 / DP4 / EP4; decode TP8 / DP8 |
| **MoE backend** | FlashInfer TRT-LLM | Prefill FlashInfer TRT-LLM routed; decode FlashInfer Cutlass |
| **Attention backend** | DSA | DSA TRT-LLM prefill; default decode |
| **AllReduce backend** | FlashInfer fusion | FlashInfer fusion on decode |
| **All2All backend** | None | None |
| **Routing** | KV-aware | KV-aware |
| **Speculative decoding** | Built-in EAGLE (DL=3, AL=2.69 for benchmarking) | Built-in EAGLE (DL=3, AL=2.69 for benchmarking) |
| **Context length** | 500,000 | 500,000 |
| **KV cache offloading** | SGLang HiCache CPU | SGLang HiCache CPU on prefill |
| **KV transfer** | N/A | NIXL over UCX/InfiniBand |

Deployments:

- [`sglang/agg-b200-agentic/deploy.yaml`](sglang/agg-b200-agentic/deploy.yaml)
- [`sglang/disagg-b200-agentic/deploy.yaml`](sglang/disagg-b200-agentic/deploy.yaml)

## Supported features

- Modalities: Text
- Reasoning
- Tool calling
- KV-aware routing
- CPU KV cache offloading
- Prefill/decode disaggregation

## Prerequisites

1. **Dynamo Platform installed** — see the
   [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **B200 capacity** — one B200x4 node for aggregate mode, or one B200x4
   prefill node plus one B200x8 decode node for disaggregated mode.
3. **Host memory** — at least 1 TiB available to the aggregate or prefill
   worker because `--hicache-size 200` allocates 200 GB per TP rank.
4. **InfiniBand resources for disaggregated mode** — the manifests expect the
   RDMA device plugin to expose `rdma/ib`.
5. **Hugging Face token** with access to `nvidia/GLM-5.2-NVFP4`:

   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

The deployments use the public preview image
`nvcr.io/nvidia/ai-dynamo/sglang-runtime-nightly:20260701-5245c0f` because
GLM-5.2 requires newer SGLang support than the latest stable Dynamo runtime.

## Quick Start

### 1. Create namespace

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download \
  -n ${NAMESPACE} --timeout=7200s
```

If the namespace already provides a shared model-cache PVC, skip PVC creation
and change `claimName: model-cache` in the deployment and benchmark manifests.

### 4. Deploy the DGD

Deploy the target DynamoGraphDeployment (DGD):

```bash
MODE=agg # or disagg
kubectl apply -f sglang/${MODE}-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [`perf/README.md`](perf/README.md) for the trace-staging, AIPerf replay,
concurrency-sweep, cache-reset, and artifact-fetch workflow.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Agentic coding and tool use | 64k | 400 | 90% | 50 |

The modified Mooncake trace under [`perf/traces`](perf/traces) reproduces the
agentic workload and KV reuse pattern.

## Performance results

No completed aggregate or disaggregated benchmark result is published yet.
Use the included AIPerf Job to produce results for the target cluster.

## Recipe notes

- All SGLang workers use a 500,000-token context. A 524,288-token context
  exceeded the measured 520,000-token prefill KV capacity for this
  configuration.
- JIT and compilation caches are pod-local under `/tmp/compilation-cache`;
  only model and benchmark files use the shared PVC.
- The `SGLANG_SIMULATE_ACC_*` variables force an acceptance length of 2.69 for
  performance comparison. Remove them for quality evaluation.
- Neither deployment uses fake prefill, dummy weights, or simulated uniform
  experts.

## Known issues

1. A longer disaggregation timeout does not recover an orphaned NIXL transfer.
   Monitor `State index length mismatch` and `KVPoll.WaitingForInput` during
   long trace replays.
