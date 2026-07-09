<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.2 Recipes

Recipes for [`GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2).

## Configurations

Dynamo + SGLang deployment profiles for the B200 agentic workload:

| | B200 aggregated agentic | B200 disaggregated agentic |
| --- | --- | --- |
| **GPU** (per worker) | 4x B200 | 4x B200 prefill + 8x B200 decode |
| **Mode** | Aggregated | Prefill/decode disaggregated |
| **Framework** | SGLang | SGLang |
| **Precision** | NVFP4 + FP8 KV | NVFP4 + FP8 KV |
| **Parallelism** | DTP4 | DEP4 / DTP8 |
| **Routing** | KV-aware | KV-aware |
| **Speculative decoding** | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) |
| **Context length** | 500,000 | 500,000 |
| **KV cache offloading** | HiCache CPU | HiCache CPU |
| **KV transfer** | N/A | NIXL/UCX over IB |

Deployments:

- [`sglang/agg-b200-agentic/deploy.yaml`](sglang/agg-b200-agentic/deploy.yaml)
- [`sglang/disagg-b200-agentic/deploy.yaml`](sglang/disagg-b200-agentic/deploy.yaml)

## Supported features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see the
   [Kubernetes Deployment Guide](../../docs/kubernetes/README.md).
2. **Hugging Face token** with access to `nvidia/GLM-5.2-NVFP4`:

   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

The deployments use
`nvcr.io/nvstaging/nim/alexandrem:sglang-glm5.2`.

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

### 4. Deploy the DGD

Deploy the target DynamoGraphDeployment (DGD):

```bash
MODE=agg # or disagg
kubectl apply -f sglang/${MODE}-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

Each deployment embeds its SGLang ConfigMap and mounts the role-specific YAML
files into its workers. Dynamo tool-calling and reasoning parser arguments
remain on the worker command line.

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

TBD

## Known issues

N/A
