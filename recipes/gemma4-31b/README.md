<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Gemma 4 31B Recipes

Recipes for [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B).

## Configurations

<!--
TODOs:
- vLLM
- H200
- GB200
-->

Dynamo + TensorRT-LLM deployment profile for the B200 workload:

<!-- Note: AL number comes from https://github.com/NVIDIA/TensorRT-LLM/pull/15833 -->

|                          | B200 aggregated agentic                    |
| ------------------------ | ------------------------------------------ |
| **GPU** (per worker)     | 1x B200                                    |
| **Mode**                 | Aggregated                                 |
| **Framework**            | TensorRT-LLM                               |
| **Precision**            | NVFP4 + FP8 KV                             |
| **Parallelism**          | None                                       |
| **Routing**              | KV-aware                                   |
| **Speculative decoding** | EAGLE-style MTP (DL=3, AL=2.88)            |
| **Context length**       | 256K                                       |
| **KV cache offloading**  | TensorRT-LLM managed                       |
| **KV transfer**          | N/A                                        |


## Supported features

- Modalities: Text, Images, Videos, Audio
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to `nvidia/Gemma-4-31B-IT-NVFP4` for B200, and `google/gemma-4-31B-it-assistant` for
   speculative decoding (MTP).

## Quick Start

### 1. Create namespace and secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create storage

> [!NOTE]
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a
> ReadWriteMany storage class available on the target cluster.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

Edit `model-cache/model-download.yaml` and remove any `hf download` command for
the checkpoint that does not match the target SKU, if applicable.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

Deploy the target DGD:

```bash
SKU=b200
MODE=agg
kubectl apply -f trtllm/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace staging on the PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

## Optimization targets


| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |


Modified Mooncake traces are provided to showcase the value of KV-aware routing and CPU offloading, see [perf/README.md](perf/README.md) for details.

## Performance results


| Workload             | Recipe                 | SKU  | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT P50 (ms) |
| -------------------- | ---------------------- | ---- | ----------- | ----------------------- | ----------------------- | ------------- |
| Agentic (15% subset) | Aggregated (8 workers) | B200 | 192         | 1,109.489               | 60.10                   | 3,431.78      |



## Limitations

<!-- TODO -->
