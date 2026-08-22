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
-->

Dynamo + TensorRT-LLM deployment profiles for the B200, GB200, and H200
agentic workload:

<!-- Note: AL number comes from https://github.com/NVIDIA/TensorRT-LLM/pull/15833 -->

|                          | B200 aggregated agentic         | GB200 aggregated agentic        | H200 aggregated agentic       |
| ------------------------ | ------------------------------- | -------------------------------- | ----------------------------- |
| **GPU** (per worker)     | 1x B200                         | 1x GB200                         | 4x H200                       |
| **Mode**                 | Aggregated                      | Aggregated                       | Aggregated                    |
| **Framework**            | TensorRT-LLM                    | TensorRT-LLM                     | TensorRT-LLM                  |
| **Precision**            | NVFP4 + FP8 KV                  | NVFP4 + FP8 KV                   | BF16 + 16-bit KV              |
| **Parallelism**          | TP1                             | TP1                              | TP4                           |
| **Routing**              | KV-aware                        | KV-aware                         | KV-aware                      |
| **Speculative decoding** | EAGLE-style MTP (DL=3, AL=2.88) | EAGLE-style MTP (DL=3, AL=2.88)  | Not enabled; not yet verified |
| **Context length**       | 256K                            | 256K                             | 256K                          |
| **KV cache offloading**  | TensorRT-LLM managed, 128 GiB   | TensorRT-LLM managed, 128 GiB    | TensorRT-LLM managed, 64 GiB  |
| **KV transfer**          | N/A                             | N/A                              | N/A                           |

## Supported features

- Modalities: Text, Images, Videos, Audio
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to the checkpoints for the target SKU:
   - B200 and GB200: `nvidia/Gemma-4-31B-IT-NVFP4` and
     `google/gemma-4-31B-it-assistant` for multi-token prediction (MTP).
   - H200: `google/gemma-4-31B-it`.

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

If the cluster already provides a shared ReadWriteMany model-cache PVC, skip
creating `model-cache` and replace `claimName: model-cache` in the download,
deploy, and perf manifests with the existing claim name, such as
`shared-model-cache`.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

Edit `model-cache/model-download.yaml` before creating the Job. The B200 and
GB200 downloads are enabled by default. For H200, comment out those commands
and uncomment `hf download google/gemma-4-31B-it`. Make sure the Job downloads
every checkpoint referenced by the selected deploy manifest.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

Deploy the target DGD:

```bash
SKU=b200 # or gb200 or h200
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


| Workload             | Recipe                     | SKU   | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT P50 (ms) |
| -------------------- | -------------------------- | ----- | ----------- | ----------------------- | ----------------------- | ------------- |
| Agentic (15% subset) | Aggregated (8 workers)     | B200  | 192         | 1,109.489               | 60.10                   | 3,431.78      |
| Agentic (15% subset) | Aggregated (8 workers)     | GB200 | 192         | 1,085.038               | 48.67                   | 378.79        |
| Agentic (15% subset) | Aggregated (2 TP4 workers) | H200  | 32          | 136.441                 | 61.20                   | 1,579.68      |



## Limitations

- MTP support has not been verified on H200 and is not enabled in the H200
  deployment.
