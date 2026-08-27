<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DeepSeek-V4-Pro-0813 Recipes

Recipes for [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813).

> [!NOTE]
> This is a **different checkpoint** from `deepseek-ai/DeepSeek-V4-Pro` used by the sibling
> [`../deepseek-v4-pro`](../deepseek-v4-pro) recipes. Do not share a model cache between them.

## Configurations

Dynamo + vLLM deployment profiles.

|                          | H200 aggregated (agentic + 1M) | H200 disaggregated (agentic + 1M) | GB200 aggregated       | GB200 disaggregated |
| ------------------------ | ----------------------- | -------------------------------- | ---------------------- | ------------------- |
| **GPU** (per worker)     | 8x H200                 | 8x H200 prefill + 8x H200 decode | 8x GB200 (2 nodes)     | 8x GB200 prefill + 8x GB200 decode (2 nodes each) |
| **Mode**                 | Aggregated              | Prefill/decode disaggregated     | Aggregated             | Prefill/decode disaggregated |
| **Framework**            | vLLM                    | vLLM                             | vLLM                   | vLLM |
| **Precision**            | MXFP4 experts + FP8 KV  | MXFP4 experts + FP8 KV           | MXFP4 experts + FP8 KV | MXFP4 experts + FP8 KV |
| **Parallelism**          | TP8/EP8                 | TP8/EP8 prefill / TP8/EP8 decode | TP8/EP8                | TP8/EP8 prefill / TP8/EP8 decode |
| **Routing**              | KV-aware                | KV-aware                         | KV-aware               | KV-aware |
| **Speculative decoding** | None                    | None                             | DSpark k=5             | DSpark k=5 (decode only) |
| **Context length**       | 1,048,576               | 1,048,576                        | 1,048,576              | 1,048,576 |
| **KV cache offloading**  | None                    | None                             | None                   | None |
| **KV transfer**          | N/A                     | NIXL                             | N/A                    | NIXL |

## Supported features

- Modalities: **Text** (this checkpoint is `DeepseekV4ForCausalLM`; it has no vision tower — image input is **not** supported)
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to `deepseek-ai/DeepSeek-V4-Pro-0813`.
3. No CPU KV offload is needed for 1M context on either topology — GPU KV holds ~3.1M tokens,
   about 3x a full 1M request.

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
> Edit `model-cache/model-cache.yaml` and set `storageClassName` to a ReadWriteMany storage
> class available on the target cluster. The checkpoint is ~832 GiB.

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

```bash
MODE=agg         # or disagg
SKU=h200         # or gb200 (agentic only)
WORKLOAD=agentic # or 1m
kubectl apply -f vllm/${MODE}-${SKU}-${WORKLOAD}/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace staging on the
PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

## Optimization targets

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |

The gate is **joint**: E2E ≥ 50 tok/s/user **and** TTFT p50 < 5 s, where
`E2E = OSL / (TTFT + OSL × ITL)` — the per-user rate *including* time-to-first-token.

Modified Mooncake traces are provided to showcase the value of KV-aware routing and CPU
offloading, see [perf/README.md](perf/README.md) for details.

## Performance results

| Workload | Recipe | SKU | User output tok/s (P50) | TTFT P50 (ms) |
| -------- | ------ | --- | ----------------------- | ------------- |
| Agentic 64K | `agg-gb200-agentic` | 8x GB200 | 55.906 | 383.3 |
| Agentic 64K | `disagg-gb200-agentic` | 16x GB200 | 50.042 | 2370.3 |

Both GB200 rows were measured on the same 3,541-row 15% agentic trace, each at its own
iso-SLA operating point against the joint gate of >= 50 tok/s/user and TTFT p50 < 5 s.

