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

Dynamo + vLLM deployment profiles. GB200 profiles are a separate contribution and are not
filled in yet.

|                          | H200 aggregated agentic | H200 disaggregated agentic       | H200 aggregated 1M      | H200 disaggregated 1M            | GB200 aggregated | GB200 disaggregated |
| ------------------------ | ----------------------- | -------------------------------- | ----------------------- | -------------------------------- | ---------------- | ------------------- |
| **GPU** (per worker)     | 8x H200                 | 8x H200 prefill + 8x H200 decode  | 8x H200                 | 8x H200 prefill + 8x H200 decode  | TBD              | TBD                 |
| **Mode**                 | Aggregated              | Prefill/decode disaggregated      | Aggregated              | Prefill/decode disaggregated      | TBD              | TBD                 |
| **Framework**            | vLLM                    | vLLM                              | vLLM                    | vLLM                              | TBD              | TBD                 |
| **Precision**            | MXFP4 experts + FP8 KV  | MXFP4 experts + FP8 KV            | MXFP4 experts + FP8 KV  | MXFP4 experts + FP8 KV            | TBD              | TBD                 |
| **Parallelism**          | TP8/EP8                 | TP8/EP8 prefill / TP8/EP8 decode  | TP8/EP8                 | TP8/EP8 prefill / TP8/EP8 decode  | TBD              | TBD                 |
| **Routing**              | KV-aware                | KV-aware                          | KV-aware                | KV-aware                          | TBD              | TBD                 |
| **Speculative decoding** | None                    | None                              | None                    | None                              | TBD              | TBD                 |
| **Context length**       | 86,016                  | 86,016                            | 1,048,576               | 1,048,576                         | TBD              | TBD                 |
| **KV cache offloading**  | None                    | None                              | `SimpleCPUOffload` (CPU)| `SimpleCPUOffload` (CPU, decode)  | TBD              | TBD                 |
| **KV transfer**          | N/A                     | NIXL                              | N/A                     | NIXL (via `MultiConnector`)       | TBD              | TBD                 |

## Supported features

- Modalities: **Text** (this checkpoint is `DeepseekV4ForCausalLM`; it has no vision tower — image input is **not** supported)
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to `deepseek-ai/DeepSeek-V4-Pro-0813`.
3. For the **1M context** recipes, worker nodes need **~1.4 TB of host RAM** for the CPU KV
   offload tier, in addition to the 8 GPUs.

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
MODE=agg        # or disagg
WORKLOAD=agentic # or 1m
kubectl apply -f vllm/${MODE}-h200-${WORKLOAD}/deploy.yaml -n ${NAMESPACE}
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

| Workload | Recipe | SKU | Concurrency | System output tok/s/gpu | User output tok/s (P50) | TTFT P50 (ms) |
| -------- | ------ | --- | ----------- | ----------------------- | ----------------------- | ------------- |
