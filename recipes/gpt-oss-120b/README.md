<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gpt-oss-120b Recipes

Recipes for **openai/gpt-oss-120b** (MXFP4 MoE, 128 experts / top-4, GQA + sliding-window + attention sinks).

## Configurations

Dynamo + vLLM deployment profiles for the Mooncake agentic trace (64k/400/90%-KV) across two GPU SKUs and two topologies:

|                          | B200 agentic (agg)                 | H200 agentic (agg)                 | B200 agentic (disagg)              | H200 agentic (disagg)              |
|--------------------------|------------------------------------|------------------------------------|------------------------------------|------------------------------------|
| **GPUs** (per node)      | 8x B200                            | 8x H200                            | 8x B200 (2 prefill + 6 decode)     | 8x H200 (4 prefill + 4 decode)     |
| **Mode**                 | aggregated                         | aggregated                         | disaggregated                      | disaggregated                      |
| **Framework**            | Dynamo 1.3.0 / vLLM 0.23           | Dynamo 1.3.0 / vLLM 0.23           | Dynamo 1.3.0 / vLLM 0.23           | Dynamo 1.3.0 / vLLM 0.23           |
| **Deployment**           | DGD (8 pods, KV router)            | DGD (8 pods, KV router)            | DGD (colocated 1 node)             | DGD (colocated 1 node)             |
| **Precision**            | MXFP4 + FP8 KV                     | MXFP4 + FP8 KV                     | MXFP4 + FP8 KV                     | MXFP4 + FP8 KV                     |
| **Parallelism**          | 8x TP1 replicas                    | 8x TP1 replicas                    | 2x TP1 prefill + 6x TP1 decode³    | 4x TP1 prefill + 4x TP1 decode³    |
| **MoE backend**          | flashinfer_trtllm (MXFP4xFP8)      | flashinfer_cutlass                 | flashinfer_trtllm                  | flashinfer_cutlass                 |
| **Attention backend**    | FlashInfer                         | FlashInfer                         | FlashInfer                         | FlashInfer                         |
| **Routing**              | KV-aware                           | KV-aware                           | KV-aware                           | KV-aware                           |
| **Speculative decoding** | EAGLE3-v3 (DL=3), ON               | EAGLE3-v3 (DL=3), ON               | EAGLE3-v3 (DL=3), ON (both)        | EAGLE3-v3 (DL=3), ON (both)        |
| **KV cache offloading**  | none (SimpleCPUOffload opt-in)     | **SimpleCPUOffload ON**            | none (NIXL-incompatible)           | none (NIXL-incompatible)           |


## Supported features

- Modality: Text (gpt-oss "harmony" format: reasoning + tool calls)
- Reasoning (`gpt_oss` parser)
- Tool / function calling (`harmony` tool-call parser) — `tool_calls` populates, `finish_reason: tool_calls`

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/kubernetes/README.md). The recipes pull
   `nvcr.io/nvidia/ai-dynamo/vllm-runtime` images via the `dynamo-ngc-token` image pull secret.
2. **HuggingFace token** with access to `openai/gpt-oss-120b` and `nvidia/gpt-oss-120b-Eagle3-v3`:
   ```bash
   export NAMESPACE=your-namespace
   kubectl create secret generic hf-token-secret \
     --from-literal=HF_TOKEN="your-token" \
     -n ${NAMESPACE}
   ```

## Quick Start

### 1. Create namespace

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Create Storage

> **Note:** Edit `model-cache/model-cache.yaml` first and update `storageClassName` to match your cluster (`kubectl get storageclass`).

```bash
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download model + EAGLE3 head

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s
```

### 4. Deploy

Both `agg` and `disagg` apply a DynamoGraphDeployment (`disagg` adds `podAffinity` so its prefill+decode workers colocate on one node for the NVLink NIXL path). Same command:

```bash
SKU=b200   # or h200
TOPO=agg   # or disagg

kubectl apply -f vllm/${TOPO}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```

### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace staging on the PVC, running the AIPerf trace-replay Job ([perf/perf.yaml](perf/perf.yaml)), running a concurrency sweep, and fetching artifacts.

## Optimization targets

Recipes are optimized for the following configuration, at the target user interactivity:

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
|----------|------------|------------|-------------------|-------------------|
| Agentic  | 64k        | 400        | 90%               | 50                |

The benchmark reuses the Mooncake-format traces shipped with the Kimi-K2.6 recipe
([../kimi-k2.6/perf/traces/](../kimi-k2.6/perf/traces/)); only the **agentic 15% subset** is needed. The traces
are stored in Git LFS — run `git lfs pull --include "recipes/kimi-k2.6/perf/traces/*"` to fetch the actual
files before staging. See [perf/README.md](perf/README.md) for details.

## Performance results

Measured 2026-07-09 on the agentic 15% trace (see [perf/README.md](perf/README.md)), 8 GPUs, with the
**synthetic-acceptance EAGLE3 throughput proxy** (AL=2.72 on all workers — output is garbage by design; use for
*relative* comparisons) and each recipe's KV-offload default. Per-GPU = system throughput / 8.

| SKU  | config (shipped)                 | conc | tok/s/GPU | tok/s/user | TTFT avg | note                                        |
|------|----------------------------------|------|-----------|------------|----------|---------------------------------------------|
| B200 | **agg** — 8x TP1                 | c512 | **2896**  | 58         | 4.4s     | offload quality-neutral                     |
| B200 | disagg — 2P6D colocated             | c256 | 2069      | 99         | 5.6s     | plateau; TTFT-limited above c256            |
| H200 | **agg** — 8x TP1 + CPU offload   | c256 | **1256**  | 47.5       | 1.4s     | at the 50-tps floor; offload = +9%, quality-neutral |
| H200 | disagg — 4P4D colocated             | c256 | 1046      | 43         | 4.4s     | best usable H200 split                      |

## Known issues

1. Structured output (`response_format: json_object` / `json_schema`) is not working — may return invalid
   JSON while speculative decoding is enabled; use tool calling or validate client-side.

