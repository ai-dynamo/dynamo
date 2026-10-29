<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5.3 Recipes

Recipes for [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3). The same recipes also serve [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) as a fallback (see step 4 below); `recipes/glm-5.2` is kept as a symlink to this directory.

## Configurations

Dynamo + SGLang deployment profiles for the B200 and H200 agentic workload:

|                          | B200 aggregated agentic                    | B200 disaggregated agentic                 | H200 aggregated agentic                    | H200 disaggregated agentic                 |
| ------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ | ------------------------------------------ |
| **GPU** (per worker)     | 4x B200                                    | 4x B200 prefill + 8x B200 decode           | 8x H200                                    | 8x H200 prefill + 8x H200 decode           |
| **Mode**                 | Aggregated                                 | Prefill/decode disaggregated               | Aggregated                                 | Prefill/decode disaggregated               |
| **Framework**            | SGLang                                     | SGLang                                     | SGLang                                     | SGLang                                     |
| **Precision**            | NVFP4 + FP8 KV                             | NVFP4 + FP8 KV                             | FP8 + FP8 KV                               | FP8 + FP8 KV                               |
| **Parallelism**          | DTP4                                       | DEP4 / DTP8                                | TP8/EP8                                    | TP8/EP8 prefill / TP8/DP8/EP1 decode       |
| **Routing**              | KV-aware                                   | KV-aware                                   | KV-aware                                   | KV-aware                                   |
| **Speculative decoding** | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) | EAGLE-style MTP (DL=3, SpeedBench AL=2.69) |
| **Context length**       | 500,000                                    | 500,000                                    | 250,000                                    | 250,000                                    |
| **KV cache offloading**  | HiCache CPU                                | HiCache CPU                                | None                                       | None                                       |
| **KV transfer**          | N/A                                        | Mooncake over IB                           | N/A                                        | Mooncake over IB                           |

All four variants run on `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5.1` (Dynamo 1.5.1, SGLang 0.5.18).


## Supported features

- Modalities: Text
- Reasoning
- Tool calling

## Prerequisites

1. **Dynamo Platform installed** — see [Kubernetes Deployment Guide](../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. **Hugging Face token** with access to `RadixArk/GLM-5.3-NVFP4` / `nvidia/GLM-5.2-NVFP4`
   for B200 or `zai-org/GLM-5.3` / `zai-org/GLM-5.2-FP8` for H200.

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

As shipped, `model-cache/model-download.yaml` downloads both GLM-5.3 checkpoints:
`RadixArk/GLM-5.3-NVFP4` (B200) and `zai-org/GLM-5.3` (H200). The GLM-5.2 lines
(`nvidia/GLM-5.2-NVFP4`, `zai-org/GLM-5.2-FP8`) are commented out. Edit the `hf download`
lines so that only the checkpoint(s) you deploy stay enabled: comment out the GLM-5.3
checkpoint you do not need, and for the GLM-5.2 fallback uncomment the matching GLM-5.2
line and comment out the GLM-5.3 lines.

```bash
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

### 4. Deploy the DGD

When serving GLM-5.2, the model name is hardcoded in more than one file. Update all of them:

- `sglang/${MODE}-${SKU}-agentic/deploy.yaml`: every `model-path` to
  `nvidia/GLM-5.2-NVFP4` for B200 or `zai-org/GLM-5.2-FP8` for H200, and every
  `served-model-name` to `zai-org/GLM-5.2`.
- `perf/perf.yaml`: the `TARGET_MODEL` env value to `zai-org/GLM-5.2`. The benchmark
  readiness loop and the AIPerf `-m` / `--tokenizer` flags use this value, so it must match
  `served-model-name` exactly or the benchmark Job never becomes ready.

For example, for the B200 recipes:

```bash
sed -i 's#model-path: RadixArk/GLM-5.3-NVFP4#model-path: nvidia/GLM-5.2-NVFP4#; s#served-model-name: zai-org/GLM-5.3#served-model-name: zai-org/GLM-5.2#' sglang/*-b200-agentic/deploy.yaml
sed -i 's#value: zai-org/GLM-5.3#value: zai-org/GLM-5.2#' perf/perf.yaml
```

For H200, replace `RadixArk/GLM-5.3-NVFP4` / `nvidia/GLM-5.2-NVFP4` above with
`zai-org/GLM-5.3` / `zai-org/GLM-5.2-FP8` and target `sglang/*-h200-agentic/deploy.yaml`.

Deploy the target DGD:

```bash
SKU=b200 # or h200
MODE=agg # or disagg
kubectl apply -f sglang/${MODE}-${SKU}-agentic/deploy.yaml -n ${NAMESPACE}
```



### 5. Benchmark

See [perf/README.md](perf/README.md) for the full benchmark workflow — trace staging on the PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

## Optimization targets


| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s |
| -------- | ---------- | ---------- | ----------------- | ----------------- |
| Agentic  | 64k        | 400        | 90%               | 50                |


Modified Mooncake traces are provided to showcase the value of KV-aware routing and CPU offloading, see [perf/README.md](perf/README.md) for details.

## Performance results (run on GLM-5.3)


| Workload             | Framework | Recipe                 | SKU  | GPUs | Concurrency | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
| -------------------- | --------- | ---------------------- | ---- | ---- | ----------- | ----------------------- | ----------------------- | ------------- |
| Agentic (15% subset) | SGLang    | Aggregated (4 workers) | B200 | 16   | 64          | 190.048                 | 61.923                  | 228.700       |
| Agentic (15% subset) | SGLang    | Disaggregated (3P1D)   | B200 | 20   | 128         | 323.821                 | 63.133                  | 1280.100      |
| Agentic (15% subset) | SGLang    | Aggregated (3 workers) | H200 | 24   | 32          | 60.866                  | 57.330                  | 1158.200      |
| Agentic (15% subset) | SGLang    | Disaggregated (1P1D)   | H200 | 16   | 24          | 84.335                  | 61.460                  | 1309.600      |

The rows require `SGLANG_SIMULATE_ACC_LEN=2.69`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, and `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` uncommented on the aggregated workers or the disaggregated decode workers (see [perf/README.md](perf/README.md)); keep them commented for accuracy evaluation and production. H200 rows: 3,535 of 3,541 trace requests completed; 6 requests exceed the 250K context limit.



## Limitations

- B200 recipes support up to 500K context lengths. The full 1M context length is not supported out of the box.
- H200 recipes support up to 250K context lengths.
- Structured decoding works with reasoning enabled: the generated JSON is populated in the `content` field and the chain-of-thought in `reasoning_content`. This requires both `--dyn-reasoning-parser glm45` (frontend) and `--reasoning-parser glm45` (engine), which the recipes set.
- `n>1` requests are not supported with the disaggregated recipe
- Chat Completions accepts a structured-output JSON schema with an invalid schema type or a malformed regex and returns HTTP 200 with `content` set to null.
- Chat Completions can return one more alternative than requested. With `top_logprobs=3`, some tokens include four entries in `top_logprobs`.
- Completions with more than 32 `stop_token_ids` returns HTTP 500. The same request is valid and should stop when a listed token is produced.
