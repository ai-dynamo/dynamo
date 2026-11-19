---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "GLM-5.3/5.2"
subtitle: "Serve GLM-5.3/5.2 with Dynamo and SGLang on B200 or H200 for long-context agentic workloads."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Each target below is a Dynamo + SGLang deployment of Z.AI's GLM-5.3/5.2 tuned for an agentic workload (64K median ISL / 400 median OSL, 90% KV cache hit rate) with KV-aware routing and EAGLE-style MTP speculative decoding. B200 serves the NVFP4 checkpoint at up to 500K context with HiCache CPU offload; H200 serves the FP8 checkpoint at up to 250K context. Both run aggregated or with prefill/decode disaggregation. Pick your GPU architecture and serving topology; every command on this page updates to match.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-h200" name="recipe-sku" value="h200" />
<label htmlFor="recipe-sku-h200">H200 (FP8)</label>
<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" defaultChecked />
<label htmlFor="recipe-sku-b200">B200 (NVFP4)</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="agg">
<span><b>Checkpoint</b> RadixArk/GLM-5.3-NVFP4 / nvidia/GLM-5.2-NVFP4</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>GPUs</b> 16x B200 (4 workers x 4)</span>
<span><b>Parallelism</b> DTP4</span>
<span><b>Routing</b> KV-aware</span>
<span><b>Context</b> Up to 500K, HiCache CPU offload</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="disagg">
<span><b>Checkpoint</b> RadixArk/GLM-5.3-NVFP4 / nvidia/GLM-5.2-NVFP4</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>GPUs</b> 20x B200 (3 prefill x 4 + 1 decode x 8)</span>
<span><b>Parallelism</b> DEP4 prefill / DTP8 decode</span>
<span><b>Routing</b> KV-aware, Mooncake over IB</span>
<span><b>Context</b> Up to 500K, HiCache CPU offload</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="h200" data-variant="agg">
<span><b>Checkpoint</b> zai-org/GLM-5.3 / zai-org/GLM-5.2-FP8</span>
<span><b>Precision</b> FP8 + FP8 KV cache</span>
<span><b>GPUs</b> 24x H200 (3 workers x 8)</span>
<span><b>Parallelism</b> TP8/EP8</span>
<span><b>Routing</b> KV-aware</span>
<span><b>Context</b> Up to 250K</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="h200" data-variant="disagg">
<span><b>Checkpoint</b> zai-org/GLM-5.3 / zai-org/GLM-5.2-FP8</span>
<span><b>Precision</b> FP8 + FP8 KV cache</span>
<span><b>GPUs</b> 8x H200 prefill + 8x H200 decode</span>
<span><b>Parallelism</b> TP8/EP8 prefill / TP8/DP8/EP1 decode</span>
<span><b>Routing</b> KV-aware, Mooncake over IB</span>
<span><b>Context</b> Up to 250K</span>
</div>
</div>

## Prerequisites

<div data-sku="b200">

- A Kubernetes cluster with the Dynamo platform installed and B200 GPUs available — 16x for aggregated, 20x for disaggregated. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to `RadixArk/GLM-5.3-NVFP4` / `nvidia/GLM-5.2-NVFP4`.

</div>

<div data-sku="h200">

- A Kubernetes cluster with the Dynamo platform installed and H200 GPUs available — 24x for aggregated, 16x for disaggregated. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to `zai-org/GLM-5.3` / `zai-org/GLM-5.2-FP8`.

</div>

Create the namespace and token secret:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

<Warning>
Edit `storageClassName` in `model-cache/model-cache.yaml` to a ReadWriteMany storage class on your cluster (`kubectl get storageclass`) before applying it. Review namespace, image tags, node selectors, and resource claims in the manifests as well.
</Warning>

## Deploy

Create the shared model cache, then download the checkpoint for your target SKU. Edit `model-cache/model-download.yaml` (select between GLM-5.3 and GLM-5.2 checkpoints, and between FP8 and NVFP4 precisions). Uncomment the `hf download` line for your checkpoint and remove the others:

```bash
# Edit storageClassName in model-cache/model-cache.yaml first.
kubectl apply -f recipes/glm-5.3/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/glm-5.3/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

When serving GLM-5.2, update every `model-path` in the target DGD to `nvidia/GLM-5.2-NVFP4` for B200 or `zai-org/GLM-5.2-FP8` for H200, and update every `served-model-name` to `zai-org/GLM-5.2`.

Then deploy the target DGD:

<div data-sku="b200" data-variant="agg">

```bash
kubectl apply -f recipes/glm-5.3/sglang/agg-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="b200" data-variant="disagg">

```bash
kubectl apply -f recipes/glm-5.3/sglang/disagg-b200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="h200" data-variant="agg">

```bash
kubectl apply -f recipes/glm-5.3/sglang/agg-h200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="h200" data-variant="disagg">

```bash
kubectl apply -f recipes/glm-5.3/sglang/disagg-h200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

## Smoke Test

Send a test request to verify the deployment serves traffic. First forward the frontend port for your target:

<div data-sku="b200" data-variant="agg">

```bash
kubectl port-forward svc/glm53-agg-b200-agentic-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="b200" data-variant="disagg">

```bash
kubectl port-forward svc/glm53-disagg-b200-agentic-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="h200" data-variant="agg">

```bash
kubectl port-forward svc/glm53-agg-h200-agentic-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="h200" data-variant="disagg">

```bash
kubectl port-forward svc/glm53-disagg-h200-agentic-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

Set `MODEL_NAME` to the `served-model-name` configured in the DGD: `zai-org/GLM-5.3` for GLM-5.3 or `zai-org/GLM-5.2` for GLM-5.2:

```bash
export MODEL_NAME=zai-org/GLM-5.3 # or zai-org/GLM-5.2
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a one-sentence readiness check.\"}],\"max_tokens\":64}"
```

GLM-5.3/5.2 reasons before answering. To get the answer in the `content` field instead of `reasoning_content`, disable thinking:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 times 24?\"}],\"chat_template_kwargs\":{\"enable_thinking\":false},\"max_tokens\":64}"
```

## Benchmark

A single AIPerf trace-replay Job — `perf/perf.yaml` — covers all four DGDs. It replays a Mooncake-format agentic trace (64K ISL / 400 OSL, 90% KV cache hit rate) at one concurrency value and writes artifacts to the shared `model-cache` PVC. The benchmark pod is co-located with a DGD frontend through `podAffinity`.

Edit the `env` block in `perf/perf.yaml` to target your deployed DGD — set `ENDPOINT` to the matching frontend service, `SYNTHESIS_MAX_ISL` to its context limit, and `CONCURRENCY` to the value for that target. The `CONCURRENCY` values below reproduce the [Expected Performance (run on GLM-5.3)](#expected-performance-run-on-glm-53) numbers. The rows also require `SGLANG_SIMULATE_ACC_LEN=2.69`, `SGLANG_SIMULATE_ACC_METHOD=match-expected`, and `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` uncommented on the aggregated workers or the disaggregated decode workers (see [perf/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/perf/README.md)); keep them commented for accuracy evaluation and production:

| Target | `ENDPOINT` | `SYNTHESIS_MAX_ISL` | `CONCURRENCY` |
| --- | --- | --- | --- |
| B200 aggregated | `glm53-agg-b200-agentic-frontend:8000` | `500000` | `64` |
| B200 disaggregated | `glm53-disagg-b200-agentic-frontend:8000` | `500000` | `128` |
| H200 aggregated | `glm53-agg-h200-agentic-frontend:8000` | `250000` | `32` |
| H200 disaggregated | `glm53-disagg-h200-agentic-frontend:8000` | `250000` | `24` |

Then run the Job:

```bash
kubectl apply -f recipes/glm-5.3/perf/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/glm53-bench -n ${NAMESPACE} --timeout=7200s
```

For trace staging, concurrency sweeps, and fetching artifacts, see the [benchmark README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/perf/README.md).

## Expected Performance (run on GLM-5.3)

Measured on the 15% agentic trace subset. System throughput is per-GPU output tokens per second; user throughput is the P50 per-request output rate.

| | Concurrency | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
|---|---|---|---|---|
| <span data-sku="b200" data-variant="agg">**B200 aggregated** (4 workers)</span><span data-sku="b200" data-variant="disagg">**B200 disaggregated** (3P1D)</span><span data-sku="h200" data-variant="agg">**H200 aggregated** (3 workers)</span><span data-sku="h200" data-variant="disagg">**H200 disaggregated** (1P1D)</span> | <span data-sku="b200" data-variant="agg">64</span><span data-sku="b200" data-variant="disagg">128</span><span data-sku="h200" data-variant="agg">32</span><span data-sku="h200" data-variant="disagg">24</span> | <span data-sku="b200" data-variant="agg">190.048</span><span data-sku="b200" data-variant="disagg">323.821</span><span data-sku="h200" data-variant="agg">60.866</span><span data-sku="h200" data-variant="disagg">84.335</span> | <span data-sku="b200" data-variant="agg">61.923</span><span data-sku="b200" data-variant="disagg">63.133</span><span data-sku="h200" data-variant="agg">57.330</span><span data-sku="h200" data-variant="disagg">61.460</span> | <span data-sku="b200" data-variant="agg">228.700</span><span data-sku="b200" data-variant="disagg">1280.100</span><span data-sku="h200" data-variant="agg">1158.200</span><span data-sku="h200" data-variant="disagg">1309.600</span> |

<div data-sku="h200">

3,535 of 3,541 trace requests completed; 6 requests exceed the 250K context limit.

</div>

## Compare All Targets

All four targets serve GLM-5.3/5.2 on SGLang with KV-aware routing and EAGLE-style MTP speculative decoding (draft length 3, SpeedBench acceptance length 2.69), benchmarked on the same agentic trace:

| | B200 aggregated | B200 disaggregated | H200 aggregated | H200 disaggregated |
|---|---|---|---|---|
| **Checkpoint** | RadixArk/GLM-5.3-NVFP4 / nvidia/GLM-5.2-NVFP4 | RadixArk/GLM-5.3-NVFP4 / nvidia/GLM-5.2-NVFP4 | zai-org/GLM-5.3 / zai-org/GLM-5.2-FP8 | zai-org/GLM-5.3 / zai-org/GLM-5.2-FP8 |
| **Precision** | NVFP4 + FP8 KV | NVFP4 + FP8 KV | FP8 + FP8 KV | FP8 + FP8 KV |
| **GPUs** | 16x B200 (4 workers x 4) | 20x B200 (3 prefill x 4 + 1 decode x 8) | 24x H200 (3 workers x 8) | 16x H200 (8 prefill + 8 decode) |
| **Parallelism** | DTP4 | DEP4 / DTP8 | TP8/EP8 | TP8/EP8 prefill / TP8/DP8/EP1 decode |
| **KV offload** | HiCache CPU | HiCache CPU | None | None |
| **Max context** | 500K | 500K | 250K | 250K |
| **Image** | `sglang-runtime:1.5.1` | `sglang-runtime:1.5.1` | `sglang-runtime:1.5.1` | `sglang-runtime:1.5.1` |

## Notes

- Speculative decoding uses EAGLE-style MTP with draft length 3, measured at a SpeedBench acceptance length of 2.69.
- All four targets use KV-aware routing at the frontend. The B200 targets add HiCache CPU offload; the agentic traces are shaped to show the value of KV-aware routing and offloading.
- The disaggregated targets transfer KV over Mooncake on InfiniBand.
- The H200 disaggregated decode worker uses `mem-fraction-static: 0.95`; at 0.88 the per-rank KV pool is smaller than one 250K-token request.

## Limitations

- B200 targets support up to 500K context; the full 1M context length is not supported out of the box. H200 targets support up to 250K context.
- Structured decoding works with reasoning enabled: the generated JSON is populated in the `content` field and the chain-of-thought in `reasoning_content`. This requires both `--dyn-reasoning-parser glm45` (frontend) and `--reasoning-parser glm45` (engine), which the recipes set.
- `n>1` requests are not supported with the disaggregated targets.

## Source

- Source README: [recipes/glm-5.3/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/README.md)
- Benchmark README: [recipes/glm-5.3/perf/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/perf/README.md)
- Aggregated B200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/sglang/agg-b200-agentic/deploy.yaml)
- Disaggregated B200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/sglang/disagg-b200-agentic/deploy.yaml)
- Aggregated H200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/sglang/agg-h200-agentic/deploy.yaml)
- Disaggregated H200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/sglang/disagg-h200-agentic/deploy.yaml)
- Setup assets: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/model-cache/model-download.yaml)
- Benchmark manifest: [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3/perf/perf.yaml)
