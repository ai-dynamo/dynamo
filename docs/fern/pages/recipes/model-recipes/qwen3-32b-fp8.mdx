---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Qwen3-32B FP8"
subtitle: "Serve Qwen/Qwen3-32B-FP8 with Dynamo on TensorRT-LLM or vLLM, aggregated or disaggregated."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Each target below is a validated FP8 deployment of Qwen3-32B — from a 2-GPU TensorRT-LLM aggregate to 8-GPU disaggregated prefill/decode setups on TensorRT-LLM or vLLM — each with a checked-in AIPerf benchmark Job. The targets use different traffic shapes and GPU counts, so this page is not a backend benchmark. Pick your target; every command on this page updates to match.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Target</span>
<input type="radio" id="recipe-variant-trtllm-agg" name="recipe-variant" value="trtllm-agg" defaultChecked />
<label htmlFor="recipe-variant-trtllm-agg">TRT-LLM aggregated (2 GPU) <span className="dynamo-target-picker-hint">Recommended</span></label>
<input type="radio" id="recipe-variant-trtllm-disagg" name="recipe-variant" value="trtllm-disagg" />
<label htmlFor="recipe-variant-trtllm-disagg">TRT-LLM disaggregated (8 GPU)</label>
<input type="radio" id="recipe-variant-vllm-disagg" name="recipe-variant" value="vllm-disagg" />
<label htmlFor="recipe-variant-vllm-disagg">vLLM disaggregated (8 GPU)</label>
</div>
<div className="dynamo-target-picker-summary" data-variant="trtllm-agg">
<span><b>Checkpoint</b> Qwen/Qwen3-32B-FP8</span>
<span><b>GPUs</b> 2x H100/H200/A100, single TP2 worker</span>
<span><b>Routing</b> Round-robin</span>
<span><b>Techniques</b> CUDA graphs, FP8 KV cache</span>
<span><b>Workload</b> 4K ISL / 500 OSL, concurrency 4</span>
</div>
<div className="dynamo-target-picker-summary" data-variant="trtllm-disagg">
<span><b>Checkpoint</b> Qwen/Qwen3-32B-FP8</span>
<span><b>GPUs</b> 8x H100/H200/A100</span>
<span><b>Topology</b> 4x prefill (TP1) + 2x decode (TP2)</span>
<span><b>Routing</b> Round-robin</span>
<span><b>Workload</b> 4K ISL / 500 OSL, concurrency 48</span>
</div>
<div className="dynamo-target-picker-summary" data-variant="vllm-disagg">
<span><b>Checkpoint</b> Qwen/Qwen3-32B-FP8</span>
<span><b>GPUs</b> 8x H100/H200/A100, single node</span>
<span><b>Topology</b> 2x prefill (TP2) + 1x decode (TP4)</span>
<span><b>KV transfer</b> NIXL (NixlConnector)</span>
<span><b>Workload</b> 2K ISL / 500 OSL, concurrency 8</span>
</div>
</div>

## Prerequisites

<div data-variant="trtllm-agg">

- A Kubernetes cluster with the Dynamo platform installed and **2x H100/H200/A100-class GPUs** available.
- A Hugging Face token with access to `Qwen/Qwen3-32B-FP8`.

</div>

<div data-variant="trtllm-disagg">

- A Kubernetes cluster with the Dynamo platform installed and **8x H100/H200/A100-class GPUs** available.
- A Hugging Face token with access to `Qwen/Qwen3-32B-FP8`.

</div>

<div data-variant="vllm-disagg">

- A Kubernetes cluster with the Dynamo platform installed and **8x H100/H200/A100-class GPUs on a single node** — all prefill and decode workers must be co-located for NIXL KV transfer.
- A Hugging Face token with access to `Qwen/Qwen3-32B-FP8`.

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
Edit namespace, storage class, image tags, node selectors, resource claims, and cluster-specific placement in the manifests before applying them.
</Warning>

## Deploy

Prepare the model cache and download the checkpoint:

```bash
# Edit storageClassName in model-cache/model-cache.yaml to match your cluster first
# (kubectl get storageclass).
kubectl apply -f recipes/qwen3-32b-fp8/model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=1800s
```

Then deploy:

<div data-variant="trtllm-agg">

A single TP2 TensorRT-LLM worker with round-robin routing and CUDA graphs enabled:

```bash
kubectl apply -f recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-variant="trtllm-disagg">

4x prefill workers (TP1) and 2x decode workers (TP2):

```bash
kubectl apply -f recipes/qwen3-32b-fp8/trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-variant="vllm-disagg">

2x prefill workers (TP2) and 1x decode worker (TP4) using NixlConnector KV transfer; all workers must land on one node:

```bash
kubectl apply -f recipes/qwen3-32b-fp8/vllm/disagg/deploy.yaml -n ${NAMESPACE}
```

</div>

## Smoke Test

Send a test request to verify the deployment serves traffic:

<div data-variant="trtllm-agg">

```bash
kubectl port-forward svc/qwen3-32b-fp8-agg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-variant="trtllm-disagg">

```bash
kubectl port-forward svc/qwen3-32b-fp8-disagg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-variant="vllm-disagg">

```bash
kubectl port-forward svc/qwen3-32b-fp8-vllm-disagg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-32B-FP8","messages":[{"role":"user","content":"Write a one-sentence readiness check."}],"max_tokens":64}'
```

## Benchmark

Each target ships its own `perf.yaml` AIPerf Job, sized at a fixed concurrency per GPU with `request-count = 10x concurrency`. Artifacts land on the `model-cache` PVC under `/model-cache/perf`.

<div data-variant="trtllm-agg">

The Job wraps this AIPerf run — 4K ISL / 500 OSL at concurrency 4 (2 per GPU x 2 GPUs):

```bash
aiperf profile \
  --model Qwen/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B-FP8 \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --url http://qwen3-32b-fp8-agg-frontend:8000 --streaming \
  --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 500 --output-tokens-stddev 0 \
  --extra-inputs max_tokens:500 --extra-inputs min_tokens:500 --extra-inputs ignore_eos:true \
  --concurrency 4 --request-count 40 --warmup-request-count 4 \
  --random-seed 100
```

```bash
kubectl apply -f recipes/qwen3-32b-fp8/trtllm/agg/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-32b-fp8-bench -n ${NAMESPACE} --timeout=7200s
```

</div>

<div data-variant="trtllm-disagg">

The Job wraps this AIPerf run — 4K ISL / 500 OSL at concurrency 48 (6 per GPU x 8 GPUs):

```bash
aiperf profile \
  --model Qwen/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B-FP8 \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --url http://qwen3-32b-fp8-disagg-frontend:8000 --streaming \
  --synthetic-input-tokens-mean 4000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 500 --output-tokens-stddev 0 \
  --extra-inputs max_tokens:500 --extra-inputs min_tokens:500 --extra-inputs ignore_eos:true \
  --concurrency 48 --request-count 480 --warmup-request-count 48 \
  --random-seed 100
```

```bash
kubectl apply -f recipes/qwen3-32b-fp8/trtllm/disagg/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-32b-fp8-bench -n ${NAMESPACE} --timeout=7200s
```

</div>

<div data-variant="vllm-disagg">

The Job wraps this AIPerf run — 2K ISL / 500 OSL at concurrency 8 (1 per GPU x 8 GPUs); note the shorter input length than the TRT-LLM targets:

```bash
aiperf profile \
  --model Qwen/Qwen3-32B-FP8 --tokenizer Qwen/Qwen3-32B-FP8 \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --url http://qwen3-32b-fp8-vllm-disagg-frontend:8000 --streaming \
  --synthetic-input-tokens-mean 2000 --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 500 --output-tokens-stddev 0 \
  --extra-inputs max_tokens:500 --extra-inputs min_tokens:500 --extra-inputs ignore_eos:true \
  --concurrency 8 --request-count 80 --warmup-request-count 8 \
  --random-seed 100
```

```bash
kubectl apply -f recipes/qwen3-32b-fp8/vllm/disagg/perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/qwen3-32b-fp8-vllm-disagg-perf -n ${NAMESPACE} --timeout=7200s
```

</div>

## Compare All Targets

All three targets serve `Qwen/Qwen3-32B-FP8`; they differ in runtime, GPU count, topology, and benchmark traffic:

| | TRT-LLM aggregated | TRT-LLM disaggregated | vLLM disaggregated |
|---|---|---|---|
| **GPUs** | 2x H100/H200/A100 | 8x H100/H200/A100 | 8x H100/H200/A100, single node |
| **Topology** | 1x worker, TP2 | 4x prefill (TP1) + 2x decode (TP2) | 2x prefill (TP2) + 1x decode (TP4) |
| **Routing / KV transfer** | Round-robin | Round-robin | NIXL (NixlConnector) |
| **Workload** | 4K / 500, conc. 4 | 4K / 500, conc. 48 | 2K / 500, conc. 8 |

## Notes

- This page is the FP8 alternative to the BF16 [Qwen3-32B](qwen3-32b.mdx) recipe.
- The TRT-LLM and vLLM targets use different traffic shapes (4K vs 2K ISL) and different GPU counts; normalize traffic before making backend performance claims.
- The aggregated config uses CUDA graphs for optimized inference, and KV cache uses FP8 dtype for memory efficiency.
- `--max-model-len 8192` is set in `vllm/disagg/deploy.yaml` for A100 40 GB compatibility; remove or increase it on H100/H200.
- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying.

## Source

- Source README: [recipes/qwen3-32b-fp8/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/README.md)
- TRT-LLM aggregated: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/agg/deploy.yaml) and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/agg/perf.yaml)
- TRT-LLM disaggregated: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/disagg/deploy.yaml) and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/trtllm/disagg/perf.yaml)
- vLLM disaggregated: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/vllm/disagg/deploy.yaml) and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/vllm/disagg/perf.yaml)
- Setup assets: [model-cache/model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/model-cache/model-cache.yaml) and [model-cache/model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/qwen3-32b-fp8/model-cache/model-download.yaml)
