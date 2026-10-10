---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Motif-3 NVFP4"
subtitle: "Serve Motif-3 NVFP4 with Dynamo and the Motif vLLM runtime on NVIDIA B200 GPUs."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

This experimental recipe serves [Motif-Technologies/Motif-3-NVFP4](https://huggingface.co/Motif-Technologies/Motif-3-NVFP4) on two NVIDIA B200 GPUs with vLLM tensor parallelism, expert parallelism, Dynamo MTP2 speculative decoding, and a 262K-token context. Only an aggregated target is provided; there is no disaggregated target for Motif-3 yet.

<div className="dynamo-target-picker static">
<p className="dynamo-target-picker-title">Deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked disabled />
<label htmlFor="recipe-variant-agg">Aggregated <span className="dynamo-target-picker-hint">Recommended</span></label>
</div>
<div className="dynamo-target-picker-summary">
<span><b>Hardware</b> 2x NVIDIA B200</span>
<span><b>Runtime</b> Motif vLLM image, Dynamo 1.5.0</span>
<span><b>Serving</b> TP2, expert parallelism, Dynamo MTP2</span>
<span><b>Cache</b> Shared model cache, FP8 KV cache</span>
</div>
</div>

## Prerequisites

- A Kubernetes cluster with the Dynamo platform installed and 2xB200 GPUs available.
- Create a namespace and an `hf-token-secret` containing access to the model. The token is used only by the model-download Job.

```bash
export NAMESPACE=your-namespace
kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n "${NAMESPACE}"
```

## Deploy

Edit `storageClassName` in the [model-cache manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/model-cache/model-cache.yaml), then create the cache and download the checkpoint:

```bash
kubectl apply -f recipes/motif-3/model-cache/model-cache.yaml -n "${NAMESPACE}"
kubectl apply -f recipes/motif-3/model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/model-download -n "${NAMESPACE}" --timeout=7200s
```

Apply the [aggregated chat manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml):

```bash
kubectl apply -f recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml -n "${NAMESPACE}"
kubectl get dgd motif3-agg-b200 -n "${NAMESPACE}" -w
```

The base manifest has no cluster-specific scheduling. To add node selectors or tolerations for your cluster, use the [Kustomization](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3/vllm/agg-b200-chat/kustomize) described in the [recipe README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/README.md).

## Smoke Test

First, forward the frontend port for your target:

```bash
kubectl port-forward svc/motif3-agg-b200-frontend 8000:8000 -n "${NAMESPACE}"
```

Send a request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Motif-Technologies/Motif-3-NVFP4",
    "messages": [{"role": "user", "content": "Write a one-sentence readiness check."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Benchmark

The [AIPerf manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/perf/perf.yaml) replays the [`8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` trace](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl) at concurrency 11 against the Dynamo MTP2, TP2 deployment. Stage that trace at `/shared-model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl`, then apply the Job:

```bash
kubectl apply -f recipes/motif-3/perf/perf.yaml -n "${NAMESPACE}"
kubectl logs -f job/motif3-aiperf-job -n "${NAMESPACE}"
```

The deployment uses actual MTP verification by default. For benchmark-only synthetic acceptance-length trials, change the `SPECULATIVE_CONFIG` ConfigMap key in the [deployment manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml) from `speculative-config` to `speculative-config-synthetic`. These acceptance lengths were calculated by running [SPEED-Bench](https://huggingface.co/blog/nvidia/speed-bench) on the coding domain:

| MTP speculative tokens | Acceptance length |
|:---:|:---:|
| 1 | 1.75 |
| 2 | 2.13 |
| 3 | 2.21 |

## Expected Performance

<Warning>
This is a benchmark-only synthetic proxy, not the expected performance of the shipped manifests. It was measured with `speculative-config-synthetic` (MTP2, acceptance length 2.13 from SPEED-Bench coding) on the chat trace. The deployment manifest uses real MTP verification, and the AIPerf Job does not select the synthetic configuration, so applying them as shipped will not reproduce these figures. A real-MTP chat-trace result is not yet available.
</Warning>

Under the synthetic proxy, the run meets user output throughput P50 ≥ 50 tokens/second/user and TTFT P50 < 5 seconds. Thirty-five trace requests exceeded the configured 262,144-token context limit.

| Workload | Recipe | GPU | Concurrency | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
|---|---|---|---:|---:|---:|---:|
| Chat trace (synthetic AL 2.13 proxy) | Dynamo MTP2, TP2 | B200 | 11 | 293.96 | 66.27 | 388.18 |

## Notes

- The recipe uses the public `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-motif-3-dev.1` image; no image-pull secret is required.
- The model image carries Motif-specific vLLM compatibility code. Do not replace its vLLM package with an unrelated nightly wheel.

## Source

- [Motif-3 recipe files](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3)
- [Motif-3-NVFP4 model card](https://huggingface.co/Motif-Technologies/Motif-3-NVFP4)
