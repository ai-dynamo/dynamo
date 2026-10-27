---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Motif-3 NVFP4"
subtitle: "Deploy aggregated or 3P1D KV-aware Motif-3 serving on NVIDIA B200 GPUs."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

**Experimental.** Serve [Motif-3 NVFP4](https://huggingface.co/Motif-Technologies/Motif-3-NVFP4) with NVIDIA Dynamo and vLLM on B200. Choose one aggregated worker or three prefill workers and one decode worker (3P1D) with KV-aware routing. Each worker uses two GPUs (TP2), expert parallelism, and multi-token prediction with two speculative tokens (MTP2).

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated 3P1D</label>
</div>
<div className="dynamo-target-picker-summary" data-variant="agg">
<span><b>Hardware</b> 2x NVIDIA B200</span>
<span><b>Serving</b> One TP2 worker, expert parallelism, MTP2</span>
<span><b>Cache</b> NVFP4 weights, FP8 KV cache, shared model cache</span>
</div>
<div className="dynamo-target-picker-summary" data-variant="disagg">
<span><b>Hardware</b> 8x NVIDIA B200</span>
<span><b>Serving</b> 3 TP2 prefill + 1 TP2 decode, expert parallelism, MTP2</span>
<span><b>Routing</b> KV-aware, NIXL/UCX transfer, FP8 KV cache</span>
</div>
</div>

## Prerequisites

- A Kubernetes cluster with the [Dynamo platform installed](../../kubernetes/getting-started/quickstart.mdx) and a shared ReadWriteMany model-cache volume.
- A Hugging Face token with access to the checkpoint, used by the model-download Job.

<div data-variant="agg">

- Two available B200 GPUs on one node.

</div>

<div data-variant="disagg">

- Eight available B200 GPUs. Each TP2 worker occupies two GPUs on one node; configure the fabric for NIXL/UCX transfer between nodes.

</div>

Run the following commands from the repository root. Create a namespace and download secret if they do not already exist:

```bash
export NAMESPACE=your-namespace
kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n "${NAMESPACE}"
```

## Deploy

Reuse a populated `shared-model-cache` PVC when available. Otherwise, set `storageClassName` in the [model-cache manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/model-cache/model-cache.yaml), then create the cache and download the checkpoint. For Nscale, use `storageClassName: vast`.

```bash
kubectl apply -f recipes/motif-3/model-cache/model-cache.yaml -n "${NAMESPACE}"
kubectl apply -f recipes/motif-3/model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=Complete job/model-download -n "${NAMESPACE}" --timeout=7200s
```

<div data-variant="agg">

Apply the [aggregated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml):

```bash
kubectl apply -f recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml -n "${NAMESPACE}"
kubectl get dgd motif3-agg-b200 -n "${NAMESPACE}" -w
```

For Nscale B200 selection and GPU tolerations, apply the [aggregated Kustomization](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3/vllm/agg-b200-chat/kustomize) instead:

```bash
kubectl apply -k recipes/motif-3/vllm/agg-b200-chat/kustomize -n "${NAMESPACE}"
```

Rendering the aggregated Kustomization requires Git and network access.

</div>

<div data-variant="disagg">

Configure GPU node selection and fabric resources for your cluster, then apply the [generic 3P1D manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/disagg-b200-chat/deploy-generic.yaml):

```bash
kubectl apply -f recipes/motif-3/vllm/disagg-b200-chat/deploy-generic.yaml -n "${NAMESPACE}"
kubectl get dgd motif3-disagg-b200 -n "${NAMESPACE}" -w
```

For Nscale, use the optional [site component](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3/vllm/disagg-b200-chat/kustomize/components/nscale) to add B200 node selection and RDMA settings:

```bash
python3 scripts/kustomize-matrix.py compose \
  recipes/motif-3/vllm/disagg-b200-chat/kustomize/overlays/generic \
  recipes/motif-3/vllm/disagg-b200-chat/kustomize/components/nscale \
  | kubectl apply -f - -n "${NAMESPACE}"
```

Wait for the frontend and all four workers to become ready.

</div>

## Smoke Test

Forward the frontend port for the selected target:

<div data-variant="agg">

```bash
kubectl port-forward svc/motif3-agg-b200-frontend 8000:8000 -n "${NAMESPACE}"
```

</div>

<div data-variant="disagg">

```bash
kubectl port-forward svc/motif3-disagg-b200-frontend 8000:8000 -n "${NAMESPACE}"
```

</div>

In another terminal, verify model discovery and a completion:

```bash
curl --fail http://localhost:8000/v1/models
curl --fail http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Motif-Technologies/Motif-3-NVFP4",
    "messages": [{"role": "user", "content": "Write a one-sentence readiness check."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

## Benchmark

Stage the complete [`8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl` trace](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.6/perf/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl) at `/shared-model-cache/traces/8k_1k_70kv_chat_new_noschedule_short_15perc.jsonl`.

<div data-variant="agg">

The [aggregated AIPerf Job](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/perf/perf.yaml) uses concurrency 11:

```bash
kubectl apply -f recipes/motif-3/perf/perf.yaml -n "${NAMESPACE}"
kubectl logs -f job/motif3-aiperf-job -n "${NAMESPACE}"
```

</div>

<div data-variant="disagg">

Follow the [disaggregated benchmark workflow](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/perf/README.md#disaggregated-3p1d) to prepare the client and reset worker and router caches before each independent run.

The runner sends 32 warmup requests at concurrency 32, then measures all 1,805 trace rows at concurrency 64. It retains worker and router caches between phases and saves separate artifacts; warmup requests are excluded from the results.

</div>

Both targets default to **real MTP verification**. To use synthetic acceptance length 2.13 for benchmarking, change the `SPECULATIVE_CONFIG` ConfigMap key from `speculative-config` to `speculative-config-synthetic` in the deployment source, regenerate any derived manifest, and redeploy. For disaggregation, change **both worker roles**; see the [recipe README](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3/vllm/disagg-b200-chat) for regeneration commands.

## Expected Performance

<Note>
These chat-trace results use synthetic MTP2 acceptance length 2.13. They do not measure real MTP acceptance or output quality. The default manifests use real MTP verification.
</Note>

TTFT is Time To First Token. Per-user output rate excludes TTFT; throughput per GPU includes every serving GPU.

<div data-variant="agg">

| Deployment | GPUs | Concurrency | Output tokens/sec/GPU | P50 tokens/sec/user | P50 TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| Aggregated | 2 | 11 | 293.96 | 66.27 | 388.18 |

Thirty-five requests exceeded the 262,144-token context limit.

</div>

<div data-variant="disagg">

| Deployment | GPUs | Concurrency | Output tokens/sec/GPU | P50 tokens/sec/user | P50 TTFT (ms) |
|---|---:|---:|---:|---:|---:|
| Disaggregated 3P1D, KV-aware | 8 | 64 | 319.26 | 51.49 | 4305.08 |

</div>

## Compare All Targets

| Target | Workers | B200 GPUs | Deployment |
|---|---|---:|---|
| Aggregated | 1 × TP2 | 2 | [Generic](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/agg-b200-chat/base/deploy.yaml) · [Nscale](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3/vllm/agg-b200-chat/kustomize) |
| Disaggregated, KV-aware | 3 prefill × TP2 + 1 decode × TP2 | 8 | [Generic](https://github.com/ai-dynamo/dynamo/blob/main/recipes/motif-3/vllm/disagg-b200-chat/deploy-generic.yaml) |

## Notes

- Both targets use NVFP4 weights, FP8 KV cache, prefix caching, and a 262,144-token context limit. Frontends explicitly set `--kv-cache-block-size 128`, matching `--block-size=128` on every worker.
- Both targets use the public `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0-motif-3-dev.2` image; no NGC API key or image-pull secret is required. The Hugging Face secret is used only to download the model.
- Populate the shared model cache before starting the workers, which run in offline mode.
- The disaggregated recipe has been validated with TP2 on both prefill and decode workers.

## Source

- [Motif-3 recipe files](https://github.com/ai-dynamo/dynamo/tree/main/recipes/motif-3)
- [Motif-3-NVFP4 model card](https://huggingface.co/Motif-Technologies/Motif-3-NVFP4)
