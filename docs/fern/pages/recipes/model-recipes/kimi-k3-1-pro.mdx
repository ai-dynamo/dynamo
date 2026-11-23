---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Kimi-K3.1-Pro"
subtitle: "Serve Kimi-K3.1-Pro with Dynamo using vLLM or SGLang."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

<style>{`
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-agg:checked) .dynamo-kimi-comparison tr[data-highlight-target="vllm-gb300-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-disagg:checked) .dynamo-kimi-comparison tr[data-highlight-target="vllm-gb300-disagg"],
body:has(#recipe-framework-sglang:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-agg:checked) .dynamo-kimi-comparison tr[data-highlight-target="sglang-gb300-agg"],
body:has(#recipe-framework-sglang:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-disagg:checked) .dynamo-kimi-comparison tr[data-highlight-target="sglang-gb300-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200:checked):has(#recipe-variant-agg:checked) .dynamo-kimi-comparison tr[data-highlight-target="vllm-b200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-vr200:checked):has(#recipe-variant-agg:checked) .dynamo-kimi-comparison tr[data-highlight-target="vllm-vr200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-vr200:checked):has(#recipe-variant-disagg:checked) .dynamo-kimi-comparison tr[data-highlight-target="vllm-vr200-disagg"] {
  background: color-mix(in srgb, var(--nv-color-green) 18%, transparent);
  font-weight: 600;
}
`}</style>

Serve `moonshotai/Kimi-K3.1-Pro` with NVIDIA Dynamo using vLLM or SGLang. Pick your framework, GPU, and serving topology; deployment commands update to match. SGLang targets are experimental, text-only, and functional-only; no performance claims are made.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Framework</span>
<input type="radio" id="recipe-framework-vllm" name="recipe-framework" value="vllm" defaultChecked />
<label htmlFor="recipe-framework-vllm">vLLM</label>
<input type="radio" id="recipe-framework-sglang" name="recipe-framework" value="sglang" />
<label htmlFor="recipe-framework-sglang">SGLang</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" />
<label htmlFor="recipe-sku-b200">B200</label>
<input type="radio" id="recipe-sku-gb300" name="recipe-sku" value="gb300" />
<label htmlFor="recipe-sku-gb300">GB300</label>
<input type="radio" id="recipe-sku-vr200" name="recipe-sku" value="vr200" defaultChecked />
<label htmlFor="recipe-sku-vr200">VR200</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">
<span><b>GPUs</b> 8x GB300 (1 worker)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> Disabled</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">
<span><b>GPUs</b> 24x GB300 (2P1D)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> Disabled</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="gb300" data-variant="agg">
<span><b>GPUs</b> 8x GB300 (1 worker)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> Disabled</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="gb300" data-variant="disagg">
<span><b>GPUs</b> 16x GB300 (1P1D)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> Disabled</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200" data-variant="agg">
<span><b>GPUs</b> 16x B200 (1 worker)</span>
<span><b>Parallelism</b> TP8 × PP2 across 2 nodes</span>
<span><b>Speculative decoding</b> DSpark (7 draft tokens)</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">
<span><b>GPUs</b> 16x VR200 (2 workers)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> DSpark (7 draft tokens)</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">
<span><b>GPUs</b> 24x VR200 (2P1D)</span>
<span><b>Parallelism</b> TP8 across 2 nodes per worker</span>
<span><b>Speculative decoding</b> DSpark (7 draft tokens)</span>
</div>
</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="disagg">

<Note>No B200 disaggregated recipe is published. Select aggregated serving or another GPU.</Note>

</div>

<div data-recipe-framework="sglang" data-sku="b200 vr200">

<Note>SGLang recipes on this page cover GB300 only. Select GB300 or switch to vLLM.</Note>

</div>

## Prerequisites

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 8 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 24 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="sglang" data-sku="gb300" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 8 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="sglang" data-sku="gb300" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 16 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 16 B200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 16 VR200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 24 VR200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).

</div>

- A Hugging Face token with access to `moonshotai/Kimi-K3.1-Pro`.
- Access to the runtime images referenced by the selected manifest.

<div data-sku="gb300 vr200">

- The NVIDIA DRA driver with ComputeDomain support for multi-node NVLink serving. Every GB300 and VR200 manifest includes a ComputeDomain and worker channel claims.

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
Review storage paths, node placement, registry credentials, and resource claims before applying the selected manifest.
</Warning>

## Deploy

<div data-sku="b200 gb300 vr200">

Create the shared model cache and download the main and draft checkpoints:

<Warning>
Edit `storageClassName` in `recipes/kimi-k3.1-pro/model-cache/model-cache.yaml` to a ReadWriteMany class on your cluster before you apply it.
</Warning>

```bash
kubectl apply -f recipes/kimi-k3.1-pro/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/kimi-k3.1-pro/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

The download Job populates the Hugging Face cache with `moonshotai/Kimi-K3.1-Pro` and `Inferact/Kimi-K3-DSpark`, using `HF_HOME=/shared-model-cache`. Workers resolve these Hugging Face IDs offline from the same PVC. To change the checkpoints, update `MODEL_NAME` and `DRAFT_MODEL_NAME` in the Job and the corresponding worker settings.

</div>

<div data-sku="b200">

The B200 worker uses the draft checkpoint downloaded by the Job.

</div>

<div data-sku="vr200">

Both VR200 topologies use the draft checkpoint downloaded by the Job and mount `shared-model-cache` only in the workers; frontends do not mount the PVC.

</div>

Deploy the selected DynamoGraphDeployment (DGD):

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">

```bash
export DGD=kimi-k3-1-pro-vllm-gb300-agg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-gb300-agentic/deploy-gb300.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">

For image and video input, apply the multimodal overlay instead:

```bash
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-gb300-agentic/deploy-gb300-multimodal.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">

```bash
export DGD=kimi-k3-1-pro-vllm-gb300-disagg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/vllm/disagg-gb300-agentic/deploy-gb300.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">

For image and video input, apply the multimodal overlay instead:

```bash
kubectl apply -f recipes/kimi-k3.1-pro/vllm/disagg-gb300-agentic/deploy-gb300-multimodal.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="sglang" data-sku="gb300" data-variant="agg">

```bash
export DGD=kimi-k3-1-pro-sglang-gb300-agg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/sglang/agg-gb300-agentic/deploy-gb300.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="sglang" data-sku="gb300" data-variant="disagg">

```bash
export DGD=kimi-k3-1-pro-sglang-gb300-disagg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/sglang/disagg-gb300-agentic/deploy-gb300.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="agg">

```bash
export DGD=kimi-k3-1-pro-vllm-b200-agg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-b200-agentx/deploy-b200.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">

```bash
export DGD=kimi-k3-1-pro-vllm-vr200-agg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-vr200-agentx/deploy-vr200.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">

```bash
export DGD=kimi-k3-1-pro-vllm-vr200-disagg-agentic
kubectl apply -f recipes/kimi-k3.1-pro/vllm/disagg-vr200-agentx/deploy-vr200.yaml -n ${NAMESPACE}
```

</div>

Each target's Kustomize overlay produces the corresponding manifest above. Customize node placement and registry credentials for your cluster before deployment; the published manifests do not include node selectors, tolerations, or image pull secrets.

B200 and VR200 targets enable DSpark with natural acceptance. Their manifests include a commented synthetic-acceptance alternative for benchmarking.

## Smoke Test

The deploy command sets `DGD` to the selected resource name. Wait for readiness and forward the frontend port:

```bash
kubectl wait --for=condition=Ready dynamographdeployment/${DGD} \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}
```

Send a chat completion with the served model name `moonshotai/Kimi-K3.1-Pro`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"moonshotai/Kimi-K3.1-Pro","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":256}'
```

## Benchmark

Use [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/perf/perf.yaml) to run AIPerf 0.13.0 against the deployment. It runs `aiperf profile` with the chat endpoint type and streaming on the AgentX agentic trace (`semianalysis_cc_traces_weka_062126`), scenario `inferencex-agentx-mvp`, with an agentic warmup and a 1200 s profiling phase. The tokenizer is resolved by Hugging Face ID with `HF_HOME=/shared-model-cache`.

The performance results use text-only AgentX traffic. For B200 and VR200 benchmarking, edit the selected target's `kustomize/base/deploy.yaml`: uncomment both lines of the synthetic `SPECULATIVE_CONFIG` alternative and comment out both lines of the active natural-acceptance entry. Keep exactly one active entry per worker; update both prefill and decode for disaggregated VR200. Regenerate and apply the selected manifest before running AIPerf:

<div data-recipe-framework="vllm" data-sku="b200" data-variant="agg">

```bash
python3 scripts/kustomize-matrix.py render recipes/kimi-k3.1-pro/vllm/agg-b200-agentx/.kustomize-matrix.yaml
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-b200-agentx/deploy-b200.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">

```bash
python3 scripts/kustomize-matrix.py render recipes/kimi-k3.1-pro/vllm/agg-vr200-agentx/.kustomize-matrix.yaml
kubectl apply -f recipes/kimi-k3.1-pro/vllm/agg-vr200-agentx/deploy-vr200.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">

```bash
python3 scripts/kustomize-matrix.py render recipes/kimi-k3.1-pro/vllm/disagg-vr200-agentx/.kustomize-matrix.yaml
kubectl apply -f recipes/kimi-k3.1-pro/vllm/disagg-vr200-agentx/deploy-vr200.yaml -n ${NAMESPACE}
```

</div>

Wait for the replacement workers to become Ready before measuring. Restore natural acceptance, regenerate, and reapply the manifest before returning to normal serving.

The synthetic acceptance length of 3.84 was measured offline on the SpeedBench (qualitative) coding subset using the same DSpark draft length as this recipe: 7 draft tokens. The AgentX dataset and scenario above describe the throughput benchmark, not the offline acceptance-length measurement.

Set these values in the Job `env` section for your target:

- `ENDPOINT`: the frontend Service of the DynamoGraphDeployment, `${DGD}-frontend:8000` (substitute the selected deployment name).
- `CONCURRENCY`: GB300 uses 32 for vLLM aggregated and 48 for disaggregated. Reported VR200 points use 180 for aggregated and 174 for disaggregated; B200 uses 48.

The benchmark Job uses the same `shared-model-cache` PVC and tokenizer files as the deployment.

```bash
kubectl apply -f recipes/kimi-k3.1-pro/perf/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/kimi-k3-1-aiperf -n ${NAMESPACE}
```

Results are written to `/shared-model-cache/perf/<epoch>_<job-name>/`. Restart the workers between independent runs to reset KV-cache and routing state.

### Optimization Targets

The AgentX benchmark targets use the following interactivity goals:

| Workload | User output tok/s (P50) | TTFT P50 |
| --- | ---: | ---: |
| AgentX (inferencex-agentx-mvp) | ≥ 50 | < 2 s |

### Performance Results

All results below measure text-only traffic; the selected target is highlighted. B200 and VR200 use synthetic draft acceptance (AL 3.84); GB300 has speculative decoding disabled.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table dynamo-kimi-comparison">
<thead><tr><th>Workload</th><th>Recipe</th><th>Framework</th><th>SKU</th><th>Concurrency</th><th>System Output tok/s/GPU</th><th>User output tok/s (P50)</th><th>TTFT P50 (ms)</th></tr></thead>
<tbody>
<tr data-highlight-target="vllm-vr200-agg"><td>AgentX (inferencex-agentx-mvp)</td><td>aggregated, 2 replicas</td><td>vLLM</td><td>VR200</td><td>180</td><td>153.01</td><td>57.71</td><td>1,534</td></tr>
<tr data-highlight-target="vllm-vr200-disagg"><td>AgentX (inferencex-agentx-mvp)</td><td>disaggregated 2P1D</td><td>vLLM</td><td>VR200</td><td>174</td><td>122.19</td><td>73.93</td><td>1,761</td></tr>
<tr data-highlight-target="vllm-b200-agg"><td>AgentX (inferencex-agentx-mvp)</td><td>aggregated, 1 replica</td><td>vLLM</td><td>B200</td><td>48</td><td>≈26.41</td><td>64.02</td><td>550.45</td></tr>
<tr data-highlight-target="vllm-gb300-agg"><td>AgentX (inferencex-agentx-mvp)</td><td>aggregated TP8, 1 replica</td><td>vLLM</td><td>GB300</td><td>32</td><td>39.89</td><td>76.7</td><td>428</td></tr>
<tr data-highlight-target="vllm-gb300-disagg"><td>AgentX (inferencex-agentx-mvp)</td><td>disaggregated 2P1D</td><td>vLLM</td><td>GB300</td><td>48</td><td>23.08</td><td>87.1</td><td>656</td></tr>
<tr data-highlight-target="vllm-gb300-agg"><td>AgentX (inferencex-agentx-mvp)</td><td>aggregated multimodal overlay, 1 replica (pre-release)</td><td>vLLM</td><td>GB300</td><td>32</td><td>39.56</td><td>70.9</td><td>440</td></tr>
<tr data-highlight-target="vllm-gb300-disagg"><td>AgentX (inferencex-agentx-mvp)</td><td>disaggregated multimodal overlay (pre-release)</td><td>vLLM</td><td>GB300</td><td>48</td><td>23.38</td><td>85.7</td><td>693</td></tr>
</tbody>
</table>
</div>




## Compare All Targets

| Target | GPUs | Parallelism | Validation |
| --- | ---: | --- | --- |
| [GB300 vLLM aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-gb300-agentic/deploy-gb300.yaml) | 8 | TP8 per worker | Text-only benchmark |
| [GB300 vLLM aggregated multimodal](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-gb300-agentic/deploy-gb300-multimodal.yaml) | 8 | TP8 per worker | Text-only benchmark with pre-release weights; image input validated |
| [GB300 vLLM disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/disagg-gb300-agentic/deploy-gb300.yaml) | 24 | TP8 per worker | Text-only benchmark |
| [GB300 vLLM disaggregated multimodal](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/disagg-gb300-agentic/deploy-gb300-multimodal.yaml) | 24 | TP8 per worker | Text-only benchmark with pre-release weights; image input validated |
| [GB300 SGLang aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/sglang/agg-gb300-agentic/deploy-gb300.yaml) | 8 | TP8 per worker | Functional only; no performance claim |
| [GB300 SGLang disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/sglang/disagg-gb300-agentic/deploy-gb300.yaml) | 16 | TP8 per worker | Functional only; no performance claim |
| [B200 vLLM aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-b200-agentx/deploy-b200.yaml) | 16 | TP8 × PP2 | Text-only benchmark |
| [VR200 vLLM aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-vr200-agentx/deploy-vr200.yaml) | 16 | TP8 per worker | Text-only benchmark |
| [VR200 vLLM disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/disagg-vr200-agentx/deploy-vr200.yaml) | 24 | TP8 per worker | Text-only benchmark |

GPU counts below describe the checked-in manifests. P counts prefill workers; D counts decode workers.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table dynamo-kimi-comparison">
<thead><tr><th>Framework</th><th>GPU</th><th>Topology</th><th>Total GPUs</th><th>Parallelism</th><th>Speculative Decoding</th><th>Storage</th></tr></thead>
<tbody>
<tr data-highlight-target="vllm-gb300-agg"><td>vLLM</td><td>GB300</td><td>1 worker</td><td>8</td><td>TP8 across 2 nodes per worker</td><td>Disabled</td><td>Shared PVC</td></tr>
<tr data-highlight-target="vllm-gb300-disagg"><td>vLLM</td><td>GB300</td><td>2P1D</td><td>24</td><td>TP8 across 2 nodes per worker</td><td>Disabled</td><td>Shared PVC</td></tr>
<tr data-highlight-target="sglang-gb300-agg"><td>SGLang</td><td>GB300</td><td>1 worker</td><td>8</td><td>TP8 across 2 nodes per worker</td><td>Disabled</td><td>Shared PVC</td></tr>
<tr data-highlight-target="sglang-gb300-disagg"><td>SGLang</td><td>GB300</td><td>1P1D</td><td>16</td><td>TP8 across 2 nodes per worker</td><td>Disabled</td><td>Shared PVC</td></tr>
<tr data-highlight-target="vllm-b200-agg"><td>vLLM</td><td>B200</td><td>1 worker</td><td>16</td><td>TP8 × PP2 across 2 nodes</td><td>DSpark (7 draft tokens)</td><td>Shared PVC</td></tr>
<tr data-highlight-target="vllm-vr200-agg"><td>vLLM</td><td>VR200</td><td>2 workers</td><td>16</td><td>TP8 across 2 nodes per worker</td><td>DSpark (7 draft tokens)</td><td>Shared PVC</td></tr>
<tr data-highlight-target="vllm-vr200-disagg"><td>vLLM</td><td>VR200</td><td>2P1D</td><td>24</td><td>TP8 across 2 nodes per worker</td><td>DSpark (7 draft tokens)</td><td>Shared PVC</td></tr>
</tbody>
</table>
</div>

## Notes

- SGLang targets are experimental, text-only, and functional-only. No SGLang performance claims are made. Use the vLLM multimodal overlays for image and video input.
- Synthetic draft acceptance is for benchmarking. Keep the natural-acceptance configuration active for normal serving.
- Disaggregated performance depends on your fabric; RDMA tuning is cluster-specific.

## Source

- Deployment assets: [Kimi-K3.1-Pro recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes/kimi-k3.1-pro)
- Setup assets: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/model-cache/model-download.yaml)
- Benchmark manifest: [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/perf/perf.yaml)

- B200 aggregated: [deployment manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-b200-agentx/deploy-b200.yaml)
- VR200 aggregated: [deployment manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/agg-vr200-agentx/deploy-vr200.yaml)
- VR200 disaggregated: [deployment manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k3.1-pro/vllm/disagg-vr200-agentx/deploy-vr200.yaml)
