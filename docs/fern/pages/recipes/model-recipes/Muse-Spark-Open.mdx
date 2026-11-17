---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Muse-Spark-Open-816B-A42B"
subtitle: "Serve Muse-Spark-Open-816B-A42B with Dynamo using SGLang or vLLM on GB200 and GB300, or TensorRT-LLM on GB200 and GB300."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

This page covers serving Muse-Spark-Open-816B-A42B with SGLang (recommended), or experimental vLLM and TensorRT-LLM, on GB200 and GB300.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Framework</span>
<input type="radio" id="recipe-engine-sglang" name="recipe-engine" value="sglang" defaultChecked />
<label htmlFor="recipe-engine-sglang">SGLang <span className="dynamo-target-picker-hint">Recommended</span></label>
<input type="radio" id="recipe-engine-vllm" name="recipe-engine" value="vllm" />
<label htmlFor="recipe-engine-vllm">vLLM <span className="dynamo-target-picker-hint">Experimental</span></label>
<input type="radio" id="recipe-engine-trtllm" name="recipe-engine" value="trtllm" />
<label htmlFor="recipe-engine-trtllm">TensorRT-LLM <span className="dynamo-target-picker-hint">Experimental</span></label>
</div>
<div className="dynamo-target-picker-row" data-engine="sglang vllm trtllm">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-gb200" name="recipe-sku" value="gb200" defaultChecked />
<label htmlFor="recipe-sku-gb200">GB200</label>
<input type="radio" id="recipe-sku-gb300" name="recipe-sku" value="gb300" />
<label htmlFor="recipe-sku-gb300">GB300</label>
</div>
<div className="dynamo-target-picker-row" data-engine="sglang">
<span className="dynamo-target-picker-dim">Deployment mode</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
<input type="radio" id="recipe-variant-kv-router" name="recipe-variant" value="kv-router" />
<label htmlFor="recipe-variant-kv-router">KV-aware routing</label>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb200" data-variant="agg">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>GPUs</b> 4x GB200, TP4/EP4</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb200" data-variant="disagg">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>Topology</b> 1P1D, TP4/EP4 per worker</span>
<span><b>KV transfer</b> NIXL</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb200" data-variant="kv-router">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>GPUs</b> 8x GB200, 2 TP4/EP4 replicas</span>
<span><b>Routing</b> KV-aware</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb300" data-variant="agg">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>GPUs</b> 4x GB300, TP4/EP4</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb300" data-variant="disagg">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>Topology</b> 1P1D, TP4/EP4 per worker</span>
<span><b>KV transfer</b> NIXL</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="sglang" data-sku="gb300" data-variant="kv-router">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> SGLang</span>
<span><b>GPUs</b> 8x GB300, 2 TP4/EP4 replicas</span>
<span><b>Routing</b> KV-aware</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="vllm" data-sku="gb200">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> vLLM (experimental)</span>
<span><b>GPUs</b> 4x GB200, TP4/EP4</span>
<span><b>Topology</b> Aggregated</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="vllm" data-sku="gb300">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> vLLM (experimental)</span>
<span><b>GPUs</b> 4x GB300, TP4/EP4</span>
<span><b>Topology</b> Aggregated</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="trtllm" data-sku="gb200">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> TensorRT-LLM (experimental)</span>
<span><b>GPUs</b> 4x GB200, TP4/EP4</span>
<span><b>Topology</b> Aggregated</span>
</div>
<div className="dynamo-target-picker-summary" data-engine="trtllm" data-sku="gb300">
<span><b>Model</b> Muse-Spark-Open-816B-A42B</span>
<span><b>Precision</b> NVFP4 + FP8 KV cache</span>
<span><b>Framework</b> TensorRT-LLM (experimental)</span>
<span><b>GPUs</b> 4x GB300, TP4/EP4</span>
<span><b>Topology</b> Aggregated</span>
</div>
</div>

<div data-engine="sglang" data-variant="disagg">

<Warning>
Video requests fail with the referenced Dynamo image because it does not include `decord2`, the package that provides SGLang's `decord` video decoder. Frontend decoding (`DYN_SGL_FRONTEND_DECODING=true`) is not supported for image or video inputs in these disaggregated recipes. Select an aggregated SGLang recipe for video requests.
</Warning>

</div>

## Prerequisites

<div data-engine="sglang" data-sku="gb200">

- A Kubernetes cluster with the Dynamo platform installed and GB200 GPUs available. Use 4 GPUs (one node) for aggregated serving and 8 GPUs (two nodes) for either the disaggregated or KV-aware routed target.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the SGLang runtime image referenced by the deployment manifest.
- ComputeDomain support with Dynamic Resource Allocation (DRA) enabled.

</div>

<div data-engine="sglang" data-sku="gb300">

- A Kubernetes cluster with the Dynamo platform installed and GB300 GPUs available. Use 4 GPUs for aggregated serving and 8 GPUs for either the disaggregated or KV-aware routed target.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the SGLang runtime image referenced by the deployment manifest.
- For disaggregated serving, ComputeDomain support with Dynamic Resource Allocation (DRA) enabled.

</div>

<div data-engine="vllm" data-sku="gb200">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB200 GPUs available.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the vLLM runtime image referenced by the deployment manifest.

</div>

<div data-engine="vllm" data-sku="gb300">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB300 GPUs available.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the vLLM runtime image referenced by the deployment manifest.

</div>

<div data-engine="trtllm" data-sku="gb200">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB200 GPUs available.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the TensorRT-LLM runtime image referenced by the deployment manifest.

</div>

<div data-engine="trtllm" data-sku="gb300">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB300 GPUs available.
- Access to the Muse-Spark-Open-816B-A42B checkpoint and tokenizer. The public model identifier is `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`.
- Access to the TensorRT-LLM runtime image referenced by the deployment manifest.

</div>

## Deploy

### 1. Create the namespace and Hugging Face secret

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

### 2. Create shared storage

<Warning>
Edit **`recipes/muse-spark-open-816b-a42b/model-cache/model-cache.yaml`** and set **`storageClassName`** to a ReadWriteMany storage class available on the target cluster.
</Warning>

Then create the `shared-model-cache` PVC:

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/model-cache/model-cache.yaml -n ${NAMESPACE}
```

### 3. Download the model

The model-download manifest is preconfigured with the public model identifier. Apply it to populate the PVC and wait for the download to finish:

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download \
  -n ${NAMESPACE} --timeout=10800s
```

All worker manifests mount the `shared-model-cache` PVC at `/model`.

### 4. Deploy the selected target

<div data-engine="sglang">

<div data-sku="gb200" data-variant="agg">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/agg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb200" data-variant="disagg">

This base omits provider-specific networking. Configure [RDMA](../../kubernetes/installation/rdma-setup/overview.md) for long-context requests using a [cluster Kustomization](https://github.com/ai-dynamo/dynamo/blob/main/recipes/templates/kustomize/README.md).

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/disagg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb200" data-variant="kv-router">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/agg-gb200-agentic-kv-router/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="agg">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/agg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="disagg">

This base omits provider-specific networking. Configure [RDMA](../../kubernetes/installation/rdma-setup/overview.md) for long-context requests using a [cluster Kustomization](https://github.com/ai-dynamo/dynamo/blob/main/recipes/templates/kustomize/README.md).

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/disagg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="kv-router">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/sglang/agg-gb300-agentic-kv-router/deploy.yaml -n ${NAMESPACE}
```

</div>

</div>

<div data-engine="vllm">

The vLLM recipes use aggregated TP4/EP4 serving on 4 GPUs and mount the same `shared-model-cache` PVC.

<div data-sku="gb200">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/vllm/agg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/vllm/agg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

</div>

<div data-engine="trtllm">

The experimental TensorRT-LLM recipes use aggregated TP4/EP4 serving on 4 GB200 or GB300 GPUs and mount the same `shared-model-cache` PVC. Apply the manifest for your GPU.

<div data-sku="gb200">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/trtllm/agg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-sku="gb300">

```bash
kubectl apply -f recipes/muse-spark-open-816b-a42b/trtllm/agg-gb300-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

</div>

## Smoke Test

Wait for the selected deployment to become ready, then forward its frontend Service:

<div data-engine="sglang">

<div data-sku="gb200" data-variant="agg">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-aggregate-gb200 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-aggregate-gb200-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb200" data-variant="disagg">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-disagg-gb200 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-disagg-gb200-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb200" data-variant="kv-router">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-aggregate-gb200-kv \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-aggregate-gb200-kv-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="agg">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-aggregate-gb300 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-aggregate-gb300-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="disagg">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-disagg-gb300 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-disagg-gb300-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb300" data-variant="kv-router">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-sglang-aggregate-gb300-kv \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-sglang-aggregate-gb300-kv-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

</div>

<div data-engine="vllm">

<div data-sku="gb200">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-vllm-aggregate-gb200 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-vllm-aggregate-gb200-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb300">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-vllm-aggregate-gb300 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-vllm-aggregate-gb300-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

</div>

<div data-engine="trtllm">

<div data-sku="gb200">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-trtllm-aggregate-gb200 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-trtllm-aggregate-gb200-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-sku="gb300">

```bash
kubectl wait --for=condition=Ready dynamographdeployment/muse-spark-trtllm-aggregate-gb300 \
  -n ${NAMESPACE} --timeout=7200s
kubectl port-forward svc/muse-spark-trtllm-aggregate-gb300-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

</div>

With either frontend forwarded, send a deterministic chat completion using the served model name `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"meta-models/Muse-Spark-Open-816B-A42B-NVFP4","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":64}'
```

## Benchmark

Use [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/perf/perf.yaml) to run AIPerf against the selected deployment. The agentic trace has approximately 64K input tokens, 400 output tokens, and 90% KV-cache reuse.

### Select the workload

The selector provides the endpoint and workload settings for your recipe:

<div data-engine="sglang">

<div data-sku="gb200" data-variant="agg">

```bash
export ENDPOINT=muse-spark-sglang-aggregate-gb200-frontend:8000
export CONCURRENCY=12
export REQUEST_COUNT=3541
```

</div>

<div data-sku="gb200" data-variant="kv-router">

```bash
export ENDPOINT=muse-spark-sglang-aggregate-gb200-kv-frontend:8000
export CONCURRENCY=22
export REQUEST_COUNT=3541
```

</div>

<div data-sku="gb200" data-variant="disagg">

```bash
export ENDPOINT=muse-spark-sglang-disagg-gb200-frontend:8000
export CONCURRENCY=16
export REQUEST_COUNT=3541
```

</div>

<div data-sku="gb300" data-variant="agg">

```bash
export ENDPOINT=muse-spark-sglang-aggregate-gb300-frontend:8000
export CONCURRENCY=16
export REQUEST_COUNT=3541
```

</div>

<div data-sku="gb300" data-variant="kv-router">

```bash
export ENDPOINT=muse-spark-sglang-aggregate-gb300-kv-frontend:8000
export CONCURRENCY=20
export REQUEST_COUNT=3541
```

</div>

<div data-sku="gb300" data-variant="disagg">

```bash
export ENDPOINT=muse-spark-sglang-disagg-gb300-frontend:8000
export CONCURRENCY=16
export REQUEST_COUNT=3541
```

</div>

</div>

<div data-engine="vllm">

<div data-sku="gb200">

```bash
export ENDPOINT=muse-spark-vllm-aggregate-gb200-frontend:8000
export CONCURRENCY=24
export REQUEST_COUNT=500
```

</div>

<div data-sku="gb300">

```bash
export ENDPOINT=muse-spark-vllm-aggregate-gb300-frontend:8000
export CONCURRENCY=24
export REQUEST_COUNT=500
```

</div>

</div>

<div data-engine="trtllm">

<div data-sku="gb200">

```bash
export ENDPOINT=muse-spark-trtllm-aggregate-gb200-frontend:8000
export CONCURRENCY=24
export REQUEST_COUNT=500
```

</div>

<div data-sku="gb300">

```bash
export ENDPOINT=muse-spark-trtllm-aggregate-gb300-frontend:8000
export CONCURRENCY=24
export REQUEST_COUNT=500
```

</div>

</div>

<div data-engine="vllm trtllm">

These settings replay the first 500 requests for functional validation, rather than the full trace.

</div>

### Dataset

Place the [benchmark trace](https://media.githubusercontent.com/media/ai-dynamo/dynamo/main/recipes/kimi-k2.6/perf/traces/64k_400_90kv_agent_new_noschedule_short_15perc.jsonl) at `/model-cache/agentic.jsonl` on the `shared-model-cache` PVC. The Job replays it with `--custom-dataset-type mooncake_trace`.

### Run the benchmark

Restart workers between independent runs to reset KV-cache and routing state. After the previous Job completes, run from the repository root with the selected settings:

```bash
kubectl delete job muse-spark-aiperf -n ${NAMESPACE} --ignore-not-found
kubectl set env --local -f recipes/muse-spark-open-816b-a42b/perf/perf.yaml \
  ENDPOINT="${ENDPOINT}" CONCURRENCY="${CONCURRENCY}" \
  REQUEST_COUNT="${REQUEST_COUNT}" -o yaml | \
  kubectl create -f - -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/muse-spark-aiperf \
  -n ${NAMESPACE} --timeout=10800s
```

The Job runs five sequential warmup requests before replaying the trace.

## Expected Performance

Disaggregated results were measured with cluster-specific RDMA networking.

<div data-engine="sglang">

The table lists the measured operating point for each of the six recipe targets.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Recipe</th><th>GPUs</th><th>Workload</th><th>Concurrency</th><th>Requests</th><th>Passing</th><th>Failing</th><th>Output throughput (tok/s)</th><th>Output throughput/GPU (tok/s/GPU)</th><th>Output throughput/user p50 (tok/s/user)</th><th>TTFT p50 (ms)</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="agg"><td>sglang/agg-gb200-agentic</td><td>4</td><td>trace</td><td>12</td><td>3,541</td><td>3,526</td><td>15</td><td>514.30</td><td>128.58</td><td>57.43</td><td>221.70</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="disagg"><td>sglang/disagg-gb200-agentic</td><td>8</td><td>trace</td><td>16</td><td>3,541</td><td>3,526</td><td>15</td><td>768.50</td><td>96.07</td><td>58.77</td><td>1,335.20</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="kv-router"><td>sglang/agg-gb200-agentic-kv-router</td><td>8</td><td>trace</td><td>22</td><td>3,541</td><td>3,526</td><td>15</td><td>934.20</td><td>116.78</td><td>54.51</td><td>239.20</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="agg"><td>sglang/agg-gb300-agentic</td><td>4</td><td>trace</td><td>16</td><td>3,541</td><td>3,526</td><td>15</td><td>635.80</td><td>158.95</td><td>50.97</td><td>177.90</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="disagg"><td>sglang/disagg-gb300-agentic</td><td>8</td><td>trace</td><td>16</td><td>3,541</td><td>3,526</td><td>15</td><td>835.80</td><td>104.48</td><td>62.12</td><td>289.40</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="kv-router"><td>sglang/agg-gb300-agentic-kv-router</td><td>8</td><td>trace</td><td>20</td><td>3,541</td><td>3,526</td><td>15</td><td>1,054.70</td><td>131.84</td><td>66.70</td><td>158.40</td></tr>
</tbody>
</table>
</div>

</div>

<div data-engine="vllm">

These results cover the first 500 trace requests and establish functional operation. They are not full-trace performance runs and are not directly comparable to the SGLang results.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Recipe</th><th>GPUs</th><th>Workload</th><th>Concurrency</th><th>Requests</th><th>Passing</th><th>Failing</th><th>Output throughput (tok/s)</th><th>Output throughput/GPU (tok/s/GPU)</th><th>Output throughput/user p50 (tok/s/user)</th><th>TTFT p50 (ms)</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row" data-highlight-sku="gb200"><td>vllm/agg-gb200-agentic</td><td>4</td><td>trace (first 500; functional)</td><td>24</td><td>500</td><td>499</td><td>1</td><td>475.20</td><td>118.81</td><td>27.23</td><td>2,339.90</td></tr>
<tr className="dynamo-performance-row" data-highlight-sku="gb300"><td>vllm/agg-gb300-agentic</td><td>4</td><td>trace (first 500; functional)</td><td>24</td><td>500</td><td>499</td><td>1</td><td>564.80</td><td>141.20</td><td>28.84</td><td>292.90</td></tr>
</tbody>
</table>
</div>

</div>

<div data-engine="trtllm">

These results cover the first 500 trace requests and are functional validation, not a full-trace benchmark.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Recipe</th><th>GPUs</th><th>Workload</th><th>Concurrency</th><th>Requests</th><th>Passing</th><th>Failing</th><th>Output throughput (tok/s)</th><th>Output throughput/GPU (tok/s/GPU)</th><th>Output throughput/user p50 (tok/s/user)</th><th>TTFT p50 (ms)</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row" data-highlight-sku="gb200"><td>trtllm/agg-gb200-agentic</td><td>4</td><td>trace (first 500; functional)</td><td>24</td><td>500</td><td>499</td><td>1</td><td>384.58</td><td>96.15</td><td>30.82</td><td>11,533.66</td></tr>
<tr className="dynamo-performance-row" data-highlight-sku="gb300"><td>trtllm/agg-gb300-agentic</td><td>4</td><td>trace (first 500; functional)</td><td>24</td><td>500</td><td>499</td><td>1</td><td>583.23</td><td>145.81</td><td>31.37</td><td>600.34</td></tr>
</tbody>
</table>
</div>

</div>

## Compare All Targets

<div data-engine="sglang">

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Target</th><th>Framework</th><th>Topology</th><th>Parallelism</th><th>KV cache</th><th>KV transfer</th><th>GPU count</th><th>Context length</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="agg"><td>GB200 aggregated</td><td>SGLang</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="disagg"><td>GB200 disaggregated</td><td>SGLang</td><td>1P1D disaggregated</td><td>TP4/EP4 per worker</td><td>FP8</td><td>NIXL</td><td>8</td><td>1M</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb200" data-muse-variant="kv-router"><td>GB200 KV-aware routing</td><td>SGLang</td><td>Aggregated, 2 replicas</td><td>2 × TP4/EP4</td><td>FP8</td><td>N/A</td><td>8</td><td>1M</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="agg"><td>GB300 aggregated</td><td>SGLang</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="disagg"><td>GB300 disaggregated</td><td>SGLang</td><td>1P1D disaggregated</td><td>TP4/EP4 per worker</td><td>FP8</td><td>NIXL</td><td>8</td><td>1M</td></tr>
<tr className="dynamo-performance-row dynamo-muse-spark-performance-row" data-muse-sku="gb300" data-muse-variant="kv-router"><td>GB300 KV-aware routing</td><td>SGLang</td><td>Aggregated, 2 replicas</td><td>2 × TP4/EP4</td><td>FP8</td><td>N/A</td><td>8</td><td>1M</td></tr>
</tbody>
</table>
</div>

</div>

<div data-engine="vllm">

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Target</th><th>Framework</th><th>Topology</th><th>Parallelism</th><th>KV cache</th><th>KV transfer</th><th>GPU count</th><th>Context length</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row" data-highlight-sku="gb200"><td>GB200 aggregated</td><td>vLLM</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
<tr className="dynamo-performance-row" data-highlight-sku="gb300"><td>GB300 aggregated</td><td>vLLM</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
</tbody>
</table>
</div>

</div>

<div data-engine="trtllm">

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table">
<thead><tr><th>Target</th><th>Framework</th><th>Topology</th><th>Parallelism</th><th>KV cache</th><th>KV transfer</th><th>GPU count</th><th>Context length</th></tr></thead>
<tbody>
<tr className="dynamo-performance-row" data-highlight-sku="gb200"><td>GB200 aggregated</td><td>TensorRT-LLM</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
<tr className="dynamo-performance-row" data-highlight-sku="gb300"><td>GB300 aggregated</td><td>TensorRT-LLM</td><td>Aggregated</td><td>TP4/EP4</td><td>FP8</td><td>N/A</td><td>4</td><td>1M</td></tr>
</tbody>
</table>
</div>

</div>

## Source

<div data-engine="sglang">

- Deployment assets: [Muse-Spark-Open-816B-A42B SGLang manifests](https://github.com/ai-dynamo/dynamo/tree/main/recipes/muse-spark-open-816b-a42b/sglang)
- Setup assets: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-download.yaml)
- Benchmark manifest: [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/perf/perf.yaml)

</div>

<div data-engine="vllm">

- Deployment assets: [Muse-Spark-Open-816B-A42B vLLM manifests](https://github.com/ai-dynamo/dynamo/tree/main/recipes/muse-spark-open-816b-a42b/vllm)
- Setup assets: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-download.yaml)
- Benchmark manifest: [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/perf/perf.yaml)

</div>

<div data-engine="trtllm">

- Deployment assets: [Muse-Spark-Open-816B-A42B TensorRT-LLM GB200 manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/trtllm/agg-gb200-agentic/deploy.yaml) and [GB300 manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/trtllm/agg-gb300-agentic/deploy.yaml)
- Setup assets: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/model-cache/model-download.yaml)
- Benchmark manifest: [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/muse-spark-open-816b-a42b/perf/perf.yaml)

</div>
