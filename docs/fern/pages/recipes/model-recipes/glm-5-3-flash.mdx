---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "GLM-5.3-Flash"
subtitle: "Serve GLM-5.3-Flash with Dynamo and SGLang on B200 and GB200, or vLLM on GB200 and H200."
---

This recipe references an upstream third-party vLLM container image. NVIDIA does not publish or distribute this image. Users should review the upstream image’s open-source license and codec terms before use or redistribution.

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Deploy Z.AI's GLM-5.3-Flash with NVIDIA Dynamo using SGLang or vLLM. The SGLang B200 and GB200 recipes use a pinned NVFP4 checkpoint, FP8 KV cache, EAGLE ReplaySSM speculative decoding, and HiCache CPU offload. Choose single-node B200 or GB200 TP4 aggregated serving, or GB200 disaggregated serving with two TP4/EP4 prefill workers plus one TP4 decode worker, KV events, and Mooncake NVLink transfer. The vLLM targets use an FP8 checkpoint with BF16 KV cache on GB200 or H200. Pick your framework, GPU architecture, and serving topology.

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
<input type="radio" id="recipe-sku-h200" name="recipe-sku" value="h200" />
<label htmlFor="recipe-sku-h200">H200</label>
<input type="radio" id="recipe-sku-gb200" name="recipe-sku" value="gb200" defaultChecked />
<label htmlFor="recipe-sku-gb200">GB200 <span className="dynamo-target-picker-hint">Recommended</span></label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h200" data-variant="agg">
<span><b>Checkpoint</b> zai-org/GLM-5.3-Flash</span>
<span><b>Precision</b> FP8-quantized checkpoint + BF16 KV cache</span>
<span><b>GPUs</b> 8x H200, one worker</span>
<span><b>Parallelism</b> TP8</span>
<span><b>Speculation</b> MTP7</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">
<span><b>Checkpoint</b> zai-org/GLM-5.3-Flash</span>
<span><b>Precision</b> FP8-quantized checkpoint + BF16 KV cache</span>
<span><b>GPUs</b> 8x H200 prefill + 8x H200 decode</span>
<span><b>Parallelism</b> TP8 prefill / TP8 decode</span>
<span><b>Routing</b> KV-aware, NIXL KV transfer</span>
<span><b>Speculation</b> MTP7 on prefill and decode</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">
<span><b>Checkpoint</b> zai-org/GLM-5.3-Flash</span>
<span><b>Precision</b> FP8-quantized checkpoint + BF16 KV cache</span>
<span><b>GPUs</b> 4x GB200, one worker</span>
<span><b>Parallelism</b> TP4 over MNNVL</span>
<span><b>Routing</b> KV-aware</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">
<span><b>Checkpoint</b> zai-org/GLM-5.3-Flash</span>
<span><b>Precision</b> FP8-quantized checkpoint + BF16 KV cache</span>
<span><b>GPUs</b> 4x GB200 prefill + 4x GB200 decode</span>
<span><b>Parallelism</b> TP4 prefill / TP4 decode over MNNVL</span>
<span><b>Routing</b> KV-aware, NIXL/UCX KV transfer over TCP</span>
<span><b>Context</b> Up to 1,048,576 tokens</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="gb200" data-variant="agg">
<span><b>Checkpoint</b> RadixArk/GLM-5.3-Flash-NVFP4</span>
<span><b>Precision</b> NVFP4 weights + FP8 E4M3 KV cache</span>
<span><b>GPUs</b> 4x GB200, one TP4 worker</span>
<span><b>Routing</b> Round-robin</span>
<span><b>Speculation</b> EAGLE, 3 steps / 4 draft tokens, ReplaySSM</span>
<span><b>HiCache</b> 100 GB per TP rank; ~400 GB per worker Pod</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="gb200" data-variant="disagg">
<span><b>Checkpoint</b> RadixArk/GLM-5.3-Flash-NVFP4</span>
<span><b>Precision</b> NVFP4 weights + FP8 E4M3 KV cache</span>
<span><b>GPUs</b> 8x GB200 prefill + 4x GB200 decode</span>
<span><b>Parallelism</b> 2x prefill TP4/EP4 + 1x decode TP4</span>
<span><b>Routing</b> KV-aware routing with worker events, Mooncake NVLink transfer</span>
<span><b>Speculation</b> EAGLE ReplaySSM on decode</span>
<span><b>HiCache</b> 100 GB per prefill TP rank; ~400 GB per prefill Pod</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="b200" data-variant="agg">
<span><b>Checkpoint</b> RadixArk/GLM-5.3-Flash-NVFP4</span>
<span><b>Precision</b> NVFP4 weights + FP8 E4M3 KV cache</span>
<span><b>GPUs</b> 4x B200 on one amd64 node, one TP4 worker</span>
<span><b>Routing</b> Round-robin</span>
<span><b>Speculation</b> EAGLE, 3 steps / 4 draft tokens, ReplaySSM</span>
<span><b>HiCache</b> 100 GB per TP rank; ~400 GB per worker Pod</span>
<span><b>Overlay</b> nscale, single-node NVLink without ComputeDomain</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="b200" data-variant="disagg">
<span>For B200, select Aggregated. For SGLang disaggregated serving, select GB200.</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200">
<span>For B200, select SGLang and Aggregated. The vLLM recipes target GB200 and H200.</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="h200">
<span>SGLang recipes are available for B200 aggregated serving and GB200 aggregated/disaggregated serving. Choose vLLM for H200.</span>
</div>
</div>

## Prerequisites

<div data-recipe-framework="vllm" data-sku="gb200">

- A Kubernetes cluster with the Dynamo platform installed and GB200 GPUs available: 4x on one ARM64 node for aggregated, or 8x across two ARM64 nodes for disaggregated. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- The NVIDIA DRA driver with ComputeDomain support installed. Both GB200 targets require a ComputeDomain spanning the full NVL72 rack.
- Access to `zai-org/GLM-5.3-Flash` and the `vllm/vllm-openai:glm53-flash` image.
- A ReadWriteMany PVC named `model-cache`, populated with the checkpoint in the standard Hugging Face cache layout.

</div>

<div data-recipe-framework="vllm" data-sku="h200">

- A Kubernetes cluster with the Dynamo platform installed and H200 GPUs available: 8x on one node for aggregated, or 16x across two nodes for disaggregated. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- Eight RDMA resource claims on each GPU node for disaggregated serving. Update `rdma/ib` if your device plugin advertises a different resource name.
- Access to `zai-org/GLM-5.3-Flash` and the `vllm/vllm-openai:glm53-flash` image.
- A ReadWriteMany PVC named `model-cache`, populated with the checkpoint in the standard Hugging Face cache layout.

</div>

<div data-recipe-framework="sglang" data-sku="gb200">

- A Kubernetes cluster with the Dynamo operator supporting `nvidia.com/v1beta1`. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- GB200 arm64 nodes with four available GPUs per worker Pod: 4 GPUs for aggregated serving or 12 GPUs for 2P1D disaggregated serving.
- The NVIDIA DRA driver and `resource.nvidia.com/v1beta1` ComputeDomain CRD. Disaggregated workers must fit in one NVLink clique; the overlay uses Pod affinity on the `nvidia.com/gpu.clique` node label.
- Standalone Kustomize v5.8.1 and a checkout of this repository.
- Access to `RadixArk/GLM-5.3-Flash-NVFP4` at revision `f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`.
- A ReadWriteMany PVC named `shared-model-cache`, accessible from all worker nodes.

The scheduling component selects arm64 nodes and `nvidia.com/gpu.product=NVIDIA-GB200` for workers. It tolerates `kubernetes.io/arch=arm64:NoSchedule` and, on workers, `nvidia.com/gpu=present:NoSchedule`. Keep site-specific namespace, storage class, pull Secrets, priority class, and scheduling overrides in a private cluster overlay.

</div>

<div data-recipe-framework="sglang" data-sku="b200" data-variant="agg">

- A Kubernetes cluster with the Dynamo operator supporting `nvidia.com/v1beta1` and four available B200 GPUs on one amd64 node. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- Standalone Kustomize v5.8.1 and a checkout of this repository.
- A populated ReadWriteMany PVC named `shared-model-cache`, accessible from the worker Pod, containing `RadixArk/GLM-5.3-Flash-NVFP4` at revision `f46cf340d35a22d0d83d0c1dac8957cf2b1bcd35`.
- The `inference` priority class, or a private overlay selecting your cluster's priority class.

The `nscale` overlay selects amd64 nodes and `nvidia.com/gpu.product=NVIDIA-B200` for the worker. Both Pods tolerate `nvidia.com/gpu=true:NoSchedule`. Adapt placement and registry access to your cluster. This single-node recipe uses NVLink/NVLS with MNNVL disabled and creates no ComputeDomain or DRA channel claims.

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
Match storage, scheduling, registry access, and resource claims to your cluster. For SGLang, edit the Kustomize sources or use a private cluster overlay; regenerate manifests instead of editing the generated copies.
</Warning>

## Deploy

<div data-recipe-framework="vllm">

All four vLLM targets read the checkpoint from the `model-cache` PVC with `HF_HUB_OFFLINE=1`. Create and populate the PVC with the recipe's [storage manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/model-cache/model-cache.yaml) and [model-download job](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/model-cache/model-download.yaml):

```bash
kubectl apply -f recipes/glm-5.3-flash/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/glm-5.3-flash/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s
```

</div>

Apply the manifest or Kustomize overlay for your target. Run these commands from the repository root.

<div data-recipe-framework="vllm" data-sku="h200" data-variant="agg">

```bash
kubectl apply -f recipes/glm-5.3-flash/vllm/agg-h200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">

```bash
kubectl apply -f recipes/glm-5.3-flash/vllm/disagg-h200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">

```bash
kubectl apply -f recipes/glm-5.3-flash/vllm/agg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">

```bash
kubectl apply -f recipes/glm-5.3-flash/vllm/disagg-gb200-agentic/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="sglang" data-sku="b200 gb200">

Reuse `shared-model-cache` if it exists. For a new cache, set the storage class in the [SGLang storage manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/model-cache/model-cache.yaml) before applying it. The [download Job](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/model-cache/model-download.yaml) downloads the pinned NVFP4 revision, which differs from the vLLM checkpoint:

```bash
kubectl apply -f recipes/glm-5.3-flash/sglang/model-cache/model-cache.yaml -n "$NAMESPACE"
kubectl apply -f recipes/glm-5.3-flash/sglang/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl wait --for=condition=Complete job/glm53-flash-nvfp4-model-download \
  -n "$NAMESPACE" --timeout=7200s
```

All SGLang workers resolve `--model-path RadixArk/GLM-5.3-Flash-NVFP4` and the pinned `--revision` from the shared cache with offline mode enabled. All SGLang frontends obtain model metadata from workers without mounting the cache.

</div>

<div data-recipe-framework="sglang" data-sku="gb200">

<div data-variant="agg">

```bash
export RECIPE=recipes/glm-5.3-flash/sglang/agg-gb200-agentic
```

</div>

<div data-variant="disagg">

```bash
export RECIPE=recipes/glm-5.3-flash/sglang/disagg-gb200-agentic
```

The [base deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/disagg-gb200-agentic/kustomize/base/deploy.yaml) runs two prefill workers and one decode worker (2P1D), using 12 GB200 GPUs in one NVLink clique.

</div>

Render and apply the selected GB200 overlay:

```bash
kustomize build "$RECIPE/kustomize/overlays/gb200" > /tmp/glm53-sglang.yaml
kubectl apply --dry-run=server -f /tmp/glm53-sglang.yaml -n "$NAMESPACE"
kubectl apply -f /tmp/glm53-sglang.yaml -n "$NAMESPACE"
kubectl get dynamographdeployment,pods -n "$NAMESPACE"
```

The overlay creates a ComputeDomain and connects its channel to the worker claims. Alternatively, apply the generated `"$RECIPE/deploy-gb200.yaml"` manifest. The [SGLang README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/README.md#sglang) documents the base/component layout and regeneration commands.

</div>

<div data-recipe-framework="sglang" data-sku="b200" data-variant="agg">

Render and apply the B200 `nscale` overlay:

```bash
export RECIPE=recipes/glm-5.3-flash/sglang/agg-b200-agentic
kustomize build "$RECIPE/kustomize/overlays/nscale" > /tmp/glm53-sglang-b200.yaml
kubectl apply --dry-run=server -f /tmp/glm53-sglang-b200.yaml -n "$NAMESPACE"
kubectl apply -f /tmp/glm53-sglang-b200.yaml -n "$NAMESPACE"
kubectl get dynamographdeployment,pods -n "$NAMESPACE"
```

Alternatively, apply the generated `"$RECIPE/deploy-nscale.yaml"` manifest. See the [SGLang README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/README.md#sglang) for overlay settings and regeneration commands.

</div>

First worker launch loads the model and captures CUDA graphs and can take tens of minutes.

## Smoke Test

Send a test request to verify the deployment serves traffic. First forward the frontend port for your target:

<div data-recipe-framework="vllm" data-sku="h200" data-variant="agg">

```bash
kubectl port-forward svc/glm53-flash-agg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">

```bash
kubectl port-forward svc/glm53-flash-disagg-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">

```bash
kubectl port-forward svc/glm53-flash-agg-gb200-agentic-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">

```bash
kubectl port-forward svc/glm53-flash-disagg-gb200-frontend 8000:8000 -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm">

All vLLM targets serve under the same model name, `zai-org/GLM-5.3-Flash`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"Reply with exactly: READY"}],"temperature":0,"max_tokens":64}'
```

GLM-5.3-Flash reasons before answering. The deployments support streaming chat, completions, tool calls, one image per prompt, and schema-constrained output through the `glm45` reasoning and `glm47` tool-call parsers.

</div>

<div data-recipe-framework="sglang" data-sku="gb200">

Wait for the frontend and all workers to become ready, then forward the frontend Service for the selected topology:

<div data-variant="agg">

```bash
kubectl port-forward svc/glm53-flash-sglang-agg-gb200-frontend 8000:8000 -n "$NAMESPACE"
```

</div>

<div data-variant="disagg">

```bash
kubectl port-forward svc/glm53-flash-sglang-disagg-gb200-frontend 8000:8000 -n "$NAMESPACE"
```

</div>

</div>

<div data-recipe-framework="sglang" data-sku="b200" data-variant="agg">

Wait for both Pods to become ready, then forward the B200 frontend:

```bash
kubectl port-forward svc/glm53-flash-sglang-agg-b200-frontend 8000:8000 -n "$NAMESPACE"
```

</div>

<div data-recipe-framework="sglang" data-sku="b200 gb200">

In another terminal, verify registration and send a request using the SGLang served model name:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"RadixArk/GLM-5.3-Flash-NVFP4","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":128,"temperature":0}'
```

The SGLang workers use the `glm45` reasoning parser and `glm47` tool-call parser. All three recipes enable `--enable-multimodal` and `--mm-enable-dp-encoder`. Aggregated workers enable `--frontend-decoding`, so the Dynamo frontend decodes images and videos before passing them to the backend. Disaggregated workers omit `--frontend-decoding` on both prefill and decode workers and support text and image input only.

Video input is unavailable in the SGLang disaggregated recipe because frontend decoding is unsupported on this path and the runtime image omits SGLang's video decoder dependencies.

</div>

## Benchmark

<div data-recipe-framework="sglang" data-sku="b200 gb200">

Use the [AIPerf benchmark guide](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/perf/README.md) and [benchmark manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/perf/perf.yaml) for agentic text trace replay.

### Synthetic Acceptance Results

These measurements describe the source configurations with simulated acceptance length `2.5657`, method `match-expected`, and token mode `real-draft-token`. The default deployment overlays use real speculative verification. To configure synthetic acceptance, follow the [compose, dry-run, and apply procedure](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/README.md#synthetic-acceptance) with the optional component for [B200 aggregated](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/agg-b200-agentic/kustomize/components/synthetic-acceptance), [GB200 aggregated](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/agg-gb200-agentic/kustomize/components/synthetic-acceptance), or [GB200 disaggregated](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/disagg-gb200-agentic/kustomize/components/synthetic-acceptance). Wait for replacement workers to become ready before measurement.

Set the `CONCURRENCY` environment variable in the benchmark Job to **56 for either aggregated target** or **128 for GB200 disaggregated**, as listed below. The Job's default is **64** and does not match these rows. Select the matching endpoint and frontend pod affinity using the benchmark guide.

| Workload | Framework | Recipe | SKU | `CONCURRENCY` | System output tok/s/GPU | User output tok/s (P50) | TTFT P50 (ms) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Agentic (15% subset) | SGLang | Aggregated (1 worker) | B200 | 56 | 579.1 | 50.7 | 189 |
| Agentic (15% subset) | SGLang | Aggregated (1 worker) | GB200 | 56 | 590.34 | 51.05 | 316 |
| Agentic (15% subset) | SGLang | Disaggregated (2P1D) | GB200 | 128 | 619.7 | 80.1 | 4,680 |

</div>

## Compare All Targets

### vLLM

All four vLLM targets serve the same FP8-quantized GLM-5.3-Flash checkpoint on vLLM with BF16 KV cache. The GB200 aggregated target and both H200 targets enable prefix caching. The H200 targets add MTP7 speculative decoding, DeepGEMM, and FLASHMLA_SPARSE attention. The GB200 targets use MNNVL for TP4 collectives.

| | GB200 Aggregated | GB200 Disaggregated | H200 Aggregated | H200 Disaggregated |
|---|---|---|---|---|
| **Topology** | DynamoGraphDeployment | DynamoGraphDeployment, 1P1D | DynamoGraphDeployment | DynamoGraphDeployment, 1P1D |
| **GPUs** | 4x GB200 | 4x GB200 prefill + 4x GB200 decode | 8x H200 | 8x H200 prefill + 8x H200 decode |
| **Parallelism** | TP4 | TP4 prefill / TP4 decode | TP8 | TP8 prefill / TP8 decode |
| **Precision** | FP8 checkpoint + BF16 KV | FP8 checkpoint + BF16 KV | FP8 checkpoint + BF16 KV | FP8 checkpoint + BF16 KV |
| **Speculation** | None | None | MTP7 | MTP7 on prefill and decode |
| **Maximum context** | 1,048,576 | 1,048,576 | 1,048,576 | 1,048,576 |
| **Maximum sequences** | 256 | 32 prefill / 256 decode | 32 | 16 prefill / 128 decode |
| **Maximum batched tokens** | 8,192 | 16,384 prefill / 8,192 decode | 8,192 | 8,192 prefill / 2,048 decode |
| **GPU memory utilization** | 0.92 | 0.92 prefill / 0.92 decode | 0.80 | 0.80 prefill / 0.85 decode |
| **KV transfer** | N/A | NIXL/UCX over TCP | N/A | NIXL with GPU buffers and RDMA resources |
| **Routing** | KV-aware | KV-aware | Round-robin | KV-aware |

### SGLang

All three SGLang targets use the pinned NVFP4 checkpoint and FP8 E4M3 KV cache. Worker Pods request 16 CPUs and four GPUs, with a 64 GiB shared-memory volume. Aggregated workers and disaggregated prefill workers have a host-memory request and limit of 448 GiB.


| Setting | B200 Aggregated | GB200 Aggregated | GB200 Disaggregated |
| --- | --- | --- | --- |
| **Topology** | 1 worker on one amd64 node | 1 worker on arm64 | 2 prefill + 1 decode on arm64 |
| **GPUs** | 4 | 4 | 12 |
| **Parallelism** | TP4 | TP4 | Prefill TP4/EP4, decode TP4 |
| **Speculation** | EAGLE, 3 steps / 4 draft tokens, ReplaySSM | Same | Same, decode only |
| **HiCache CPU capacity** | 100 GB per TP rank; ~400 GB per worker Pod | Same | 100 GB per prefill TP rank; ~400 GB per prefill Pod; none on decode |
| **Host memory request / limit per worker Pod** | 448 GiB / 448 GiB | 448 GiB / 448 GiB | Prefill: 448 GiB / 448 GiB; decode: 256 GiB request, no explicit limit |
| **Maximum running requests** | 64 | 64 | 64 per prefill worker / 128 decode |
| **Maximum decode CUDA graph batch size** | 64 | 64 | 128 on decode |
| **Chunked prefill size** | 16,384 tokens | 16,384 tokens | 16,384 tokens |
| **Static memory fraction** | 0.70 | 0.70 | Prefill 0.70 / decode 0.82 |
| **Routing** | Round-robin | Round-robin | KV-aware with worker KV events |
| **KV transfer** | Not applicable | Not applicable | Mooncake over NVLink with MNNVL |
| **ComputeDomain** | None | Created by overlay | Created by overlay |
| **Overlay** | `nscale` | `gb200` | `gb200` |
| **Input modalities** | Text, images, video | Text, images, video | Text, images |

## Related Feature Benchmarks

No controlled feature benchmark is linked to this recipe. Browse the [Feature Benchmarks](../feature-benchmarks/browse-all-benchmarks.mdx) for comparable KV-routing, disaggregation, and speculative-decoding studies.

## Notes

### vLLM

- The H200 targets enable prefix caching with a match unit of 128 and use MTP with seven speculative tokens.
- The H200 disaggregated target sets the same MTP configuration, `VLLM_SSM_CONV_STATE_LAYOUT=DS`, and `VLLM_KV_CACHE_LAYOUT=HND` on prefill and decode. These values must match for the NIXL compatibility handshake and KV layout.
- The GB200 targets use MNNVL for TP4 collectives. The disaggregated target excludes `cuda_ipc` from UCX and transfers KV data over `cuda_copy` and TCP because the GLM image's UCX build lacks MNNVL IPC support.
- To enable MNNVL IPC for GB200 KV transfer, rebuild on the vLLM runtime base and set `UCX_TLS=cuda_copy,cuda_ipc,tcp` with `UCX_CUDA_IPC_ENABLE_MNNVL=y`.
- A `sitecustomize.py` overlay disables cuDNN to avoid a convolution crash during multimodal initialization with the GLM-specific vLLM image.
- `n>1` requests are not supported by the disaggregated targets.

### SGLang

- The pinned SGLang runtime image includes the cuDNN workaround for the vision encoder downsampling operation.
- All worker bases set `DYN_HEALTH_CHECK_REQUEST_TIMEOUT=300` and an explicit `/live` liveness probe on port `system`, with period 5 seconds, timeout 4 seconds, and failure threshold 60. GB200 components provide placement and ComputeDomain wiring; startup and readiness probes use operator defaults.
- GB200 disaggregated routing uses worker KV events. Both prefill and decode workers set `--kv-events-config` to a ZMQ publisher on `tcp://*:5557` with topic `kv-events`. The router block size and SGLang page size are both 64.
- All three SGLang default overlays omit the source configurations' `SGLANG_SIMULATE_ACC_*` variables. Add the optional component only when reproducing synthetic acceptance measurements.

## Source

- Source README: [recipes/glm-5.3-flash/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/README.md)
- vLLM aggregated GB200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/vllm/agg-gb200-agentic/deploy.yaml)
- vLLM disaggregated GB200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/vllm/disagg-gb200-agentic/deploy.yaml)
- vLLM aggregated H200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/vllm/agg-h200-agentic/deploy.yaml)
- vLLM disaggregated H200: [deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/vllm/disagg-h200-agentic/deploy.yaml)
- SGLang guide: [README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/README.md#sglang)
- SGLang aggregated GB200: [Kustomize overlay](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/agg-gb200-agentic/kustomize/overlays/gb200), [generated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/agg-gb200-agentic/deploy-gb200.yaml)
- SGLang disaggregated GB200: [Kustomize overlay](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/disagg-gb200-agentic/kustomize/overlays/gb200), [generated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/disagg-gb200-agentic/deploy-gb200.yaml)
- SGLang aggregated B200: [Kustomize overlay](https://github.com/ai-dynamo/dynamo/tree/main/recipes/glm-5.3-flash/sglang/agg-b200-agentic/kustomize/overlays/nscale), [generated manifest](https://github.com/ai-dynamo/dynamo/blob/main/recipes/glm-5.3-flash/sglang/agg-b200-agentic/deploy-nscale.yaml)
