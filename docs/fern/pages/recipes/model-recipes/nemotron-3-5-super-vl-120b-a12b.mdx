---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Nemotron 3.5 Super VL"
subtitle: "Serve Nemotron 3.5 Super VL with Dynamo on VR200, GB300, B300, GB200, B200, H200 or H100."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

<style>{`
/* Highlight the selected framework, GPU, and topology in both comparison tables. */
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-h100:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-h100-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-h100:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-h100-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-h200:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-h200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-h200:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-h200-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b200-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200-certified:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b200-certified-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200-certified:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b200-certified-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb200:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-gb200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb200:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-gb200-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b300:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b300-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-gb300-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b300:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-b300-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-gb300:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-gb300-disagg"],
body:has(#recipe-framework-trtllm:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="trtllm-b200-agg"],
body:has(#recipe-framework-trtllm:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="trtllm-b200-disagg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-vr200:checked):has(#recipe-variant-agg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-vr200-agg"],
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-vr200:checked):has(#recipe-variant-disagg:checked) .dynamo-super-vl-comparison tr[data-highlight-target="vllm-vr200-disagg"] {
  background: color-mix(in srgb, var(--nv-color-green) 18%, transparent);
  font-weight: 600;
}

/* The certified B200 targets: show only their blocks, and hide the open-source model-cache steps. */
body:has(#recipe-sku-b200-certified:checked) [data-sku]:not([data-sku~="b200-certified"]),
body:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200-certified:checked) .dynamo-super-vl-open-source,
body:not(:has(#recipe-framework-vllm:checked):has(#recipe-sku-b200-certified:checked)) .dynamo-super-vl-certified {
  display: none;
}
`}</style>

Each target below is a Dynamo + vLLM or TensorRT-LLM deployment of NVIDIA Nemotron 3.5 Super VL, a multimodal MoE model serving up to 1M-token context. Pick your framework, GPU architecture, and serving topology. Every command on this page updates to match.

The **B200 Certified** targets use only [Dynamo Enterprise Support](../../enterprise/overview.mdx) images pinned to release 1.7.0, and NIM Factory certified both of them. Commercial support covers the enterprise images, not the recipe or configurations derived from it; see [Scope Boundaries](../../enterprise/scope-boundaries.mdx).

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>

<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Framework</span>

<input type="radio" id="recipe-framework-vllm" name="recipe-framework" value="vllm" defaultChecked />
<label htmlFor="recipe-framework-vllm">vLLM</label>

<input type="radio" id="recipe-framework-trtllm" name="recipe-framework" value="trtllm" />
<label htmlFor="recipe-framework-trtllm">TensorRT-LLM</label>

</div>

<div className="dynamo-target-picker-row" data-recipe-framework="vllm">
<span className="dynamo-target-picker-dim">GPU</span>

<input type="radio" id="recipe-sku-h100" name="recipe-sku" value="h100" />
<label htmlFor="recipe-sku-h100">H100</label>

<input type="radio" id="recipe-sku-h200" name="recipe-sku" value="h200" />
<label htmlFor="recipe-sku-h200">H200</label>

<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" />
<label htmlFor="recipe-sku-b200">B200</label>

<input type="radio" id="recipe-sku-b200-certified" name="recipe-sku" value="b200-certified" />
<label htmlFor="recipe-sku-b200-certified">B200 <span className="dynamo-target-picker-hint">Certified</span></label>

<input type="radio" id="recipe-sku-gb200" name="recipe-sku" value="gb200" defaultChecked />
<label htmlFor="recipe-sku-gb200">GB200</label>

<input type="radio" id="recipe-sku-b300" name="recipe-sku" value="b300" />
<label htmlFor="recipe-sku-b300">B300</label>

<input type="radio" id="recipe-sku-gb300" name="recipe-sku" value="gb300" />
<label htmlFor="recipe-sku-gb300">GB300</label>

<input type="radio" id="recipe-sku-vr200" name="recipe-sku" value="vr200" />
<label htmlFor="recipe-sku-vr200">VR200</label>

</div>

<div className="dynamo-target-picker-row" data-recipe-framework="trtllm">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-trtllm-sku-b200" name="recipe-trtllm-sku" value="b200" defaultChecked />
<label htmlFor="recipe-trtllm-sku-b200">B200</label>
</div>

<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>

<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>

<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>

</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h100" data-variant="agg">
<span><b>GPUs</b> 4x H100 (2 workers)</span>
<span><b>Parallelism</b> TP2 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h100" data-variant="disagg">
<span><b>GPUs</b> 4x H100 (1P1D)</span>
<span><b>Parallelism</b> TP2 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h200" data-variant="agg">
<span><b>GPUs</b> 2x H200 (2 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">
<span><b>GPUs</b> 2x H200 (1P1D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200" data-variant="agg">
<span><b>GPUs</b> 4x B200 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200" data-variant="disagg">
<span><b>GPUs</b> 5x B200 (3P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200-certified" data-variant="agg">
<span><b>GPUs</b> 4x B200 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
<span><b>Images</b> vllm-runtime-enterprise:1.7.0, dynamo-frontend-enterprise:1.7.0</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200-certified" data-variant="disagg">
<span><b>GPUs</b> 5x B200 (3P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
<span><b>Images</b> vllm-runtime-enterprise:1.7.0, dynamo-frontend-enterprise:1.7.0</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">
<span><b>GPUs</b> 4x GB200 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">
<span><b>GPUs</b> 5x GB200 (3P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP3</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b300" data-variant="agg">
<span><b>GPUs</b> 4x B300 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">
<span><b>GPUs</b> 4x GB300 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b300" data-variant="disagg">
<span><b>GPUs</b> 5x B300 (3P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP3</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">
<span><b>GPUs</b> 5x GB300 (3P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP3</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">
<span><b>GPUs</b> 11x VR200 (11 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">
<span><b>GPUs</b> 11x VR200 (7P4D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP5</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="trtllm" data-variant="agg">
<span><b>GPUs</b> 4x B200 (4 workers)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP3</span>
</div>

<div className="dynamo-target-picker-summary" data-recipe-framework="trtllm" data-variant="disagg">
<span><b>GPUs</b> 4x B200 (2P2D)</span>
<span><b>Parallelism</b> TP1 per worker</span>
<span><b>Speculative decoding</b> MTP3</span>
</div>

</div>

<div className="dynamo-super-vl-deployment">

## Prerequisites

<div data-recipe-framework="vllm" data-sku="h100" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 H100 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="h100" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 4 H100 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 2 H200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 2 H200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 B200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 5 B200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="b200-certified" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform 1.7.0 installed from the enterprise Helm chart and 4 B200 GPUs available. See [Install](../../enterprise/install.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.
- Room for 16 CPUs and 16 GiB of memory for the frontend pod, and 16 CPUs, 328 GiB of memory, and 128 GiB of ephemeral storage per worker pod.

</div>

<div data-recipe-framework="vllm" data-sku="b200-certified" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform 1.7.0 installed from the enterprise Helm chart and 5 B200 GPUs available. See [Install](../../enterprise/install.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.
- Room for 16 CPUs and 16 GiB of memory for the frontend pod, and 16 CPUs, 328 GiB of memory, and 128 GiB of ephemeral storage per worker pod.
- An RDMA device plugin that exposes `rdma/shared_hca`.

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 5 GB200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- The NVIDIA DRA driver with ComputeDomain support installed (required for multi-node NVLink).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="b300" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 B300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="b300" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 5 B300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 5 GB300 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- The NVIDIA DRA driver with ComputeDomain support installed (required for multi-node NVLink).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 11 VR200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 11 VR200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- The NVIDIA DRA driver with ComputeDomain support installed (required for multi-node NVLink).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="trtllm" data-variant="agg">

- A Kubernetes cluster with the Dynamo platform installed and 4 B200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

<div data-recipe-framework="trtllm" data-variant="disagg">

- A Kubernetes cluster with the Dynamo platform installed and 4 B200 GPUs available. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- A Hugging Face token with access to the checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4`.

</div>

Create the namespace and token secret:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}
```

<div className="dynamo-super-vl-open-source">

<Warning>
The recipes require the `shared-model-cache` PVC. Edit `storageClassName` in `model-cache/model-cache.yaml` to a ReadWriteMany storage class on your cluster (`kubectl get storageclass`) before applying it. Review namespace, image tags, and resource claims in the manifests as well.
</Warning>

</div>

## Deploy

<div className="dynamo-super-vl-open-source">

Create the `shared-model-cache` PVC and download the checkpoint:

```bash
# Edit storageClassName in model-cache/model-cache.yaml first.
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=14400s
```

The workers mount the PVC at `/shared-model-cache` and resolve the checkpoint from the local Hugging Face cache with `HF_HUB_OFFLINE=1`.

</div>

<div className="dynamo-super-vl-certified">

Each worker downloads the checkpoint into its own `emptyDir` volume at startup, so no model-cache PVC is needed. The startup probe waits up to 4 hours for the download.

</div>

Deploy the selected DynamoGraphDeployment (DGD):

<div data-recipe-framework="vllm" data-sku="h100" data-variant="agg">

```bash
export DGD=super-3-5-vllm-h100-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-h100-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="h100" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-h100-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-h100-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="agg">

```bash
export DGD=super-3-5-vllm-h200-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-h200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="h200" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-h200-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-h200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="agg">

```bash
export DGD=super-3-5-vllm-b200-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b200" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-b200-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b200-certified" data-variant="agg">

```bash
export DGD=super-3-5-vllm-b200-agg-swebenchpro
kubectl apply -f recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b200-certified" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-b200-disagg-swebenchpro
kubectl apply -f recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="agg">

```bash
export DGD=super-3-5-vllm-gb200-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-gb200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb200" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-gb200-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-gb200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b300" data-variant="agg">

```bash
export DGD=super-3-5-vllm-b300-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b300-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="agg">

```bash
export DGD=super-3-5-vllm-gb300-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-gb300-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="b300" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-b300-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b300-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-gb300-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-gb300-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="agg">

```bash
export DGD=super-3-5-vllm-vr200-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-vr200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="vllm" data-sku="vr200" data-variant="disagg">

```bash
export DGD=super-3-5-vllm-vr200-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-vr200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="trtllm" data-variant="agg">

```bash
export DGD=super-3-5-trtllm-b200-agg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/trtllm/agg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

<div data-recipe-framework="trtllm" data-variant="disagg">

```bash
export DGD=super-3-5-trtllm-b200-disagg-swebenchpro
kubectl apply -f recipes/nemotron-3.5-super-vl-120b-a12b/trtllm/disagg-b200-swebenchpro/deploy.yaml -n ${NAMESPACE}
```

</div>

The published `deploy.yaml` recipes are generic base recipes agnostic to cluster-specific settings. Adapt to your cluster scheduling, tolerations, and networking by using [kustomizations](https://github.com/ai-dynamo/dynamo/tree/main/recipes/templates/kustomize).

Wait for the deployment to become ready. Workers load a roughly 70 GiB checkpoint before they report ready, so this can take a while; watch progress with `kubectl get pods -n ${NAMESPACE} -w`:

```bash
kubectl wait --for=condition=Ready dynamographdeployment/${DGD} -n ${NAMESPACE} --timeout=14400s
```

## Smoke Test

Send a test request to verify the selected deployment serves traffic. The deploy command sets `DGD` to its resource name. Forward the frontend port:

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n ${NAMESPACE}
```

All Nemotron 3.5 Super VL recipes serve under the same model name, `nvidia/nemotron-3.5-super-vl-120b-a12b`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/nemotron-3.5-super-vl-120b-a12b","messages":[{"role":"user","content":"Write a one-sentence readiness check."}],"max_tokens":64}'
```

Nemotron 3.5 Super VL reasons before answering, and tool calling is supported — the deployment uses the `nemotron_v3` reasoning and `qwen3_coder` tool-call parsers at the frontend.

</div>

## Benchmark

See [perf/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/perf/README.md) for the full benchmark workflow: staging the trace on the PVC, running the AIPerf trace-replay Job, running a concurrency sweep, and fetching artifacts.

<div className="dynamo-super-vl-certified">

The benchmark Job reads the trace and tokenizer from the `shared-model-cache` PVC, which the certified recipes do not create. Create it from the [model-cache](https://github.com/ai-dynamo/dynamo/tree/main/recipes/nemotron-3.5-super-vl-120b-a12b/model-cache) manifests before running the Job, and set `ENDPOINT` in `perf.yaml` to `http://${DGD}-frontend:8000`.

</div>

### Optimization Targets

Recipes are optimized for the following agentic workload and target user interactivity / TTFT:

| Workload | Median ISL | Median OSL | KV cache hit rate | User output tok/s | TTFT P50 |
| -------- | ---------: | ---------: | ----------------: | ----------------: | -------: |
| SWEBench-Pro | 222,076 | 188 | 99% | > 50 | < 5 s |

The benchmark replays the Weka-format agentic trace described in [perf/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/perf/README.md).

### Performance Results

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table dynamo-super-vl-comparison">
<thead>
<tr><th>Workload</th><th>Recipe</th><th>Framework</th><th>SKU</th><th>Concurrency</th><th>System Output Tokens/s/GPU</th><th>User Output Tokens/s (P50)</th><th>TTFT P50 (ms)</th></tr>
</thead>
<tbody>
<tr data-highlight-target="vllm-h100-agg"><td>SWEBench-Pro</td><td>Aggregated (2 workers)</td><td>vLLM</td><td>H100</td><td>16</td><td>270</td><td>102.2</td><td>490</td></tr>
<tr data-highlight-target="vllm-h100-disagg"><td>SWEBench-Pro</td><td>Disaggregated (1P1D)</td><td>vLLM</td><td>H100</td><td>16</td><td>259</td><td>106</td><td>1,300</td></tr>
<tr data-highlight-target="vllm-h200-agg"><td>SWEBench-Pro</td><td>Aggregated (2 workers)</td><td>vLLM</td><td>H200</td><td>16</td><td>376</td><td>61</td><td>710</td></tr>
<tr data-highlight-target="vllm-h200-disagg"><td>SWEBench-Pro</td><td>Disaggregated (1P1D)</td><td>vLLM</td><td>H200</td><td>16</td><td>361</td><td>61.6</td><td>1,870</td></tr>
<tr data-highlight-target="vllm-b200-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>vLLM</td><td>B200</td><td>128</td><td>1,378.9</td><td>68.86</td><td>396</td></tr>
<tr data-highlight-target="vllm-b200-disagg"><td>SWEBench-Pro</td><td>Disaggregated (3P2D)</td><td>vLLM</td><td>B200</td><td>128</td><td>1,050.44</td><td>60.84</td><td>1,073</td></tr>
<tr data-highlight-target="vllm-b200-certified-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>vLLM</td><td>B200 Certified</td><td>128</td><td>1,411.0</td><td>74.75</td><td>369</td></tr>
<tr data-highlight-target="vllm-b200-certified-disagg"><td>SWEBench-Pro</td><td>Disaggregated (3P2D)</td><td>vLLM</td><td>B200 Certified</td><td>128</td><td>1,163.6</td><td>82.73</td><td>990</td></tr>
<tr data-highlight-target="vllm-gb200-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>vLLM</td><td>GB200</td><td>128</td><td>1,503.57</td><td>76.15</td><td>403</td></tr>
<tr data-highlight-target="vllm-gb200-disagg"><td>SWEBench-Pro</td><td>Disaggregated (3P2D)</td><td>vLLM</td><td>GB200</td><td>128</td><td>1,269.93</td><td>82.09</td><td>836</td></tr>
<tr data-highlight-target="vllm-b300-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>vLLM</td><td>B300</td><td>128</td><td>1,536</td><td>78.5</td><td>438</td></tr>
<tr data-highlight-target="vllm-gb300-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>vLLM</td><td>GB300</td><td>160</td><td>1,719.51</td><td>67.9</td><td>381</td></tr>
<tr data-highlight-target="vllm-gb300-disagg"><td>SWEBench-Pro</td><td>Disaggregated (3P2D)</td><td>vLLM</td><td>GB300</td><td>128</td><td>1,356.51</td><td>82.76</td><td>649</td></tr>
<tr data-highlight-target="trtllm-b200-agg"><td>SWEBench-Pro</td><td>Aggregated (4 workers)</td><td>TensorRT-LLM</td><td>B200</td><td>80</td><td>995</td><td>78.98</td><td>655</td></tr>
<tr data-highlight-target="trtllm-b200-disagg"><td>SWEBench-Pro</td><td>Disaggregated (2P2D)</td><td>TensorRT-LLM</td><td>B200</td><td>80</td><td>1,119.13</td><td>144.61</td><td>2,366</td></tr>
<tr data-highlight-target="vllm-vr200-agg"><td>SWEBench-Pro</td><td>Aggregated (11 workers)</td><td>vLLM</td><td>VR200</td><td>616</td><td>2,969</td><td>96.6</td><td>490</td></tr>
<tr data-highlight-target="vllm-vr200-disagg"><td>SWEBench-Pro</td><td>Disaggregated (7P4D)</td><td>vLLM</td><td>VR200</td><td>528</td><td>3,276</td><td>155.9</td><td>1,120</td></tr>
<tr data-highlight-target="vllm-b300-disagg"><td>SWEBench-Pro</td><td>Disaggregated (3P2D)</td><td>vLLM</td><td>B300</td><td>160</td><td>1,317</td><td>68.1</td><td>1,390</td></tr>
</tbody>
</table>
</div>

The vLLM disaggregated results for B200, GB200, B300, and GB300 were measured before [PR #17977](https://github.com/ai-dynamo/dynamo/pull/17977) removed the decode CPU KV cache.

The B200 Certified rows come from NIM Factory release candidate `01M7PXDSP2S8A2X2EPVD8Z200M` on the 1.7.0 enterprise images. They replay the text-only SWE-bench Pro trace with a synthetic MTP acceptance length of 4.06 and five draft tokens; the manifests default to natural acceptance, and the synthetic setting is a commented line. The runs did not set `--mm-ipc-gpu-memory-gb`, matching the manifests. There are no performance results for image and video requests.

The VR200 rows replay SWE-bench Pro agentic multi-turn traces. Throughput is raw output tokens.
Each VR200 row is the highest output per GPU with at least 50 output tokens/s per user (P50) and TTFT P50 under 5 s.
The VR200 results used a synthetic MTP acceptance length of 4.063 with five draft tokens.
The VR200 recipes default to natural acceptance; the synthetic setting is a commented line.
VR200 artifacts are not published.
At the same GPU count, 7P4D gives +10.3% output per GPU (observed across separate runs).
The 7P4D row ran on the previous image with FlashInfer #5267 and vLLM #58231 added, plus a runtime patch for the KV split that FlashInfer #6331 and vLLM #61077 provide.
The aggregated row ran on the image in the manifest.

## Compare All Targets

GPU counts below use the standalone manifests. P counts prefill workers. D counts decode workers.

<div className="dynamo-variant-table-wrap">
<table className="dynamo-variant-table dynamo-super-vl-comparison">
<thead>
<tr><th>Framework</th><th>GPU</th><th>Topology</th><th>Total GPUs</th><th>Parallelism</th><th>Precision</th><th>Routing</th><th>Speculative Decoding</th><th>Context length</th></tr>
</thead>
<tbody>
<tr data-highlight-target="vllm-h100-agg"><td>vLLM</td><td>H100</td><td>2 workers</td><td>4</td><td>TP2 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-h100-disagg"><td>vLLM</td><td>H100</td><td>1P1D</td><td>4</td><td>TP2 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-h200-agg"><td>vLLM</td><td>H200</td><td>2 workers</td><td>2</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-h200-disagg"><td>vLLM</td><td>H200</td><td>1P1D</td><td>2</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b200-agg"><td>vLLM</td><td>B200</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b200-disagg"><td>vLLM</td><td>B200</td><td>3P2D</td><td>5</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b200-certified-agg"><td>vLLM</td><td>B200 Certified</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b200-certified-disagg"><td>vLLM</td><td>B200 Certified</td><td>3P2D</td><td>5</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-gb200-agg"><td>vLLM</td><td>GB200</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-gb200-disagg"><td>vLLM</td><td>GB200</td><td>3P2D</td><td>5</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP3</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b300-agg"><td>vLLM</td><td>B300</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-b300-disagg"><td>vLLM</td><td>B300</td><td>3P2D</td><td>5</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP3</td><td>1M</td></tr>
<tr data-highlight-target="vllm-gb300-agg"><td>vLLM</td><td>GB300</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-gb300-disagg"><td>vLLM</td><td>GB300</td><td>3P2D</td><td>5</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP3</td><td>1M</td></tr>
<tr data-highlight-target="vllm-vr200-agg"><td>vLLM</td><td>VR200</td><td>11 workers</td><td>11</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="vllm-vr200-disagg"><td>vLLM</td><td>VR200</td><td>7P4D</td><td>11</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>KV-aware</td><td>MTP5</td><td>1M</td></tr>
<tr data-highlight-target="trtllm-b200-agg"><td>TensorRT-LLM</td><td>B200</td><td>4 workers</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>Load + session affinity</td><td>MTP3</td><td>1M</td></tr>
<tr data-highlight-target="trtllm-b200-disagg"><td>TensorRT-LLM</td><td>B200</td><td>2P2D</td><td>4</td><td>TP1 per worker</td><td>NVFP4 weights, FP8 KV</td><td>Load + session affinity</td><td>MTP3</td><td>1M</td></tr>
</tbody>
</table>
</div>

## Certified Targets

The B200 Certified targets differ from the open-source B200 targets as follows. The [certified recipe README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/certified/nemotron-3.5-super-vl-120b-a12b/README.md) has the details.

- They use the 1.7.0 enterprise images, which publish on NGC with the Dynamo 1.7.0 release.
- Each worker downloads the checkpoint into an `emptyDir` volume, in place of the `shared-model-cache` PVC. To download it once and reuse it across pods, mount the PVC instead; see [Shared Model Cache](https://github.com/ai-dynamo/dynamo/blob/main/recipes/certified/nemotron-3.5-super-vl-120b-a12b/README.md#shared-model-cache).
- To serve image or video requests, add `--mm-ipc-gpu-memory-gb 10` to every worker and raise the request size limits; see [Multimodal Workloads](https://github.com/ai-dynamo/dynamo/blob/main/recipes/certified/nemotron-3.5-super-vl-120b-a12b/README.md#multimodal-workloads). NIM Factory image and video accuracy testing, in release candidate `01M7QFWA5YFWDAQ7TBP7AHQSF0`, used these settings.
- To serve video on the disaggregated target, also raise `DYN_TCP_REQUEST_TIMEOUT` to 120 s and the NIXL KV lease (`kv_lease_duration`) to 2400 s; the same README section shows how. With the defaults, large videos time out on the way to prefill, and decode can read KV after prefill has released it.
- The disaggregated prefill workers offload KV cache to CPU, 200 GB per worker, and the decode workers use `NixlConnector` alone.

## Source

- Model cache: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/model-cache/model-download.yaml)

- [vLLM H100 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-h100-swebenchpro/deploy.yaml)

- [vLLM H100 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-h100-swebenchpro/deploy.yaml)

- [vLLM H200 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-h200-swebenchpro/deploy.yaml)

- [vLLM H200 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-h200-swebenchpro/deploy.yaml)

- [vLLM B200 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml)

- [vLLM B200 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml)

- [vLLM B200 Certified aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b200-swebenchpro/deploy.yaml)

- [vLLM B200 Certified disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/certified/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b200-swebenchpro/deploy.yaml)

- [vLLM GB200 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-gb200-swebenchpro/deploy.yaml)

- [vLLM GB200 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-gb200-swebenchpro/deploy.yaml)

- [vLLM B300 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-b300-swebenchpro/deploy.yaml)

- [vLLM GB300 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-gb300-swebenchpro/deploy.yaml)

- [vLLM GB300 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-gb300-swebenchpro/deploy.yaml)

- [vLLM VR200 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/agg-vr200-swebenchpro/deploy.yaml)

- [vLLM VR200 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-vr200-swebenchpro/deploy.yaml)

- [TensorRT-LLM B200 aggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/trtllm/agg-b200-swebenchpro/deploy.yaml)

- [TensorRT-LLM B200 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/trtllm/disagg-b200-swebenchpro/deploy.yaml)

- [vLLM B300 disaggregated](https://github.com/ai-dynamo/dynamo/blob/main/recipes/nemotron-3.5-super-vl-120b-a12b/vllm/disagg-b300-swebenchpro/deploy.yaml)
