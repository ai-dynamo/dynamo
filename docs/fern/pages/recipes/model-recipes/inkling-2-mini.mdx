---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Inkling 2 Mini"
subtitle: "Serve Inkling 2 Mini with Dynamo and SGLang on B200 or VR200."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Each target below is a Dynamo + SGLang deployment of Thinking Machines Lab's Inkling 2 Mini, a multimodal (text and image in, text out) thinking MoE model with NVFP4 routed experts. Pick your GPU and topology to see the deployment, then follow the steps below.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" defaultChecked />
<label htmlFor="recipe-sku-b200">B200</label>
<input type="radio" id="recipe-sku-vr200" name="recipe-sku" value="vr200" />
<label htmlFor="recipe-sku-vr200">VR200</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Topology</span>
<input type="radio" id="recipe-variant-agg" name="recipe-variant" value="agg" defaultChecked />
<label htmlFor="recipe-variant-agg">Aggregated</label>
<input type="radio" id="recipe-variant-disagg" name="recipe-variant" value="disagg" />
<label htmlFor="recipe-variant-disagg">Disaggregated</label>
</div>
<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="agg">
<span><b>Checkpoint</b> thinkingmachines/Inkling-2-Mini</span>
<span><b>Precision</b> NVFP4 routed experts</span>
<span><b>GPUs</b> 2x B200 (1 replica)</span>
<span><b>Parallelism</b> TP2, EP2</span>
<span><b>Features</b> Reasoning, tool calling, structured output, image input</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="b200" data-variant="disagg">
<span><b>Checkpoint</b> thinkingmachines/Inkling-2-Mini</span>
<span><b>Precision</b> NVFP4 routed experts</span>
<span><b>GPUs</b> 4x B200 (1P1D)</span>
<span><b>Parallelism</b> TP2, EP2 per role</span>
<span><b>KV transfer</b> NIXL over UCX, KV-aware routing</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="vr200" data-variant="agg">
<span><b>Checkpoint</b> thinkingmachines/Inkling-2-Mini</span>
<span><b>Precision</b> NVFP4 routed experts</span>
<span><b>GPUs</b> 2x VR200 (1 replica)</span>
<span><b>Parallelism</b> TP2, EP2</span>
<span><b>Features</b> Reasoning, tool calling, structured output, image input</span>
</div>
<div className="dynamo-target-picker-summary" data-sku="vr200" data-variant="disagg">
<span><b>Checkpoint</b> thinkingmachines/Inkling-2-Mini</span>
<span><b>Precision</b> NVFP4 routed experts</span>
<span><b>GPUs</b> 4x VR200 (1P1D)</span>
<span><b>Parallelism</b> TP2, EP2 per role</span>
<span><b>KV transfer</b> Mooncake over NVLink, KV-aware routing</span>
</div>
</div>

## Prerequisites

<div data-sku="b200" data-variant="agg">

- A Kubernetes cluster with the [Dynamo platform](../../kubernetes/getting-started/quickstart.mdx) installed and 2 B200 GPUs available.

</div>

<div data-sku="b200" data-variant="disagg">

- A Kubernetes cluster with the [Dynamo platform](../../kubernetes/getting-started/quickstart.mdx) installed and 4 B200 GPUs available.
- InfiniBand access for the prefill and decode workers. The manifest tunes UCX for it, but the `rdma/ib` resource and any node placement are cluster-owned, so add them for your cluster. Without RDMA, KV moves over TCP and time to first token grows with prompt length.

</div>

<div data-sku="vr200" data-variant="agg">

- A Kubernetes cluster with the [Dynamo platform](../../kubernetes/getting-started/quickstart.mdx) installed and 2 VR200 GPUs available on arm64 nodes.

</div>

<div data-sku="vr200" data-variant="disagg">

- A Kubernetes cluster with the [Dynamo platform](../../kubernetes/getting-started/quickstart.mdx) installed and 4 VR200 GPUs available on arm64 nodes.
- Both workers in one NVLink domain, on one NVL72 node or with a ComputeDomain channel claim per worker. KV moves between the workers with the Mooncake backend forced onto NVLink, and the manifest does not enforce the placement, so a cross-domain placement can fail the transfer.

</div>

- A ReadWriteMany model-cache volume of at least 100 GiB.
- A Hugging Face token only if the checkpoint repository is private or gated for your account.

The runtime image is `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.8.0-inkling-2-mini-dev.1`. It carries the model owners' SGLang runtime for this model, so no source patching happens at startup.

Run commands from the repository root:

```bash
export CONTEXT="<your-kubernetes-context>"
export NAMESPACE="<your-namespace>"

kubectl --context "$CONTEXT" create namespace "$NAMESPACE"
```

### Cluster-specific settings

The manifests contain no credentials, image-pull Secrets, or priority classes. The VR200 manifests select workers with `kubernetes.io/arch: arm64` and `nvidia.com/gpu.machine: VR-NVL72`; replace those if your GPU nodes carry different labels. The B200 manifests set no node selector. Add what your cluster needs in a private Kustomize overlay; the usual cases are an image-pull Secret for a private registry mirror, a node selector for the GPU nodes, and a priority class on clusters with a preemptible default.

## Populate the model cache

Skip this step if `shared-model-cache` already holds `thinkingmachines/Inkling-2-Mini` at `/shared-model-cache/Inkling-2-Mini`. PVC storage class and access mode fields are immutable, so keep an existing cache.

For a new cache, replace the placeholder `storageClassName` in [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/model-cache/model-cache.yaml) with a ReadWriteMany-capable storage class. If the repository is private or gated, create the token Secret first:

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" create secret generic hf-token \
  --from-literal=HF_TOKEN="<your-hugging-face-token>"
```

Then download the checkpoint:

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/model-cache/model-cache.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" delete job \
  inkling-2-mini-model-download --ignore-not-found
kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/model-cache/model-download.yaml
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait \
  --for=condition=Complete job/inkling-2-mini-model-download --timeout=7200s
```

## Deploy

<div data-sku="b200" data-variant="agg">

```bash
export DGD="inkling-2-mini-sglang-agg"

kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/sglang/agg-b200/deploy.yaml
```

</div>

<div data-sku="b200" data-variant="disagg">

```bash
export DGD="inkling-2-mini-sglang-disagg"

kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/sglang/disagg-b200/deploy.yaml
```

</div>

<div data-sku="vr200" data-variant="agg">

```bash
export DGD="inkling-2-mini-sglang-agg"

kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/sglang/agg-vr/deploy.yaml
```

</div>

<div data-sku="vr200" data-variant="disagg">

```bash
export DGD="inkling-2-mini-sglang-disagg"

kubectl --context "$CONTEXT" -n "$NAMESPACE" apply \
  -f recipes/inkling-2-mini/sglang/disagg-vr/deploy.yaml
```

</div>

Wait for the deployment to become ready. Loading the weights and capturing CUDA graphs takes several minutes:

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" wait \
  --for=condition=Ready "dynamographdeployment/${DGD}" --timeout=3600s
```

## Verify

Forward the frontend Service. The command keeps running:

```bash
kubectl --context "$CONTEXT" -n "$NAMESPACE" port-forward \
  "service/${DGD}-frontend" 8000:8000
```

In a second terminal, check that it serves the model:

```bash
curl -fsS http://localhost:8000/v1/models
```

The response lists `inkling-2-mini`.

## Smoke test

Send a chat request. The model thinks before it answers, and the reasoning parser returns the thinking in `reasoning_content` and the answer in `content`:

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    "max_tokens": 1024
  }'
```

Check structured output. The response `content` is only the JSON object; the thinking is in `reasoning_content`:

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": "Return a JSON object with keys name and age for a fictional person."}],
    "response_format": {"type": "json_object"},
    "max_tokens": 3000
  }'
```

Check image input with an image URL or a base64 data URL:

```bash
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": [
      {"type": "text", "text": "What color is this image? Answer in one word."},
      {"type": "image_url", "image_url": {"url": "<image-url-or-data-url>"}}
    ]}],
    "max_tokens": 1024
  }'
```

## Notes

- **Supported features.** Streaming and non-streaming chat and completions, a reasoning parser, tool calling (single, parallel, required, named, and streamed), structured output (`json_object` and `json_schema`), logprobs, stop sequences, and image input. The recipes pass `--grammar-backend xgrammar` and `--enable-multimodal`. All recipes use the Dynamo frontend's default parser (`--dyn-chat-processor dynamo`), with `--dyn-reasoning-parser inkling` and `--dyn-tool-call-parser inkling` on the workers.
- **Parallelism.** The NVFP4 expert kernel needs expert parallelism, so every worker runs `--tp 2 --ep-size 2`. A single-GPU worker is rejected.
- **Frontend limits.** The Dynamo frontend accepts only `n=1`. A fixed `seed` does not give reproducible output. Audio and video input are not supported.
- **Vera Rubin first request.** The first request that sets `top_p`, `top_k`, or `min_p` on VR200 compiles a sampling kernel, which takes about a minute. Later requests are not affected.
- **Disaggregated serving.** The frontend uses KV-aware routing with session affinity, and the prefill worker publishes KV events. Both workers need `--host 0.0.0.0` for the handoff, which the manifests set.
- **Status.** This recipe is experimental.

## Source

- [Recipe overview](https://github.com/ai-dynamo/dynamo/tree/main/recipes/inkling-2-mini)
- [B200 aggregated deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/sglang/agg-b200/deploy.yaml)
- [B200 disaggregated deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/sglang/disagg-b200/deploy.yaml)
- [VR200 aggregated deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/sglang/agg-vr/deploy.yaml)
- [VR200 disaggregated deployment](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/sglang/disagg-vr/deploy.yaml)
- [Model cache](https://github.com/ai-dynamo/dynamo/blob/main/recipes/inkling-2-mini/model-cache/model-cache.yaml)
