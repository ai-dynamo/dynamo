---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "MiniMax M3.1"
subtitle: "Serve MiniMax M3.1 with Dynamo and vLLM, SGLang, or TensorRT-LLM on B200 or GB300."
---

import { RecipeStyles } from "@/components/RecipeStyles";

<RecipeStyles />

Each target below is a Dynamo deployment of the MiniMax-M3.1-Flash checkpoint, a multimodal MoE model with sparse attention. Every target runs one aggregated worker with tensor parallelism (TP) 4 and expert parallelism on four GPUs of a single B200 or GB300 node. Choose your framework and GPU; the summary and commands update to match. Every target serves the model's full 1,048,576-token context with prefix caching; the vLLM and SGLang targets also use an FP8 KV cache and KV-aware routing. These are day-0 profiles, not performance-tuned deployments.

<div className="dynamo-target-picker">
<p className="dynamo-target-picker-title">Choose your deployment target</p>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">Framework</span>
<input type="radio" id="recipe-framework-vllm" name="recipe-framework" value="vllm" defaultChecked />
<label htmlFor="recipe-framework-vllm">vLLM</label>
<input type="radio" id="recipe-framework-sglang" name="recipe-framework" value="sglang" />
<label htmlFor="recipe-framework-sglang">SGLang</label>
<input type="radio" id="recipe-framework-trtllm" name="recipe-framework" value="trtllm" />
<label htmlFor="recipe-framework-trtllm">TensorRT-LLM</label>
</div>
<div className="dynamo-target-picker-row">
<span className="dynamo-target-picker-dim">GPU</span>
<input type="radio" id="recipe-sku-b200" name="recipe-sku" value="b200" />
<label htmlFor="recipe-sku-b200">B200</label>
<input type="radio" id="recipe-sku-gb300" name="recipe-sku" value="gb300" defaultChecked />
<label htmlFor="recipe-sku-gb300">GB300</label>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="b200">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x B200, one worker</span>
<span><b>Parallelism</b> TP4 + EP, CUDA graphs</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="vllm" data-sku="gb300">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x GB300, one worker</span>
<span><b>Parallelism</b> TP4 + EP, CUDA graphs</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="b200">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x B200, one worker</span>
<span><b>Parallelism</b> TP4 + EP4, MegaMoE experts</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="sglang" data-sku="gb300">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x GB300, one worker</span>
<span><b>Parallelism</b> TP4 + EP4, MegaMoE experts</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="trtllm" data-sku="b200">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x B200, one worker</span>
<span><b>Parallelism</b> TP4 + EP4, MSA sparse attention</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
<div className="dynamo-target-picker-summary" data-recipe-framework="trtllm" data-sku="gb300">
<span><b>Checkpoint</b> MiniMax-M3.1-Flash (Hugging Face)</span>
<span><b>Precision</b> NVFP4 MoE experts</span>
<span><b>GPUs</b> 4x GB300, one worker</span>
<span><b>Parallelism</b> TP4 + EP4, MSA sparse attention</span>
<span><b>Modalities</b> Text + image input, text output</span>
<span><b>Limits</b> 1,048,576-token context, 256 concurrent requests</span>
</div>
</div>

## Prerequisites

Every target requires:

- A Kubernetes cluster with the Dynamo platform installed and `nvidia.com/v1beta1` DynamoGraphDeployment support. See the [Kubernetes Deployment Guide](../../kubernetes/getting-started/quickstart.mdx).
- Four free GPUs on one node of the selected type. Each worker also requests 16 CPUs and 128 GiB of host memory, and runs as UID 0.
- A Hugging Face token that can read `MiniMaxAI/MiniMax-M3.1-Flash`.
- The recipe's Dynamo runtime image from NGC: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0-minimax-m3.1-flash-dev.1`, `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.0-minimax-m3.1-flash-dev.1`, `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.6.0-minimax-m3.1-flash-dev.1`, for vLLM, SGLang, and TensorRT-LLM.
- Standalone Kustomize v5.8.1, Python 3.9 or later with PyYAML, and a private cluster Kustomization with node placement for your cluster.

Set the namespace, then create a Secret for the model download:

```bash
export NAMESPACE=your-namespace
kubectl create namespace "${NAMESPACE}"
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "${NAMESPACE}"
```

If the namespace and Secret already exist, use them instead. Run the following commands from the Dynamo repository root.

<Warning>
Every target loads the checkpoint from a populated `shared-model-cache` PVC in the selected namespace. Set `storageClassName` in `model-cache/model-cache.yaml` to a ReadWriteMany storage class on your cluster (`kubectl get storageclass`) before applying it.
</Warning>

## Deploy

Create the cache and download the checkpoint (72 files, about 252 GB):

```bash
kubectl apply -f recipes/minimax-m3.1/model-cache/model-cache.yaml -n "${NAMESPACE}"
kubectl apply -f recipes/minimax-m3.1/model-cache/model-download.yaml -n "${NAMESPACE}"
kubectl wait --for=condition=complete job/minimax-m3-1-model-download \
  -n "${NAMESPACE}" --timeout=14400s
```

The Job writes the checkpoint to `/shared-model-cache/MiniMax-M3.1-Flash` and fails if any shard listed in `model.safetensors.index.json` is missing.

Select the recipe for your target:

<div data-recipe-framework="vllm" data-sku="b200">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/vllm/agg-b200"
export DGD=minimax-m3-1-vllm-agg-b200
```

</div>

<div data-recipe-framework="vllm" data-sku="gb300">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/vllm/agg-gb300"
export DGD=minimax-m3-1-vllm-agg-gb300
```

</div>

<div data-recipe-framework="sglang" data-sku="b200">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/sglang/agg-b200"
export DGD=minimax-m3-1-sglang-agg-b200
```

</div>

<div data-recipe-framework="sglang" data-sku="gb300">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/sglang/agg-gb300"
export DGD=minimax-m3-1-sglang-agg-gb300
```

</div>

<div data-recipe-framework="trtllm" data-sku="b200">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/trtllm/agg-b200"
export DGD=minimax-m3-1-trtllm-agg-b200
```

</div>

<div data-recipe-framework="trtllm" data-sku="gb300">

```bash
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/trtllm/agg-gb300"
export DGD=minimax-m3-1-trtllm-agg-gb300
```

</div>

The recipe's `deploy.yaml` contains no pull Secret, scheduling, or storage bindings. Copy the [beta cluster Kustomization starter](https://github.com/ai-dynamo/dynamo/blob/main/recipes/templates/kustomize/README.md) outside the repository, point its `resources` at the absolute path of `$RECIPE_DIR/deploy.yaml`, and select the `/agg` Components. At minimum, fill `scheduling` with your node placement. Select `registry-credentials` only when you pull the image from a private mirror, and `cache-binding` only when your populated claim is not named `shared-model-cache`. Keep filled cluster values outside the repository.

Set `CLUSTER_KUSTOMIZATION` to that directory, then validate, render, and apply it:

```bash
export CLUSTER_KUSTOMIZATION=/absolute/path/to/private/minimax-m3.1-cluster
python3 scripts/validate-recipe-kustomization.py \
  "$RECIPE_DIR/deploy.yaml" \
  "$CLUSTER_KUSTOMIZATION/kustomization.yaml"
kustomize build --load-restrictor LoadRestrictionsNone \
  "$CLUSTER_KUSTOMIZATION" > "$CLUSTER_KUSTOMIZATION/rendered.yaml"
kubectl apply --dry-run=server -f "$CLUSTER_KUSTOMIZATION/rendered.yaml" -n "${NAMESPACE}"
kubectl apply -f "$CLUSTER_KUSTOMIZATION/rendered.yaml" -n "${NAMESPACE}"
```

Use `LoadRestrictionsNone` only with reviewed paths. It allows the private composition to reference the portable recipe.

## Smoke Test

Wait for the deployment, then forward its frontend port. The worker spends several minutes in engine initialization after the weights load:

```bash
kubectl wait --for=condition=Ready "dynamographdeployment/${DGD}" \
  -n "${NAMESPACE}" --timeout=3600s
kubectl port-forward "service/${DGD}-frontend" 8000:8000 -n "${NAMESPACE}"
```

In another terminal, check model discovery and send a reasoning request. Every target serves the model as `minimax-m3.1`:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":1024,"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"reasoning_effort":"high"}}'
```

Expect HTTP 200 and the answer in `choices[0].message.content`. The model decides per request whether to reason, and any reasoning arrives in a separate `reasoning_content` field; `reasoning_effort` (`low`, `medium`, `high`, `xhigh`, or `max`) sets how much it reasons. `temperature` 1.0 and `top_p` 0.95 are the model's recommended sampling settings.

## Supported Features

| Feature | vLLM | SGLang | TensorRT-LLM | Notes |
| --- | --- | --- | --- | --- |
| Text input | ✅ | ✅ | ✅ | |
| Image input | ✅ | ✅ | ✅ | `image_url` content parts |

## Compare All Targets

Every target runs one aggregated worker on four GPUs, serves a 1,048,576-token context, and admits up to 256 concurrent requests.

| Framework | GPU | Parallelism | Modalities |
| --- | --- | --- | --- |
| vLLM | GB300 | TP4 + EP, CUDA graphs | Text, image |
| vLLM | B200 | TP4 + EP, CUDA graphs | Text, image |
| SGLang | GB300 | TP4 + EP4, MegaMoE | Text, image |
| SGLang | B200 | TP4 + EP4, MegaMoE | Text, image |
| TensorRT-LLM | GB300 | TP4 + EP4, MSA sparse attention | Text, image |
| TensorRT-LLM | B200 | TP4 + EP4, MSA sparse attention | Text, image |

## Notes

- On the TensorRT-LLM targets, a request that needs an attention kernel the engine did not compile at startup compiles it on the spot, which stalls the worker and its GPUs for one to two minutes. Dynamo's health check can time out during the stall and restart the worker, which discards the compiled kernel; set `DYN_HEALTH_CHECK_ENABLED=false` on the Worker to let the compilation finish.
- The checkpoint ships a DSpark draft model, which these profiles do not enable.

## Source

- Recipe README: [recipes/minimax-m3.1/README.md](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/README.md)
- vLLM: [B200 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/vllm/agg-b200/deploy.yaml) and [GB300 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/vllm/agg-gb300/deploy.yaml)
- SGLang: [B200 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/sglang/agg-b200/deploy.yaml) and [GB300 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/sglang/agg-gb300/deploy.yaml)
- TensorRT-LLM: [B200 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/trtllm/agg-b200/deploy.yaml) and [GB300 deploy.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/trtllm/agg-gb300/deploy.yaml)
- Model cache: [model-cache.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/model-cache/model-cache.yaml) and [model-download.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/minimax-m3.1/model-cache/model-download.yaml)
