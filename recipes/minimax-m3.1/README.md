<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M3.1

These recipes serve the MiniMax-M3.1-Flash checkpoint with Dynamo on NVIDIA B200 or GB300 GPUs, using vLLM, SGLang, or TensorRT-LLM. Each profile runs one aggregated worker with tensor parallelism (TP) 4 and expert parallelism on four GPUs of a single node. Every profile serves the model's full 1,048,576-token context and up to 256 concurrent requests with prefix caching. The vLLM and SGLang profiles also use an FP8 KV cache and KV-aware routing in the frontend.

## Deployment Profiles

| Profile | Framework | Topology | GPUs | Inputs |
| --- | --- | --- | ---: | --- |
| `vllm/agg-b200` | vLLM | 1 aggregated TP4 + EP worker, CUDA graphs | 4x B200 | Text, image |
| `vllm/agg-gb300` | vLLM | 1 aggregated TP4 + EP worker, CUDA graphs | 4x GB300 | Text, image |
| `sglang/agg-b200` | SGLang | 1 aggregated TP4 + EP4 worker, MegaMoE experts | 4x B200 | Text, image |
| `sglang/agg-gb300` | SGLang | 1 aggregated TP4 + EP4 worker, MegaMoE experts | 4x GB300 | Text, image |
| `trtllm/agg-b200` | TensorRT-LLM | 1 aggregated TP4 + EP4 worker, MSA sparse attention | 4x B200 | Text, image |
| `trtllm/agg-gb300` | TensorRT-LLM | 1 aggregated TP4 + EP4 worker, MSA sparse attention | 4x GB300 | Text, image |

The B200 and GB300 profiles of a framework differ only in their resource names; each uses the same multi-architecture image, and the node selects the amd64 or arm64 variant.

Every profile serves the model as `minimax-m3.1` behind the Dynamo frontend and parses reasoning and tool calls with the `minimax_m3` parsers.

## Prepare the Model Cache

The checkpoint is the Hugging Face model `MiniMaxAI/MiniMax-M3.1-Flash` (72 files, about 252 GB, including a DSpark draft model these profiles do not enable). From the Dynamo repository root, set `CONTEXT` to your cluster context, select a namespace, and create a Secret holding a Hugging Face token that can read that model:

```bash
export NAMESPACE=your-namespace
kubectl --context "${CONTEXT}" create namespace "$NAMESPACE"
kubectl --context "${CONTEXT}" create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n "$NAMESPACE"
```

Set `storageClassName` in `model-cache/model-cache.yaml`, then create the ReadWriteMany cache and download the checkpoint:

```bash
kubectl --context "${CONTEXT}" apply -f recipes/minimax-m3.1/model-cache/model-cache.yaml -n "$NAMESPACE"
kubectl --context "${CONTEXT}" apply -f recipes/minimax-m3.1/model-cache/model-download.yaml -n "$NAMESPACE"
kubectl --context "${CONTEXT}" wait --for=condition=complete job/minimax-m3-1-model-download \
  -n "$NAMESPACE" --timeout=14400s
```

The Job writes the checkpoint to `/shared-model-cache/MiniMax-M3.1-Flash` and fails if any shard listed in `model.safetensors.index.json` is missing. Every deployment profile loads the model from that path.

## Deploy

The cluster requires the NVIDIA Dynamo platform with `nvidia.com/v1beta1` DynamoGraphDeployment support. Each worker requests four B200 or GB300 GPUs on one node, 16 CPUs, and 128 GiB of host memory, and runs as UID 0.

Each `deploy.yaml` runs its framework's Dynamo runtime image for MiniMax M3.1 from NGC:

- vLLM: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0-minimax-m3.1-flash-dev.1`
- SGLang: `nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.6.0-minimax-m3.1-flash-dev.1`
- TensorRT-LLM: `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.6.0-minimax-m3.1-flash-dev.1`

Select a profile and a private cluster Kustomization directory:

```bash
export FRAMEWORK=vllm # or sglang, trtllm
export GPU=gb300 # or b200
export RECIPE_DIR="$(pwd)/recipes/minimax-m3.1/$FRAMEWORK/agg-$GPU"
export CLUSTER_KUSTOMIZATION=/absolute/path/to/private/minimax-m3.1-cluster
export DGD="minimax-m3-1-$FRAMEWORK-agg-$GPU"
```

The profile's `deploy.yaml` is the portable source and contains no pull Secret, scheduling, or storage bindings. Copy the [beta cluster Kustomization starter](../templates/kustomize/README.md) outside the repository, point its `resources` at the absolute path of `$RECIPE_DIR/deploy.yaml`, and select the `/agg` Components. At minimum, fill `scheduling` with your B200 or GB300 node placement; select `registry-credentials` only when you pull the image from a private mirror, and `cache-binding` only when your populated claim is not named `shared-model-cache`. Follow the starter's Component order and placeholder preflight.

Validate and render from the repository root with standalone Kustomize v5.8.1, Python 3.9 or later, and PyYAML, then apply:

```bash
python3 scripts/validate-recipe-kustomization.py \
  "$RECIPE_DIR/deploy.yaml" \
  "$CLUSTER_KUSTOMIZATION/kustomization.yaml"
kustomize build --load-restrictor LoadRestrictionsNone \
  "$CLUSTER_KUSTOMIZATION" > "$CLUSTER_KUSTOMIZATION/rendered.yaml"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply --dry-run=server \
  -f "$CLUSTER_KUSTOMIZATION/rendered.yaml"
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" apply \
  -f "$CLUSTER_KUSTOMIZATION/rendered.yaml"
```

Keep filled cluster values and rendered manifests outside the repository.

## Smoke Test

Wait for the deployment, then forward its frontend port. The worker spends several minutes in engine initialization after the weights load:

```bash
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" wait --for=condition=Ready \
  "dynamographdeployment/${DGD}" --timeout=3600s
kubectl --context "${CONTEXT}" -n "${NAMESPACE}" port-forward \
  "service/${DGD}-frontend" 8000:8000
```

In another terminal, verify model discovery and a reasoning completion:

```bash
curl --fail-with-body http://localhost:8000/v1/models
curl --fail-with-body http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-m3.1","messages":[{"role":"user","content":"What is 2 + 2?"}],"max_tokens":1024,"temperature":1.0,"top_p":0.95,"chat_template_kwargs":{"reasoning_effort":"high"}}'
```

Expect HTTP 200 and the answer in `choices[0].message.content`. The model decides per request whether to reason, and any reasoning arrives in a separate `reasoning_content` field; `reasoning_effort` (`low`, `medium`, `high`, `xhigh`, or `max`) sets how much it reasons. `temperature` 1.0 and `top_p` 0.95 are the model's recommended sampling settings.

## Known Issues

- On the TensorRT-LLM profiles, a request that needs an attention kernel the engine did not compile at startup compiles it on the spot, which stalls the worker and its GPUs for one to two minutes. Dynamo's health check can time out during the stall and restart the worker, which discards the compiled kernel; set `DYN_HEALTH_CHECK_ENABLED=false` on the Worker to let the compilation finish.
