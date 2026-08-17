<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning Recipes

Kubernetes recipes for serving
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` with Dynamo. The repository
contains vLLM recipes for aggregate and disaggregated deployments, plus aggregate
TensorRT-LLM recipes.

## Configurations

| Backend | Hardware | Deployment modes | Speculative decoding |
|---|---|---|---|
| vLLM | H100, H200, B200, GB200 | Aggregate and selected 1P/1D disaggregated | MTP, DFlash, DSpark |
| TensorRT-LLM | H100, H200, B200, GB200 | Aggregate | None, MTP |

The vLLM recipes are organized as `vllm/{agg,disagg}-{hardware}-{specdec}`.
TensorRT-LLM recipes are organized as `trtllm/agg-{hardware}[-{specdec}]`.

## Repository layout

| Directory | Contents |
|---|---|
| `model-cache/` | Shared cache PVC and the model-download Job |
| `vllm/` | Aggregate and 1P/1D disaggregated vLLM recipes |
| `trtllm/` | Aggregate TensorRT-LLM recipes |
| `perf/` | Benchmark trace reference and trace results |

See [vLLM recipes](vllm/README.md) and [TensorRT-LLM recipes](trtllm/README.md)
for the complete per-platform inventory and runtime configuration.

## Prerequisites

1. Dynamo Platform and the `DynamoGraphDeployment` CRD installed on the target
   Kubernetes cluster.
2. A namespace with access to the required GPU nodes.
3. A Hugging Face secret named `hf-token-secret` with access to the target and
   draft-model repositories.
4. An image-pull secret when required by the cluster. Replace
   `your-image-pull-secret` in a manifest with the local secret name.

## Quick start

Create the shared model cache and populate it with the target and draft models:

```bash
export NAMESPACE=your-namespace
# Replace your-storage-class-name in model-cache/model-cache.yaml first.
kubectl apply -n "${NAMESPACE}" -f model-cache/model-cache.yaml
kubectl apply -n "${NAMESPACE}" -f model-cache/model-download.yaml
kubectl wait -n "${NAMESPACE}" --for=condition=Complete job/model-download --timeout=3600s
```

Deploy a recipe:

```bash
RECIPE=vllm/agg-b200-dspark
kubectl apply -n "${NAMESPACE}" -f "${RECIPE}/deploy.yaml"
kubectl get dgd -n "${NAMESPACE}" vllm-agg-b200-dspark -w
```

Port-forward the frontend service and send a chat request:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  svc/vllm-agg-b200-dspark-frontend 8000:8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

## Model cache

[`model-cache/model-download.yaml`](model-cache/model-download.yaml) downloads:

- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`

## Notes

- vLLM DFlash recipes use the NVFP4 DFlash checkpoint.
- GB200 recipes require ARM-node scheduling.
- Disaggregated vLLM recipes keep prefill and decode speculative-decoding
  settings aligned so NIXL cache metadata remains compatible.
