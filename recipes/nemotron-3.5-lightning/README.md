<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning Recipes

Kubernetes recipes for serving the NVFP4 and BF16 variants of
`NVIDIA-Nemotron-3.5-Lightning-30B-A3B` with Dynamo. The repository contains
vLLM recipes for aggregate and disaggregated deployments, plus aggregate
TensorRT-LLM recipes.

## Configurations

| Backend | Hardware | Deployment modes | Speculative decoding |
|---|---|---|---|
| vLLM | H100, H200, B200, GB200 | Aggregate and selected 1P/1D disaggregated | MTP, DFlash, DSpark; NVFP4 on H100/H200 and BF16 on B200/GB200 |
| TensorRT-LLM | H100, H200, B200, GB200 | Aggregate | None, MTP; NVFP4 on H100/H200 and BF16 on B200/GB200 |

The NVFP4 recipes are organized as `vllm/{agg,disagg}-{hardware}-{specdec}` and
`trtllm/agg-{hardware}[-{specdec}]`. B200 and GB200 recipes use BF16 weights
and include `-bf16` in the recipe directory and Kubernetes resource names.

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
kubectl apply -n "${NAMESPACE}" -f model-cache/model-download-bf16.yaml
kubectl wait -n "${NAMESPACE}" --for=condition=Complete job/model-download-bf16 --timeout=3600s
```

Deploy a recipe:

```bash
RECIPE=vllm/agg-b200-dspark-bf16
kubectl apply -n "${NAMESPACE}" -f "${RECIPE}/deploy.yaml"
kubectl get dgd -n "${NAMESPACE}" vllm-agg-b200-dspark-bf16 -w
```

Port-forward the frontend service and send a chat request:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  svc/vllm-agg-b200-dspark-bf16-frontend 8000:8000
```

```bash
MODEL=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
# Replace -BF16 with -NVFP4 for H100 or H200 recipes.

curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d @- <<EOF
{
  "model": "${MODEL}",
  "messages": [{"role": "user", "content": "Hello!"}],
  "max_tokens": 128,
  "temperature": 0
}
EOF
```

## Model cache

[`model-cache/model-download.yaml`](model-cache/model-download.yaml) downloads:

- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash`
- `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark`

[`model-cache/model-download-bf16.yaml`](model-cache/model-download-bf16.yaml)
downloads the corresponding BF16 base, DFlash, and DSpark checkpoints. Both
jobs use the shared model cache; ensure its PVC has capacity for every
checkpoint you intend to retain.

## Notes

- H100 and H200 DFlash recipes use the NVFP4 DFlash checkpoint.
- B200 and GB200 recipes use the BF16 base checkpoint with BF16 DFlash or
  DSpark drafts where applicable.
- GB200 recipes require ARM-node scheduling.
- Disaggregated vLLM recipes keep prefill and decode speculative-decoding
  settings aligned so NIXL cache metadata remains compatible.
