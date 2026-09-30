<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron 3.5 Lightning TensorRT-LLM Recipes

Aggregate Dynamo TensorRT-LLM recipes for Nemotron 3.5 Lightning. H100 and H200
use the NVFP4 checkpoint. B200 and GB200 use
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`.

## Configurations

| Hardware | Recipe | Speculative decoding |
|---|---|---|
| H100 | `agg-h100-mtp/deploy.yaml` | MTP |
| H200 | `agg-h200-mtp/deploy.yaml` | MTP |
| B200 | `agg-b200-bf16/deploy.yaml` | None |
| B200 | `agg-b200-mtp-bf16/deploy.yaml` | MTP |
| GB200 | `agg-gb200-bf16/deploy.yaml` | None |
| GB200 | `agg-gb200-mtp-bf16/deploy.yaml` | MTP |

B200 and GB200 recipes include `-bf16` in the recipe directory and Kubernetes
resource names, for example `agg-b200-mtp-bf16/deploy.yaml`.

## Platform notes

GB200 manifests select ARM nodes.

## Prerequisites

1. Dynamo Platform and the `DynamoGraphDeployment` CRD installed on the target
   Kubernetes cluster.
2. The shared model cache created from `../model-cache/model-cache.yaml` and
   populated with the matching NVFP4 or BF16 download Job.
3. A Hugging Face secret named `hf-token-secret` and an image-pull secret when
   required by the cluster.

## Quick start

```bash
export NAMESPACE=your-namespace
RECIPE=agg-b200-mtp-bf16

kubectl apply -n "${NAMESPACE}" -f "${RECIPE}/deploy.yaml"
kubectl get dgd -n "${NAMESPACE}" trtllm-agg-b200-mtp-bf16 -w
```

Port-forward the frontend service:

```bash
kubectl port-forward -n "${NAMESPACE}" \
  svc/trtllm-agg-b200-mtp-bf16-frontend 8000:8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    "messages": [{"role": "user", "content": "Say OK."}],
    "max_tokens": 128,
    "temperature": 0
  }'
```

## Notes

- MTP and no-spec recipes are separate serving configurations with their own
  manifest-level engine settings.
- Replace `your-image-pull-secret` with the image-pull secret available in the
  target namespace.
