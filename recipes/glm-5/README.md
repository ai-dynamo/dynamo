<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GLM-5 FP8 Recipes

TensorRT-LLM deployment recipes for [zai-org/GLM-5-FP8](https://huggingface.co/zai-org/GLM-5-FP8).

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg/deploy.yaml**](trtllm/agg/deploy.yaml) | 8x B200 GPU | Aggregated | FP8 baseline |
| [**trtllm/agg/deploy-specdec.yaml**](trtllm/agg/deploy-specdec.yaml) | 8x B200 GPU | Aggregated | FP8 with speculative decoding (MTP) |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with 8 B200 GPUs available to a single worker pod
3. **HuggingFace token** with access to the GLM-5 model

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache/model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy one configuration
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
# OR MTP SpecDec configuration deployment:
# kubectl apply -f trtllm/agg/deploy-specdec.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Baseline
kubectl port-forward svc/glm5-fp8-trtllm-agg-frontend 8000:8000 -n ${NAMESPACE}

# Speculative decode variant
# kubectl port-forward svc/glm5-fp8-specdec-trtllm-agg-frontend 8000:8000 -n ${NAMESPACE}

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-5-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `zai-org/GLM-5-FP8`
- **Pinned revision**: `7ca2d2f1f1703aa0b189977fe3c126caf18b70e1`
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Tensor parallel**: 8
- **Expert parallel**: 8

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying.
- Update the container image tag in the deploy manifests to the Dynamo TRTLLM container tag of your choice.
- These recipes use the GLM-5 custom tokenizer path:
  - `--trtllm.custom_tokenizer glm_moe_dsa`
  - `--trtllm.tokenizer <pinned local snapshot path>`
