<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Nemotron-3-Super NVFP4 Recipes

Functional recipes for **nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4**.

## Available Configurations

| Configuration | GPUs | Backend | Mode | Description |
|--------------|------|---------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 1x B200/B300 GPU | TensorRT-LLM | Aggregated | Single-worker long-context deployment with MTP enabled |

## Prerequisites

1. **Dynamo Platform installed** - See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **Blackwell GPU cluster** with at least 1x B200 180GB GPU and Kubernetes access
3. **HuggingFace token** with access to NVIDIA models

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Create model cache PVC
# Update storageClassName in model-cache/model-cache.yaml before applying.
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}

# Download the pinned checkpoint revision
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy the TRT-LLM aggregated recipe
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
kubectl port-forward svc/nemotron-super-fp4-trtllm-agg-frontend 8000:8000 -n ${NAMESPACE}
```

Basic chat:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 128
  }'
```

Tool calling:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [{"role": "user", "content": "What is the weather in SF?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 256
  }'
```

Disable thinking:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "chat_template_kwargs": {"enable_thinking": false},
    "temperature": 1.0,
    "top_p": 0.95,
    "max_tokens": 64
  }'
```

## Model Details

- **Model**: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` pinned at revision `0f974885d5045d79ad56d539c5f8ee6cba2a1513`
- **Quantization**: NVFP4 with FP8 KV cache
- **Speculative decoding**: MTP enabled with `num_nextn_predict_layers: 3`
