<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# GPT-OSS-120B Recipes

Deploy `openai/gpt-oss-120b` on Blackwell GPUs with Dynamo.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 4x GB200 | Aggregated | WideEP, ARM64 |
| [**trtllm/disagg**](trtllm/disagg/) | 5x Blackwell (GB200/B200) | Disaggregated | Prefill/Decode split |
| [**vllm/agg-snapshot-gms**](vllm/agg-snapshot-gms/) | 1x B200 | Aggregated | Experimental Snapshot and GMS restore |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with GB200 (Blackwell) GPUs
3. **HuggingFace token** with access to the model

The experimental vLLM configuration has additional Snapshot, GMS, driver, DRA,
storage, and image-build prerequisites in its leaf README.

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

# Deploy
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/gpt-oss-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- The TensorRT-LLM recipe requires ARM64 GB200 nodes; the vLLM Snapshot recipe
  requires an x86_64 B200 node.
- Update container image tags in `deploy.yaml` to match your Dynamo release
  version.
