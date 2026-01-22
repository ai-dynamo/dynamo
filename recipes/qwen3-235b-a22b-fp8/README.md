# Qwen3-235B-A22B-FP8 Recipes

Production-ready deployments for **Qwen3-235B-A22B** (MoE model with 22B active parameters) using TensorRT-LLM.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 16x GPU | Aggregated | TP4, EP4, KV-aware routing |
| [**trtllm/disagg**](trtllm/disagg/) | 16x GPU | Disaggregated | Prefill/decode separation |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with H100/H200 GPUs (high memory recommended)
3. **HuggingFace token** with access to Qwen models

## Quick Start

```bash
# Set namespace
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}

# Download model (update storageClassName in model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy (choose one configuration)
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/qwen3-235b-a22b-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-235B-A22B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `Qwen/Qwen3-235B-A22B-FP8`
- **Architecture**: 235B parameter Mixture-of-Experts (MoE)
- **Active parameters**: ~22B per token
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Parallelism**: TP4 × EP4 (Expert Parallel)

## Hardware Requirements

This is a large MoE model requiring significant GPU resources:

| Configuration | GPUs | Min GPU VRAM (Total) |
|--------------|------|----------------------|
| Aggregated | 16x H100/H200 | ~1.3TB |
| Disaggregated | 16x H100/H200 | ~1.3TB |

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- Model download may take 30-60 minutes
- Uses KV-aware routing for efficient cache utilization
- Chunked prefill enabled for aggregated mode (disabled for disaggregated)
