# Qwen3-32B-FP8 Recipes

Production-ready deployments for **Qwen3-32B** with FP8 quantization using TensorRT-LLM.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 2x GPU | Aggregated | TP2, round-robin routing |
| [**trtllm/disagg**](trtllm/disagg/) | 8x GPU | Disaggregated | Prefill/decode separation |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with H100/H200/A100 GPUs
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
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=1800s

# Deploy (choose one configuration)
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/qwen3-32b-fp8-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-32B-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `Qwen/Qwen3-32B-FP8`
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Quantization**: FP8
- **Tensor Parallel**: 2

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- The aggregated config uses CUDA graphs for optimized inference
- KV cache uses FP8 dtype for memory efficiency
