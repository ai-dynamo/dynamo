# GPT-OSS-120B Recipes

Production-ready deployment for **GPT-OSS-120B** using TensorRT-LLM on Blackwell (GB200) hardware.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 4x GB200 | Aggregated | WideEP, ARM64 |
| [**trtllm/disagg**](trtllm/disagg/) | 5x Blackwell (GB200/B200) | Disaggregated | Prefill/Decode split |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with GB200 (Blackwell) GPUs
3. **HuggingFace token** with access to the model

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
- This recipe requires ARM64 (GB200) nodes — it will not run on x86 Hopper/Ampere hardware
- Update the container image tag in `deploy.yaml` to match your Dynamo release version
