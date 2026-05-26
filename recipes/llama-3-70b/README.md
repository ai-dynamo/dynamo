# Llama-3.3-70B Recipes

Production-ready deployments for **Llama-3.3-70B-Instruct** using vLLM with FP8 dynamic quantization.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**vllm/agg**](vllm/agg/) | 4x H100/H200 | Aggregated | Single-node, TP4 |
| [**vllm/disagg-single-node**](vllm/disagg-single-node/) | 8x H100/H200 | Disaggregated | Prefill/decode separation on one node |
| [**vllm/disagg-multi-node**](vllm/disagg-multi-node/) | 16x H100/H200 | Disaggregated | 2 nodes, 8 GPUs each |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with H100 or H200 GPUs matching the configuration requirements
3. **HuggingFace token** with access to Llama models

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
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f vllm/disagg-single-node/deploy.yaml -n ${NAMESPACE}
# OR: kubectl apply -f vllm/disagg-multi-node/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/llama3-70b-agg-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`
- **Quantization**: FP8 dynamic (applied at runtime)
- **Context length**: Default model context

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` to match your cluster before deploying
- Model download takes approximately 15-30 minutes depending on network speed
- For GAIE (Gateway API Inference Extension) integration, `kubectl apply` the files from the corresponding subfolder i.e. [vllm/agg/gaie/](vllm/agg/gaie/)
