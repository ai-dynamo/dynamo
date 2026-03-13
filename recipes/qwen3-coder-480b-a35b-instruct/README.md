# Qwen3-Coder-480B-A35B-Instruct Recipes

Production-ready deployments for **Qwen3-Coder-480B** (MoE model with 35B active parameters) using TensorRT-LLM. Uses the NVIDIA FP4 quantized variant for reduced memory footprint.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/deploy.yaml) | 4x GPU | Aggregated | TP4, EP4, KV-aware routing |
| [**trtllm/agg (KVBM)**](trtllm/agg/deploy-kvbm.yaml) | 4x GPU | Aggregated + KVBM | Same as above with KV block manager |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with H100/H200 GPUs
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

# Download model (update storageClassName in model-cache.yaml first!)
kubectl apply -f model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=3600s

# Deploy (choose one configuration)
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
# OR with KVBM: kubectl apply -f trtllm/agg/deploy-kvbm.yaml -n ${NAMESPACE}
```

**Note:** If updating `model-download.yaml`, delete the existing job first (Job spec is immutable):
```bash
kubectl delete job model-download -n ${NAMESPACE} --ignore-not-found
```

## Test the Deployment

```bash
# Port-forward the frontend
kubectl port-forward svc/qwen3-coder-480b-agg-frontend 8000:8000 -n ${NAMESPACE}
# For KVBM: kubectl port-forward svc/qwen3-coder-480b-agg-kvbm-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Model Details

- **Model**: `nvidia/Qwen3-Coder-480B-A35B-Instruct-NVFP4`
- **Architecture**: 480B parameter Mixture-of-Experts (MoE)
- **Active parameters**: ~35B per token
- **Quantization**: FP4 via NVIDIA ModelOpt
- **Backend**: TensorRT-LLM (PyTorch backend)
- **Parallelism**: TP4 × EP4 (Expert Parallel)
- **Revision**: Pinned to `7cd997a9ba42019dd3da402a106744f9b50d26c6` for reproducible downloads

## Hardware Requirements

| Configuration | GPUs | Min GPU VRAM |
|--------------|------|--------------|
| Aggregated | 4x H100/H200 | ~320GB total |

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- Model download uses `fsGroup: 1000` so the cache is readable by Dynamo workers (non-root)
- If the cache was populated before `fsGroup` was added, run the fix-permissions job:
  ```bash
  kubectl apply -f model-cache/fix-cache-permissions.yaml -n ${NAMESPACE}
  kubectl wait --for=condition=Complete job/fix-model-cache-perms -n ${NAMESPACE} --timeout=300s
  kubectl delete pod -n ${NAMESPACE} -l nvidia.com/dynamo-component=TrtllmWorker
  ```
- GPU nodes typically require a toleration for `nvidia.com/gpu:NoSchedule`; the deploy manifests include this
- Model download may take 30-60 minutes
