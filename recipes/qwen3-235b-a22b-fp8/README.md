# Qwen3-235B-A22B-FP8 Recipes

Production-ready deployments for **Qwen3-235B-A22B** (MoE model with 22B active parameters) using TensorRT-LLM.

## Available Configurations

| Configuration | GPUs | Mode | Description |
|--------------|------|------|-------------|
| [**trtllm/agg**](trtllm/agg/) | 16x GPU | Aggregated | TP4, EP4, KV-aware routing |
| [**trtllm/disagg**](trtllm/disagg/) | 16x GPU | Disaggregated | Prefill/decode separation |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with Blackwell GPUs (B100/B200; SM100+) — see [Hardware Requirements](#hardware-requirements)
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

This recipe uses `moe_config.backend: DEEPGEMM`, which requires **Blackwell GPUs (SM100+, e.g. B100/B200)**.
DeepGEMM's FP8 grouped-GEMM kernels are designed for SM100/SM103 only and will crash on Hopper (SM90).

> **Note:** To run on Hopper (H100/H200, SM90), remove the `moe_config` block from the ConfigMaps in
> `trtllm/agg/deploy.yaml` and `trtllm/disagg/deploy.yaml`. This falls back to the default MoE backend at a modest throughput reduction.

| Configuration | GPUs | Min GPU VRAM (Total) |
|--------------|------|----------------------|
| Aggregated | 16x B100/B200 | ~1.3TB |
| Disaggregated | 16x B100/B200 | ~1.3TB |

## Notes

- Update `storageClassName` in `model-cache/model-cache.yaml` before deploying
- Model download may take 30-60 minutes
- Uses KV-aware routing for efficient cache utilization
- Chunked prefill enabled for aggregated mode (disabled for disaggregated)
