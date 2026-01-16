# DeepSeek-R1 Recipes

Production-ready deployments for **DeepSeek-R1** (671B MoE) across multiple backends and hardware configurations.

## Available Configurations

| Configuration | GPUs | Backend | Mode | Description |
|--------------|------|---------|------|-------------|
| [**sglang/disagg-8gpu**](sglang/disagg-8gpu/) | 8x H200 | SGLang | Disaggregated WideEP | Single-node, expert parallel |
| [**sglang/disagg-16gpu**](sglang/disagg-16gpu/) | 16x H200 | SGLang | Disaggregated WideEP | Multi-node, expert parallel |
| [**vllm/disagg**](vllm/disagg/) | 16x H200 | vLLM | Disaggregated DEP16 | Multi-node, data-expert parallel |
| [**trtllm/disagg/wide_ep/gb200**](trtllm/disagg/wide_ep/gb200/) | 36x GB200 | TensorRT-LLM | Disaggregated WideEP | 8 decode + 1 prefill nodes |

## Prerequisites

1. **Dynamo Platform installed** — See [Kubernetes Deployment Guide](../../docs/kubernetes/README.md)
2. **GPU cluster** with H200 or GB200 GPUs matching the configuration requirements
3. **HuggingFace token** with access to DeepSeek models
4. **High-bandwidth networking** — InfiniBand or RoCE recommended for multi-node deployments

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
# For SGLang deployments:
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download-sglang.yaml -n ${NAMESPACE}

# For vLLM/TRT-LLM deployments:
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}

# Wait for download (this is a large model - may take 1+ hours)
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=7200s

# Deploy (choose one configuration)
kubectl apply -f sglang/disagg-8gpu/deploy.yaml -n ${NAMESPACE}
```

## Test the Deployment

```bash
# Port-forward the frontend (service name varies by deployment)
kubectl port-forward svc/sgl-dsr1-8gpu-frontend 8000:8000 -n ${NAMESPACE}

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Model Details

- **Model**: `deepseek-ai/DeepSeek-R1`
- **Architecture**: 671B parameter Mixture-of-Experts (MoE)
- **Active parameters**: ~37B per token
- **Recommended**: FP8 quantization for production deployments

## Hardware Requirements

DeepSeek-R1 is a very large model requiring significant GPU memory:

| Configuration | Min GPU Memory | Recommended |
|--------------|----------------|-------------|
| 8x H200 | 640GB total | H200 SXM (141GB each) |
| 16x H200 | 1.28TB total | H200 SXM (141GB each) |
| 36x GB200 | ~2.5TB total | GB200 NVL72 |

## Notes

- **Model download time**: DeepSeek-R1 is ~1.3TB; expect 1-2 hours for download
- **NCCL errors**: Usually indicate OOM. Reduce `--mem-fraction-static` in worker args
- **Multi-node**: Requires InfiniBand/IBGDA enabled. See [vLLM EP docs](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- **Storage class**: Update `storageClassName` in `model-cache/model-cache.yaml` before deploying

## Backend-Specific Notes

### SGLang
- Uses WideEP (Wide Expert Parallel) for efficient MoE inference
- See [sglang/README.md](sglang/README.md) for SGLang-specific configuration

### vLLM
- Uses DEP (Data-Expert Parallel) with hybrid load balancing
- See [vllm/disagg/README.md](vllm/disagg/README.md) for detailed setup

### TensorRT-LLM
- Requires FP4 quantized checkpoint
- GB200-specific optimizations
