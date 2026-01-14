# DeepSeek-R1

Production-ready recipes for DeepSeek-R1, a 671B parameter Mixture-of-Experts model, across multiple backends.

## Available Configurations

| Backend | Mode | GPUs | Hardware | Description |
|---------|------|------|----------|-------------|
| [SGLang](sglang/disagg-8gpu/) | Disagg WideEP | 8x H200 | Hopper | Single-node disaggregated |
| [SGLang](sglang/disagg-16gpu/) | Disagg WideEP | 16x H200 | Hopper | Multi-node disaggregated |
| [vLLM](vllm/disagg/) | Disaggregated | 16x H200 | Hopper | vLLM backend |
| [TRT-LLM](trtllm/disagg/wide_ep/gb200/) | Disagg WideEP | 36x GB200 | Blackwell | 8 decode + 1 prefill nodes |

## Model Characteristics

- **Architecture**: Mixture-of-Experts (MoE)
- **Total parameters**: 671B
- **Active parameters**: ~37B per token
- **Memory requirement**: Requires multi-GPU/multi-node deployment

## Prerequisites

- **Model**: `deepseek-ai/DeepSeek-R1`
- **Storage**: ~1.5TB for model weights
- **HuggingFace token**: Required
- **Networking**: High-bandwidth interconnect (IB/RoCE) recommended

## Model Download

Different backends may require different download configurations:

```bash
export NAMESPACE=deepseek-demo
kubectl create namespace ${NAMESPACE}

# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}

# For SGLang backend:
kubectl apply -f model-cache/model-download-sglang.yaml -n ${NAMESPACE}

# For vLLM/TRT-LLM backends:
kubectl apply -f model-cache/model-download.yaml -n ${NAMESPACE}

# Monitor progress (this will take a while for 1.5TB)
kubectl logs -f job/model-download -n ${NAMESPACE}
```

## Backend Selection Guide

### SGLang (Recommended for Hopper)
Best for H100/H200 deployments with WideEP support.

**Pros**:
- Native WideEP support
- Optimized for DeepSeek architecture
- DP-attention support

**See**: [sglang/README.md](sglang/README.md) for detailed instructions.

### vLLM
General-purpose backend with broad compatibility.

**Pros**:
- Familiar API
- Good community support

**See**: [vllm/disagg/README.md](vllm/disagg/README.md) for detailed instructions.

### TensorRT-LLM (GB200/Blackwell)
Optimized for NVIDIA Blackwell architecture.

**Pros**:
- Maximum performance on GB200
- WideEP with NIXL support

**See**: [trtllm/disagg/wide_ep/gb200/](trtllm/disagg/wide_ep/gb200/) for configuration.

## SGLang Quick Start (8x H200)

```bash
# 1. Download model (use SGLang-specific download)
kubectl apply -f model-cache/model-cache.yaml -n ${NAMESPACE}
kubectl apply -f model-cache/model-download-sglang.yaml -n ${NAMESPACE}

# 2. Wait for download
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=36000s

# 3. Update image in deploy.yaml, then:
kubectl apply -f sglang/disagg-8gpu/deploy.yaml -n ${NAMESPACE}

# 4. Test
kubectl port-forward svc/sgl-dsr1-8gpu-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## Key Configuration

### Image Tags
Replace `my-registry/sglang-runtime:my-tag` with your actual registry and tag:
```yaml
image: nvcr.io/nvidia/ai-dynamo/sglang-runtime:0.7.0
```

### WideEP Settings (SGLang)

```yaml
args:
  - --tp
  - "16"
  - --dp
  - "16"
  - --enable-dp-attention
  - --ep-size
  - "16"
```

### Memory Configuration

```yaml
--mem-fraction-static 0.75  # Reduce if you see NCCL/OOM errors
```

> **Warning**: If you see NCCL errors when sending requests, it's usually caused by OOM. Reduce `--mem-fraction-static`.

## Multi-Node Networking

DeepSeek-R1 deployments require high-bandwidth interconnect:

- **Minimum**: 100Gbps Ethernet
- **Recommended**: InfiniBand or RoCE
- **NVSHMEM**: Required for WideEP configurations

Ensure your Kubernetes cluster has proper RDMA/IB configuration.

## Troubleshooting

**NCCL errors during inference**:
```bash
# Usually OOM - reduce memory fraction
--mem-fraction-static 0.70
```

**Model download timeout**:
- 671B model requires 1.5TB+ transfer
- Use a dedicated download pod with extended timeout
- Consider pre-downloading to shared storage

**Pods stuck in Init**:
- Check network connectivity between nodes
- Verify NVSHMEM/IB drivers are installed
- Check GPU operator status

**Tokenization mismatches (SGLang)**:
```yaml
# Use SGLang's tokenizer to avoid issues:
--skip-tokenizer-init  # Let SGLang handle tokenization
```
