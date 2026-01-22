# Qwen3-235B-A22B-FP8 with TensorRT-LLM

Production-ready recipes for Qwen3-235B-A22B, a Mixture-of-Experts (MoE) model with FP8 quantization, using the TensorRT-LLM backend.

## Available Configurations

| Mode | GPUs | Workers | Description |
|------|------|---------|-------------|
| [Aggregated](trtllm/agg/) | 16x (4x TP4) | 4 replicas | Expert parallel across 4 GPUs per worker |
| [Disaggregated](trtllm/disagg/) | 20x | 4x prefill + 2x decode | Prefill/decode separation |

## Model Characteristics

- **Architecture**: Mixture-of-Experts (MoE)
- **Active parameters**: 22B per token (out of 235B total)
- **Expert parallelism**: Required for efficient inference
- **Memory**: ~500GB total weights, distributed across GPUs

## Prerequisites

- **Model**: `Qwen/Qwen3-235B-A22B-FP8`
- **Storage**: ~500GB for model weights
- **HuggingFace token**: Required for model download
- **Multi-node cluster**: Recommended for production

## Quick Start

```bash
export NAMESPACE=qwen-moe-demo
kubectl create namespace ${NAMESPACE}

# 1. Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}

# 2. Update storageClassName in model-cache/model-cache.yaml
# NOTE: Ensure you have ~500GB available
kubectl apply -f model-cache/ -n ${NAMESPACE}

# 3. Wait for model download (can take 1-2 hours for 235B)
kubectl logs -f job/model-download -n ${NAMESPACE}

# 4. Update image tag in deploy.yaml, then deploy:
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}

# 5. Test
kubectl port-forward svc/qwen3-235b-a22b-agg-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## Key Configuration

### Image Tag
Replace `my-tag` with your Dynamo version:
```yaml
image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.7.0
```

### MoE-Specific Settings

```yaml
# In the ConfigMap:
tensor_parallel_size: 4
moe_expert_parallel_size: 4  # Distribute experts across GPUs
moe_tensor_parallel_size: 1
```

### Memory Requirements

```yaml
sharedMemory:
  size: 256Gi  # Required for MoE expert communication
```

This model requires significant shared memory for cross-GPU expert communication.

## Engine Configuration

The recipes include optimized TRT-LLM settings for MoE:

```yaml
backend: pytorch
trust_remote_code: true
enable_chunked_prefill: true
build_config:
  max_batch_size: 128
  max_num_tokens: 8192
  max_seq_len: 8192
kv_cache_config:
  enable_block_reuse: true
  free_gpu_memory_fraction: 0.8
```

## Deployment Modes

### Aggregated (16 GPUs)

```bash
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

**Architecture**: 4x TP4 workers with expert parallelism.

**Resource per worker**:
- 4 GPUs
- 256Gi shared memory

### Disaggregated (20 GPUs)

```bash
kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

**Architecture**:
- **Prefill**: Optimized for long context processing
- **Decode**: Optimized for token generation latency

## Performance Considerations

### Expert Parallelism
MoE models benefit significantly from expert parallelism. The default configuration distributes experts across 4 GPUs:

```yaml
moe_expert_parallel_size: 4
```

### Chunked Prefill
Enabled by default to handle long sequences efficiently:

```yaml
enable_chunked_prefill: true
```

### KV Cache
Block reuse is enabled for better memory efficiency:

```yaml
kv_cache_config:
  enable_block_reuse: true
```

## Benchmarking

```bash
kubectl apply -f trtllm/agg/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/qwen3-235b-a22b-agg-perf -n ${NAMESPACE}
```

## Troubleshooting

**Model download fails or times out**:
- Ensure sufficient storage (~500GB)
- Check network bandwidth to HuggingFace
- Consider using a separate download pod with higher timeout

**OOM during expert routing**:
- Increase shared memory allocation
- Reduce `free_gpu_memory_fraction` in the ConfigMap

**Slow startup**:
- MoE models take longer to initialize due to expert weight distribution
- First startup includes engine compilation (30-60 minutes typical)

**NCCL timeout errors**:
- Check inter-node networking
- Ensure IB/RoCE is configured correctly for multi-node setups
