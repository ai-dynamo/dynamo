# Qwen3-32B-FP8 with TensorRT-LLM

Production-ready recipes for Qwen3-32B with FP8 quantization using the TensorRT-LLM backend.

## Available Configurations

| Mode | GPUs | Prefill | Decode | Description |
|------|------|---------|--------|-------------|
| [Aggregated](trtllm/agg/) | 4x | - | 2x TP2 workers | Simple deployment |
| [Disaggregated](trtllm/disagg/) | 8x | 4x TP1 | 2x TP2 | Prefill/decode separation |

## Prerequisites

- **Model**: `Qwen/Qwen3-32B-FP8`
- **Storage**: ~70GB for model weights
- **HuggingFace token**: Required for model download

## Quick Start

```bash
export NAMESPACE=qwen-demo
kubectl create namespace ${NAMESPACE}

# 1. Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}

# 2. Update storageClassName in model-cache/model-cache.yaml, then:
kubectl apply -f model-cache/ -n ${NAMESPACE}

# 3. Wait for model download
kubectl logs -f job/model-download -n ${NAMESPACE}

# 4. Update image tag in deploy.yaml, then deploy:
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}

# 5. Test
kubectl port-forward svc/qwen3-32b-fp8-agg-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## Key Configuration

### Image Tag
Replace `my-tag` with your Dynamo version:
```yaml
image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.7.0
```

### Engine Configuration

The recipes include embedded ConfigMaps with TRT-LLM engine settings:

```yaml
# Key settings in the ConfigMap:
tensor_parallel_size: 2
max_batch_size: 128
max_num_tokens: 7800
max_seq_len: 7800
kv_cache_config:
  free_gpu_memory_fraction: 0.7
  dtype: fp8
```

### GPU Memory

```yaml
--free-gpu-memory-fraction 0.9  # In worker args
```

Reduce to `0.7-0.8` if you encounter OOM errors.

## Deployment Modes

### Aggregated (4 GPUs)

```bash
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

**Architecture**: 2x TP2 workers with round-robin routing.

### Disaggregated (8 GPUs)

```bash
kubectl apply -f trtllm/disagg/deploy.yaml -n ${NAMESPACE}
```

**Architecture**:
- **Prefill**: 4x TP1 workers (optimized for throughput)
- **Decode**: 2x TP2 workers (optimized for latency)

**When to use**: Workloads with long input sequences where you want to isolate decode latency from prefill bursts.

## TensorRT-LLM Specific Settings

### Environment Variables

```yaml
env:
  - name: TRTLLM_ENABLE_PDL
    value: "1"  # Enable PDL for better performance
  - name: TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL
    value: "True"  # Required for some model configurations
```

### CUDA Graph Configuration

The ConfigMap includes CUDA graph batch sizes for optimal performance:

```yaml
cuda_graph_config:
  enable_padding: true
  batch_sizes: [1, 2, 4, 8, 16, 32, 64, 128]
```

## Benchmarking

```bash
kubectl apply -f trtllm/agg/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/qwen3-32b-fp8-agg-perf -n ${NAMESPACE}
```

## Troubleshooting

**Engine compilation takes too long**:
TRT-LLM compiles engines on first startup. This can take 10-30 minutes. Check logs:
```bash
kubectl logs -f <worker-pod> -n ${NAMESPACE}
```

**NCCL errors**:
Usually indicates OOM. Reduce `free_gpu_memory_fraction` in the ConfigMap.

**Pods crash during warmup**:
Increase liveness/readiness probe timeouts or check resource limits.
