# Llama-3.3-70B with vLLM

Production-ready recipes for Llama-3.3-70B-Instruct using the vLLM backend with FP8 dynamic quantization.

## Available Configurations

| Mode | GPUs | Description | Best For |
|------|------|-------------|----------|
| [Aggregated](vllm/agg/) | 4x H100/H200 | Single TP4 worker | Low-latency, simple setup |
| [Disagg Single-Node](vllm/disagg-single-node/) | 8x H100/H200 | 2x prefill (TP2) + 1x decode (TP4) | High throughput, single node |
| [Disagg Multi-Node](vllm/disagg-multi-node/) | 16x H100/H200 | Multi-node prefill/decode separation | Maximum throughput |

## Prerequisites

- **Model**: `RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic`
- **Storage**: ~150GB for model weights
- **HuggingFace token**: Required for model download

## Quick Start

```bash
export NAMESPACE=llama-demo
kubectl create namespace ${NAMESPACE}

# 1. Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" -n ${NAMESPACE}

# 2. Update storageClassName in model-cache/model-cache.yaml, then:
kubectl apply -f model-cache/ -n ${NAMESPACE}

# 3. Wait for model download (~15-30 min for 70B)
kubectl logs -f job/model-download -n ${NAMESPACE}

# 4. Update image tag in deploy.yaml, then deploy:
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}

# 5. Test
kubectl port-forward svc/llama3-70b-agg-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## Key Configuration

### Image Tag
All `deploy.yaml` files use `image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`. Replace `my-tag` with your Dynamo version (e.g., `0.7.0`).

### GPU Memory Utilization

```yaml
# In deploy.yaml args:
--gpu-memory-utilization 0.90  # Default for most configs
```

> **Warning**: The UX review found that `0.95` can cause OOM on Llama-3.3-70B. Use `0.90` or lower if you see CUDA OOM errors during warmup.

### Tensor Parallelism

| Mode | Prefill TP | Decode TP | Total GPUs |
|------|-----------|-----------|------------|
| Aggregated | - | TP4 | 4 |
| Disagg Single-Node | TP2 (x2 replicas) | TP4 | 8 |
| Disagg Multi-Node | TP4 (x2 replicas) | TP8 | 16 |

## Deployment Modes

### Aggregated (4 GPUs)
Simplest deployment - single worker handles both prefill and decode.

```bash
kubectl apply -f vllm/agg/deploy.yaml -n ${NAMESPACE}
```

**When to use**: Development, low-traffic production, latency-sensitive workloads.

### Disaggregated Single-Node (8 GPUs)
Separates prefill and decode workers on a single node.

```bash
kubectl apply -f vllm/disagg-single-node/deploy.yaml -n ${NAMESPACE}
```

**When to use**: High throughput on single node, workloads with long input sequences.

**Key benefit**: Decode workers are isolated from prefill latency spikes.

### Disaggregated Multi-Node (16 GPUs)
Scales prefill and decode across multiple nodes.

```bash
kubectl apply -f vllm/disagg-multi-node/deploy.yaml -n ${NAMESPACE}
```

**When to use**: Maximum throughput, production workloads at scale.

## Inference Gateway (GAIE) Integration

The aggregated deployment includes optional GAIE integration for advanced routing:

```bash
# After deploying the base recipe:
kubectl apply -R -f vllm/agg/gaie/k8s-manifests -n ${NAMESPACE}
```

See [gaie/k8s-manifests/](vllm/agg/gaie/k8s-manifests/) for the full configuration.

## Benchmarking

Each deployment mode includes a `perf.yaml` for standardized benchmarking:

```bash
# Run after deployment is ready
kubectl apply -f vllm/agg/perf.yaml -n ${NAMESPACE}

# Monitor progress
kubectl logs -f job/llama3-70b-agg-perf -n ${NAMESPACE}
```

**Default benchmark config**:
- Input sequence length: 8192 tokens
- Output sequence length: 1024 tokens
- Concurrency: 16 per GPU

## Troubleshooting

**OOM during warmup**:
```yaml
# Reduce gpu-memory-utilization:
--gpu-memory-utilization 0.85
```

**Pods stuck in Pending**:
```bash
kubectl describe pod <pod-name> -n ${NAMESPACE}
# Check for GPU availability and resource constraints
```

**Model download timeout**:
```bash
# Increase job timeout or check HF token permissions
kubectl logs job/model-download -n ${NAMESPACE}
```
