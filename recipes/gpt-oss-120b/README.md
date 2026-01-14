# GPT-OSS-120B with TensorRT-LLM

Production-ready recipes for GPT-OSS-120B using the TensorRT-LLM backend, optimized for NVIDIA GB200 (Blackwell) hardware.

## Available Configurations

| Mode | GPUs | Hardware | Description |
|------|------|----------|-------------|
| [Aggregated](trtllm/agg/) | 4x GB200 | Blackwell | WideEP deployment |
| [Disaggregated](trtllm/disagg/) | TBD | Blackwell | Engine configs only (no K8s manifest) |

## Hardware Requirements

> **Note**: This recipe is specifically designed for **GB200 (Blackwell)** GPUs with ARM64 architecture.

- **GPUs**: 4x NVIDIA GB200
- **Architecture**: ARM64
- **Interconnect**: NVLink/NVSwitch recommended

## Prerequisites

- **Model**: GPT-OSS-120B
- **Storage**: ~250GB for model weights
- **HuggingFace token**: Required for model download
- **GB200 cluster**: ARM64 Blackwell GPUs

## Quick Start

### Option A: Using run.sh Script

```bash
cd recipes
./run.sh --model gpt-oss-120b --framework trtllm agg
```

### Option B: Step-by-Step

```bash
export NAMESPACE=gpt-oss-demo
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
kubectl port-forward svc/gpt-oss-120b-agg-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

## Key Configuration

### Image Tag

This recipe requires the ARM64 TensorRT-LLM runtime container:

```yaml
# For Dynamo 0.5.1+:
image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1

# Pre-release (before 0.5.1):
image: nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:0.5.1-rc0.pre3
```

### Storage Class

The recipe does not specify a storage class by default. Update `model-cache/model-cache.yaml`:

```yaml
spec:
  storageClassName: "your-storage-class-name"
```

Find available storage classes:
```bash
kubectl get storageclass
```

## Deployment Mode

### Aggregated (4x GB200)

```bash
kubectl apply -f trtllm/agg/deploy.yaml -n ${NAMESPACE}
```

**Features**:
- WideEP (Wide Expert Parallelism) for efficient MoE inference
- Optimized for Blackwell architecture
- Single-command deployment

### Disaggregated

The disaggregated mode currently provides engine configuration files only. Full Kubernetes manifests are pending.

See [trtllm/disagg/](trtllm/disagg/) for available engine configurations.

## Benchmarking

```bash
kubectl apply -f trtllm/agg/perf.yaml -n ${NAMESPACE}
kubectl logs -f job/gpt-oss-120b-agg-perf -n ${NAMESPACE}
```

The benchmark uses a pinned version of aiperf for reproducible results.

## Troubleshooting

**Image pull errors**:
- Ensure you're using ARM64-compatible images
- Verify NGC credentials are configured

**Storage class not found**:
```bash
# List available storage classes
kubectl get storageclass
# Update model-cache.yaml with a valid class
```

**GPU scheduling issues**:
- Verify GB200 GPUs are available: `kubectl describe nodes | grep -i gpu`
- Check GPU operator status

## Notes

1. This recipe is optimized for GB200 hardware and may not work on other GPU architectures.
2. The benchmarking container uses a specific aiperf commit to ensure reproducibility.
3. WideEP requires proper NVLink/NVSwitch configuration for optimal performance.
