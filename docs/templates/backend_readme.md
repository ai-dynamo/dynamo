---
orphan: true
---

# <Backend> Backend

<!-- 2-3 sentence overview of this backend integration -->

## Feature Matrix

| Feature | Status | Notes |
|---------|--------|-------|
| Prefill/Decode Disagg | ✅ | |
| KV Cache Routing | ✅ | |
| Multimodal | ✅ | Vision models |
| LoRA | 🚧 | Experimental |
| Speculative Decoding | ❌ | |

## Quick Start

### Prerequisites

- <Backend> installed (`pip install <backend>`)
- Model downloaded

### Python

```bash
python -m dynamo.<backend> --model <model-path>
```

### Kubernetes

```yaml
apiVersion: dynamo.nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
metadata:
  name: <backend>-example
spec:
  # Minimal configuration
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | required | Model path or HuggingFace ID |
| `--tensor-parallel-size` | `1` | GPUs for tensor parallelism |

## Next Steps

| Document | Path | Description |
|----------|------|-------------|
| `<Backend> Guide` | `<backend>_guide.md` | Advanced configuration |
| Backend Comparison | `../README.md` | Compare backends |

<!-- Convert to links: [vLLM Guide](vllm_guide.md) -->
