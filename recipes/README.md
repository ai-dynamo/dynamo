# Dynamo Production-Ready Recipes

Production-tested Kubernetes deployment recipes for LLM inference using NVIDIA Dynamo.

> **Prerequisites:** This guide assumes you have already installed the Dynamo Kubernetes Platform.  
> If not, follow the **[Kubernetes Deployment Guide](../docs/kubernetes/README.md)** first.

## 📊 Available Recipes

| Model | Framework | Mode | GPUs | Deployment | Benchmark Recipe | Notes |
|-------|-----------|------|------|------------|------------------|-------|
| **Llama-3-70B** | vLLM | Aggregated | 4x H100/H200 | ✅ | ✅ | FP8 dynamic quantization |
| **Llama-3-70B** | vLLM | Disagg (Single-Node) | 8x H100/H200 | ✅ | ✅ | Prefill + Decode separation |
| **Llama-3-70B** | vLLM | Disagg (Multi-Node) | 16x H100/H200 | ✅ | ✅ | 2 nodes, 8 GPUs each |
| **Qwen3-32B-FP8** | TensorRT-LLM | Aggregated | 4x GPU | ✅ | ✅ | FP8 quantization |
| **Qwen3-32B-FP8** | TensorRT-LLM | Disaggregated | 8x GPU | ✅ | ✅ | Prefill + Decode separation |
| **GPT-OSS-120B** | TensorRT-LLM | Aggregated | 4x GB200 | ✅ | ✅ | Blackwell only, WideEP |
| **GPT-OSS-120B** | TensorRT-LLM | Disaggregated | TBD | ❌ | ❌ | Engine configs only, no K8s manifest |
| **DeepSeek-R1** | SGLang | Disagg WideEP | 8x H200 | ✅ | ❌ | Benchmark recipe pending |
| **DeepSeek-R1** | SGLang | Disagg WideEP | 16x H200 | ✅ | ❌ | Benchmark recipe pending |
| **DeepSeek-R1** | TensorRT-LLM | Disagg WideEP (GB200) | 32+4 GB200 | ✅ | ✅ | Multi-node: 8 decode + 1 prefill nodes |

**Legend:**
- **Deployment**: ✅ = Complete `deploy.yaml` manifest available | ❌ = Missing or incomplete
- **Benchmark Recipe**: ✅ = Includes `perf.yaml` for running AIPerf benchmarks | ❌ = No benchmark recipe provided

## 📁 Recipe Structure

Each complete recipe follows this standard structure:

```
<model-name>/
├── README.md (optional)           # Model-specific deployment notes
├── model-cache/
│   ├── model-cache.yaml          # PersistentVolumeClaim for model storage
│   └── model-download.yaml       # Job to download model from HuggingFace
└── <framework>/                  # vllm, sglang, or trtllm
    └── <deployment-mode>/        # agg, disagg, disagg-single-node, etc.
        ├── deploy.yaml           # Complete DynamoGraphDeployment manifest
        └── perf.yaml (optional)  # AIPerf benchmark job
```

## 🚀 Quick Start

### Prerequisites

**1. Dynamo Platform Installed**

The recipes require the Dynamo Kubernetes Platform to be installed. Follow the installation guide:

- **[Kubernetes Deployment Guide](../docs/kubernetes/README.md)** - Quickstart (~10 minutes)
- **[Detailed Installation Guide](../docs/kubernetes/installation_guide.md)** - Advanced options

**2. GPU Cluster Requirements**

Ensure your cluster has:
- GPU nodes matching recipe requirements (see table above)
- GPU operator installed
- Appropriate GPU drivers and container runtime

**3. HuggingFace Access**

Configure authentication to download models:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}

# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token-here" \
  -n ${NAMESPACE}
```

**4. Storage Configuration**

Update the `storageClassName` in `<model>/model-cache/model-cache.yaml` to match your cluster:

```bash
# Find your storage class name
kubectl get storageclass

# Edit the model-cache.yaml file and update:
# spec:
#   storageClassName: "your-actual-storage-class"
```

### Deploy a Recipe

**Step 1: Download Model**

```bash
# Update storageClassName in model-cache.yaml first!
kubectl apply -f <model>/model-cache/ -n ${NAMESPACE}

# Wait for download to complete (may take 10-60 minutes depending on model size)
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s

# Monitor progress
kubectl logs -f job/model-download -n ${NAMESPACE}
```

**Step 2: Deploy Service**

```bash
kubectl apply -f <model>/<framework>/<mode>/deploy.yaml -n ${NAMESPACE}

# Check deployment status
kubectl get dynamographdeployment -n ${NAMESPACE}

# Check pod status
kubectl get pods -n ${NAMESPACE}

# Wait for pods to be ready
kubectl wait --for=condition=ready pod -l nvidia.com/dynamo-graph-deployment-name=<deployment-name> -n ${NAMESPACE} --timeout=600s
```

**Step 3: Test Deployment**

```bash
# Port forward to access the service locally
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}

# In another terminal, test the endpoint
curl http://localhost:8000/v1/models

# Send a test request
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name>",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Step 4: Run Benchmark (Optional)**

```bash
# Only if perf.yaml exists in the recipe directory
kubectl apply -f <model>/<framework>/<mode>/perf.yaml -n ${NAMESPACE}

# Monitor benchmark progress
kubectl logs -f job/<benchmark-job-name> -n ${NAMESPACE}

# View results after completion
kubectl logs job/<benchmark-job-name> -n ${NAMESPACE} | tail -50
```

## 📖 Example Deployments

### Llama-3-70B with vLLM (Aggregated)

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}

# Deploy
kubectl apply -f llama-3-70b/model-cache/ -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s
kubectl apply -f llama-3-70b/vllm/agg/deploy.yaml -n ${NAMESPACE}

# Test
kubectl port-forward svc/llama3-70b-agg-frontend 8000:8000 -n ${NAMESPACE}
```

### DeepSeek-R1 on GB200 (Multi-node)

See [deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml](deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml) for the complete multi-node WideEP configuration.

## 🛠️ Customization

Each `deploy.yaml` contains:
- **ConfigMap**: Engine-specific configuration (embedded in the manifest)
- **DynamoGraphDeployment**: Kubernetes resource definitions
- **Resource limits**: GPU count, memory, CPU requests/limits
- **Image references**: Container images with version tags

### Key Customization Points

**Model Configuration:**
```yaml
# In deploy.yaml under worker args:
args:
  - python3 -m dynamo.vllm --model <your-model-path> --served-model-name <name>
```

**GPU Resources:**
```yaml
resources:
  limits:
    gpu: "4"  # Adjust based on your requirements
  requests:
    gpu: "4"
```

**Scaling:**
```yaml
services:
  VllmDecodeWorker:
    replicas: 2  # Scale to multiple workers
```

**Router Mode:**
```yaml
# In Frontend args:
args:
  - python3 -m dynamo.frontend --router-mode kv --http-port 8000
# Options: round-robin, kv (KV-aware routing)
```

**Container Images:**
```yaml
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1
# Update version tag as needed
```

## 🔧 Troubleshooting

### Common Issues

**Pods stuck in Pending:**
- Check GPU availability: `kubectl describe node <node-name>`
- Verify storage class exists: `kubectl get storageclass`
- Check resource requests vs. available resources

**Model download fails:**
- Verify HuggingFace token is correct
- Check network connectivity from cluster
- Review job logs: `kubectl logs job/model-download -n ${NAMESPACE}`

**Workers fail to start:**
- Check GPU compatibility (driver version, CUDA version)
- Verify image pull secrets if using private registries
- Review pod logs: `kubectl logs <pod-name> -n ${NAMESPACE}`

**For more troubleshooting:**
- [Kubernetes Deployment Guide](../docs/kubernetes/README.md#troubleshooting)
- [Observability Documentation](../docs/kubernetes/observability/)

## 📖 Related Documentation

- **[Kubernetes Deployment Guide](../docs/kubernetes/README.md)** - Platform installation and concepts
- **[API Reference](../docs/kubernetes/api_reference.md)** - DynamoGraphDeployment CRD specification
- **[vLLM Backend Guide](../docs/backends/vllm/README.md)** - vLLM-specific features
- **[SGLang Backend Guide](../docs/backends/sglang/README.md)** - SGLang-specific features
- **[TensorRT-LLM Backend Guide](../docs/backends/trtllm/README.md)** - TensorRT-LLM features
- **[Observability](../docs/kubernetes/observability/)** - Monitoring and logging
- **[Benchmarking Guide](../docs/benchmarks/benchmarking.md)** - Performance testing

## 🤝 Contributing

We welcome contributions of new recipes! See [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Recipe submission guidelines
- Required components checklist
- Testing and validation requirements
- Documentation standards

### Recipe Quality Standards

A production-ready recipe must include:
- ✅ Complete `deploy.yaml` with DynamoGraphDeployment
- ✅ Model cache PVC and download job
- ✅ Benchmark recipe (`perf.yaml`) for performance testing
- ✅ Verification on target hardware
- ✅ Documentation of GPU requirements
