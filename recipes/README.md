# Dynamo Production-Ready Recipes

Production-tested Kubernetes deployment recipes for LLM inference using NVIDIA Dynamo.

> **Prerequisites:** This guide assumes you have already installed the Dynamo Kubernetes Platform.
> If not, follow the **[Kubernetes Deployment Guide](../docs/kubernetes/README.md)** first.

## Available Recipes

### Multi-Feature Recipe

This recipe combines multiple Dynamo performance features (disaggregated serving + KV-aware routing):

| Model | Framework | Configuration | GPUs | Features |
|-------|-----------|---------------|------|----------|
| **[Qwen3-32B](qwen3-32b/)** | vLLM | Disagg + KV-Router | 16x H200 | **Disaggregated Serving + KV-Aware Routing** — includes benchmark comparison with real-world Mooncake traces |

### Aggregated & Disaggregated Recipes

These recipes demonstrate aggregated or disaggregated serving:

**GAIE Column**: Indicates whether the recipe includes integration with the [Gateway API Inference Extension (GAIE)](../deploy/inference-gateway/README.md) — a Kubernetes SIG project that extends the Gateway API for AI inference workloads, providing load balancing, model routing, and request management.

| Model | Framework | Mode | GPUs | Deployment | Benchmark Recipe | Notes | GAIE |
|-------|-----------|------|------|------------|------------------|-------|------|
| **[Llama-3-70B](llama-3-70b/vllm/agg/)** | vLLM | Aggregated | 4x H100/H200 | ✅ | ✅ | FP8 dynamic quantization | ✅ |
| **[Llama-3-70B](llama-3-70b/vllm/disagg-single-node/)** | vLLM | Disagg (Single-Node) | 8x H100/H200 | ✅ | ✅ | Prefill + Decode separation | ❌ |
| **[Llama-3-70B](llama-3-70b/vllm/disagg-multi-node/)** | vLLM | Disagg (Multi-Node) | 16x H100/H200 | ✅ | ✅ | 2 nodes, 8 GPUs each | ❌ |
| **[Qwen3-32B-FP8](qwen3-32b-fp8/trtllm/agg/)** | TensorRT-LLM | Aggregated | 2x GPU | ✅ | ✅ | FP8 quantization | ❌ |
| **[Qwen3-32B-FP8](qwen3-32b-fp8/trtllm/disagg/)** | TensorRT-LLM | Disaggregated | 8x GPU | ✅ | ✅ | Prefill + Decode separation | ❌ |
| **[Qwen3-235B-A22B-FP8](qwen3-235b-a22b-fp8/trtllm/agg/)** | TensorRT-LLM | Aggregated | 16x GPU | ✅ | ✅ | MoE model, TP4×EP4 | ❌ |
| **[Qwen3-235B-A22B-FP8](qwen3-235b-a22b-fp8/trtllm/disagg/)** | TensorRT-LLM | Disaggregated | 16x GPU | ✅ | ✅ | MoE model, Prefill + Decode | ❌ |
| **[GPT-OSS-120B](gpt-oss-120b/trtllm/agg/)** | TensorRT-LLM | Aggregated | 4x GB200 | ✅ | ✅ | Blackwell only, WideEP | ❌ |
| **[GPT-OSS-120B](gpt-oss-120b/trtllm/disagg/)** | TensorRT-LLM | Disaggregated | TBD | ❌ | ❌ | Engine configs only, no K8s manifest | ❌ |
| **[DeepSeek-R1](deepseek-r1/sglang/disagg-8gpu/)** | SGLang | Disagg WideEP | 16x H200 | ✅*1 | ❌ | TP=8 per worker, single-node | ❌ |
| **[DeepSeek-R1](deepseek-r1/sglang/disagg-16gpu/)** | SGLang | Disagg WideEP | 32x H200 | ✅*1 | ❌ | TP=16 per worker, multi-node | ❌ |
| **[DeepSeek-R1](deepseek-r1/trtllm/disagg/wide_ep/gb200/)** | TensorRT-LLM | Disagg WideEP (GB200) | 32+4 GB200 | ✅ | ✅ | Multi-node: 8 decode + 1 prefill nodes | ❌ |
| **[DeepSeek-R1](deepseek-r1/vllm/disagg/)** | vLLM | Disagg DEP16 | 32x H200 | ✅ | ❌ | Multi-node, data-expert parallel | ❌ |

*1: Please use `deepseek-r1/model-cache/model-download-sglang.yaml` to download the model into the PVC.

**Legend:**
- **Deployment**: ✅ = Complete `deploy.yaml` manifest available | ❌ = Missing or incomplete
- **Benchmark Recipe**: ✅ = Includes `perf.yaml` for running AIPerf benchmarks | ❌ = No benchmark recipe provided

## Recipe Structure

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

## Quick Start

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
cd recipes
# Update storageClassName in model-cache.yaml first!
kubectl apply -f <model>/model-cache/ -n ${NAMESPACE}

# Create model cache PVC
kubectl apply -f <model>/model-cache/model-download.yaml -n ${NAMESPACE}

# Wait for download to complete (may take 10-60 minutes depending on model size)
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s

# Monitor progress
kubectl logs -f job/model-download -n ${NAMESPACE}
```

**Step 2: Deploy Service**

Update the image in `<model>/<framework>/<mode>/deploy.yaml`.

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


## Example Deployments

### Llama-3-70B with vLLM (Aggregated)

```bash
export NAMESPACE=dynamo-demo
kubectl create namespace ${NAMESPACE}

# Create HF token secret
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="your-token" \
  -n ${NAMESPACE}

# Deploy
cd recipes
kubectl apply -f llama-3-70b/model-cache/ -n ${NAMESPACE}
kubectl apply -f llama-3-70b/model-cache/model-download.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/model-download -n ${NAMESPACE} --timeout=6000s
kubectl apply -f llama-3-70b/vllm/agg/deploy.yaml -n ${NAMESPACE}

# Test
kubectl port-forward svc/llama3-70b-agg-frontend 8000:8000 -n ${NAMESPACE}
```

### Inference Gateway (GAIE) Integration (Optional)**

For Llama-3-70B with vLLM (Aggregated), an example of integration with the Inference Gateway is provided.

First, deploy the Dynamo Graph per instructions above.

Then follow [Deploy Inference Gateway Section 2](../deploy/inference-gateway/README.md#2-deploy-inference-gateway) to install GAIE.

Update the containers.epp.image in the deployment file, i.e. llama-3-70b/vllm/agg/gaie/k8s-manifests/epp/deployment.yaml. It should match the release tag and be in the format `nvcr.io/nvidia/ai-dynamo/frontend:<version>` e.g. `nvcr.io/nvidia/ai-dynamo/frontend:0.9.0`
The recipe assumes you are using Kubernetes discovery backend and sets the `DYN_DISCOVERY_BACKEND` env variable in the epp deployment. If you want to use etcd enable the lines below and remove the DYN_DISCOVERY_BACKEND env var.
```bash
- name: ETCD_ENDPOINTS
  value: "dynamo-platform-etcd.$(PLATFORM_NAMESPACE):2379" #  update dynamo-platform to appropriate namespace
```

```bash
export DEPLOY_PATH=llama-3-70b/vllm/agg/
# DEPLOY_PATH=<model>/<framework>/<mode>/
kubectl apply -R -f "$DEPLOY_PATH/gaie/k8s-manifests" -n "$NAMESPACE"
```

### DeepSeek-R1 on GB200 (Multi-node)

See [deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml](deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml) for the complete multi-node WideEP configuration.

## Customization

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
image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:x.y.z
# Update version tag as needed
```

## Troubleshooting

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

## Related Documentation

- **[Kubernetes Deployment Guide](../docs/kubernetes/README.md)** - Platform installation and concepts
- **[API Reference](../docs/kubernetes/api_reference.md)** - DynamoGraphDeployment CRD specification
- **[vLLM Backend Guide](../docs/backends/vllm/README.md)** - vLLM-specific features
- **[SGLang Backend Guide](../docs/backends/sglang/README.md)** - SGLang-specific features
- **[TensorRT-LLM Backend Guide](../docs/backends/trtllm/README.md)** - TensorRT-LLM features
- **[Observability](../docs/kubernetes/observability/)** - Monitoring and logging
- **[Benchmarking Guide](../docs/benchmarks/benchmarking.md)** - Performance testing

## Contributing

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
