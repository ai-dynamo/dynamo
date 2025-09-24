# Dynamo Quickstart Guide

Get up and running with NVIDIA Dynamo in minutes! This guide provides the fastest paths to deploy Dynamo for different use cases.

## Choose Deployment Path

| Use Case | Time to Deploy | Best For | Path |
|----------|----------------|----------|------|
| **Local Development** | 5 minutes | Testing, development, getting started | [Local Quickstart](#local-quickstart) |
| **Kubernetes Production** | 15-20 minutes | Production deployments, scaling | [Kubernetes Quickstart](#kubernetes-quickstart) |

---

## Local Quickstart

**Perfect for**: Development, testing, learning Dynamo concepts

### Prerequisites
- Ubuntu 24.04 (recommended) or compatible Linux
- NVIDIA GPU with CUDA support
- Docker & Docker Compose
- Python 3.9+

### 1. Install Dynamo

```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install Dynamo
uv venv venv
source venv/bin/activate
uv pip install "ai-dynamo[sglang]==0.5.0"  # or [vllm], [trtllm]
```

### 2. Start Infrastructure Services

Dynamo uses **etcd** and **NATS** for distributed communication at data center scale. Even for local development, these services are required for component discovery and message passing.

```bash
# Start etcd and NATS using Docker Compose
curl -fsSL -o docker-compose.yml https://raw.githubusercontent.com/ai-dynamo/dynamo/release/0.5.0/deploy/docker-compose.yml
docker compose -f docker-compose.yml up -d
```

**What this sets up:**
- **etcd**: Distributed key-value store for service discovery and metadata storage
- **NATS**: High-performance message broker for inter-component communication

### 3. Deploy Your First Model

**Terminal 1 - Start the Frontend:**
```bash
python -m dynamo.frontend --http-port 8000
```

**Terminal 2 - Start the Backend Worker:**
```bash
python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B
```

### 4. Test Your Deployment

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "max_tokens": 50}'
```

**Success!** You now have a working Dynamo deployment.

### Cleanup
```bash
# Stop Dynamo components (Ctrl+C in each terminal)
# Stop infrastructure services
docker compose -f docker-compose.yml down
```

---

<details>
<summary><strong>Framework-Specific Quickstarts</strong> (Click to expand)</summary>

### vLLM Backend
```bash
# Install
uv pip install "ai-dynamo[vllm]"

# Run
python -m dynamo.vllm --model Qwen/Qwen3-0.6B
```

### SGLang Backend
```bash
# Install dependencies
apt install -y libnuma-dev

# Install
uv pip install "ai-dynamo[sglang]"

# Run
python -m dynamo.sglang --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### TensorRT-LLM Backend

**Note**: TensorRT-LLM requires the NVIDIA PyTorch container as a base, which needs NGC login.

```bash
# 1. Login to NVIDIA NGC (required for PyTorch container)
docker login nvcr.io
# Enter your NGC username and API key when prompted

# 2. Install prerequisites
uv pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install "cuda-python>=12,<13"
sudo apt-get -y install libopenmpi-dev

# 3. Install
uv pip install "ai-dynamo[trtllm]"

# 4. Run
python -m dynamo.trtllm --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

**NGC Setup**: Get your NGC username and API key from [NGC Console](https://ngc.nvidia.com/setup/api-key)

</details>

---

## Kubernetes Quickstart

**Perfect for**: Production deployments, scaling, multi-node setups

### Prerequisites
- Kubernetes cluster (1.24+)
- NVIDIA GPU operator installed
- kubectl configured
- Helm 3.0+

### 1. Install Dynamo Platform

```bash
# Set environment
export NAMESPACE=dynamo-kubernetes
export RELEASE_VERSION=0.5.0

# Install CRDs
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-${RELEASE_VERSION}.tgz
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default

# Install Platform
helm fetch https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace
```

### 2. Deploy Your Model

Choose your backend and deployment pattern:

#### **Aggregated Serving** (Single-node, all-in-one)
Prefill and decode phases run on the same worker - simplest deployment pattern.

| Backend | Configuration | Deploy Command |
|---------|---------------|----------------|
| **vLLM** | [Aggregated](components/backends/vllm/deploy/agg.yaml) | `kubectl apply -f components/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}` |
| **vLLM** | [Aggregated + Router](components/backends/vllm/deploy/agg_router.yaml) | `kubectl apply -f components/backends/vllm/deploy/agg_router.yaml -n ${NAMESPACE}` |
| **SGLang** | [Aggregated](components/backends/sglang/deploy/agg.yaml) | `kubectl apply -f components/backends/sglang/deploy/agg.yaml -n ${NAMESPACE}` |
| **SGLang** | [Aggregated + Router](components/backends/sglang/deploy/agg_router.yaml) | `kubectl apply -f components/backends/sglang/deploy/agg_router.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Aggregated](components/backends/trtllm/deploy/agg.yaml) | `kubectl apply -f components/backends/trtllm/deploy/agg.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Aggregated + Router](components/backends/trtllm/deploy/agg_router.yaml) | `kubectl apply -f components/backends/trtllm/deploy/agg_router.yaml -n ${NAMESPACE}` |

#### **Disaggregated Serving** (Multi-node, specialized workers)
Prefill and decode phases run on separate workers - higher performance and scalability.

| Backend | Configuration | Deploy Command |
|---------|---------------|----------------|
| **vLLM** | [Disaggregated](components/backends/vllm/deploy/disagg.yaml) | `kubectl apply -f components/backends/vllm/deploy/disagg.yaml -n ${NAMESPACE}` |
| **vLLM** | [Disaggregated + Router](components/backends/vllm/deploy/disagg_router.yaml) | `kubectl apply -f components/backends/vllm/deploy/disagg_router.yaml -n ${NAMESPACE}` |
| **vLLM** | [Disaggregated + Planner](components/backends/vllm/deploy/disagg_planner.yaml) | `kubectl apply -f components/backends/vllm/deploy/disagg_planner.yaml -n ${NAMESPACE}` |
| **SGLang** | [Disaggregated](components/backends/sglang/deploy/disagg.yaml) | `kubectl apply -f components/backends/sglang/deploy/disagg.yaml -n ${NAMESPACE}` |
| **SGLang** | [Disaggregated + Planner](components/backends/sglang/deploy/disagg_planner.yaml) | `kubectl apply -f components/backends/sglang/deploy/disagg_planner.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Disaggregated](components/backends/trtllm/deploy/disagg.yaml) | `kubectl apply -f components/backends/trtllm/deploy/disagg.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Disaggregated + Router](components/backends/trtllm/deploy/disagg_router.yaml) | `kubectl apply -f components/backends/trtllm/deploy/disagg_router.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Disaggregated + Planner](components/backends/trtllm/deploy/disagg_planner.yaml) | `kubectl apply -f components/backends/trtllm/deploy/disagg_planner.yaml -n ${NAMESPACE}` |

#### **Multi-node Deployment** (Distributed across multiple nodes)
Scale disaggregated serving across multiple Kubernetes nodes for maximum performance.

| Backend | Configuration | Deploy Command |
|---------|---------------|----------------|
| **vLLM** | [Multi-node](components/backends/vllm/deploy/disagg-multinode.yaml) | `kubectl apply -f components/backends/vllm/deploy/disagg-multinode.yaml -n ${NAMESPACE}` |
| **SGLang** | [Multi-node](components/backends/sglang/deploy/disagg-multinode.yaml) | `kubectl apply -f components/backends/sglang/deploy/disagg-multinode.yaml -n ${NAMESPACE}` |
| **TensorRT-LLM** | [Multi-node](components/backends/trtllm/deploy/disagg-multinode.yaml) | `kubectl apply -f components/backends/trtllm/deploy/disagg-multinode.yaml -n ${NAMESPACE}` |

### 3. Test Your Deployment

```bash
# Check status
kubectl get dynamoGraphDeployment -n ${NAMESPACE}

# Test it
kubectl port-forward svc/agg-vllm-frontend 8000:8000 -n ${NAMESPACE}
curl http://localhost:8000/v1/models
```

**Success!** Your Dynamo deployment is running on Kubernetes.

### Cleanup
```bash
kubectl delete dynamoGraphDeployment agg-vllm -n ${NAMESPACE}
helm uninstall dynamo-platform -n ${NAMESPACE}
helm uninstall dynamo-crds --namespace default
```

---

## Next Steps

### For Local Development Users

**Dive deeper into Dynamo's architecture and Python development:**

- **[Architecture Guide](docs/architecture/)** - Understand Dynamo's design and components
- **[Disaggregated Serving](examples/basics/disaggregated_serving/)** - Try advanced serving patterns locally
- **[Multi-node Deployment](examples/basics/multinode/)** - Scale across multiple local nodes
- **[Custom Backend Examples](examples/custom_backend/)** - Build your own Dynamo components
- **[Runtime Examples](lib/bindings/python/README.md)** - Low-level Python<>Rust bindings
- **[KV-Aware Routing](docs/architecture/kv_cache_routing.md)** - Understand intelligent request routing

### For Kubernetes Production Users

**Production deployment and operations:**

- **[Kubernetes Documentation](docs/kubernetes/)** - Complete K8s deployment guide
- **[API Reference](docs/kubernetes/api_reference.md)** - DynamoGraphDeployment CRD specifications
- **[Installation Guide](docs/kubernetes/installation_guide.md)** - Detailed platform setup
- **[Monitoring Setup](docs/kubernetes/metrics.md)** - Observability and metrics
- **[Logging Configuration](docs/kubernetes/logging.md)** - Centralized logging setup
- **[Multi-node Deployment](docs/kubernetes/multinode-deployment.md)** - Scale across K8s nodes
- **[Performance Tuning](docs/benchmarks/)** - Optimize for your workload

---

## Troubleshooting

### Common Issues

**"Connection refused" errors:**
```bash
# Check if etcd and NATS are running
docker ps | grep -E "(etcd|nats)"

# Restart infrastructure services
docker compose -f docker-compose.yml down
docker compose -f docker-compose.yml up -d
```

**GPU not detected:**
```bash
# Check GPU availability
nvidia-smi

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Kubernetes deployment stuck:**
```bash
# Check pod status
kubectl get pods -n dynamo-kubernetes

# Check logs for any given component
kubectl logs -f deployment/agg-vllm-frontend -n dynamo-kubernetes
```

**Model download issues:**
```bash
# Set HuggingFace token for private models
export HUGGINGFACE_HUB_TOKEN=your_token_here

# Or use local model path
python -m dynamo.vllm --model /path/to/local/model
```

### Getting Help

- **[GitHub Issues](https://github.com/ai-dynamo/dynamo/issues)** - Report bugs and request features
- **[Discord Community](https://discord.gg/D92uqZRjCZ)** - Get help from the community
- **[Documentation](https://docs.nvidia.com/dynamo/latest/)** - Comprehensive guides and API docs

---

## System Requirements

For detailed compatibility information, see the [Support Matrix](docs/support_matrix.md).

