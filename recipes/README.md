# Dynamo Model Serving Recipes

This repository contains production-ready recipes for deploying large language models using the Dynamo platform. Each recipe includes deployment configurations, performance benchmarking, and model caching setup.

## 📋 Available Models

| Model Family    | Framework | Deployment Mode      | GPU Requirements | Status | Benchmark |
|-----------------|-----------|---------------------|------------------|--------|-----------|
| llama-3-70b     | vllm      | agg                 | 4x H100/H200     | ✅     | ✅        |
| llama-3-70b     | vllm      | disagg-single-node  | 4x H100/H200     | ✅     | ✅        |
| llama-3-70b     | vllm      | disagg-multi-node   | 4x H100/H200     | ✅     | ✅        |
| gpt-oss-120b    | trtllm    | agg                 | 4x GB200         | ✅     | ✅        |
| deepseek-r1     | sglang    | disagg (8 GPU)      | 8x H200          | 🚧     | 🚧        |
| deepseek-r1     | sglang    | disagg (16 GPU)     | 16x H200         | 🚧     | 🚧        |

**Legend:**
- ✅ Fully functional and tested
- 🚧 Under development or requires manual setup
- ❌ Not available

## 🚀 Quick Start

### Option 1: Automated Deployment

Use the `run.sh` script for automated deployment:

```bash
# Deploy Llama-3-70B with vLLM in aggregated mode
./run.sh --model llama-3-70b --framework vllm agg

# Deploy GPT-OSS-120B with TensorRT-LLM
./run.sh --model gpt-oss-120b --framework trtllm agg

# Deploy with existing model cache (skip download)
./run.sh --model llama-3-70b --framework vllm agg --skip-model-cache
```

### Option 2: Manual Deployment

For more control or troubleshooting, deploy manually:

```bash
# 1. Set up environment
export NAMESPACE=your-namespace

# 2. Create model cache and download model
kubectl apply -n $NAMESPACE -f <model>/model-cache/

# 3. Deploy the model
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/deploy.yaml

# 4. Run performance benchmark
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/perf.yaml
```

## 📋 Prerequisites

### 1. Environment Setup

Create a Kubernetes namespace and set environment variable:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Dynamo Platform

Install the Dynamo Cloud Platform following the [Quickstart Guide](../docs/kubernetes/README.md).

### 3. GPU Cluster

Ensure your Kubernetes cluster has:
- GPU nodes with appropriate GPU types (see model requirements above)
- GPU operator installed
- Sufficient GPU memory and compute resources

### 4. Container Registry Access

Ensure access to NVIDIA container registry for runtime images:
- `nvcr.io/nvidia/ai-dynamo/vllm-runtime:x.y.z`
- `nvcr.io/nvidia/ai-dynamo/trtllm-runtime:x.y.z`

### 5. HuggingFace Access

Set up HuggingFace token for model access:

```bash
# Update the token in the secret file
vim hf_hub_secret/hf_hub_secret.yaml

# Apply the secret
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}
```

### 6. Storage Configuration

Configure persistent storage for model caching:

```bash
# Check available storage classes
kubectl get storageclass

# Update storage class in model cache files
# Edit: <model>/model-cache/model-cache.yaml
# Replace "your-storage-class-name" with your actual storage class
```

## Usage

### Using run.sh Script

The `run.sh` script provides a unified interface for all models:

```bash
./run.sh [OPTIONS] --model <model> --framework <framework> <deployment-type>
```

**Arguments:**
- `<deployment-type>`: Deployment mode (agg, disagg-single-node, disagg-multi-node)

**Required Options:**
- `--model <model>`: Model name (llama-3-70b, gpt-oss-120b, deepseek-r1)
- `--framework <framework>`: Backend framework (vllm, trtllm, sglang)

**Optional Options:**
- `--namespace <namespace>`: Kubernetes namespace (default: dynamo)
- `--skip-model-cache`: Skip model download (assumes cache exists)
- `--dry-run`: Show commands without executing
- `-h, --help`: Show help message

**Environment Variables:**
- `NAMESPACE`: Kubernetes namespace (default: dynamo)

### Examples

```bash
# Basic deployment
./run.sh --model llama-3-70b --framework vllm agg

# With custom namespace
./run.sh --namespace my-namespace --model llama-3-70b --framework vllm agg

# Skip model download (use existing cache)
./run.sh --skip-model-cache --model llama-3-70b --framework vllm agg

# Dry run to see what would be executed
./run.sh --dry-run --model llama-3-70b --framework vllm agg

# Disaggregated deployment
./run.sh --model llama-3-70b --framework vllm disagg-single-node
```

### Manual Deployment Steps

For each model, follow this consistent pattern:

#### Step 1: Prepare Model Cache

```bash
# Navigate to model directory
cd <model-name>/

# Update storage class in model-cache.yaml if needed
vim model-cache/model-cache.yaml

# Create PVC and download model
kubectl apply -n $NAMESPACE -f model-cache/
```

#### Step 2: Deploy Model

```bash
# Deploy the model service
kubectl apply -n $NAMESPACE -f <framework>/<deployment-mode>/deploy.yaml

# Wait for deployment to be ready
kubectl wait --for=condition=available deployment/<deployment-name> -n $NAMESPACE --timeout=600s
```

#### Step 3: Run Performance Benchmark

```bash
# Launch benchmark job
kubectl apply -n $NAMESPACE -f <framework>/<deployment-mode>/perf.yaml

# Monitor benchmark progress
kubectl logs -f job/<benchmark-job-name> -n $NAMESPACE
```

### Model-Specific Examples

#### Llama-3-70B

```bash
# Automated
./run.sh --model llama-3-70b --framework vllm agg

# Manual
cd llama-3-70b/
kubectl apply -n $NAMESPACE -f model-cache/
kubectl apply -n $NAMESPACE -f vllm/agg/deploy.yaml
kubectl apply -n $NAMESPACE -f vllm/agg/perf.yaml
```

#### GPT-OSS-120B

```bash
# Automated
./run.sh --model gpt-oss-120b --framework trtllm agg

# Manual
cd gpt-oss-120b/
kubectl apply -n $NAMESPACE -f model-cache/
kubectl apply -n $NAMESPACE -f trtllm/agg/deploy.yaml
kubectl apply -n $NAMESPACE -f trtllm/agg/perf.yaml
```

#### DeepSeek-R1 (Manual Setup Required)

```bash
# Note: Requires custom container build
# See deepseek-r1/sglang-wideep/README.md for container setup

cd deepseek-r1/
kubectl apply -n $NAMESPACE -f model_cache/
kubectl apply -n $NAMESPACE -f sglang-wideep/tep8p-dep8d-disagg.yaml
```

## 🔧 Configuration

### Storage Class Configuration

Before deploying, update the storage class in model cache files:

```yaml
# In <model>/model-cache/model-cache.yaml
spec:
  storageClassName: "your-actual-storage-class"  # Replace this
```

### Resource Requirements

Each model has specific GPU and memory requirements:

| Model         | GPUs | GPU Memory | Shared Memory | Storage |
|---------------|------|------------|---------------|---------|
| llama-3-70b   | 4    | 80GB each  | 20Gi          | 100Gi   |
| gpt-oss-120b  | 4    | 80GB each  | 80Gi          | 100Gi   |
| deepseek-r1   | 8-16 | 80GB each  | 80Gi          | 1000Gi  |

## 🔍 Monitoring and Troubleshooting

### Check Deployment Status

```bash
# Check pod status
kubectl get pods -n $NAMESPACE

# Check deployment status
kubectl get deployments -n $NAMESPACE

# View logs
kubectl logs -f deployment/<deployment-name> -n $NAMESPACE
```

### Common Issues

1. **Storage Class Not Found**
   ```bash
   # List available storage classes
   kubectl get storageclass
   # Update model-cache.yaml with correct storage class
   ```

2. **Insufficient GPU Resources**
   ```bash
   # Check GPU availability
   kubectl describe nodes | grep nvidia.com/gpu
   ```

3. **Model Download Failures**
   ```bash
   # Check HuggingFace token secret
   kubectl get secret hf-token-secret -n $NAMESPACE
   # Check model download job logs
   kubectl logs job/model-download-<model> -n $NAMESPACE
   ```

4. **Container Image Pull Errors**
   ```bash
   # Ensure access to NVIDIA container registry
   kubectl get pods -n $NAMESPACE
   kubectl describe pod <pod-name> -n $NAMESPACE
   ```

### Validation Commands

```bash
# Validate prerequisites
kubectl get storageclass
kubectl get secret hf-token-secret -n $NAMESPACE
kubectl describe nodes | grep nvidia.com/gpu

# Test model endpoint (after deployment)
kubectl port-forward service/<service-name> 8000:8000 -n $NAMESPACE
curl http://localhost:8000/v1/models
```

## Additional Resources

- [Dynamo Platform Documentation](../docs/kubernetes/README.md)
- [Model-specific READMEs](./): Each model directory contains detailed configuration notes
- [Performance Tuning Guide](../docs/performance/): Optimization recommendations
- [Troubleshooting Guide](../docs/troubleshooting/): Common issues and solutions

## Contributing

When adding new recipes, ensure they follow the standard structure:
```
<model-name>/
├── model-cache/
│   ├── model-cache.yaml
│   └── model-download.yaml
├── <framework>/
│   └── <deployment-mode>/
│       ├── deploy.yaml
│       └── perf.yaml
└── README.md (optional)
```