# Dynamo Model Serving Recipes

This repository contains production-ready recipes for deploying large language models using the Dynamo platform. Each recipe includes deployment configurations, performance benchmarking, and model caching setup.

## Available Models

| Model Family    | Framework | Deployment Mode      | GPU Requirements | Status | Benchmark |
|-----------------|-----------|---------------------|------------------|--------|-----------|
| llama-3-70b     | vllm      | agg                 | 4x H100/H200     | ✅     | ✅        |
| llama-3-70b     | vllm      | disagg (1 node)      | 8x H100/H200    | ✅     | ✅        |
| llama-3-70b     | vllm      | disagg (multi-node)     | 16x H100/H200    | ✅     | ✅        |
| deepseek-r1     | sglang    | disagg (1 node, wide-ep)     | 8x H200          | ✅     | 🚧        |
| deepseek-r1     | sglang    | disagg (multi-node, wide-ep)     | 16x H200        | ✅     | 🚧        |
| gpt-oss-120b    | trtllm    | agg                 | 4x GB200         | ✅     | ✅        |

**Legend:**
- ✅ Functional
- 🚧 Under development

## Quick Start

Choose your preferred deployment method:

### Option 1: Automated Deployment

Use the `run.sh` script for fully automated deployment:

```bash
# Set up environment
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}

# Configure HuggingFace token
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}

# Deploy model with automatic download and benchmarking
./run.sh --model llama-3-70b --framework vllm agg

# Or skip model download if model has been already downloaded to model cache PVC
./run.sh --skip-model-cache --model llama-3-70b --framework vllm agg
```

### Option 2: Manual Deployment

For step-by-step control, deploy manually:

```bash
# 1. Set up environment
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}

# 2. Configure storage and secrets (see Prerequisites)
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}

# 3. Download model (see Model Download section)
kubectl apply -n $NAMESPACE -f <model>/model-cache/

# 4. Deploy model (see Deployment section)
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/deploy.yaml

# 5. Run benchmarks (optional, if perf.yaml exists)
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/perf.yaml
```

## Prerequisites

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
- `nvcr.io/nvidia/ai-dynamo/sglang-runtime:x.y.z`

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
```

> Note: Update storage class in model cache files
> Edit: <model>/model-cache/model-cache.yaml

# Replace "your-storage-class-name" with your actual storage class

## Model Download

Before deploying any model, you must download the model weights to a persistent volume. This process involves creating a PVC and running a download job.

### Step 1: Configure Storage Class

Update the storage class in the model cache configuration:

```bash
# Navigate to your chosen model directory
cd <model-name>/

# Edit the model cache configuration
vim model-cache/model-cache.yaml

# Replace "your-storage-class-name" with your actual storage class
```

### Step 2: Create Model Cache PVC

Create the persistent volume claim for storing model weights:

```bash
# Create the PVC
kubectl apply -n $NAMESPACE -f model-cache/model-cache.yaml

# Verify PVC creation
kubectl get pvc -n $NAMESPACE
```

### Step 3: Start Model Download Job

Launch the model download job:

```bash
# Start the download job
kubectl apply -n $NAMESPACE -f model-cache/model-download.yaml

# Verify job creation
kubectl get jobs -n $NAMESPACE
```

### Step 4: Wait for Download Completion

Monitor and wait for the model download to complete:

```bash
# Get the job name (usually "model-download" or "model-download-<model>")
JOB_NAME=$(kubectl get jobs -n $NAMESPACE -o jsonpath='{.items[?(@.metadata.labels.app=="model-download")].metadata.name}')

# Wait for job completion (timeout after 100 minutes)
kubectl wait --for=condition=Complete job/$JOB_NAME -n $NAMESPACE --timeout=6000s

# Check job status
kubectl get job $JOB_NAME -n $NAMESPACE

# View download logs
kubectl logs job/$JOB_NAME -n $NAMESPACE
```

### Step 5: Verify Model Download

Confirm the model was downloaded successfully:

```bash
# Check if download job completed successfully
kubectl get job $JOB_NAME -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}'

# If the above returns "True", the download was successful
# You can also check the PVC contents (optional)
kubectl exec -n $NAMESPACE -it $(kubectl get pods -n $NAMESPACE -l job-name=$JOB_NAME -o jsonpath='{.items[0].metadata.name}') -- ls -la /model-store/
```

## Using run.sh Script

The `run.sh` script provides a unified interface for automated deployment:

### Script Usage

```bash
./run.sh [OPTIONS] --model <model> --framework <framework> <deployment-type>
```

**Arguments:**
- `<deployment-type>`: Deployment mode (agg, disagg-single-node, disagg-multi-node, disagg-8gpu, disagg-16gpu)

**Required Options:**
- `--model <model>`: Model name (llama-3-70b, gpt-oss-120b, deepseek-r1)
- `--framework <framework>`: Backend framework (vllm, trtllm, sglang)

**Optional Options:**
- `--namespace <namespace>`: Kubernetes namespace (default: dynamo)
- `--skip-model-cache`: Skip model download (assumes cache exists)
- `--dry-run`: Show commands without executing
- `-h, --help`: Show help message

### Examples

```bash
# Deploy Llama-3-70B with vLLM (aggregated mode)
./run.sh --model llama-3-70b --framework vllm agg

# Deploy GPT-OSS-120B with TensorRT-LLM
./run.sh --model gpt-oss-120b --framework trtllm agg

# Deploy DeepSeek-R1 with SGLang (8 GPU disaggregated)
./run.sh --model deepseek-r1 --framework sglang disagg-8gpu

# Deploy with custom namespace and skip model download
./run.sh --namespace my-namespace --skip-model-cache --model llama-3-70b --framework vllm agg

# Dry run to see what would be executed
./run.sh --dry-run --model llama-3-70b --framework vllm agg
```

**Note:** The script automatically runs performance benchmarks if a `perf.yaml` file is present in the deployment directory.

## Manual Model Deployment

For step-by-step control, deploy manually following these steps:

### Step 1: Deploy Model Service

```bash
# Navigate to the specific deployment configuration
cd <model>/<framework>/<deployment-mode>/

# Deploy the model service
kubectl apply -n $NAMESPACE -f deploy.yaml

# Verify deployment creation
kubectl get deployments -n $NAMESPACE
```

### Step 2: Wait for Deployment Ready

```bash
# Get deployment name from the deploy.yaml file
DEPLOYMENT_NAME=$(grep "name:" deploy.yaml | head -1 | awk '{print $2}')

# Wait for deployment to be ready (timeout after 10 minutes)
kubectl wait --for=condition=available deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=600s

# Check deployment status
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE

# Check pod status
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
```

### Step 3: Verify Model Service

```bash
# Check if service is running
kubectl get services -n $NAMESPACE

# Test model endpoint (port-forward to test locally)
kubectl port-forward service/${DEPLOYMENT_NAME}-frontend 8000:8000 -n $NAMESPACE &

# Test the model API (in another terminal)
curl http://localhost:8000/v1/models

# Stop port-forward when done
pkill -f "kubectl port-forward"
```

## Performance Benchmarking (Optional)

Run performance benchmarks to evaluate model performance. Note that benchmarking is only available for models that include a `perf.yaml` file:

### Step 1: Launch Benchmark Job

```bash
# From the deployment directory
kubectl apply -n $NAMESPACE -f perf.yaml

# Verify benchmark job creation
kubectl get jobs -n $NAMESPACE
```

### Step 2: Monitor Benchmark Progress

```bash
# Get benchmark job name
PERF_JOB_NAME=$(grep "name:" perf.yaml | head -1 | awk '{print $2}')

# Monitor benchmark logs in real-time
kubectl logs -f job/$PERF_JOB_NAME -n $NAMESPACE

# Wait for benchmark completion (timeout after 100 minutes)
kubectl wait --for=condition=Complete job/$PERF_JOB_NAME -n $NAMESPACE --timeout=6000s
```

### Step 3: View Benchmark Results

```bash
# Check final benchmark results
kubectl logs job/$PERF_JOB_NAME -n $NAMESPACE | tail -50

# Get benchmark artifacts (if stored in PVC)
kubectl exec -n $NAMESPACE -it $(kubectl get pods -n $NAMESPACE -l job-name=$PERF_JOB_NAME -o jsonpath='{.items[0].metadata.name}') -- ls -la /model-cache/perf/
```

## Model-Specific Examples

### Llama-3-70B (vLLM)

```bash
# Set environment
export NAMESPACE=my-namespace

# Download model
cd llama-3-70b/
kubectl apply -n $NAMESPACE -f model-cache/model-cache.yaml
kubectl apply -n $NAMESPACE -f model-cache/model-download.yaml

# Wait for download
kubectl wait --for=condition=Complete job/model-download -n $NAMESPACE --timeout=6000s

# Deploy aggregated mode
kubectl apply -n $NAMESPACE -f vllm/agg/deploy.yaml
kubectl wait --for=condition=available deployment/llama3-70b-agg -n $NAMESPACE --timeout=600s

# Run benchmark
kubectl apply -n $NAMESPACE -f vllm/agg/perf.yaml
kubectl logs -f job/llama3-70b-agg-perf -n $NAMESPACE
```

### GPT-OSS-120B (TensorRT-LLM)

```bash
# Set environment
export NAMESPACE=my-namespace

# Download model
cd gpt-oss-120b/
kubectl apply -n $NAMESPACE -f model-cache/model-cache.yaml
kubectl apply -n $NAMESPACE -f model-cache/model-download.yaml

# Wait for download
kubectl wait --for=condition=Complete job/model-download -n $NAMESPACE --timeout=6000s

# Deploy aggregated mode
kubectl apply -n $NAMESPACE -f trtllm/agg/deploy.yaml
kubectl wait --for=condition=available deployment/gpt-oss-agg -n $NAMESPACE --timeout=600s

# Run benchmark
kubectl apply -n $NAMESPACE -f trtllm/agg/perf.yaml
kubectl logs -f job/gpt-oss-120b-bench -n $NAMESPACE
```

### DeepSeek-R1 (SGLang)

```bash
# Note: Requires custom container build
# See deepseek-r1/sglang-wideep/README.md for container setup

# Set environment
export NAMESPACE=my-namespace

# Download model
cd deepseek-r1/
kubectl apply -n $NAMESPACE -f model_cache/model-cache.yaml
kubectl apply -n $NAMESPACE -f model_cache/model-download.yaml

# Wait for download
kubectl wait --for=condition=Complete job/model-download -n $NAMESPACE --timeout=6000s

# Deploy 8-GPU disaggregated mode
kubectl apply -n $NAMESPACE -f sglang-wideep/tep8p-dep8d-disagg.yaml
kubectl wait --for=condition=available deployment/sgl-dsr1-8gpu -n $NAMESPACE --timeout=600s
```

## Configuration

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

## Monitoring and Troubleshooting

### Check Deployment Status

```bash
# Check pod status
kubectl get pods -n $NAMESPACE

# Check deployment status
kubectl get deployments -n $NAMESPACE

# Check job status
kubectl get jobs -n $NAMESPACE

# View logs
kubectl logs -f deployment/<deployment-name> -n $NAMESPACE
```

### Common Issues

1. **Model Download Job Fails**
   ```bash
   # Check job status and logs
   kubectl describe job/<job-name> -n $NAMESPACE
   kubectl logs job/<job-name> -n $NAMESPACE

   # Common causes:
   # - Invalid HuggingFace token
   # - Insufficient storage space
   # - Network connectivity issues
   ```

2. **Storage Class Not Found**
   ```bash
   # List available storage classes
   kubectl get storageclass
   # Update model-cache.yaml with correct storage class
   ```

3. **Insufficient GPU Resources**
   ```bash
   # Check GPU availability
   kubectl describe nodes | grep nvidia.com/gpu
   ```

4. **Container Image Pull Errors**
   ```bash
   # Ensure access to NVIDIA container registry
   kubectl get pods -n $NAMESPACE
   kubectl describe pod <pod-name> -n $NAMESPACE
   ```

5. **Job Timeout Issues**
   ```bash
   # For large models, increase timeout values
   kubectl wait --for=condition=Complete job/<job-name> -n $NAMESPACE --timeout=12000s
   ```

### Validation Commands

```bash
# Validate prerequisites
kubectl get storageclass
kubectl get secret hf-token-secret -n $NAMESPACE
kubectl describe nodes | grep nvidia.com/gpu

# Check model download completion
kubectl get jobs -n $NAMESPACE
kubectl get pvc -n $NAMESPACE

# Test model endpoint (after deployment)
kubectl port-forward service/<service-name> 8000:8000 -n $NAMESPACE
curl http://localhost:8000/v1/models
```

## Cleanup

To remove deployed models and free up resources:

```bash
# Delete benchmark jobs
kubectl delete jobs --all -n $NAMESPACE

# Delete model deployments
kubectl delete deployments --all -n $NAMESPACE

# Delete services
kubectl delete services --all -n $NAMESPACE

# Delete PVCs (this will delete downloaded models)
kubectl delete pvc --all -n $NAMESPACE

# Delete namespace (removes everything)
kubectl delete namespace $NAMESPACE
```

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