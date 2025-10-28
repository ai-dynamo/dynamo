# Dynamo Model Serving Recipes

This repository contains production-ready recipes for deploying large language models using the Dynamo platform. Each recipe includes deployment configurations, performance benchmarking, and model caching setup.

## Contents
- [Available Models](#available-models)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- Deployment Methods
   - [Option 1: Automated Deployment](#option-1-automated-deployment)
   - [Option 2: Manual Deployment](#option-2-manual-deployment)


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


**Recipe Directory Structure:**
Recipes are organized into a directory structure that follows the pattern:
```text
<model-name>/
├── model-cache/
│   ├── model-cache.yaml         # PVC for model cache
│   └── model-download.yaml      # Job for model download
├── <framework>/
│   └── <deployment-mode>/
│       ├── deploy.yaml          # DynamoGraphDeployment CRD and optional configmap for custom configuration
│       └── perf.yaml (optional) # Performance benchmark
└── README.md (optional)         # Model documentation
```

## Quick Start

Follow the instructions in the [Prerequisites](#prerequisites) section to set up your environment.

Choose your preferred deployment method: using the `run.sh` script or manual deployment steps.


## Prerequisites

### 1. Environment Setup

Create a Kubernetes namespace and set environment variable:

```bash
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
```

### 2. Deploy Dynamo Platform

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

### 5. HuggingFace Access and Kubernetes Secret Creation

Set up a kubernetes secret with the HuggingFace token for model download:

```bash
# Update the token in the secret file
vim hf_hub_secret/hf_hub_secret.yaml

# Apply the secret
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}
```

### 6. Configure Storage Class

Configure persistent storage for model caching:

```bash
# Check available storage classes
kubectl get storageclass
```

Replace "your-storage-class-name" with your actual storage class in the file: `<model>/model-cache/model-cache.yaml`

```yaml
# In <model>/model-cache/model-cache.yaml
spec:
  storageClassName: "your-actual-storage-class"  # Replace this
```

## Option 1: Automated Deployment

Use the `run.sh` script for fully automated deployment:

**Note:** The script automatically:
- Create model cache PVC and downloads the model
- Deploy the model service
- Runs performance benchmark if a `perf.yaml` file is present in the deployment directory


#### Script Usage

```bash
./run.sh [OPTIONS] --model <model> --framework <framework> --deployment <deployment-type>
```

**Required Options:**
- `--model <model>`: Model name matching the directory name in the recipes directory (e.g., llama-3-70b, gpt-oss-120b, deepseek-r1)
- `--framework <framework>`: Backend framework (`vllm`, `trtllm`, `sglang`)
- `--deployment <deployment-type>`: Deployment mode (e.g., agg, disagg, disagg-single-node, disagg-multi-node)

**Optional Options:**
- `--namespace <namespace>`: Kubernetes namespace (default: dynamo)
- `--dry-run`: Show commands without executing them
- `-h, --help`: Show help message

**Environment Variables:**
- `NAMESPACE`: Kubernetes namespace (default: dynamo)

#### Example Usage
```bash
# Set up environment
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
# Configure HuggingFace token
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}

# use run.sh script to deploy the model
# Deploy Llama-3-70B with vLLM (aggregated mode)
./run.sh --model llama-3-70b --framework vllm --deployment agg

# Deploy GPT-OSS-120B with TensorRT-LLM
./run.sh --model gpt-oss-120b --framework trtllm --deployment agg

# Deploy DeepSeek-R1 with SGLang (disaggregated mode)
./run.sh --model deepseek-r1 --framework sglang --deployment disagg

# Deploy with custom namespace
./run.sh --namespace my-namespace --model llama-3-70b --framework vllm --deployment agg

# Dry run to see what would be executed
./run.sh --dry-run --model llama-3-70b --framework vllm --deployment agg
```


## Option 2: Manual Deployment

For step-by-step manual deployment follow these steps :

```bash
# 0. Set up environment (see Prerequisites section)
export NAMESPACE=your-namespace
kubectl create namespace ${NAMESPACE}
kubectl apply -f hf_hub_secret/hf_hub_secret.yaml -n ${NAMESPACE}

# 1. Download model (see Model Download section)
kubectl apply -n $NAMESPACE -f <model>/model-cache/

# 2. Deploy model (see Deployment section)
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/deploy.yaml

# 3. Run benchmarks (optional, if perf.yaml exists)
kubectl apply -n $NAMESPACE -f <model>/<framework>/<mode>/perf.yaml
```

### Step 1: Download Model

```bash
# Start the download job
kubectl apply -n $NAMESPACE -f <model>/model-cache

# Verify job creation
kubectl get jobs -n $NAMESPACE | grep model-download
```

Monitor and wait for the model download to complete:

```bash

# Wait for job completion (timeout after 100 minutes)
kubectl wait --for=condition=Complete job/model-download -n $NAMESPACE --timeout=6000s

# Check job status
kubectl get job model-download -n $NAMESPACE

# View download logs
kubectl logs job/model-download -n $NAMESPACE
```

### Step 2: Deploy Model Service

```bash
# Navigate to the specific deployment configuration
cd <model>/<framework>/<deployment-mode>/

# Deploy the model service
kubectl apply -n $NAMESPACE -f deploy.yaml

# Verify deployment creation
kubectl get deployments -n $NAMESPACE
```

#### Wait for Deployment Ready

```bash
# Get deployment name from the deploy.yaml file
DEPLOYMENT_NAME=$(grep "name:" deploy.yaml | head -1 | awk '{print $2}')

# Wait for deployment to be ready (timeout after 10 minutes)
kubectl wait --for=condition=available deployment/$DEPLOYMENT_NAME -n $NAMESPACE --timeout=1200s

# Check deployment status
kubectl get deployment $DEPLOYMENT_NAME -n $NAMESPACE

# Check pod status
kubectl get pods -n $NAMESPACE -l app=$DEPLOYMENT_NAME
```

#### Verify Model Service

```bash
# Check if service is running
kubectl get services -n $NAMESPACE

# Test model endpoint (port-forward to test locally)
kubectl port-forward service/${DEPLOYMENT_NAME}-frontend 8000:8000 -n $NAMESPACE

# Test the model API (in another terminal)
curl http://localhost:8000/v1/models

# Stop port-forward when done
pkill -f "kubectl port-forward"
```

### Step 3: Performance Benchmarking (Optional)

Run performance benchmarks to evaluate model performance. Note that benchmarking is only available for models that include a `perf.yaml` file (optional):

#### Launch Benchmark Job

```bash
# From the deployment directory
kubectl apply -n $NAMESPACE -f perf.yaml

# Verify benchmark job creation
kubectl get jobs -n $NAMESPACE
```

#### Monitor Benchmark Progress

```bash
# Get benchmark job name
PERF_JOB_NAME=$(grep "name:" perf.yaml | head -1 | awk '{print $2}')

# Monitor benchmark logs in real-time
kubectl logs -f job/$PERF_JOB_NAME -n $NAMESPACE

# Wait for benchmark completion (timeout after 100 minutes)
kubectl wait --for=condition=Complete job/$PERF_JOB_NAME -n $NAMESPACE --timeout=6000s
```

#### View Benchmark Results

```bash
# Check final benchmark results
kubectl logs job/$PERF_JOB_NAME -n $NAMESPACE | tail -50
```