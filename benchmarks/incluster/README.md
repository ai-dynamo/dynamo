<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. -->

# In-Cluster Benchmarking

This in-cluster benchmarking solution runs Dynamo benchmarks directly within a Kubernetes cluster, eliminating the need for port forwarding and providing better resource utilization.

## What This Tool Does

The in-cluster benchmarking solution:
- Runs benchmarks directly within the Kubernetes cluster using internal service URLs
- Uses Kubernetes service DNS for direct communication (no port forwarding required)
- Leverages the existing benchmarking infrastructure (`benchmarks.utils.benchmark`)
- Stores results persistently using `dynamo-pvc`
- Provides isolated execution environment with configurable resources

## Prerequisites

1. **Kubernetes cluster** with NVIDIA GPUs and Dynamo namespace setup (see [Dynamo Cloud/Platform docs](../../docs/guides/dynamo_deploy/README.md))
2. **Storage and service account** PersistentVolumeClaim and service account configured with appropriate permissions (see [deploy/utils README](../../deploy/utils/README.md))
3. **Docker image** containing the Dynamo benchmarking tools

## Quick Start

### Step 1: Deploy Your DynamoGraphDeployment
Deploy your DynamoGraphDeployment using the [deployment documentation](../../components/backends/). Ensure it has a frontend service exposed.

### Step 2: Deploy and Run Benchmark Job

**Option A: Set environment variables (recommended for multiple commands)**
```bash
# Set environment variables for your deployment
export NAMESPACE=benchmarking
export MODEL_NAME=Qwen/Qwen3-0.6B
export INPUT_NAME=qwen-vllm-agg
export SERVICE_URL=vllm-agg-frontend:8000
export DOCKER_IMAGE=nvcr.io/nvidian/dynamo-dev/vllm-runtime:dyn-973.0

# Deploy the benchmark job
envsubst < benchmark_job.yaml | kubectl apply -f -

# Monitor the job
kubectl logs -f job/dynamo-benchmark -n $NAMESPACE

# Check job status
kubectl get jobs -n $NAMESPACE
```

**Option B: One-liner deployment**
```bash
NAMESPACE=benchmarking MODEL_NAME=Qwen/Qwen3-0.6B INPUT_NAME=qwen-vllm-agg SERVICE_URL=vllm-agg-frontend:8000 DOCKER_IMAGE=nvcr.io/nvidian/dynamo-dev/vllm-runtime:dyn-973.0 envsubst < benchmarks/incluster/benchmark_job.yaml | kubectl apply -f -
```

### Step 3: Retrieve Results
```bash
# Download results from PVC (recommended)
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./benchmarks/results/${INPUT_NAME} \
  --folder /data/results/${INPUT_NAME} \
  --no-config
```

## Configuration

The benchmark job is fully configurable through environment variables:

### Required Environment Variables

- **NAMESPACE**: Kubernetes namespace where the benchmark will run
- **MODEL_NAME**: Hugging Face model identifier (e.g., `Qwen/Qwen3-0.6B`)
- **INPUT_NAME**: Name identifier for the benchmark input (e.g., `qwen-agg`)
- **SERVICE_URL**: Internal service URL for the DynamoGraphDeployment frontend
- **DOCKER_IMAGE**: Docker image containing the Dynamo benchmarking tools

## Understanding Your Results

Results are stored in `/data/results` and follow the same structure as local benchmarking:

```text
/data/results/
├── plots/                           # Performance visualization plots
│   ├── SUMMARY.txt                  # Human-readable benchmark summary
│   ├── p50_inter_token_latency_vs_concurrency.png
│   ├── avg_inter_token_latency_vs_concurrency.png
│   ├── request_throughput_vs_concurrency.png
│   ├── efficiency_tok_s_gpu_vs_user.png
│   └── avg_time_to_first_token_vs_concurrency.png
└── dsr1/                           # Results for dsr1 input
    ├── c1/                          # Concurrency level 1
    │   └── profile_export_genai_perf.json
    ├── c2/                          # Concurrency level 2
    └── ...                          # Other concurrency levels
```

## Monitoring and Debugging

### Check Job Status
```bash
kubectl get jobs -n $NAMESPACE
kubectl describe job dynamo-benchmark -n $NAMESPACE
```

### View Logs
```bash
# Follow logs in real-time
kubectl logs -f job/dynamo-benchmark -n $NAMESPACE

# Get logs from specific container
kubectl logs job/dynamo-benchmark -c benchmark-runner -n $NAMESPACE
```

### Debug Failed Jobs
```bash
# Check pod status
kubectl get pods -n $NAMESPACE -l job-name=dynamo-benchmark

# Describe failed pod
kubectl describe pod <pod-name> -n $NAMESPACE
```

## Comparison with Local Benchmarking

| Feature | Local Benchmarking | In-Cluster Benchmarking |
|---------|-------------------|------------------------|
| Port Forwarding | Required | Not needed |
| Resource Usage | Local machine | Cluster resources |
| Network Latency | Higher (port-forward) | Lower (direct service) |
| Scalability | Limited | High |
| Isolation | Shared environment | Isolated job |
| Results Storage | Local filesystem | Persistent PVC |

The in-cluster approach is recommended for:
- Production benchmarking
- Multiple deployment comparisons
- Resource-constrained environments
- Automated CI/CD pipelines

## Troubleshooting

### Common Issues

1. **Service not found**: Ensure your DynamoGraphDeployment frontend service is running
2. **Service account permissions**: Verify `dynamo-sa` has necessary RBAC permissions
3. **PVC access**: Check that `dynamo-pvc` is properly configured and accessible
4. **Image pull issues**: Ensure the Docker image is accessible from the cluster
5. **Resource constraints**: Adjust resource limits if the job is being evicted

### Debug Commands

```bash
# Check PVC status
kubectl get pvc dynamo-pvc -n $NAMESPACE

# Verify service account
kubectl get sa dynamo-sa -n $NAMESPACE

# Check service endpoints
kubectl get svc -n $NAMESPACE

# Verify your service URL is accessible
kubectl get svc $SERVICE_URL -n $NAMESPACE
```