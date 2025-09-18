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
2. **dynamo-pvc** PersistentVolumeClaim configured (see [deploy/utils README](../../deploy/utils/README.md))
3. **Service account** (`dynamo-sa`) with appropriate permissions (see [deploy/utils README](../../deploy/utils/README.md))
4. **Docker image** containing the Dynamo benchmarking tools

## Quick Start

### Step 1: Deploy Your DynamoGraphDeployment
Deploy your DynamoGraphDeployment using the [deployment documentation](../../components/backends/). Ensure it has a frontend service exposed.

### Step 2: Deploy and Run Benchmark Job
```bash
# Deploy the benchmark job with your namespace
NAMESPACE=your-namespace envsubst < benchmark_job.yaml | kubectl apply -f -

# Monitor the job
kubectl logs -f job/dynamo-benchmark -n your-namespace

# Check job status
kubectl get jobs -n your-namespace
```

### Step 3: Retrieve Results
```bash
# Download results from PVC (recommended)
python3 -m deploy.utils.download_pvc_results \
  --namespace your-namespace \
  --output-dir ./benchmark_results \
  --folder /data/results \
  --no-config

# Alternative: Copy results directly (requires pod name)
kubectl cp <pod-name>:/data/results ./benchmark_results -n your-namespace
```

## Configuration

The job manifest uses these default parameters:
- **Model**: `Qwen/Qwen3-0.6B`
- **Input sequence length**: 2000 tokens
- **Output sequence length**: 256 tokens
- **Input**: `dsr1=${NAMESPACE}-dsr1-frontend:8000` (internal service URL)

### Customizing the Job Manifest

Edit `benchmark_job.yaml` to modify:

```yaml
# Change model
args:
  - --model
  - "meta-llama/Meta-Llama-3-8B"

# Change sequence lengths
args:
  - --isl
  - "1500"
  - --osl
  - "200"

# Change input service
args:
  - --input
  - my-service=${NAMESPACE}-my-service:8000
```

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
kubectl get jobs -n <namespace>
kubectl describe job dynamo-benchmark -n <namespace>
```

### View Logs
```bash
# Follow logs in real-time
kubectl logs -f job/dynamo-benchmark -n <namespace>

# Get logs from specific container
kubectl logs job/dynamo-benchmark -c benchmark-runner -n <namespace>
```

### Debug Failed Jobs
```bash
# Check pod status
kubectl get pods -n <namespace> -l job-name=dynamo-benchmark

# Describe failed pod
kubectl describe pod <pod-name> -n <namespace>
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
kubectl get pvc dynamo-pvc -n <namespace>

# Verify service account
kubectl get sa dynamo-sa -n <namespace>

# Check service endpoints
kubectl get svc -n <namespace>
```