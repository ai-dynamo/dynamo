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

# Dynamo Benchmarking Guide

This benchmarking framework lets you compare performance across any combination of:
- **DynamoGraphDeployments**
- **External HTTP endpoints** (existing services deployed following standard documentation from vLLM, llm-d, AIBrix, etc.)

## Choosing Your Benchmarking Approach

Dynamo provides two benchmarking approaches to suit different use cases: **client-side** and **server-side**. Client-side refers to running benchmarks on your local machine and connecting to Kubernetes deployments via port-forwarding, while server-side refers to running benchmarks directly within the Kubernetes cluster using internal service URLs. Which method to use depends on your use case.

**TLDR:**
Need high performance/load testing? Server-side.
Just quick testing/comparison? Client-side.

### Use Client-Side Benchmarking When:
- You want to quickly test deployments
- You want immediate access to results on your local machine
- You're comparing external services or deployments (not necessarily just Dynamo deployments)
- You need to run benchmarks from your laptop/workstation

→ **[Go to Client-Side Benchmarking (Local)](#client-side-benchmarking-local)**

### Use Server-Side Benchmarking When:
- You have a development environment with kubectl access
- You're doing performance validation with high load/speed requirements
- You're experiencing timeouts or performance issues with client-side benchmarking
- You want optimal network performance (no port-forwarding overhead)
- You're running automated CI/CD pipelines
- You need isolated execution environments
- You're doing resource-intensive benchmarking
- You want persistent result storage in the cluster

→ **[Go to Server-Side Benchmarking (In-Cluster)](#server-side-benchmarking-in-cluster)**

### Quick Comparison

| Feature | Client-Side | Server-Side |
|---------|-------------|-------------|
| **Location** | Your local machine | Kubernetes cluster |
| **Network** | Port-forwarding required | Direct service DNS |
| **Setup** | Quick and simple | Requires cluster resources |
| **Performance** | Limited by local resources, may timeout under high load | Optimal cluster performance, handles high load |
| **Isolation** | Shared environment | Isolated job execution |
| **Results** | Local filesystem | Persistent volumes |
| **Best for** | Light load | High load |

## What This Tool Does

The framework is a Python-based wrapper around `aiperf` that:
- Benchmarks any HTTP endpoints
- Runs concurrency sweeps across configurable load levels
- Generates comparison plots with your custom labels
- Works with any HuggingFace-compatible model on NVIDIA GPUs (H200, H100, A100, etc.)
- Provides direct Python script execution for maximum flexibility

**Default sequence lengths**: Input: 2000 tokens, Output: 256 tokens (configurable with `--isl` and `--osl`)

**Important**: The `--model` parameter configures AIPerf for benchmarking and provides logging context. The default `--model` value in the benchmarking script is `Qwen/Qwen3-0.6B`, but it must match the model deployed at the endpoint(s).

---

# Client-Side Benchmarking (Local)

Client-side benchmarking runs on your local machine and connects to Kubernetes deployments via port-forwarding.

## Prerequisites

1. **Dynamo container environment** - You must be running inside a Dynamo container with the benchmarking tools pre-installed.

2. **HTTP endpoints** - Ensure you have HTTP endpoints available for benchmarking. These can be:
   - DynamoGraphDeployments exposed via HTTP endpoints
   - External services (vLLM, llm-d, AIBrix, etc.)
   - Any HTTP endpoint serving HuggingFace-compatible models

3. **Benchmark dependencies** - Since benchmarks run locally, you need to install the required Python dependencies. Install them using:
   ```bash
   pip install -r deploy/utils/requirements.txt
   ```

## User Workflow

Follow these steps to benchmark Dynamo deployments using client-side benchmarking:

### Step 1: Establish Kubernetes Cluster and Install Dynamo
Set up your Kubernetes cluster with NVIDIA GPUs and install the Dynamo Cloud platform. First follow the [installation guide](/docs/kubernetes/installation_guide.md) to install Dynamo Cloud, then use [deploy/utils/README](../../deploy/utils/README.md) to set up benchmarking resources.

### Step 2: Deploy DynamoGraphDeployments
Deploy your DynamoGraphDeployments separately using the [deployment documentation](../../examples/backends/). Each deployment should have a frontend service exposed.

### Step 3: Port-Forward and Benchmark Deployment A
```bash
# Port-forward the frontend service for deployment A
kubectl port-forward -n <namespace> svc/<frontend-service-name> 8000:8000 > /dev/null 2>&1 &
# Note: remember to stop the port-forward process after benchmarking.

# Benchmark deployment A using Python scripts
python3 -m benchmarks.utils.benchmark \
   --benchmark-name deployment-a \
   --endpoint-url http://localhost:8000 \
   --model "your-model-name" \
   --output-dir ./benchmarks/results
```

### Step 4: [If Comparative] Teardown Deployment A and Establish Deployment B
If comparing multiple deployments, teardown deployment A and deploy deployment B with a different configuration.

### Step 5: [If Comparative] Port-Forward and Benchmark Deployment B
```bash
# Port-forward the frontend service for deployment B
kubectl port-forward -n <namespace> svc/<frontend-service-name> 8001:8000 > /dev/null 2>&1 &

# Benchmark deployment B using Python scripts
python3 -m benchmarks.utils.benchmark \
   --benchmark-name deployment-b \
   --endpoint-url http://localhost:8001 \
   --model "your-model-name" \
   --output-dir ./benchmarks/results
```

### Step 6: Generate Summary and Visualization
```bash
# Generate plots and summary using Python plotting script
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results

# Or plot only specific benchmark experiments
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results --benchmark-name experiment-a --benchmark-name experiment-b
```

## Use Cases

The benchmarking framework supports various comparative analysis scenarios:

- **Compare multiple DynamoGraphDeployments of a single backend** (e.g., aggregated vs disaggregated configurations)
- **Compare different backends** (e.g., vLLM vs TensorRT-LLM vs SGLang)
- **Compare Dynamo vs other platforms** (e.g., Dynamo vs llm-d vs AIBrix)
- **Compare different models** (e.g., Llama-3-8B vs Llama-3-70B vs Qwen-3-0.6B)
- **Compare different hardware configurations** (e.g., H100 vs A100 vs H200)
- **Compare different parallelization strategies** (e.g., different GPU counts or memory configurations)

## Configuration and Usage

### Command Line Options

```bash
python3 -m benchmarks.utils.benchmark --benchmark-name <name> --endpoint-url <endpoint_url> [OPTIONS]

REQUIRED:
  --benchmark-name NAME           Name/label for this benchmark (used in plots and results)
  --endpoint-url URL              HTTP endpoint URL to benchmark (e.g., http://localhost:8000)

OPTIONS:
  -h, --help                    Show help message and examples
  -m, --model MODEL             Model name for AIPerf configuration and logging (default: Qwen/Qwen3-0.6B)
                                NOTE: This must match the model deployed at the endpoint
  -i, --isl LENGTH              Input sequence length (default: 2000)
  -s, --std STDDEV              Input sequence standard deviation (default: 10)
  -o, --osl LENGTH              Output sequence length (default: 256)
  -d, --output-dir DIR          Output directory (default: ./benchmarks/results)
  --verbose                     Enable verbose output
```

### Important Notes

- **Benchmark Name**: The benchmark name becomes the label in plots and results
- **Name Restrictions**: Names can only contain letters, numbers, hyphens, and underscores. The name `plots` is reserved.
- **Port-Forwarding**: You must have an exposed endpoint before benchmarking
- **Model Parameter**: The `--model` parameter configures AIPerf for testing and logging, and must match the model deployed at the endpoint
- **Sequential Benchmarking**: For comparative benchmarks, deploy and benchmark each configuration separately

### What Happens During Benchmarking

The Python benchmarking module:
1. **Connects** to your port-forwarded endpoint
2. **Benchmarks** using AIPerf at various concurrency levels (default: 1, 2, 5, 10, 50, 100, 250)
3. **Measures** key metrics: latency, throughput, time-to-first-token
4. **Saves** results to an output directory organized by benchmark name

The Python plotting module:
1. **Generates** comparison plots using your benchmark name in `<OUTPUT_DIR>/plots/`
2. **Creates** summary statistics and visualizations

### Plotting Options

The plotting script supports several options for customizing which experiments to visualize:

```bash
# Plot all benchmark experiments in the data directory
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results

# Plot only specific benchmark experiments
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results --benchmark-name experiment-a --benchmark-name experiment-b

# Specify custom output directory for plots
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results --output-dir ./custom-plots
```

**Available Options:**
- `--data-dir`: Directory containing benchmark results (required)
- `--benchmark-name`: Specific benchmark experiment name to plot (can be specified multiple times). Names must match subdirectory names under the data dir.
- `--output-dir`: Custom output directory for plots (defaults to data-dir/plots)

**Note**: If `--benchmark-name` is not specified, the script will plot all subdirectories found in the data directory.

### Using Your Own Models and Configuration

The benchmarking framework supports any HuggingFace-compatible LLM model. Specify your model in the benchmark script's `--model` parameter. It must match the model name of the deployment. You can override the default sequence lengths (2000/256 tokens) with `--isl` and `--osl` flags if needed for your specific workload.

The benchmarking framework is built around Python modules that provide direct control over the benchmark workflow. The Python benchmarking module connects to your existing endpoints, runs the benchmarks, and can generate plots. Deployment is user-managed and out of scope for this tool.

### Comparison Limitations

The plotting system supports up to 12 different benchmarks in a single comparison.

### Concurrency Configuration

You can customize the concurrency levels using the CONCURRENCIES environment variable:

```bash
# Custom concurrency levels
CONCURRENCIES="1,5,20,50" python3 -m benchmarks.utils.benchmark \
    --benchmark-name my-test \
    --endpoint-url http://localhost:8000

# Or set permanently
export CONCURRENCIES="1,2,5,10,25,50,100"
python3 -m benchmarks.utils.benchmark \
    --benchmark-name test \
    --endpoint-url http://localhost:8000
```

## Understanding Your Results

After benchmarking completes, check `./benchmarks/results/` (or your custom output directory):

### Plot Labels and Organization

The plotting script uses the `--benchmark-name` as the experiment name in all generated plots. For example:
- `--benchmark-name aggregated` → plots will show "aggregated" as the label
- `--benchmark-name vllm-disagg` → plots will show "vllm-disagg" as the label

This allows you to easily identify and compare different configurations in the visualization plots.

### Summary and Plots

```text
benchmarks/results/plots
├── SUMMARY.txt                                     # Quick overview of all results
├── p50_inter_token_latency_vs_concurrency.png      # Token generation speed
├── avg_time_to_first_token_vs_concurrency.png      # Response time
├── request_throughput_vs_concurrency.png           # Requests per second
├── efficiency_tok_s_gpu_vs_user.png                # GPU efficiency
└── avg_inter_token_latency_vs_concurrency.png      # Average latency
```

### Data Files

Raw data is organized by deployment/benchmark type and concurrency level:

**For Any Benchmarking (uses your custom benchmark name):**
```text
results/                         # Client-side: ./benchmarks/results/ or custom dir
├── plots/                       # Server-side: /data/results/
│   ├── SUMMARY.txt              # Performance visualization plots
│   ├── p50_inter_token_latency_vs_concurrency.png
│   ├── avg_inter_token_latency_vs_concurrency.png
│   ├── request_throughput_vs_concurrency.png
│   ├── efficiency_tok_s_gpu_vs_user.png
│   └── avg_time_to_first_token_vs_concurrency.png
├── <your-benchmark-name>/       # Results for your benchmark (uses your custom name)
│   ├── c1/                      # Concurrency level 1
│   │   └── profile_export_aiperf.json
│   ├── c2/                      # Concurrency level 2
│   ├── c5/                      # Concurrency level 5
│   └── ...                      # Other concurrency levels (10, 50, 100, 250)
└── <your-benchmark-name-N>/     # Results for additional benchmarking runs
    └── c*/                      # Same structure as above
```

**Example with actual benchmark names:**
```text
results/
├── plots/
├── experiment-a/                  # --benchmark-name experiment-a
├── experiment-b/                  # --benchmark-name experiment-b
└── experiment-c/                  # --benchmark-name experiment-c
```

Each concurrency directory contains:
- **`profile_export_aiperf.json`** - Structured metrics from AIPerf
- **`profile_export_aiperf.csv`** - CSV format metrics from AIPerf
- **`profile_export.json`** - Raw AIPerf results
- **`inputs.json`** - Generated test inputs

---

# Server-Side Benchmarking (In-Cluster)

Server-side benchmarking runs directly within the Kubernetes cluster, eliminating the need for port forwarding and providing better resource utilization.

## What Server-Side Benchmarking Does

The server-side benchmarking solution:
- Runs benchmarks directly within the Kubernetes cluster using internal service URLs
- Uses Kubernetes service DNS for direct communication (no port forwarding required)
- Leverages the existing benchmarking infrastructure (`benchmarks.utils.benchmark`)
- Stores results persistently using `dynamo-pvc`
- Provides isolated execution environment with configurable resources
- Handles high load/speed requirements without timeout issues
- **Note**: Each benchmark job runs within a single Kubernetes namespace, but can benchmark services across multiple namespaces using the full DNS format `svc_name.namespace.svc.cluster.local`

## Prerequisites

1. **Kubernetes cluster** with NVIDIA GPUs and Dynamo namespace setup (see [Dynamo Cloud/Platform docs](/docs/kubernetes/README.md))
2. **Storage** PersistentVolumeClaim configured with appropriate permissions (see [deploy/utils README](../../deploy/utils/README.md))
3. **Docker image** containing the Dynamo benchmarking tools

## Quick Start

### Step 1: Deploy Your DynamoGraphDeployment
Deploy your DynamoGraphDeployment using the [deployment documentation](../../examples/backends/). Ensure it has a frontend service exposed.

### Step 2: Deploy and Run Benchmark Job

**Note**: The server-side benchmarking job requires a Docker image containing the Dynamo benchmarking tools. Before the 0.5.1 release, you must build your own Docker image using the [container build instructions](../../container/README.md), push it to your container registry, then update the `image` field in `benchmarks/incluster/benchmark_job.yaml` to use your built image tag.

```bash
export NAMESPACE=benchmarking

# Deploy the benchmark job with default settings
kubectl apply -f benchmarks/incluster/benchmark_job.yaml -n $NAMESPACE

# Monitor the job, wait for it to complete
kubectl logs -f job/dynamo-benchmark -n $NAMESPACE
```

#### Customize the job configuration

To customize the benchmark parameters, edit the `benchmarks/incluster/benchmark_job.yaml` file and modify:

- **Model name**: Change `"Qwen/Qwen3-0.6B"` in the args section
- **Benchmark name**: Change `"qwen3-0p6b-vllm-agg"` to your desired benchmark name
- **Service URL**: Change `"vllm-agg-frontend:8000"` so the service URL matches your deployed service
- **Docker image**: Change the image field if needed

Then deploy:
```bash
kubectl apply -f benchmarks/incluster/benchmark_job.yaml -n $NAMESPACE
```

### Step 3: Retrieve Results
```bash
# Download results from PVC (recommended)
python3 -m deploy.utils.download_pvc_results \
  --namespace $NAMESPACE \
  --output-dir ./benchmarks/results/<benchmark-name> \
  --folder /data/results/<benchmark-name> \
  --no-config
```

### Step 4: Generate Plots
```bash
# Generate performance plots from the downloaded results
python3 -m benchmarks.utils.plot \
  --data-dir ./benchmarks/results
```

This will create visualization plots. For more details on interpreting these plots, see the [Summary and Plots](#summary-and-plots) section above.

## Cross-Namespace Service Access

Server-side benchmarking can benchmark services across multiple namespaces from a single job using Kubernetes DNS. When referencing services in other namespaces, use the full DNS format:

```bash
# Access service in same namespace
SERVICE_URL=vllm-agg-frontend:8000

# Access service in different namespace
SERVICE_URL=vllm-agg-frontend.production.svc.cluster.local:8000
```

**DNS Format**: `<service-name>.<namespace>.svc.cluster.local:port`

This allows you to:
- Benchmark multiple services across different namespaces in a single job
- Compare services running in different environments (dev, staging, production)
- Test cross-namespace integrations without port-forwarding
- Run comprehensive cross-namespace performance comparisons

## Configuration

The benchmark job is configured directly in the YAML file.

### Default Configuration

- **Model**: `Qwen/Qwen3-0.6B`
- **Benchmark Name**: `qwen3-0p6b-vllm-agg`
- **Service**: `vllm-agg-frontend:8000`
- **Docker Image**: `nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag`

### Customizing the Job

To customize the benchmark, edit `benchmarks/incluster/benchmark_job.yaml`:

1. **Change the model**: Update the `--model` argument
2. **Change the benchmark name**: Update the `--benchmark-name` argument
3. **Change the service URL**: Update the `--endpoint-url` argument (use `<svc_name>.<namespace>.svc.cluster.local:port` for cross-namespace access)
4. **Change Docker image**: Update the image field if needed

### Example: Multi-Namespace Benchmarking

To benchmark services across multiple namespaces, you would need to run separate benchmark jobs for each service since the format supports one benchmark per job. However, the results are stored in the same PVC and may be accessed together.

```yaml
# Job 1: Production service
args:
  - --model
  - "Qwen/Qwen3-0.6B"
  - --benchmark-name
  - "prod-vllm"
  - --endpoint-url
  - "vllm-agg-frontend.production.svc.cluster.local:8000"
  - --output-dir
  - /data/results

# Job 2: Staging service
args:
  - --model
  - "Qwen/Qwen3-0.6B"
  - --benchmark-name
  - "staging-vllm"
  - --endpoint-url
  - "vllm-agg-frontend.staging.svc.cluster.local:8000"
  - --output-dir
  - /data/results
```

## Understanding Your Results

Results are stored in `/data/results` and follow the same structure as client-side benchmarking:

```text
/data/results/
└── <benchmark-name>/                # Results for your benchmark name
    ├── c1/                          # Concurrency level 1
    │   └── profile_export_aiperf.json
    ├── c2/                          # Concurrency level 2
    └── ...                          # Other concurrency levels
```

## Monitoring and Debugging

### Check Job Status
```bash
kubectl describe job dynamo-benchmark -n $NAMESPACE
```

### View Logs
```bash
# Follow logs in real-time
kubectl logs -f job/dynamo-benchmark -n $NAMESPACE
```

### Debug Failed Jobs
```bash
# Check pod status
kubectl get pods -n $NAMESPACE -l job-name=dynamo-benchmark

# Describe failed pod
kubectl describe pod <pod-name> -n $NAMESPACE
```

## Troubleshooting

### Common Issues

1. **Service not found**: Ensure your DynamoGraphDeployment frontend service is running
3. **PVC access**: Check that `dynamo-pvc` is properly configured and accessible
4. **Image pull issues**: Ensure the Docker image is accessible from the cluster
5. **Resource constraints**: Adjust resource limits if the job is being evicted

### Debug Commands

```bash
# Check PVC status
kubectl get pvc dynamo-pvc -n $NAMESPACE

# Check service endpoints
kubectl get svc -n $NAMESPACE

# Verify your service exists and has endpoints
SVC_NAME="${SERVICE_URL%%:*}"
kubectl get svc "$SVC_NAME" -n "$NAMESPACE"
kubectl get endpoints "$SVC_NAME" -n "$NAMESPACE"
```

---

## Customize Benchmarking Behavior

The built-in Python workflow connects to endpoints, benchmarks with aiperf, and generates plots. If you want to modify the behavior:

1. **Extend the workflow**: Modify `benchmarks/utils/workflow.py` to add custom deployment types or metrics collection

2. **Generate different plots**: Modify `benchmarks/utils/plot.py` to generate a different set of plots for whatever you wish to visualize.

3. **Direct module usage**: Use individual Python modules (`benchmarks.utils.benchmark`, `benchmarks.utils.plot`) for granular control over each step of the benchmarking process.

The Python benchmarking module provides a complete end-to-end benchmarking experience with full control over the workflow.

---

## Testing with Mocker Backend

For development and testing purposes, Dynamo provides a [mocker backend](../../components/src/dynamo/mocker/) that simulates LLM inference without requiring actual GPU resources. This is useful for:

- **Testing deployments** without expensive GPU infrastructure
- **Developing and debugging** router, planner, or frontend logic
- **CI/CD pipelines** that need to validate infrastructure without model execution
- **Benchmarking framework validation** to ensure your setup works before using real backends

The mocker backend mimics the API and behavior of real backends (vLLM, SGLang, TensorRT-LLM) but generates mock responses instead of running actual inference.

See the [mocker directory](../../components/src/dynamo/mocker/) for usage examples and configuration options.
