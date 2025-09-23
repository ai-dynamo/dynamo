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

## What This Tool Does

The framework is a Python-based wrapper around `genai-perf` that:
- Benchmarks any HTTP endpoints
- Runs concurrency sweeps across configurable load levels
- Generates comparison plots with your custom labels
- Works with any HuggingFace-compatible model on NVIDIA GPUs (H200, H100, A100, etc.)
- Runs locally and connects to your Kubernetes deployments/endpoints
- Provides direct Python script execution for maximum flexibility

**Default sequence lengths**: Input: 2000 tokens, Output: 256 tokens (configurable with `--isl` and `--osl`)

**Important**: The `--model` parameter configures GenAI-Perf for benchmarking and provides logging context. The default `--model` value in the benchmarking script is `Qwen/Qwen3-0.6B`, but it must match the model deployed at the endpoint(s).

## Prerequisites

1. **Dynamo container environment** - You must be running inside a Dynamo container with the benchmarking tools pre-installed.

2. **Ubuntu 24.04** - GenAI-Perf requires Ubuntu 24.04 or higher to work properly. If you are on Ubuntu 22.04 or lower, you will need to build perf_analyzer [from source](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/install.md#build-from-source).

3. **kubectl access** - You need `kubectl` installed and configured to access your Kubernetes cluster.

4. **HTTP endpoints** - Ensure you have HTTP endpoints available for benchmarking. These can be:
   - DynamoGraphDeployments exposed via HTTP endpoints
   - External services (vLLM, llm-d, AIBrix, etc.)
   - Any HTTP endpoint serving HuggingFace-compatible models

5. **Benchmark dependencies** - Since benchmarks run locally, you need to install the required Python dependencies. Install them using:
   ```bash
   pip install -r deploy/utils/requirements.txt
   ```

## User Workflow

Follow these steps to benchmark Dynamo deployments:

### Step 1: Establish Kubernetes Cluster and Install Dynamo
Set up your Kubernetes cluster with NVIDIA GPUs and install the Dynamo Cloud platform. First follow the [installation guide](/docs/guides/dynamo_deploy/installation_guide.md) to install Dynamo Cloud, then use [deploy/utils/README](../../deploy/utils/README.md) to set up benchmarking resources.

### Step 2: Deploy DynamoGraphDeployments
Deploy your DynamoGraphDeployments separately using the [deployment documentation](../../components/backends/). Each deployment should have a frontend service exposed.

### Step 3: Port-Forward and Benchmark Deployment A
```bash
# Port-forward the frontend service for deployment A
kubectl port-forward -n <namespace> svc/<frontend-service-name> 8000:8000 > /dev/null 2>&1 &
# Note: remember to stop the port-forward process after benchmarking.

# Benchmark deployment A using Python scripts
python3 -m benchmarks.utils.benchmark \
   --input deployment-a=http://localhost:8000 \
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
   --input deployment-b=http://localhost:8001 \
   --model "your-model-name" \
   --output-dir ./benchmarks/results
```

### Step 6: Generate Summary and Visualization
```bash
# Generate plots and summary using Python plotting script
python3 -m benchmarks.utils.plot --data-dir ./benchmarks/results
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
python3 -m benchmarks.utils.benchmark --input <label>=<endpoint_url> [--input <label>=<endpoint_url>]... [OPTIONS]

REQUIRED:
  --input <label>=<endpoint_url>     Benchmark input with custom label
                                        - <label>: becomes the name/label in plots (see Important Notes for restrictions)
                                        - <endpoint_url>: HTTP endpoint URL (e.g., http://localhost:8000)
                                        Can be specified multiple times for comparisons

OPTIONS:
  -h, --help                    Show help message and examples
  -m, --model MODEL             Model name for GenAI-Perf configuration and logging (default: Qwen/Qwen3-0.6B)
                                NOTE: This must match the model deployed at the endpoint
  -i, --isl LENGTH              Input sequence length (default: 2000)
  -s, --std STDDEV              Input sequence standard deviation (default: 10)
  -o, --osl LENGTH              Output sequence length (default: 256)
  -d, --output-dir DIR          Output directory (default: ./benchmarks/results)
  --verbose                     Enable verbose output
```

### Important Notes

- **Custom Labels**: Each input must have a unique label that becomes the name in plots and results
- **Label Restrictions**: Labels can only contain letters, numbers, hyphens, and underscores. The label `plots` is reserved.
- **Port-Forwarding**: You must have an exposed endpoint before benchmarking
- **Model Parameter**: The `--model` parameter configures GenAI-Perf for testing and logging, and must match the model deployed at the endpoint
- **Sequential Benchmarking**: For comparative benchmarks, deploy and benchmark each configuration separately

### What Happens During Benchmarking

The Python benchmarking module:
1. **Connects** to your port-forwarded endpoint
2. **Benchmarks** using GenAI-Perf at various concurrency levels (default: 1, 2, 5, 10, 50, 100, 250)
3. **Measures** key metrics: latency, throughput, time-to-first-token
4. **Saves** results to an output directory organized by input labels

The Python plotting module:
1. **Generates** comparison plots using your custom labels in `<OUTPUT_DIR>/plots/`
2. **Creates** summary statistics and visualizations

### Using Your Own Models and Configuration

The benchmarking framework supports any HuggingFace-compatible LLM model. Specify your model in the benchmark script's `--model` parameter. It must match the model name of the deployment. You can override the default sequence lengths (2000/256 tokens) with `--isl` and `--osl` flags if needed for your specific workload.

The benchmarking framework is built around Python modules that provide direct control over the benchmark workflow. The Python benchmarking module connects to your existing endpoints, runs the benchmarks, and can generate plots. Deployment is user-managed and out of scope for this tool.

### Comparison Limitations

The plotting system supports up to 12 different inputs in a single comparison. If you need to compare more than 12 different deployments/endpoints, consider running separate benchmark sessions or grouping related comparisons together.

### Concurrency Configuration

You can customize the concurrency levels using the CONCURRENCIES environment variable:

```bash
# Custom concurrency levels
CONCURRENCIES="1,5,20,50" python3 -m benchmarks.utils.benchmark \
    --input my-test=http://localhost:8000

# Or set permanently
export CONCURRENCIES="1,2,5,10,25,50,100"
python3 -m benchmarks.utils.benchmark \
    --input test=http://localhost:8000
```

## Understanding Your Results

After benchmarking completes, check `./benchmarks/results/` (or your custom output directory):

### Plot Labels and Organization

The plotting script uses the `--input` labels (the keys before the `=` sign) as the experiment names in all generated plots. For example:
- `--input aggregated=http://localhost:8000` → plots will show "aggregated" as the label
- `--input vllm-disagg=http://localhost:8001` → plots will show "vllm-disagg" as the label

This allows you to easily identify and compare different configurations in the visualization plots.

### Summary and Plots

```text
benchmarks/results/
├── SUMMARY.txt          # Quick overview of all results
└── plots/               # Visual comparisons (these are what you want!)
    ├── p50_inter_token_latency_vs_concurrency.png      # Token generation speed
    ├── avg_time_to_first_token_vs_concurrency.png      # Response time
    ├── request_throughput_vs_concurrency.png           # Requests per second
    ├── efficiency_tok_s_gpu_vs_user.png                # GPU efficiency
    └── avg_inter_token_latency_vs_concurrency.png      # Average latency
```

### Data Files

Raw data is organized by deployment/benchmark type and concurrency level:

**For Any Benchmarking (uses your custom labels):**
```text
benchmarks/results/
├── plots/                       # Performance visualization plots
│   ├── SUMMARY.txt             # Human-readable benchmark summary
│   ├── p50_inter_token_latency_vs_concurrency.png
│   ├── avg_inter_token_latency_vs_concurrency.png
│   ├── request_throughput_vs_concurrency.png
│   ├── efficiency_tok_s_gpu_vs_user.png
│   └── avg_time_to_first_token_vs_concurrency.png
├── <your-label-1>/              # Results for first input (uses your custom label)
│   ├── c1/                      # Concurrency level 1
│   │   └── profile_export_genai_perf.json
│   ├── c2/                      # Concurrency level 2
│   ├── c5/                      # Concurrency level 5
│   └── ...                      # Other concurrency levels (10, 50, 100, 250)
├── <your-label-2>/              # Results for second input (if provided)
│   └── c*/                      # Same structure as above
└── <your-label-N>/              # Results for additional inputs
    └── c*/                      # Same structure as above
```

**Example with actual labels:**
```text
benchmarks/results/
├── plots/
├── experiment-a/                  # --input experiment-a=http://localhost:8000
├── experiment-b/                  # --input experiment-b=http://localhost:8001
└── experiment-c/                  # --input experiment-c=http://localhost:8002
```

Each concurrency directory contains:
- **`profile_export_genai_perf.json`** - Structured metrics from GenAI-Perf
- **`profile_export.json`** - Raw GenAI-Perf results
- **`inputs.json`** - Generated test inputs

## Customize Benchmarking Behavior

The built-in Python workflow connects to endpoints, benchmarks with genai-perf, and generates plots. If you want to modify the behavior:

1. **Extend the workflow**: Modify `benchmarks/utils/workflow.py` to add custom deployment types or metrics collection

2. **Generate different plots**: Modify `benchmarks/utils/plot.py` to generate a different set of plots for whatever you wish to visualize.

3. **Direct module usage**: Use individual Python modules (`benchmarks.utils.benchmark`, `benchmarks.utils.plot`) for granular control over each step of the benchmarking process.

The Python benchmarking module provides a complete end-to-end benchmarking experience with full control over the workflow.
