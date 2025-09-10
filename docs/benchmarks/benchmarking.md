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
- **DynamoGraphDeployments** (automatically deployed from your manifests)
- **External HTTP endpoints** (existing services, vLLM, TensorRT-LLM, etc.)

You can mix and match these in a single benchmark run using custom labels. Configure your DynamoGraphDeployment manifests for your specific models, hardware, and parallelization needs.

## What This Tool Does

The framework is a wrapper around `genai-perf` that:
- Deploys user-specified `DynamoGraphDeployments` automatically
- Benchmarks any HTTP endpoints (no deployment needed)
- Runs concurrency sweeps across configurable load levels
- Generates comparison plots with your custom labels
- Works with any HuggingFace-compatible model on NVIDIA GPUs (H200, H100, A100, etc.)
- Runs locally and connects to your Kubernetes deployments/endpoints

**Default sequence lengths**: Input: 2000 tokens, Output: 256 tokens (configurable with `--isl` and `--osl`)

**Important**: The `--model` parameter configures GenAI-Perf for benchmarking and provides logging context. The actual model loaded is determined by your deployment manifests. Only one model can be benchmarked at a time across all inputs to ensure fair comparison. The default `--model` value in the benchmarking script is `Qwen/Qwen3-0.6B`, but it must match the model in the manifest(s) and the model deployed at the endpoint(s).

## Prerequisites

1. **Kubernetes cluster with NVIDIA GPUs and Dynamo namespace setup** - You need a Kubernetes cluster with eligible NVIDIA GPUs and a properly configured namespace for Dynamo benchmarking. See the [deploy/utils/README](../../deploy/utils/README.md) for complete setup instructions.

2. **kubectl access** - You need `kubectl` installed and configured to access your Kubernetes cluster. All other required tools (GenAI-Perf, Python, etc.) are included in the Dynamo containers. If you are not working within a Dynamo container, you can install the necessary requirements using `deploy/utils/requirements.txt`. *Note: if you are on Ubuntu 22.04 or lower, you will also need to build perf_analyzer [from source](https://github.com/triton-inference-server/perf_analyzer/blob/main/docs/install.md#build-from-source).*

## Quick Start Examples

The tool can be used to deploy, benchmark and compare Dynamo deployments (DynamoGraphDeployments) on a Kubernetes cluster as well as benchmark and compare servers deployed separately given a URL. In the examples below, Dynamo deployments are specified with a yaml and servers deployed separately by URL.

```bash
export NAMESPACE=benchmarking

# Compare multiple DynamoGraphDeployments of a single backend
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input agg=components/backends/vllm/deploy/agg.yaml \
   --input disagg=components/backends/vllm/deploy/disagg.yaml

# Compare different backend types (vLLM vs TensorRT-LLM)
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input vllm-disagg=components/backends/vllm/deploy/disagg.yaml \
   --input trtllm-disagg=components/backends/trtllm/deploy/disagg.yaml

# Compare Dynamo deployment vs existing deployment (external endpoint)
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input dynamo=components/backends/vllm/deploy/disagg.yaml \
   --input vllm-baseline=http://localhost:8000

# Compare three different configurations
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input dynamo-agg=components/backends/vllm/deploy/agg.yaml \
   --input dynamo-disagg=components/backends/vllm/deploy/disagg.yaml \
   --input external-vllm=http://localhost:8000

# Benchmark single external endpoint
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input production-api=http://your-api:8000

# Custom model and sequence lengths
./benchmarks/benchmark.sh --namespace $NAMESPACE \
   --input my-setup=my-custom-manifest.yaml \
   --model "meta-llama/Meta-Llama-3-8B" --isl 512 --osl 256
```

**Key**: Configure your manifests for your specific models, hardware, and parallelization strategy before benchmarking.

### Important: Image Accessibility

Ensure container images in your DynamoGraphDeployment manifests are accessible:
- **Public images**: Use [Dynamo NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-dynamo/collections/ai-dynamo/artifacts) public releases
- **Custom registries**: Configure proper credentials in your Kubernetes namespace

## Configuration and Usage

### Command Line Options

```bash
./benchmarks/benchmark.sh --namespace NAMESPACE --input <label>=<manifest_path_or_endpoint> [--input <label>=<manifest_path_or_endpoint>]... [OPTIONS]

REQUIRED:
  -n, --namespace NAMESPACE           Kubernetes namespace
  --input <label>=<manifest_path_or_endpoint>  Benchmark input with custom label
                                        - <label>: becomes the name/label in plots
                                        - <manifest_path_or_endpoint>: either a DynamoGraphDeployment manifest or HTTP endpoint URL
                                        Can be specified multiple times for comparisons

OPTIONS:
  -h, --help                    Show help message and examples
  -m, --model MODEL             Model name for GenAI-Perf configuration and logging (default: Qwen/Qwen3-0.6B)
                                NOTE: This must match the model configured in your deployment manifests and endpoints
  -i, --isl LENGTH              Input sequence length (default: 2000)
  -s, --std STDDEV              Input sequence standard deviation (default: 10)
  -o, --osl LENGTH              Output sequence length (default: 256)
  -d, --output-dir DIR          Output directory (default: ./benchmarks/results)
  --verbose                     Enable verbose output
```

### Important Notes

- **Custom Labels**: Each input must have a unique label that becomes the name in plots and results
- **Label Restrictions**: Labels can only contain letters, numbers, hyphens, and underscores. The label `plots` is reserved.
- **Input Types**: Supports DynamoGraphDeployment manifests for automatic deployment, or HTTP endpoints for existing services
- **Model Parameter**: The `--model` parameter configures GenAI-Perf for testing and logging, not deployment (deployment model is determined by the manifest files)
- **Standalone Deployments**: For non-Dynamo backends (vLLM, TensorRT-LLM, SGLang, etc.), you must deploy them manually following their respective Kubernetes deployment guides. The benchmarking framework only supports automatic deployment of DynamoGraphDeployments.
- **Single Model Requirement**: Only one model can be benchmarked at a time across all inputs to ensure fair comparison.

### What Happens During Benchmarking

The script automatically:
1. **Deploys** each DynamoGraphDeployment configuration to Kubernetes if manifests are passed in
2. **Benchmarks** using GenAI-Perf at various concurrency levels (default: 1, 2, 5, 10, 50, 100, 250)
3. **Measures** key metrics: latency, throughput, time-to-first-token
4. **Generates** comparison plots using your custom labels in `./benchmarks/results/plots/`
5. **Cleans up** deployments when complete

### GPU Resource Usage

**Important**: Models are deployed and benchmarked **sequentially**, not in parallel. This means:

- **One deployment at a time**: Each DynamoGraphDeployment is deployed, benchmarked, and cleaned up before the next one starts
- **Full GPU access**: Each deployment gets exclusive access to all available GPUs during its benchmark run
- **Resource isolation**: No resource conflicts between different deployment configurations
- **Fair comparison**: Each configuration is tested under identical resource conditions

This sequential approach ensures:
- **Accurate performance measurements** without interference between deployments
- **Consistent resource allocation** for fair comparison across different configurations
- **Simplified resource management** without complex GPU scheduling
- **Reliable cleanup** between benchmark runs

If you need to benchmark multiple configurations simultaneously, consider using separate Kubernetes namespaces or running benchmarks on different clusters.

### Results Clearing Behavior

**Important**: The benchmark script automatically clears the output directory before each run to ensure clean, reproducible results. This means:
- Previous benchmark results in the same output directory will be completely removed
- Each benchmark run starts with a clean slate
- Results from different runs are not mixed or accumulated

If you want to preserve results from previous runs, use different output directories with the `--output-dir` flag.

### Using Your Own Models and Configuration

The benchmarking framework supports any HuggingFace-compatible LLM model. To benchmark your own custom deployment:

1. **Edit your deployment YAML files** to specify your model in the `--model` argument of the container command
2. **Use the corresponding model name** in the benchmark script's `--model` parameter

**Note**: You can override the default sequence lengths (2000/256 tokens) with `--isl` and `--osl` flags if needed for your specific workload.

### Direct Python Execution

For direct control over the benchmark workflow:

```bash
# Endpoint benchmarking
python3 -u -m benchmarks.utils.benchmark \
   --input trtllm=http://your-endpoint:8000 \
   --namespace $NAMESPACE \
   --isl 2000 \
   --std 10 \
   --osl 256 \
   --output-dir $OUTPUT_DIR

# Deployment benchmarking (any combination)
python3 -u -m benchmarks.utils.benchmark \
   --input agg=$AGG_CONFIG \
   --input disagg=$DISAGG_CONFIG \
   --namespace $NAMESPACE \
   --isl 2000 \
   --std 10 \
   --osl 256 \
   --output-dir $OUTPUT_DIR

# Generate plots separately
python3 -m benchmarks.utils.plot --data-dir $OUTPUT_DIR
```

### Comparison Limitations

The plotting system supports up to 12 different inputs in a single comparison. If you need to compare more than 12 different deployments/endpoints, consider running separate benchmark sessions or grouping related comparisons together.

### Concurrency Configuration

You can customize the concurrency levels using the CONCURRENCIES environment variable:

```bash
# Custom concurrency levels
CONCURRENCIES="1,5,20,50" ./benchmarks/benchmark.sh --namespace $NAMESPACE --input my-test=components/backends/vllm/deploy/disagg.yaml

# Or set permanently
export CONCURRENCIES="1,2,5,10,25,50,100"
./benchmarks/benchmark.sh --namespace $NAMESPACE --input test=disagg.yaml
```

## Understanding Your Results

After benchmarking completes, check `./benchmarks/results/` (or your custom output directory):

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
├── dynamo-agg/                  # --input dynamo-agg=agg.yaml
├── dynamo-disagg/               # --input dynamo-disagg=disagg.yaml
└── external-vllm/               # --input external-vllm=http://localhost:8000
```

Each concurrency directory contains:
- **`profile_export_genai_perf.json`** - Structured metrics from GenAI-Perf
- **`profile_export.json`** - Raw GenAI-Perf results
- **`inputs.json`** - Generated test inputs

## Customize Benchmarking Behavior

The built-in workflow handles DynamoGraphDeployment deployment, benchmarking with genai-perf, and plot generation automatically. If you want to modify the behavior:

1. **Extend the workflow**: Modify `benchmarks/utils/workflow.py` to add custom deployment types or metrics collection

2. **Generate different plots**: Modify `benchmarks/utils/plot.py` to generate a different set of plots for whatever you wish to visualize.

The `benchmark.sh` script provides a complete end-to-end benchmarking experience. For more granular control, use the Python modules directly.
