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

# Router Benchmarking Guide

This directory contains scripts for benchmarking the Dynamo router with prefix caching. The benchmarks measure performance improvements from prefix sharing across requests.

## Prerequisites

- NVIDIA GPUs (8 GPUs for default configuration)
- CUDA environment properly configured
- etcd and NATS running (required for Dynamo coordination)
- Required Python packages:
  - `dynamo` package (with vllm and frontend modules)
  - `genai-perf` for benchmarking
  - `matplotlib` for plotting results

### Setting up etcd and NATS

This benchmark requires etcd and NATS. To quickly set them up, run:

```bash
# From the repository root:
docker compose -f deploy/docker-compose.yml up -d
```

This will start both etcd and NATS with the required configurations in the background.

## Scripts Overview

- **`run_engines.sh`** - Launches multiple vLLM worker instances
- **`ping.sh`** - Simple test script to verify the setup is working
- **`prefix_ratio_benchmark.py`** - Main benchmarking script that sweeps prefix ratios
- **`plot_prefix_ratio_comparison.py`** - Generates comparison plots from benchmark results

## Usage Instructions

### Step 1: Launch vLLM Workers

First, start the vLLM worker engines in a terminal.

```bash
# Default: 8 vLLM workers with DeepSeek model (explicitly sets --block-size 64)
./run_engines.sh \
    --num-workers 8 \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Example: 4 vLLM workers with larger model using tensor parallelism (2 GPUs per worker)
./run_engines.sh \
    --num-workers 4 \
    --model-path openai/gpt-oss-120b \
    --tensor-parallel-size 2
```

#### Alternative: Launch vLLM Mock Workers

We also supports running lightweight mock engines that simulate vLLM behavior without performing actual model inference. Mocker engines are useful for testing router logic and performance without GPU requirements. Use the `--mockers` flag to run mocker engines instead of real vLLM workers.

```bash
# Example: Running mocker engines for testing (no GPU required)
./run_engines.sh --mockers \
    --num-workers 8 \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --block-size 64 \
    --speedup-ratio 2.0
```

**Note**: The `--speedup-ratio` parameter controls the inference speed of mocker engines. A higher value (e.g., 2.0) makes the mocker engines simulate faster inference, allowing benchmarks to complete more quickly. This is particularly useful for testing router performance without waiting for realistic inference times.

### Step 2: Start the Router

In a **new terminal**, launch the Dynamo router using the Python CLI:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --kv-cache-block-size 64 \
    --router-reset-states \
    --http-port 8000
```

This starts the router with:
- KV cache routing mode
- Block size of 64 (**Important:** This should match the `--block-size` used by your engines)
- `--router-reset-states` flag to clear the event cache (JetStream) from previous runs (useful for single router benchmarking)
- HTTP port 8000

To see all available router arguments, run:
```bash
python -m dynamo.frontend --help
```

For detailed explanations of router arguments (especially KV cache routing parameters), see the [KV Cache Routing documentation](../../docs/architecture/kv_cache_routing.md).

**Note**: If you're unsure whether your backend engines correctly emit KV events for certain models (e.g., hybrid models like gpt-oss or nemotron nano 2), use the `--no-kv-events` flag to disable KV event tracking and use approximate KV indexing instead:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --kv-cache-block-size 64 \
    --http-port 8000 \
    --no-kv-events
```

### Step 3: Verify Setup

In another terminal, test that everything is working:

```bash
./ping.sh
# Or specify a different port:
./ping.sh 8000
```

This sends a simple test request to the router. You should see a streamed response if everything is configured correctly.

### Step 4: Run Benchmarks

Once the setup is verified, run the prefix ratio benchmark:

```bash
python prefix_ratio_benchmark.py
```

Default configuration:
- Tests prefix ratios: 0.5 (can be customized with `--prefix-ratios 0.1 0.3 0.5 0.7 0.9`)
- Input sequence length: 14000 tokens
- Output sequence length: 200 tokens
- Requests: 200
- Concurrency: 20

You can customize the benchmark:

```bash
# Test multiple prefix ratios
python prefix_ratio_benchmark.py --prefix-ratios 0.1 0.3 0.5 0.7 0.9

# Adjust input/output lengths
python prefix_ratio_benchmark.py --isl 10000 --osl 500

# Change request count and concurrency
python prefix_ratio_benchmark.py --requests 500 --concurrency 50

# Use multiple router endpoints for parallel benchmarking (for testing multiple Router replicas)
python prefix_ratio_benchmark.py --url http://localhost:8000 http://localhost:8001

# Specify output directory
python prefix_ratio_benchmark.py --output-dir results/experiment1
```

### Benchmark Output

The benchmark script generates:

1. **Performance plots** (`prefix_ratio_performance.png`):
   - TTFT (Time to First Token) vs Prefix Ratio
   - Throughput (tokens/s) vs Prefix Ratio

2. **Results summary** (`results_summary.json`):
   - Raw data for all prefix ratios tested
   - Configuration parameters used

3. **Detailed artifacts** (in subdirectories):
   - Full genai-perf profiling data for each run

## Troubleshooting

1. **Workers fail to start**: Check CUDA_VISIBLE_DEVICES and GPU availability
2. **Router connection refused**: Ensure router is running and port is correct
3. **Benchmark timeout**: Decrease concurrency or reduce request count
4. **OOM errors**: Reduce max-num-batched-tokens or max-model-len in run_engines.sh
