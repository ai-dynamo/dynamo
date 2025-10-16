<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Router Benchmarking Guide

This directory contains scripts for benchmarking the Dynamo router with prefix caching. The benchmarks measure performance improvements from prefix sharing across requests.

## Prerequisites

- NVIDIA GPUs (8 GPUs for default configuration)
- (optional) H100 GPUs or later for gpt-oss-120b examples
- CUDA environment properly configured
- etcd and NATS running (required for Dynamo coordination)
- Required Python packages:
  - `dynamo` package (with vllm and frontend modules)
  - `aiperf` for benchmarking
  - `matplotlib` for plotting results
  - `data-generator` package (install with `pip install -e ./benchmarks` from repo root)

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
- **`real_data_benchmark.py`** - Benchmarking script that uses real mooncake-style trace data
- **`plot_prefix_ratio_comparison.py`** - Generates comparison plots from benchmark results

## Usage Instructions

### Step 1: Launch Workers

Make sure you have 8 GPUs for these examples, unless you are using mockers (see below). First, start the worker engines in a terminal.

The script supports three modes:
- **`agg` (default)**: Aggregated/monolithic workers that handle both prefill and decode
- **`decode`**: Workers dedicated to decode (token generation) phase
- **`prefill`**: Workers dedicated to prefill (prompt processing) phase

```bash
# Default: 8 aggregated workers with DeepSeek model (handles both prefill and decode)
./run_engines.sh \
    --num-workers 8 \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Example: 4 workers with larger model using tensor parallelism (2 GPUs per worker)
# NOTE: this requires having Hopper or later GPU SKUs to support MXFP4 precision.
./run_engines.sh \
    --num-workers 4 \
    --model-path openai/gpt-oss-120b \
    --tensor-parallel-size 2
```

#### Disaggregated Serving (Decode + Prefill Workers)

You can launch separate decode and prefill workers for disaggregated serving. This allows you to dedicate specific GPUs to prefill (prompt processing) and decode (token generation) tasks:

```bash
# Launch 4 decode workers (GPUs 0-3)
./run_engines.sh \
    --decode \
    --num-workers 4 \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Launch 4 prefill workers (GPUs 4-7)
./run_engines.sh \
    --prefill \
    --num-workers 4 \
    --base-gpu-offset 4 \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B
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
    --router-reset-states \
    --http-port 8000
```

This starts the router with:
- KV cache routing mode
- `--router-reset-states` flag to clear the event cache (JetStream) from previous runs (useful for single router benchmarking)
- HTTP port 8000

To see all available router arguments, run:
```bash
python -m dynamo.frontend --help
```

For detailed explanations of router arguments (especially KV cache routing parameters), see the [KV Cache Routing documentation](../../docs/architecture/kv_cache_routing.md).

#### Launching a Standalone Router for Prefill Workers (Optional)

If you're using disaggregated serving with separate prefill and decode workers, you should also launch a standalone router for prefill workers. This router handles routing prefill requests to dedicated prefill workers. When using a standalone prefill router, it's recommended to start the frontend (decode router) with `--kv-overlap-score-weight 0` for pure load balancing (as prefix-aware routing is now handled by the standalone router):

```bash
# Start the decode router with pure load balancing
python -m dynamo.frontend \
    --router-mode kv \
    --router-reset-states \
    --http-port 8000 \
    --kv-overlap-score-weight 0

# In another terminal, start the standalone router for prefill workers
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --block-size 64 \
    --router-reset-states \
    --no-track-active-blocks
```

The `--router-reset-states` flag clears any previous state, and `--no-track-active-blocks` disables active block tracking (suitable for prefill-only routing where decode load is not relevant).

**Note**: If you're unsure whether your backend engines correctly emit KV events for certain models (e.g., hybrid models like gpt-oss or nemotron nano 2), use the `--no-kv-events` flag to disable KV event tracking and use approximate KV indexing instead:

```bash
python -m dynamo.frontend \
    --router-mode kv \
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

### Step 4 (Alternative): Run Benchmarks with Real Trace Data

Instead of synthetic benchmarks with controlled prefix ratios, you can benchmark using real trace data in [mooncake-style format](https://github.com/kvcache-ai/Mooncake/blob/d21da178bae8db9651cf18a76824c084145fc725/mooncake_trace.jsonl). This approach uses actual request patterns from production traces, potentially modified with synthesis parameters.

```bash
python real_data_benchmark.py --input-dataset mooncake_trace.jsonl
```

The script can apply various modifications on top of the original trace dataset to simulate different scenarios and workload conditions. This script accepts the same synthesis parameters as the [prefix data generator](../prefix_data_generator/README.md):

**Key parameters:**
- `--num-requests`: Number of requests to synthesize from the trace (default: use all)
- `--speedup-ratio`: Speed up request arrival times (e.g., 2.0 makes requests arrive 2x faster)
- `--prefix-len-multiplier`: Scale the length of shared prefixes (e.g., 2.0 doubles prefix lengths)
- `--prefix-root-multiplier`: Replicate the prefix tree structure N times with different roots
- `--prompt-len-multiplier`: Scale the length of unique user prompts (e.g., 0.5 for shorter prompts)
- `--max-isl`: Filter out requests exceeding this input sequence length

Examples:

```bash
# Use original trace dataset as-is (no synthesis parameters specified)
python real_data_benchmark.py --input-dataset trace.jsonl

# Speed up request rate by 2x and use only first 1000 requests
python real_data_benchmark.py --input-dataset trace.jsonl --num-requests 1000 --speedup-ratio 2.0

# Double prefix lengths to test cache efficiency with longer shared contexts
python real_data_benchmark.py --input-dataset trace.jsonl --prefix-len-multiplier 2.0

# Create more diverse workload by replicating prefix tree 3 times
python real_data_benchmark.py --input-dataset trace.jsonl --prefix-root-multiplier 3
```

> [!Note]
> At the time of writing this documentation, you may need to install the latest aiperf from the main source branch to loadgen on the trace files:
> ```bash
> pip install git+https://github.com/ai-dynamo/aiperf.git#subdirectory=aiperf
> ```
> However, by the time of release, the aiperf version included in the vLLM runtime container should be up to date enough to use as-is.

## Troubleshooting

1. **Workers fail to start**: Check CUDA_VISIBLE_DEVICES and GPU availability
2. **Router connection refused**: Ensure router is running and port is correct
3. **Benchmark timeout**: Decrease concurrency or reduce request count
4. **OOM errors**: Reduce max-num-batched-tokens or max-model-len in run_engines.sh
