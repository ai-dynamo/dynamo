<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Benchmark Scripts

This folder contains scripts and utilities for benchmarking disaggregated (and baseline) inference configurations with various GPU topologies and model sizes. The primary scripts are:

1. **`bench_2GPUs_8B.sh`**
2. **`bench_8GPUs_70B.sh`**
3. **`process_results.py`**
4. **`benchmark.py`**

The workflow is organized into two main parts:
- **Running Benchmarks** (via `.sh` scripts and `benchmark.py`)
- **Processing & Visualizing Results** (via `process_results.py`)

You can also use [multi_node/](multi_node/) scripts to run benchmarks across multiple nodes.

- **Required Python modules** (used primarily by `process_results.py` and `benchmark.py`):
  - `requests`
  - `matplotlib`
  - `seaborn`
  - `pandas`

## ISL (Input Sequence Length) and OSL (Output Sequence Length)

- **ISL (Input Sequence Length)**: Total tokens in the prompt provided to the model. We split ISL into:
  - **ISL-Shared**: The portion of the prompt that is repeated across requests (sometimes called "prefix").
  - **ISL-Unique**: The portion of the prompt that is unique per request (sometimes called "suffix").

- **OSL (Output Sequence Length)**: The number of tokens the model generates in its output.

For example, if your prompt has 500 tokens of shared context (ISL-Shared) plus 250 tokens that differ for each request (ISL-Unique), then `ISL = 750`. If the model generates 128 tokens on average, `OSL = 128`.


Below is a quick reference for each script and how to use it.

---

## 1. `bench_2GPUs_8B.sh`

**Purpose**
Runs a benchmark of a small model (8B) across two GPUs. This script follows these steps:

1. Starts supporting services:
   - A local [NATS server](https://nats.io/) for internal messaging.
   - An [etcd](https://etcd.io/) server for configuration storage.

2. Registers a model with the HTTP endpoint using `llmctl http add chat-models ...`.

3. Sets up a Python virtual environment for Triton, then launches:
   - **Prefill worker** (on `CUDA_VISIBLE_DEVICES=0`)
   - **Decode worker** (on `CUDA_VISIBLE_DEVICES=1`)

4. Executes a concurrency-based load test (concurrency from 1 to 256, in powers of 2) using `benchmark.py`.

5. Kills the processes to clean up, then runs a baseline test using `vllm serve` with tensor parallel = 2 and repeats the same concurrency-based benchmark.

6. Once complete, calls `process_gap_results.py` to generate performance plots.

### How to Use

1. **Check your environment**:
   - Dynamo container includes all dependencies below so use ``build.sh`` to build it and ``run.sh`` interactive session.
   - You need to ensure the script can access `nats-server`, `etcd`, `http`, `llmctl`, `vllm`, and `Python` dependencies.

2. **Run**:
   ```bash
   bash /workspace/examples/python_rs/llm/vllm/benchmark/bench_2GPUs_8B.sh
   ```
   This will start all services, run the benchmark, generate artifacts in `./artifacts/...`, and produce performance plots.


---

## 2. `bench_8GPUs_70B.sh`

**Purpose**
Demonstrates the use of more GPUs (8 in total) and the same disaggregated architecture for a bigger model. It follows the same overall pattern as `bench_2GPUs_8B.sh`, but configures:

- **Two** prefill workers:
  - `CUDA_VISIBLE_DEVICES=0,1` (tensor parallel size = 2)
  - `CUDA_VISIBLE_DEVICES=2,3` (tensor parallel size = 2)
- **One** decode worker:
  - `CUDA_VISIBLE_DEVICES=4,5,6,7` (tensor parallel size = 4)

---

## 3. `process_results.py`

**Purpose**
Parses and visualizes the benchmark outputs. It scans each subdirectory (organized by concurrency level, timestamps, etc.) to locate the `my_profile_export_genai_perf.json` files, aggregates them, and generates a series of plots such as:

- **Pareto Frontier** (`pareto_plot.png`)
- **Inter-token latency** (`plot_itl.png`)
- **Time to first token** (`plot_ttft.png`)
- **Request throughput** (`plot_req.png`)
- **Basic throughput vs concurrency** (`plot.png`)

It also produces a CSV (`results.csv`) with aggregated data.

### Usage

Typically, each `.sh` script automatically calls `process_gap_results.py`. If you want to run it manually:

```bash
python3 process_gap_results.py /path/to/artifacts "My Awesome Benchmark"
```

Where:
- `/path/to/artifacts` is the directory containing concurrency runs, each of which has performance JSON files.
- `"My Awesome Benchmark"` is a title string that appears in generated plots.

---

## 4. `benchmark.py`

**Purpose**
This is the internal script used by the `.sh` scripts to drive concurrency- or request-based load tests against your endpoint. It:

1. Waits until the server is available (by sending a test request).
2. Spawns `genai-perf profile ...` with the correct parameters (RPS or concurrency).
3. Collects results in the specified `artifact-dir`.

### Key Arguments

- `--isl-cached`: Size of the prompt portion that is reused
- `--isl-uncached`: Size of the prompt portion that is random.
- `--osl`: Output sequence length.
- `--model`: Model name (must match your endpoint).
- `--tokenizer`: Tokenizer name (passed to `genai-perf`).
- `--url`: Endpoint base URL like ``http://localhost:8080``. The chat completions part ``v1/chat/completions`` is add inside script at the end of base URL.
- `--artifact-dir`: Where to store the run artifacts.
- `--load-type`: `rps` or `concurrency`.
- `--load-value`: For concurrency, how many concurrent requests. For RPS, the request rate per second.
- `--request-count`: Number of requests to send. If not specified in concurrency mode, the script calculates a default.

**Note**: If using `load-type=rps`, you must explicitly supply `--request-count`, since there is no easy default in RPS mode.

### Example

```bash
python benchmark.py \
  --isl-cached 0 \
  --isl-uncached 3000 \
  --osl 150 \
  --model neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic \
  --url http://localhost:8080 \
  --artifact-dir ./artifacts/test_run/ \
  --load-type concurrency \
  --load-value 8 \
  --tokenizer neuralmagic/DeepSeek-R1-Distill-Llama-8B-FP8-dynamic \
  --verbose
```

