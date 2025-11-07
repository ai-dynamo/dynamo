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

# LLM Deployment using TensorRT-LLM

This directory contains examples and reference implementations for deploying Large Language Models (LLMs) in various configurations using TensorRT-LLM.

## Use the Latest Release

We recommend using the latest stable release of dynamo to avoid breaking changes:

[![GitHub Release](https://img.shields.io/github/v/release/ai-dynamo/dynamo)](https://github.com/ai-dynamo/dynamo/releases/latest)

You can find the latest release [here](https://github.com/ai-dynamo/dynamo/releases/latest) and check out the corresponding branch with:

```bash
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))
```

---

## Table of Contents
- [Feature Support Matrix](#feature-support-matrix)
- [Quick Start](#quick-start)
- [Single Node Examples](#single-node-examples)
- [Advanced Examples](#advanced-examples)
- [KV Cache Transfer](#kv-cache-transfer-in-disaggregated-serving)
- [Client](#client)
- [Benchmarking](#benchmarking)
- [Multimodal Support](#multimodal-support)
- [Logits Processing](#logits-processing)
- [Performance Sweep](#performance-sweep)

## Feature Support Matrix

### Core Dynamo Features

| Feature | TensorRT-LLM | Notes |
|---------|--------------|-------|
| [**Disaggregated Serving**](../../../docs/design_docs/disagg_serving.md) | ✅ |  |
| [**Conditional Disaggregation**](../../../docs/design_docs/disagg_serving.md#conditional-disaggregation) | 🚧 | Not supported yet |
| [**KV-Aware Routing**](../../../docs/router/kv_cache_routing.md) | ✅ |  |
| [**SLA-Based Planner**](../../../docs/planner/sla_planner.md) | ✅ |  |
| [**Load Based Planner**](../../../docs/planner/load_planner.md) | 🚧 | Planned |
| [**KVBM**](../../../docs/kvbm/kvbm_architecture.md) | ✅ | |

### Large Scale P/D and WideEP Features

| Feature            | TensorRT-LLM | Notes                                                           |
|--------------------|--------------|-----------------------------------------------------------------|
| **WideEP**         | ✅           |                                                                 |
| **DP Rank Routing**| ✅           |                                                                 |
| **GB200 Support**  | ✅           |                                                                 |

## TensorRT-LLM Quick Start

Below we provide a guide that lets you run all of our the common deployment patterns on a single node.

### Start NATS and ETCD in the background

Start using [Docker Compose](../../../deploy/docker-compose.yml)

```bash
docker compose -f deploy/docker-compose.yml up -d
```

### Build container

```bash
# TensorRT-LLM uses git-lfs, which needs to be installed in advance.
apt-get update && apt-get -y install git git-lfs

# On an x86 machine:
./container/build.sh --framework trtllm

# On an ARM machine:
./container/build.sh --framework trtllm --platform linux/arm64

# Build the container with the default experimental TensorRT-LLM commit
# WARNING: This is for experimental feature testing only.
# The container should not be used in a production environment.
./container/build.sh --framework trtllm --tensorrtllm-git-url https://github.com/NVIDIA/TensorRT-LLM.git --tensorrtllm-commit main
```

### Run container

```bash
./container/run.sh --framework trtllm -it
```

## Single Node Examples

> [!IMPORTANT]
> Below we provide some simple shell scripts that run the components for each configuration. Each shell script is simply running the `python3 -m dynamo.frontend <args>` to start up the ingress and using `python3 -m dynamo.trtllm <args>` to start up the workers. You can easily take each command and run them in separate terminals.

This figure shows an overview of the major components to deploy:

```
+------+      +-----------+      +------------------+             +---------------+
| HTTP |----->| processor |----->|      Worker1     |------------>|    Worker2    |
|      |<-----|           |<-----|                  |<------------|               |
+------+      +-----------+      +------------------+             +---------------+
                  |    ^                  |
       query best |    | return           | publish kv events
           worker |    | worker_id        v
                  |    |         +------------------+
                  |    +---------|     kv-router    |
                  +------------->|                  |
                                 +------------------+
```

**Note:** The diagram above shows all possible components in a deployment. In disaggregated serving, Worker1 acts as the decode worker and Worker2 as the prefill worker, with the unified frontend coordinating request routing between them.

### Aggregated
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/agg.sh
```

### Aggregated with KV Routing
```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/agg_router.sh
```

### Disaggregated

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/disagg.sh
```

### Disaggregated with KV Routing

> [!IMPORTANT]
> In disaggregated workflow, requests are routed to the prefill worker to maximize KV cache reuse.

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
./launch/disagg_router.sh
```

### Aggregated with Multi-Token Prediction (MTP) and DeepSeek R1
```bash
cd $DYNAMO_HOME/examples/backends/trtllm

export AGG_ENGINE_ARGS=./recipes/deepseek-r1/trtllm/agg/mtp/mtp_agg.yaml
export SERVED_MODEL_NAME="nvidia/DeepSeek-R1-FP4"
# nvidia/DeepSeek-R1-FP4 is a large model
export MODEL_PATH="nvidia/DeepSeek-R1-FP4"
./launch/agg.sh
```

Notes:
- There is a noticeable latency for the first two inference requests. Please send warm-up requests before starting the benchmark.
- MTP performance may vary depending on the acceptance rate of predicted tokens, which is dependent on the dataset or queries used while benchmarking. Additionally, `ignore_eos` should generally be omitted or set to `false` when using MTP to avoid speculating garbage outputs and getting unrealistic acceptance rates.

## Advanced Examples

Below we provide a selected list of advanced examples. Please open up an issue if you'd like to see a specific example!

### Multinode Deployment

For comprehensive instructions on multinode serving, see the [multinode-examples.md](./multinode/multinode-examples.md) guide. It provides step-by-step deployment examples and configuration tips for running Dynamo with TensorRT-LLM across multiple nodes. While the walkthrough uses DeepSeek-R1 as the model, you can easily adapt the process for any supported model by updating the relevant configuration files. You can see [Llama4+eagle](./llama4_plus_eagle.md) guide to learn how to use these scripts when a single worker fits on the single node.

### Speculative Decoding
- **[Llama 4 Maverick Instruct + Eagle Speculative Decoding](./llama4_plus_eagle.md)**

### Kubernetes Deployment

For complete Kubernetes deployment instructions, configurations, and troubleshooting, see [TensorRT-LLM Kubernetes Deployment Guide](../../../examples/backends/trtllm/deploy/README.md).

### Client

See [client](../../../docs/backends/sglang/README.md#testing-the-deployment) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which is running `python3 -m dynamo.frontend <args>`.

### Benchmarking

To benchmark your deployment with AIPerf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../../benchmarks/llm/perf.sh)

## KV Cache Transfer in Disaggregated Serving

Dynamo with TensorRT-LLM supports two methods for transferring KV cache in disaggregated serving: UCX (default) and NIXL (experimental). For detailed information and configuration instructions for each method, see the [KV cache transfer guide](./kv-cache-transfer.md).


## Request Migration

You can enable [request migration](../../../docs/fault_tolerance/request_migration.md) to handle worker failures gracefully. Use the `--migration-limit` flag to specify how many times a request can be migrated to another worker:

```bash
# For decode and aggregated workers
python3 -m dynamo.trtllm ... --migration-limit=3
```

> [!IMPORTANT]
> **Prefill workers do not support request migration** and must use `--migration-limit=0` (the default). Prefill workers only process prompts and return KV cache state - they don't maintain long-running generation requests that would benefit from migration.

See the [Request Migration Architecture](../../../docs/fault_tolerance/request_migration.md) documentation for details on how this works.

## Request Cancellation

When a user cancels a request (e.g., by disconnecting from the frontend), the request is automatically cancelled across all workers, freeing compute resources for other requests.

### Cancellation Support Matrix

| | Prefill | Decode |
|-|---------|--------|
| **Aggregated** | ✅ | ✅ |
| **Disaggregated** | ✅ | ✅ |

For more details, see the [Request Cancellation Architecture](../../fault_tolerance/request_cancellation.md) documentation.

## Client

See [client](../../../docs/backends/sglang/README.md#testing-the-deployment) section to learn how to send request to the deployment.

NOTE: To send a request to a multi-node deployment, target the node which is running `python3 -m dynamo.frontend <args>`.

## Benchmarking

To benchmark your deployment with AIPerf, see this utility script, configuring the
`model` name and `host` based on your deployment: [perf.sh](../../../benchmarks/llm/perf.sh)

## Multimodal support

Dynamo with the TensorRT-LLM backend supports multimodal models, enabling you to process both text and images (or pre-computed embeddings) in a single request. For detailed setup instructions, example requests, and best practices, see the [Multimodal Support Guide](./multimodal_support.md).

## Logits Processing

Logits processors let you modify the next-token logits at every decoding step (e.g., to apply custom constraints or sampling transforms). Dynamo provides a backend-agnostic interface and an adapter for TensorRT-LLM so you can plug in custom processors.

### How it works
- **Interface**: Implement `dynamo.logits_processing.BaseLogitsProcessor` which defines `__call__(input_ids, logits)` and modifies `logits` in-place.
- **TRT-LLM adapter**: Use `dynamo.trtllm.logits_processing.adapter.create_trtllm_adapters(...)` to convert Dynamo processors into TRT-LLM-compatible processors and assign them to `SamplingParams.logits_processor`.
- **Examples**: See example processors in `lib/bindings/python/src/dynamo/logits_processing/examples/` ([temperature](../../../lib/bindings/python/src/dynamo/logits_processing/examples/temperature.py), [hello_world](../../../lib/bindings/python/src/dynamo/logits_processing/examples/hello_world.py)).

### Quick test: HelloWorld processor
You can enable a test-only processor that forces the model to respond with "Hello world!". This is useful to verify the wiring without modifying your model or engine code.

```bash
cd $DYNAMO_HOME/examples/backends/trtllm
export DYNAMO_ENABLE_TEST_LOGITS_PROCESSOR=1
./launch/agg.sh
```

Notes:
- When enabled, Dynamo initializes the tokenizer so the HelloWorld processor can map text to token IDs.
- Expected chat response contains "Hello world".

### Bring your own processor
Implement a processor by conforming to `BaseLogitsProcessor` and modify logits in-place. For example, temperature scaling:

```python
from typing import Sequence
import torch
from dynamo.logits_processing import BaseLogitsProcessor

class TemperatureProcessor(BaseLogitsProcessor):
    def __init__(self, temperature: float = 1.0):
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def __call__(self, input_ids: Sequence[int], logits: torch.Tensor):
        if self.temperature == 1.0:
            return
        logits.div_(self.temperature)
```

Wire it into TRT-LLM by adapting and attaching to `SamplingParams`:

```python
from dynamo.trtllm.logits_processing.adapter import create_trtllm_adapters
from dynamo.logits_processing.examples import TemperatureProcessor

processors = [TemperatureProcessor(temperature=0.7)]
sampling_params.logits_processor = create_trtllm_adapters(processors)
```

### Current limitations
- Per-request processing only (batch size must be 1); beam width > 1 is not supported.
- Processors must modify logits in-place and not return a new tensor.
- If your processor needs tokenization, ensure the tokenizer is initialized (do not skip tokenizer init).

## Performance Sweep

For detailed instructions on running comprehensive performance sweeps across both aggregated and disaggregated serving configurations, see the [TensorRT-LLM Benchmark Scripts for DeepSeek R1 model](../../../examples/backends/trtllm/performance_sweeps/README.md). This guide covers recommended benchmarking setups, usage of provided scripts, and best practices for evaluating system performance.

## Dynamo KV Block Manager Integration

Dynamo with TensorRT-LLM currently supports integration with the Dynamo KV Block Manager. This integration can significantly reduce time-to-first-token (TTFT) latency, particularly in usage patterns such as multi-turn conversations and repeated long-context requests.

Here is the instruction: [Running KVBM in TensorRT-LLM](./../../../docs/kvbm/trtllm-setup.md) .
