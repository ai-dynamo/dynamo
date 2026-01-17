<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Router Standalone

A toy implementation of KvRouter that demonstrates standalone usage without dependency on the dynamo runtime, etcd control plane, or nats event plane.

## Overview

This example shows how to use KvRouter in a standalone fashion to intelligently route requests across multiple vLLM workers based on KV cache overlap and load metrics. The router maintains a view of each worker's cached blocks and routes new requests to the worker with the best combination of cache overlap and available capacity.

> [!Tip]
> The main focus should be put on `router.py` as it contains the bulk of the non-boilerplate code and core routing logic.

## How It Works

### Core Architecture

The router uses a **RadixTree** data structure (written in Rust) to efficiently track which blocks each worker has cached. When a new request arrives, the router:

1. Uses `find_matches` to calculate overlap scores (number of matching blocks) between the request and each worker's cached blocks
2. Combines this with current load metrics to select the optimal worker
3. Routes the request to the chosen worker for processing

### Event-Driven Updates

The router receives two types of events from vLLM engines:

1. **KV Events**: Emitted automatically by vLLM engines when blocks are cached/evicted
2. **Load Metrics**: GPU usage percentage and waiting request count via custom callbacks

These events keep the router's view of worker state up-to-date in real-time.

### Alternative: Pure Predictive Routing

While not implemented in this example, the router can also operate in a pure predictive mode, estimating the radix tree state and loads based solely on the requests it receives, without relying on backend events. This requires simulating / mocking the block managing (e.g. eviction) and the scheduling policies of the backend engine. This is not recommended as there is no real-time feedback from the engines, and the router state may drift out of sync with the engine states. Nevertheless, this is WIP and can be supported in the future via our mocker engines.

## Components

> [!Note]
> This is a standalone toy implementation created for pedagogical purposes to demonstrate the core KvRouter concepts in isolation.
> Our default dynamo router is already very efficient and uses NATS for event communication and etcd for endpoint registration.
> This example intentionally avoids these production components to provide a simpler, self-contained demonstration of the routing logic and cache overlap mechanics.
>
> The toy communication pattern is as follows:
> - **OpenAI Compatible Frontend** – FastAPI application serving OpenAI compatible HTTP API.
> - **Router** – Standalone FastAPI endpoint for best worker selection, with core routines implemented in Rust exposed via Python bindings.
> - **Workers** – Served in-process within the frontend application to reduce complexity and boilerplate, rather than as separate endpoints.

### `router.py`
- **KvRouter**: Core routing logic using RadixTree
- Subscribes to KV cache events and load metrics from workers
- Implements `get_best_worker()` to select optimal routing destination
- Runs background tasks to periodically update worker states

### `worker.py`
- **VllmWorkers**: Manages multiple vLLM worker processes
- Each worker runs on a separate port with KV cache event emission enabled
- Provides `direct()` method for sending requests to specific workers
- Handles worker lifecycle and configuration

### `api.py`
- **RouterAPI**: Minimal FastAPI server providing OpenAI-compatible chat completions endpoint
- Enables in-process communication between router and workers
- Can be easily modified to use external communication (FastAPI clients, dynamo endpoints, etc.)
- Integrates with vLLM's OpenAI serving components for request preprocessing and response formatting

### `perf.sh`
- Benchmarking script using `aiperf` to test the router setup
- Configured for streaming chat completions with synthetic workloads
- Tests concurrent requests to evaluate routing performance

## Usage

1. **Install latest vLLM**:
   ```bash
   uv pip uninstall ai-dynamo-vllm
   uv pip install vllm==0.9.0
   ```
   *Note: This uninstalls the local vLLM patch (`ai-dynamo-vllm`) and replaces it with the latest standard vLLM package.*

2. **Start the router API**:
   For example:
   ```bash
   python api.py \
     --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
     --num-workers 4 \
     --block-size 64 \
     --base-kv-events-port 5557 \
     --base-metrics-port 5657 \
     --router-port 7000 \
     --http-port 8000
    ```

3. **Ping the endpoint (optional)**:
   ```bash
   ./ping.sh
   ```

4. **Run performance benchmark**:
   ```bash
   ./perf.sh
   ```
