# Router Standalone

A toy implementation of KvRouter that demonstrates standalone usage without dependency on the dynamo runtime, etcd control plane, or nats event plane.

## Overview

This example shows how to use KvRouter in a standalone fashion to intelligently route requests across multiple vLLM workers based on KV cache overlap and load metrics. The router maintains a view of each worker's cached blocks and routes new requests to the worker with the best combination of cache overlap and available capacity.

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

While not implemented in this example, the router can also operate in a pure predictive mode, estimating the radix tree state and loads based solely on the requests it receives, without relying on backend events.

## Components

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
- Benchmarking script using `genai-perf` to test the router setup
- Configured for streaming chat completions with synthetic workloads
- Tests concurrent requests to evaluate routing performance

## Usage

1. **Start the router API**:
   For example:
   ```bash
   python api.py \
     --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
     --num-workers 4 \
     --block-size 64 \
     --base-kv-events-port 5557 \
     --base-metrics-port 5657 \
     --http-port 8000
    ```

2. **Ping the endpoint (optional)**:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Accept: text/event-stream" \
     -d '{
       "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
       "messages": [{"role": "user", "content": "Hello!"}],
       "stream": true,
       "max_tokens": 100
     }'
   ```

3. **Run performance benchmark**:
   ```bash
   ./perf.sh
   ```

## Notes

This is a standalone toy implementation created for pedagogical purposes to demonstrate the core KvRouter concepts in isolation. Our default dynamo router is already very efficient and uses NATS for event communication and etcd for endpoint registration. This example intentionally avoids these production components to provide a simpler, self-contained demonstration of the routing logic and cache overlap mechanics.