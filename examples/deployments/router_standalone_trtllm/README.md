<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Router Standalone - TensorRT-LLM

A standalone implementation of KvRouter that demonstrates usage with TensorRT-LLM workers, without dependency on the dynamo runtime, etcd control plane, or nats event plane.

## Overview

This example shows how to use KvRouter with TensorRT-LLM workers to intelligently route requests across multiple GPUs based on KV cache overlap and load metrics. The router maintains a view of each worker's cached blocks and routes new requests to the worker with the best combination of cache overlap and available capacity.

This is a TensorRT-LLM version of the vLLM-based router standalone example. The core routing logic and RadixTree data structure remain the same, but the worker implementation uses TensorRT-LLM's LLM API instead of vLLM's AsyncLLM.

## Key Differences from vLLM Version

### Backend Engine
- Uses TensorRT-LLM's `LLM` API with pytorch backend
- Configured with `KvCacheConfig` for KV cache event emission
- Uses `tensorrt_llm.llmapi.tokenizer.tokenizer_factory()` for tokenization

### Event APIs
- Metrics: `llm.get_stats_async()` streams engine statistics
- KV Events: `llm.get_kv_cache_events_async()` streams cache events
- Both are published over ZMQ to the router

### Request Processing
- Manual chat template application using tokenizer's `apply_chat_template()`
- Streaming responses via `llm.generate_async()`
- OpenAI-compatible response formatting (without vLLM's serving components)

## How It Works

### Core Architecture

The router uses a **RadixTree** data structure (written in Rust) to efficiently track which blocks each worker has cached. When a new request arrives, the router:

1. Uses `find_matches` to calculate overlap scores (number of matching blocks) between the request and each worker's cached blocks
2. Combines this with current load metrics to select the optimal worker
3. Routes the request to the chosen worker for processing

### Event-Driven Updates

The router receives two types of events from TensorRT-LLM engines:

1. **KV Events**: Emitted automatically when blocks are stored/removed from cache
2. **Load Metrics**: GPU cache usage and waiting request count

These events keep the router's view of worker state up-to-date in real-time.

## Components

### `worker.py`
- **TrtllmWorkers**: Manages multiple TensorRT-LLM worker processes
- Each worker runs on a separate GPU with KV cache event emission enabled
- Publishes metrics and KV events over ZMQ
- Provides `direct()` method for sending requests to specific workers

### `router.py`
- **KvRouter**: Core routing logic using RadixTree (copied from vLLM version)
- Subscribes to KV cache events and load metrics from workers
- Implements `get_best_worker()` to select optimal routing destination
- Runs background tasks to periodically update worker states

### `api.py`
- **ServiceAPI**: FastAPI server providing OpenAI-compatible chat completions endpoint
- Uses TensorRT-LLM's tokenizer for chat template application and tokenization
- Routes requests through the router to select best worker
- Streams responses in OpenAI format

### `perf.sh`
- Benchmarking script using `aiperf` to test the router setup
- Configured for streaming chat completions with synthetic workloads
- Tests concurrent requests to evaluate routing performance

## Requirements

- TensorRT-LLM with pytorch backend
- Multiple GPUs (one per worker)
- Python 3.10+
- Required packages: fastapi, uvicorn, httpx, zmq, tensorrt_llm

## Usage

1. **Start the router API**:
   ```bash
   python api.py \
     --model Qwen/Qwen2.5-0.5B-Instruct \
     --num-workers 2 \
     --block-size 32 \
     --base-kv-events-port 5557 \
     --base-metrics-port 5657 \
     --router-port 7000 \
     --http-port 8000
   ```

   Note: TensorRT-LLM uses block_size=32 by default, not 64 like vLLM.

   The script will:
   - Initialize TensorRT-LLM engines on each GPU
   - Start ZMQ publishers for metrics and KV events
   - Start the router service
   - Start the OpenAI-compatible API server

2. **Ping the endpoint (optional)**:
   ```bash
   ./ping.sh
   ```

3. **Run performance benchmark**:
   ```bash
   ./perf.sh
   ```

## Configuration

### Command-line Arguments

- `--model`: HuggingFace model name (default: Qwen/Qwen2.5-0.5B-Instruct)
- `--num-workers`: Number of GPU workers (default: 2)
- `--block-size`: KV cache block size (default: 32, TensorRT-LLM's default)
- `--base-kv-events-port`: Base port for KV events ZMQ (default: 5557)
- `--base-metrics-port`: Base port for metrics ZMQ (default: 5657)
- `--router-port`: Router HTTP service port (default: 7000)
- `--http-port`: API server port (default: 8000)

### Port Assignment

Workers use sequential ports:
- Worker 0: KV events on 5557, metrics on 5657
- Worker 1: KV events on 5558, metrics on 5658
- Worker N: KV events on 5557+N, metrics on 5657+N

## Example Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

## Architecture Diagram

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────┐
│   API Server    │
│   (api.py)      │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│     Router      │──┐
│   (router.py)   │  │ ZMQ (KV Events)
└────────┬────────┘  │ ZMQ (Metrics)
         │           │
         │ Select    │
         │ Worker    │
         ▼           │
┌─────────────────┐ │
│  TrtllmWorkers  │ │
│   (worker.py)   │◄┘
└─────────────────┘
    │         │
    ▼         ▼
  GPU 0     GPU 1
```

## Notes

- This is a standalone toy implementation for pedagogical purposes
- Production dynamo uses NATS for events and etcd for service discovery
- Each worker needs its own GPU (set via CUDA_VISIBLE_DEVICES)
- TensorRT-LLM models may take time to compile on first run
- Block size should match the model's configuration for optimal cache reuse

## Troubleshooting

**Issue**: Workers fail to initialize
- Check GPU availability and memory
- Ensure CUDA is properly installed
- Try a smaller model if memory is limited

**Issue**: KV Event "IterationResult is not properly instantiated" error
- This is a known limitation in some TensorRT-LLM versions
- KV events may only work after processing the first request
- The system will continue to work in degraded mode (load balancing only, no cache overlap tracking)
- To fix: ensure you're using TensorRT-LLM >= 1.0.0 with pytorch backend
- Workaround: the error can be safely ignored - routing will still work based on load metrics

**Issue**: Router not receiving events
- Verify ZMQ ports are not in use
- Check firewall settings
- Review worker logs for event publishing errors
- KV events may require processing at least one request first

**Issue**: Chat template errors
- Some models may not have chat templates
- Fallback formatting will be used automatically
- You can customize `_format_messages_simple()` for your model

## Comparison with vLLM Version

| Aspect | vLLM Version | TensorRT-LLM Version |
|--------|--------------|---------------------|
| Engine | vLLM AsyncLLM | TensorRT-LLM LLM |
| Backend | vLLM v1 | pytorch backend |
| Tokenizer | vLLM's wrapper | tensorrt_llm tokenizer_factory |
| Chat Preprocessing | OpenAI serving components | Manual template application |
| Event Format | Same | Same |
| Router Logic | Same (RadixTree) | Same (RadixTree) |
| Communication | ZMQ | ZMQ |

## See Also

- [vLLM Router Standalone](../router_standalone/) - Original vLLM version
- [TensorRT-LLM Documentation](https://github.com/NVIDIA/TensorRT-LLM)
- [Dynamo Documentation](../../../docs/)

