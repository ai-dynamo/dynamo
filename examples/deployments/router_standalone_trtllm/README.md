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

Key features:
- **KV cache-aware routing**: Routes requests to workers with matching cached blocks
- **Multimodal support**: Handles vision-language models (e.g., Qwen2-VL) with image inputs
- **MM hash routing**: Identical images produce identical hashes for cache reuse

## How It Works

### Core Architecture

The router uses a **RadixTree** data structure (written in Rust) to efficiently track which blocks each worker has cached. When a new request arrives, the router:

1. Tokenizes the request and computes block hashes (including MM hashes for images)
2. Uses `find_matches` to calculate overlap scores between the request and each worker's cached blocks
3. Combines this with current load metrics to select the optimal worker
4. Routes the request to the chosen worker for processing

### Multimodal Routing

For vision-language models:
1. Images are processed using `default_multimodal_input_loader` from TensorRT-LLM
2. Image placeholders are expanded to visual tokens using HuggingFace `AutoProcessor`
3. `apply_mm_hashes` computes a content hash for each image
4. The MM hash is included in block hash computation, so identical images produce cache hits

### Event-Driven Updates

The router receives two types of events from TensorRT-LLM engines:

1. **KV Events**: Emitted automatically when blocks are stored/removed from cache (includes `mm_keys` for multimodal)
2. **Load Metrics**: GPU cache usage and waiting request count

## Components

### `worker.py`
- **TrtllmWorkers**: Manages multiple TensorRT-LLM worker processes
- Each worker runs on a separate GPU with KV cache event emission enabled
- Publishes metrics and KV events over ZMQ
- Extracts `mm_hash` from TRTLLM's `mm_keys` field for multimodal routing

### `router.py`
- **KvRouter**: Core routing logic using RadixTree
- Subscribes to KV cache events and load metrics from workers
- Implements `get_best_worker()` to select optimal routing destination

### `api.py`
- **ServiceAPI**: FastAPI server providing OpenAI-compatible chat completions endpoint
- Handles multimodal inputs (images) via `default_multimodal_input_loader`
- Computes block hashes including MM hashes for routing decisions
- Streams responses in OpenAI format

### `test_router.py`
- Comprehensive test suite for router functionality
- Includes local hash computation tests and server-side multimodal tests
- Run with `--mm-only` for multimodal-specific tests

## Requirements

- **TensorRT-LLM >= 1.2.0rc6**: You need TensorRT-LLM version 1.2.0rc6 or later, which includes multimodal information (`mm_keys`) in KV cache events. This is required for MM hash-based routing. See [PR #9604](https://github.com/NVIDIA/TensorRT-LLM/pull/9604) for details.
- TensorRT-LLM with pytorch backend
- Multiple GPUs (one per worker)
- Python 3.10+
- Required packages: fastapi, uvicorn, httpx, zmq, tensorrt_llm, transformers

## Usage

### 1. Start the API Server

```bash
python api.py \
  --model Qwen/Qwen2-VL-2B-Instruct \
  --num-workers 2 \
  --block-size 32 \
  --base-kv-events-port 5557 \
  --base-metrics-port 5657 \
  --router-port 7000 \
  --http-port 8000
```

This will:
- Initialize TensorRT-LLM engines on each GPU
- Start ZMQ publishers for metrics and KV events
- Start the router service
- Start the OpenAI-compatible API server

### 2. Test with curl

**Text-only request:**
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{"role": "user", "content": "Hello, how are you?"}],
    "max_tokens": 100,
    "stream": false
  }' | jq
```

**Multimodal request (with images):**
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe both images in detail."},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg"}},
        {"type": "image_url", "image_url": {"url": "http://images.cocodataset.org/test2017/000000000001.jpg"}}
      ]
    }],
    "max_tokens": 500,
    "stream": false
  }' | jq
```

### 3. Run Tests

```bash
# Run all tests
python test_router.py

# Run multimodal tests only
python test_router.py --mm-only

# Verbose output
python test_router.py -v
```

### 4. Check endpoint health

```bash
./ping.sh
```

## Configuration

### Command-line Arguments

- `--model`: HuggingFace model name (default: Qwen/Qwen2-VL-2B-Instruct)
- `--num-workers`: Number of GPU workers (default: 2)
- `--block-size`: KV cache block size (default: 32, TensorRT-LLM's default)
- `--base-kv-events-port`: Base port for KV events ZMQ (default: 5557)
- `--base-metrics-port`: Base port for metrics ZMQ (default: 5657)
- `--router-port`: Router HTTP service port (default: 7000)
- `--http-port`: API server port (default: 8000)

### Environment Variables

- `DYNAMO_DEBUG=1`: Enable debug file dumps to `/tmp/debug_*.txt`
- `LOGLEVEL=DEBUG`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `TRANSFORMERS_ATTN_IMPLEMENTATION=eager`: Disable FlashAttention (set automatically)

### Port Assignment

Workers use sequential ports:
- Worker 0: KV events on 5557, metrics on 5657
- Worker 1: KV events on 5558, metrics on 5658
- Worker N: KV events on 5557+N, metrics on 5657+N

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
└────────┬────────┘  │
         │           │
         │ Select    │
         │ Worker    │
         ▼           │
┌─────────────────┐  │
│  TrtllmWorkers  │  │
│   (worker.py)   │◄-┘
└─────────────────┘
    │         │
    ▼         ▼
  GPU 0     GPU 1
```

## Multimodal KV Cache Routing

When processing multimodal requests:

1. **API Layer** (`api.py`):
   - Parses OpenAI-format messages with `image_url` content
   - Uses `default_multimodal_input_loader` to process images
   - Expands image placeholders to visual tokens via `AutoProcessor`
   - Computes `mm_hash` using `apply_mm_hashes`
   - Includes `mm_hash` in block hash computation for routing

2. **Worker Layer** (`worker.py`):
   - Receives multimodal input and passes to TRTLLM
   - Extracts `mm_hash` from TRTLLM's `mm_keys` in KV events
   - Publishes KV events with `mm_extra_info` to router

3. **Router Layer** (`router.py`):
   - RadixTree matches blocks including MM hash
   - Same image content = same hash = cache hit on same worker

## Notes

- This is a standalone implementation for pedagogical purposes
- Production dynamo uses NATS for events and etcd for service discovery
- Each worker needs its own GPU
- TensorRT-LLM models may take time to compile on first run

## See Also

- [vLLM Router Standalone](../router_standalone/) - Original vLLM version
- [TensorRT-LLM KV Event Documentation](https://nvidia.github.io/TensorRT-LLM/0.21.0/examples/llm_inference_kv_events.html)
