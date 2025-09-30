<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Prefill Router

A backend-agnostic centralized prefill router service for disaggregated serving in Dynamo.

## Overview

The prefill router provides KV-aware routing for prefill workers in disaggregated deployments. Instead of each decode worker maintaining its own round-robin client to prefill workers, this service uses `KvRouter` to make intelligent routing decisions based on KV cache state, maximizing prefix cache hit rates across prefill workers.

This component is **backend-agnostic** and works with any Dynamo backend (vLLM, TensorRT-LLM, SGLang, etc.) that follows the standard prefill worker interface.

## Usage

### Command Line

```bash
python -m dynamo.prefill_router \
    --namespace dynamo \
    --block-size 64 \
    --log-level INFO
```

### Arguments

- `--namespace`: Dynamo namespace for discovering prefill workers (default: `dynamo` or `DYN_NAMESPACE` env var)
- `--block-size`: KV cache block size for routing decisions (default: 128)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, or ERROR (default: INFO)

## Architecture

The prefill router exposes two endpoints via the Dynamo runtime:

1. **`find_best_worker`**: Given a request with token IDs, returns the best prefill worker to handle it
2. **`free`**: Cleans up router state when a request completes

Decode workers query the `find_best_worker` endpoint to determine which prefill worker should process each request, then call the selected worker directly.

## Example: Disaggregated Serving

See [`components/backends/vllm/launch/disagg_router.sh`](../backends/vllm/launch/disagg_router.sh) for a complete example of using the prefill router with vLLM workers.

Basic pattern:

```bash
# Start frontend router
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0  # Use load balancing for decode routing

# Start prefill router
python -m dynamo.prefill_router \
    --namespace dynamo \
    --block-size 64

# Start decode workers (will connect to frontend router)
python -m dynamo.vllm --model MODEL_NAME --block-size 64 &

# Start prefill workers (will register with prefill router)
python -m dynamo.vllm --model MODEL_NAME --block-size 64 --is-prefill-worker &
```

## Configuration

The block size must match across:
- Prefill router (`--block-size`)
- Frontend router (`--kv-cache-block-size`)
- All worker instances (`--block-size`)

## Integration with Backends

To integrate the prefill router with a backend:

1. Decode workers should query the `prefill_router.find_best_worker` endpoint before sending requests
2. Prefill workers should register under the `prefill` component name
3. Workers should call the `prefill_router.free` endpoint when requests complete

See [`components/backends/vllm/src/dynamo/vllm/handlers.py`](../backends/vllm/src/dynamo/vllm/handlers.py) for a reference implementation.

## Monitoring

The prefill router exposes metrics with the label `service=prefill_router` for both endpoints.

