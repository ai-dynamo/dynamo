<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Standalone Router

A backend-agnostic standalone KV-aware router service for Dynamo deployments.

## Overview

The standalone router provides configurable KV-aware routing for any set of workers in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions.

This component is **fully configurable** and works with any Dynamo backend (vLLM, TensorRT-LLM, SGLang, etc.) and any worker endpoint.

## Features

- **KV-aware routing**: Routes requests to workers with maximum KV cache overlap
- **Backend-agnostic**: Works with any backend that exposes workers via the Dynamo runtime
- **Fully configurable**: All KvRouter settings exposed via CLI (overlap weight, temperature, replica sync, etc.)
- **Flexible endpoint**: Routes to any worker endpoint specified via `--endpoint` argument
- **State management**: Configurable state persistence and active block tracking
- **Resource cleanup**: Provides endpoints for freeing request resources

## Usage

### Command Line

```bash
python -m dynamo.router_standalone \
    --endpoint dynamo.prefill.generate \
    --block-size 64 \
    --router-reset-states \
    --no-track-active-blocks
```

### Arguments

**Required:**
- `--endpoint`: Full endpoint path for workers in the format `namespace.component.endpoint` (e.g., `dynamo.prefill.generate`)

**Router Configuration:**
- `--block-size`: KV cache block size for routing decisions (default: 128)
- `--kv-overlap-score-weight`: Weight for overlap score in worker selection. Higher values prioritize KV cache reuse (default: 1.0)
- `--router-temperature`: Temperature for worker sampling via softmax. Higher values promote randomness, 0 is deterministic (default: 0.0)
- `--no-kv-events`: Disable KV events. When set, uses ApproxKvRouter for predicting blocks (default: events enabled)
- `--router-replica-sync`: Enable replica synchronization across multiple router instances (default: False)
- `--router-snapshot-threshold`: Number of messages before triggering a snapshot (default: 1000000)
- `--router-reset-states`: Reset router state on startup, purging stream and object store (default: False)
- `--no-track-active-blocks`: Disable tracking of active blocks for load balancing (default: tracking enabled)
- `--log-level`: Logging level - DEBUG, INFO, WARNING, or ERROR (default: INFO)

## Architecture

The standalone router exposes two endpoints via the Dynamo runtime:

1. **`find_best_worker`**: Given a request with token IDs, returns the best worker to handle it
2. **`free`**: Cleans up router state when a request completes

Clients query the `find_best_worker` endpoint to determine which worker should process each request, then call the selected worker directly.

## Example: Disaggregated Serving with Prefill Workers

See [`components/backends/vllm/launch/disagg_router.sh`](../backends/vllm/launch/disagg_router.sh) for a complete example.

```bash
# Start frontend router for decode workers
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0  # Pure load balancing for decode

# Start standalone router for prefill workers
python -m dynamo.router_standalone \
    --endpoint dynamo.prefill.generate \
    --block-size 64 \
    --router-reset-states \
    --no-track-active-blocks

# Start decode workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 &

# Start prefill workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 --is-prefill-worker &
```

**Why `--no-track-active-blocks` for prefill routing?**

Active block tracking is used for load balancing across decode (generation) phases. For prefill-only routing, decode load is not relevant, so disabling this reduces overhead and simplifies the router state.

## Configuration Best Practices

**For Prefill Worker Routing:**
- Use `--no-track-active-blocks` (prefill has no decode phase)
- Use `--router-reset-states` on first startup to clear any previous state
- Consider `--kv-overlap-score-weight 1.0` or higher to maximize cache reuse

**For Multi-Purpose Routing:**
- Keep active block tracking enabled (default)
- Tune `--kv-overlap-score-weight` based on your TTFT vs ITL priorities
- Use `--router-replica-sync` for multi-router deployments

**Block Size Matching:**

The block size must match across:
- Standalone router (`--block-size`)
- Frontend router (`--kv-cache-block-size`)
- All worker instances (`--block-size`)

**Endpoint Matching:**

The `--endpoint` argument must match where your target workers register. For example:
- vLLM prefill workers: `dynamo.prefill.generate`
- vLLM decode workers: `dynamo.backend.generate`
- Custom workers: `<your_namespace>.<your_component>.<your_endpoint>`

## Integration with Backends

To integrate the standalone router with a backend:

1. Clients should query the `router_standalone.find_best_worker` endpoint before sending requests
2. Workers should register at the endpoint specified by the `--endpoint` argument
3. Clients should call the `router_standalone.free` endpoint when requests complete

See [`components/backends/vllm/src/dynamo/vllm/handlers.py`](../backends/vllm/src/dynamo/vllm/handlers.py) for a reference implementation (search for `prefill_router_client`).

## Monitoring

The standalone router exposes metrics with the label `service=router_standalone` for both endpoints.

## See Also

- [KV Cache Routing Architecture](../../docs/architecture/kv_cache_routing.md) - Detailed explanation of KV-aware routing
- [Frontend Router](../frontend/README.md) - Main HTTP frontend with integrated routing
- [Router Benchmarking](../../benchmarks/router/README.md) - Performance testing and tuning
