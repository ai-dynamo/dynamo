<!-- # SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Standalone Router

A backend-agnostic standalone KV-aware router service for Dynamo deployments. For details on how KV-aware routing works, see the [KV Cache Routing documentation](/docs/architecture/kv_cache_routing.md).

## Overview

The standalone router provides configurable KV-aware routing for any set of workers in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions.

This component is **fully configurable** and works with any Dynamo backend (vLLM, TensorRT-LLM, SGLang, etc.) and any worker endpoint.

## Usage

### Command Line

```bash
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --block-size 64 \
    --router-reset-states \
    --no-track-active-blocks
```

### Arguments

**Required:**
- `--endpoint`: Full endpoint path for workers in the format `namespace.component.endpoint` (e.g., `dynamo.prefill.generate`)

**Router Configuration:**
For detailed descriptions of all KV router configuration options including `--block-size`, `--kv-overlap-score-weight`, `--router-temperature`, `--no-kv-events`, `--router-replica-sync`, `--router-snapshot-threshold`, `--router-reset-states`, and `--no-track-active-blocks`, see the [KV Cache Routing documentation](/docs/architecture/kv_cache_routing.md).

## Architecture

The standalone router exposes two endpoints via the Dynamo runtime:

1. **`find_best_worker`**: Given a request with token IDs, returns the best worker to handle it
2. **`free`**: Cleans up router state when a request completes

Clients query the `find_best_worker` endpoint to determine which worker should process each request, then call the selected worker directly.

## Example: Manual Disaggregated Serving (Alternative Setup)

> [!Note]
> **This is an alternative advanced setup.** The recommended approach for disaggregated serving is to use the frontend's automatic prefill routing, which activates when you register workers with `ModelType.Prefill`. See the [KV Cache Routing documentation](/docs/architecture/kv_cache_routing.md#disaggregated-serving-prefill-and-decode) for the default setup.
>
> Use this manual setup if you need explicit control over prefill routing configuration or want to manage prefill and decode routers separately.

See [`components/backends/vllm/launch/disagg_router.sh`](/components/backends/vllm/launch/disagg_router.sh) for a complete example.

```bash
# Start frontend router for decode workers
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0  # Pure load balancing for decode

# Start standalone router for prefill workers
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --block-size 64 \
    --router-reset-states \
    --no-track-active-blocks

# Start decode workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 &

# Start prefill workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 --is-prefill-worker &
```

>[!Note]
> **Why `--no-track-active-blocks` for prefill routing?**
> Active block tracking is used for load balancing across decode (generation) phases. For prefill-only routing, decode load is not relevant, so disabling this reduces overhead and simplifies the router state.
>
> **Why `--block-size` is required for standalone routers:**
> Unlike the frontend router which can infer block size from the ModelDeploymentCard (MDC) during worker registration, standalone routers cannot access the MDC and must have the block size explicitly specified. This is a work in progress to enable automatic inference.

## Configuration Best Practices

>[!Note]
> **Block Size Matching:**
> The block size must match across:
> - Standalone router (`--block-size`)
> - All worker instances (`--block-size`)
>
> **Endpoint Matching:**
> The `--endpoint` argument must match where your target workers register. For example:
> - vLLM prefill workers: `dynamo.prefill.generate`
> - vLLM decode workers: `dynamo.backend.generate`
> - Custom workers: `<your_namespace>.<your_component>.<your_endpoint>`

## Integration with Backends

To integrate the standalone router with a backend:

1. Clients should query the `router.find_best_worker` endpoint before sending requests
2. Workers should register at the endpoint specified by the `--endpoint` argument
3. Clients should call the `router.free` endpoint when requests complete

See [`components/src/dynamo/vllm/handlers.py`](../vllm/handlers.py) for a reference implementation (search for `prefill_router_client`).

## See Also

- [KV Cache Routing Architecture](/docs/architecture/kv_cache_routing.md) - Detailed explanation of KV-aware routing
- [Frontend Router](../frontend/README.md) - Main HTTP frontend with integrated routing
- [Router Benchmarking](/benchmarks/router/README.md) - Performance testing and tuning
