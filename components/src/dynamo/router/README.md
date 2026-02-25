<!-- # SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Standalone Router

A backend-agnostic standalone KV-aware router service for Dynamo deployments. For details on how KV-aware routing works, see the [Router Guide](/docs/pages/components/router/router-guide.md).

## Overview

The standalone router provides configurable KV-aware routing for any set of workers in a Dynamo deployment. It can be used for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions.

This component is **fully configurable** and works with any Dynamo backend (vLLM, TensorRT-LLM, SGLang, etc.) and any worker endpoint.

## Usage

### Command Line

```bash
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-block-size 64 \
    --router-reset-states \
    --no-router-track-active-blocks
```

### Arguments

**Required:**
- `--endpoint`: Full endpoint path for workers in the format `namespace.component.endpoint` (e.g., `dynamo.prefill.generate`)

**Router Configuration:**
All router options use the `--router-*` prefix (e.g., `--router-block-size`, `--router-kv-overlap-score-weight`, `--router-temperature`, `--router-kv-events` / `--no-router-kv-events`, `--router-replica-sync`, `--router-snapshot-threshold`, `--router-reset-states`, `--router-track-active-blocks` / `--no-router-track-active-blocks`). Legacy names without the prefix (e.g., `--block-size`, `--kv-events`) are still accepted but deprecated. For detailed descriptions, see the [Router Guide](/docs/pages/components/router/router-guide.md).

## Architecture

The standalone router exposes two endpoints via the Dynamo runtime:

1. **`generate`**: Routes requests to the best worker and streams back generation results (KV-aware routing).
2. **`best_worker_id`**: Given token IDs, returns the best worker ID for the request without routing; useful for debugging or custom routing logic.

Clients call the `generate` endpoint to stream completions, or call `best_worker_id` to decide which worker to use and then contact that worker directly.

## Example: Manual Disaggregated Serving (Alternative Setup)

> [!Note]
> **This is an alternative advanced setup.** The recommended approach for disaggregated serving is to use the frontend's automatic prefill routing, which activates when you register workers with `ModelType.Prefill`. See the [Router Guide](/docs/pages/components/router/router-guide.md#disaggregated-serving) for the default setup.
>
> Use this manual setup if you need explicit control over prefill routing configuration or want to manage prefill and decode routers separately.

See [`examples/backends/vllm/launch/disagg_router.sh`](/examples/backends/vllm/launch/disagg_router.sh) for a complete example.

```bash
# Start frontend router for decode workers
python -m dynamo.frontend \
    --router-mode kv \
    --http-port 8000 \
    --kv-overlap-score-weight 0  # Pure load balancing for decode

# Start standalone router for prefill workers
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-block-size 64 \
    --router-reset-states \
    --no-router-track-active-blocks

# Start decode workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 &

# Start prefill workers
python -m dynamo.vllm --model MODEL_NAME --block-size 64 --disaggregation-mode prefill &
```

>[!Note]
> **Why `--no-router-track-active-blocks` for prefill routing?**
> Active block tracking is used for load balancing across decode (generation) phases. For prefill-only routing, decode load is not relevant, so disabling this reduces overhead and simplifies the router state.
>
> **Why `--router-block-size` is required for standalone routers:**
> Unlike the frontend router which can infer block size from the ModelDeploymentCard (MDC) during worker registration, standalone routers cannot access the MDC and must have the block size explicitly specified. This is a work in progress to enable automatic inference.

## Configuration Best Practices

>[!Note]
> **Block Size Matching:**
> The block size must match across:
> - Standalone router (`--router-block-size`)
> - All worker instances (backend-specific, e.g. `--block-size` for vLLM)
>
> **Endpoint Matching:**
> The `--endpoint` argument must match where your target workers register. For example:
> - vLLM prefill workers: `dynamo.prefill.generate`
> - vLLM decode workers: `dynamo.backend.generate`
> - Custom workers: `<your_namespace>.<your_component>.<your_endpoint>`

## Integration with Backends

To integrate the standalone router with a backend:

1. Workers should register at the endpoint specified by the `--endpoint` argument
2. Clients call the `router.generate` endpoint to stream completions (router selects the best worker), or call `router.best_worker_id` to get the best worker ID and then send requests to that worker
3. Router state is updated automatically as requests are routed; no separate "free" call is required

See [`components/src/dynamo/vllm/handlers.py`](../vllm/handlers.py) for a reference implementation (search for `prefill_router_client`).

## See Also

- [Router Guide](/docs/pages/components/router/router-guide.md) - Configuration and tuning for KV-aware routing
- [Router Design](/docs/pages/design-docs/router-design.md) - Architecture details and event transport modes
- [Frontend Router](../frontend/README.md) - Main HTTP frontend with integrated routing
- [Router Benchmarking](/benchmarks/router/README.md) - Performance testing and tuning
