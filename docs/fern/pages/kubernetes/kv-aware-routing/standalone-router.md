---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Standalone Router
subtitle: Run KV-aware routing as its own Dynamo service.
---

The standalone router is a backend-agnostic KV-aware router service for Dynamo deployments. It is useful when worker selection should live outside the HTTP frontend, such as manual disaggregated serving, multi-tier architectures, or custom request pipelines that need to route to a specific worker endpoint.

For the routing cost model and worker-selection behavior, see [Routing Concepts](../../developer-guide/knowledge-base/modular-components/router/routing-concepts.md). For shared tuning guidance, see [Configuration and Tuning](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md).

## When to use it

Use the frontend-embedded router for the normal HTTP serving path. It owns `/v1/chat/completions`, discovers backend workers, and routes requests inside the frontend process.

Use the standalone router when you need a dedicated routing component that targets a specific Dynamo endpoint. The standalone router does not provide the OpenAI-compatible HTTP frontend; clients call the router service through the Dynamo runtime.

| Deployment | Process | Primary use case |
|---|---|---|
| Frontend-embedded | `python -m dynamo.frontend --router-mode kv` | Standard aggregated or disaggregated serving behind the Dynamo frontend. |
| Standalone | `python -m dynamo.router` | Advanced prefill routing, custom pipelines, or routing without the HTTP frontend. |

## Start the router

Run the router with the target worker endpoint. The endpoint uses the `namespace.component.endpoint` format.

```bash
python -m dynamo.router \
  --endpoint dynamo.prefill.generate \
  --router-block-size 64 \
  --no-router-track-active-blocks
```

Required option:

- `--endpoint`: Full endpoint path for workers, such as `dynamo.prefill.generate`.

Standalone-specific options:

- `--router-block-size`: KV block size used for routing decisions. Set this explicitly so it matches the backend worker block size.
- `--no-router-track-active-blocks`: Disables decode active-block tracking. This is commonly useful for prefill-only routing because decode load is not relevant.

Most KV tuning options use the `--router-*` prefix, but shared options such as `--load-aware`, `--serve-indexer`, `--use-remote-indexer`, and `--shared-cache-*` do not. Legacy names such as `--block-size` and `--kv-events` are still accepted but deprecated. Run `python -m dynamo.router --help` for the current command surface.

## Runtime endpoints

The standalone router exposes three Dynamo runtime endpoints:

- `generate`: Routes requests to the best worker and streams generation results.
- `best_worker_id`: Returns the best worker ID for the provided token IDs without forwarding the request.
- `get_overlap_scores`: Returns per-worker and per-DP-rank matched block counts for device, host-pinned, disk, and shared-cache tiers.

Clients can use `generate` when the router should forward the request, `best_worker_id` when an external scheduler wants to make the final call itself, or `get_overlap_scores` when the scheduler needs the raw tiered overlap signal.

## Manual prefill routing example

> [!NOTE]
> This is an advanced alternative setup. For most disaggregated serving deployments, use the frontend's automatic prefill routing, which activates when workers register with `WorkerType.Prefill`. See [Disaggregated Serving](../../developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md).

Use a standalone router when you need explicit control over prefill routing configuration or want to manage prefill and decode routers separately.

```bash
# Start frontend routing for decode workers.
python -m dynamo.frontend \
  --router-mode kv \
  --http-port 8000 \
  --router-kv-overlap-score-credit 0

# Start standalone routing for prefill workers.
python -m dynamo.router \
  --endpoint dynamo.prefill.generate \
  --router-block-size 64 \
  --no-router-track-active-blocks

# Start decode workers.
python -m dynamo.vllm \
  --model MODEL_NAME \
  --block-size 64 \
  --disaggregation-mode decode \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &

# Start prefill workers.
python -m dynamo.vllm \
  --model MODEL_NAME \
  --block-size 64 \
  --disaggregation-mode prefill \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}' &
```

For an integrated frontend disaggregated example, see [`examples/backends/vllm/launch/disagg_router.sh`](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/vllm/launch/disagg_router.sh). For explicit multi-router composition, see the [Global Router README](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/global_router/README.md).

For event-driven prefix-cache state, add the backend-specific KV event publishing flags to the workers that the router indexes. Use `--no-router-kv-events` on the router only when approximate cache-state prediction is acceptable.

## Configuration guidance

> [!NOTE]
> Set `--router-block-size` on standalone routers. Unlike frontend-embedded routing, standalone routers do not infer block size from the ModelDeploymentCard during worker registration.

The block size must match across the standalone router and all target workers. For vLLM, that means `--router-block-size` on the router should match `--block-size` on the vLLM workers.

The endpoint must match where the target workers register:

- vLLM prefill workers: `dynamo.prefill.generate`
- vLLM decode workers: `dynamo.backend.generate`
- Custom workers: `<your_namespace>.<your_component>.<your_endpoint>`

To integrate a backend with the standalone router:

1. Register workers at the endpoint passed to `--endpoint`.
2. Call `router.generate` to let the router select and forward to a worker, or call `router.best_worker_id` and then send the request directly.
3. Let router state update as requests are routed; no separate free call is required.

See [`components/src/dynamo/vllm/handlers.py`](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/vllm/handlers.py) for a backend reference implementation.

## See also

- [Router Guide](../../developer-guide/knowledge-base/modular-components/router/router-guide.md)
- [Configuration and Tuning](../../developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md)
- [Disaggregated Serving](../../developer-guide/knowledge-base/modular-components/router/disaggregated-serving.md)
- [Router Design](../../developer-guide/knowledge-base/modular-components/router/router-design.md)
- [Frontend Router](../../developer-guide/knowledge-base/modular-components/frontend/overview.md)
- [Router Benchmarking](https://github.com/ai-dynamo/dynamo/blob/main/benchmarks/router/README.md)
