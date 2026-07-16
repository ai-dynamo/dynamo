---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
subtitle: KV cache-aware router that picks workers by combined prefill and decode cost to maximize throughput and minimize latency.
---

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

Event-driven KV routing requires configuration on both the frontend and the KV-routed backend
workers:

1. Start the frontend with the KV router enabled:

   ```bash
   python -m dynamo.frontend --router-mode kv --http-port 8000
   ```

2. Enable KV event publication with the configuration for your backend.

   For vLLM, pass `--kv-events-config` to each aggregated worker or disaggregated prefill worker:

   ```bash
   python -m dynamo.vllm --model Qwen/Qwen3-0.6B \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'
   ```

   For SGLang, pass `--kv-events-config` to each aggregated worker. In disaggregated deployments,
   pass it to both prefill and decode workers:

   ```bash
   python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B \
     --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
   ```

Do not pass `--kv-events-config` to vLLM decode-only workers. Use a unique ZMQ endpoint port for
workers that share a network namespace. `--kv-events-config` configures the engine-to-worker
publisher; Dynamo's worker-to-router event transport is configured separately. The
[vLLM](../../backends/vllm/vllm-reference-guide.md#kv-event-publication-for-kv-routing) and
[SGLang](../../backends/sglang/sglang-reference-guide.md#kv-events) reference guides document
backend defaults, role-specific behavior, and tuning.

For Kubernetes, set `DYN_ROUTER_MODE=kv` on the Frontend service and add the backend event flag to
the appropriate vLLM or SGLang workers as described above. For approximate routing without worker
events, omit the backend event flag and set `--no-router-kv-events` or
`DYN_ROUTER_USE_KV_EVENTS=false` on the frontend.

| Argument | Default | Description |
|----------|---------|-------------|
| `--router-mode kv` | `round-robin` | Enable KV cache-aware routing |
| `--load-aware` | disabled | Use KV active-load routing without cache-reuse signals; implies `--router-mode kv` on the frontend |
| `--router-kv-overlap-score-credit` | `1.0` | Credit multiplier for device-local prefix overlap, from 0.0 to 1.0 |
| `--router-prefill-load-scale` | `1.0` | Scale adjusted prompt-side prefill load before adding decode blocks |
| `--router-kv-events` / `--no-router-kv-events` | `--router-kv-events` | Consume worker KV events, or explicitly disable them to use approximate routing |
| `--router-queue-threshold` | disabled | Backpressure queue threshold; setting a numeric value enables queueing, where priority hints reorder waiting requests |
| `--router-queue-policy` | `fcfs` | Queue scheduling policy: `fcfs` (tail TTFT), `wspt` (avg TTFT), or `lcfs` (comparison-only reverse ordering) |
| `--no-router-track-prefill-tokens` | disabled | Ignore prompt-side prefill tokens in router load accounting; useful for decode-only routing paths |

> [!IMPORTANT]
> With the default `--router-kv-events` setting, missing publishers leave the router in
> event-driven mode without real cache state; the router does not automatically switch to
> approximate prediction. Configure the backend-specific publishing flags as shown above.
> If workers will not publish events, use `--no-router-kv-events` for approximate
> cache prediction or `--load-aware` for load-only routing.

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend). See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

For deployment modes and quick start steps, see the [Router Guide](router-guide.md). For CLI arguments and tuning guidelines, see [Configuration and Tuning](router-configuration.md). For A/B benchmarking, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

## Prerequisites and Limitations

**Requirements:**
- **Dynamic endpoints only**: KV router requires `register_model()` with `model_input=ModelInput.Tokens`. Your backend handler receives pre-tokenized requests with `token_ids` instead of raw text.
- Backend workers must call `register_model()` with `model_input=ModelInput.Tokens` (see [Backend Guide](../../development/backend-guide.md))
- Use dynamic discovery with KV routing so the router can track worker instances and KV cache state

**Multimodal Support:**
- **Image routing via multimodal hashes**: Supported in the documented TRT-LLM and vLLM router paths.
- **Other backend or modality combinations**: Check the backend-specific multimodal docs before relying on multimodal hash routing.

**Limitations:**
- Static endpoints are not supported with KV routing; use dynamic discovery so the router can track worker instances and KV cache state

For basic model registration without KV routing, use `--router-mode round-robin`, `--router-mode random`, `--router-mode least-loaded`, or `--router-mode device-aware-weighted` with both static and dynamic endpoints.

## Next Steps

- **[Router Guide](router-guide.md)**: Deployment modes, quick start, and page map
- **[Routing Concepts](router-concepts.md)**: Cost model and worker-selection behavior
- **[Router Filtering](router-filtering.md)**: Candidate eligibility, DP-rank filtering, and busy-threshold overload handling
- **[Configuration and Tuning](router-configuration.md)**: Router flags, transport modes, and metrics
- **[Deficit Round Robin Queue Scheduling](deficit-round-robin.md)**: Weighted policy-class arbitration, cursor movement, and bulk virtual rounds
- **[Priority Scheduling](priority-scheduling.md)**: Router queue, backend engine, and cache priority behavior
- **[Disaggregated Serving](router-disaggregated-serving.md)**: Prefill and decode routing setups
- **[Router Operations](router-operations.md)**: Replicas, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Test layers from Rust unit tests to fixture-backed replay and full process E2E
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service for independent scaling
- **[Standalone Selection Service](standalone-selection.md)**: Expose KV-aware selection and reservation accounting over HTTP
- **[Standalone Slot Tracker](standalone-slot-tracker.md)**: Run active-request load accounting as a separate HTTP service
- **[Router Design](../../design-docs/router-design.md)**: Architecture details, algorithms, and event transport modes
