---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
subtitle: KV cache-aware router that picks workers by combined prefill and decode cost to maximize throughput and minimize latency.
---

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

The [Frontend Configuration Reference](../frontend/configuration.md#router) is the
canonical reference for embedded-router flags, environment variables, defaults, and
boolean forms. See [Configuration and Tuning](router-configuration.md) for behavioral
guidance.

> [!IMPORTANT]
> `--router-mode kv` (or `DYN_ROUTER_MODE=kv`) enables KV routing on the
> frontend, but it does not enable KV event publishing on backend workers. With
> the default `--router-kv-events` setting, missing publishers leave the router
> in event-driven mode without real cache state; the router does not
> automatically switch to approximate prediction. Configure the backend-specific
> publishing flags in [Router Operations](router-operations.md#additional-notes).
> If workers will not publish events, use `--no-router-kv-events` for approximate
> cache prediction or `--load-aware` for load-only routing.

For Kubernetes, set `DYN_ROUTER_MODE=kv` on the Frontend service.

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend). See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

For deployment modes and quick start steps, see the [Router Guide](router-guide.md). For tuning guidelines, see [Configuration and Tuning](router-configuration.md). For A/B benchmarking, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

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

For basic model registration without KV routing, use `--router-mode round-robin`, `--router-mode random`, `--router-mode power-of-two`, `--router-mode least-loaded`, or `--router-mode device-aware-weighted` with both static and dynamic endpoints.

## Next Steps

- **[Router Guide](router-guide.md)**: Deployment modes, quick start, and page map
- **[Routing Concepts](router-concepts.md)**: Cost model and worker-selection behavior
- **[Router Filtering](router-filtering.md)**: Candidate eligibility, DP-rank filtering, and busy-threshold overload handling
- **[Frontend Configuration Reference](../frontend/configuration.md#router)**: Canonical embedded-router flags and environment variables
- **[Configuration and Tuning](router-configuration.md)**: Router behavior, transport modes, and tuning guidance
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
