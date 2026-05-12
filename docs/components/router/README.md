---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router
---

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

For Kubernetes, set `DYN_ROUTER_MODE=kv` on the Frontend service. Workers automatically report KV cache events — no worker-side configuration changes needed.

| Argument | Default | Description |
|----------|---------|-------------|
| `--router-mode kv` | `round_robin` | Enable KV cache-aware routing |
| `--router-kv-overlap-score-weight` | `1.0` | Balance prefill vs decode optimization (higher = better TTFT) |
| `--no-router-kv-events` | enabled | Fall back to approximate routing (no event consumption from workers) |
| `--router-queue-threshold` | `4.0` | Backpressure queue threshold; enables priority scheduling via `nvext.agent_hints.priority` |
| `--router-queue-policy` | `fcfs` | Queue scheduling policy: `fcfs` (tail TTFT), `wspt` (avg TTFT), or `lcfs` (comparison-only reverse ordering) |
| `--no-router-track-prefill-tokens` | disabled | Ignore prompt-side prefill tokens in router load accounting; useful for decode-only routing paths |

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend). See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

For deployment modes and quick start steps, see the [Router Guide](router-guide.md). For CLI arguments and tuning guidelines, see [Configuration and Tuning](router-configuration.md). For A/B benchmarking, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

## Prerequisites and Limitations

**Requirements:**
- **Dynamic endpoints only**: KV router requires `register_model()` with `model_input=ModelInput.Tokens`. Your backend handler receives pre-tokenized requests with `token_ids` instead of raw text.
- Backend workers must call `register_model()` with `model_input=ModelInput.Tokens` (see [Backend Guide](../../development/backend-guide.md))
- You cannot use `--static-endpoint` mode with KV routing (use dynamic discovery instead)

**Multimodal Support:**
- **TRT-LLM and vLLM**: Multimodal routing supported for images via multimodal hashes
- **SGLang**: Image routing not yet supported
- **Other modalities** (audio, video, etc.): Not yet supported

**Limitations:**
- Static endpoints not supported—KV router requires dynamic model discovery via etcd to track worker instances and their KV cache states

For basic model registration without KV routing, use `--router-mode round-robin`, `--router-mode random`, `--router-mode least-loaded`, or `--router-mode device-aware-weighted` with both static and dynamic endpoints.

## Next Steps

- **[Router Guide](router-guide.md)**: Deployment modes, quick start, and page map
- **[Routing Concepts](router-concepts.md)**: Cost model and worker-selection behavior
- **[Configuration and Tuning](router-configuration.md)**: Router flags, transport modes, and metrics
- **[Disaggregated Serving](router-disaggregated-serving.md)**: Prefill and decode routing setups
- **[Router Operations](router-operations.md)**: Replicas, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Test layers from Rust unit tests to fixture-backed replay and full process E2E
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service for independent scaling
- **[Router Design](../../design-docs/router-design.md)**: Architecture details, algorithms, and event transport modes
