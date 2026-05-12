---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Routing Concepts
subtitle: Cost model, worker selection, and routing primitives for the Dynamo router
---

This page explains how the Dynamo router evaluates workers, chooses a target, and fits into the request path. For CLI flags and tuning knobs, see [Configuration and Tuning](router-configuration.md).

## KV Cache Routing

KV cache routing optimizes large language model inference by intelligently directing requests to workers with the most relevant cached data. By maximizing cache reuse, it reduces redundant computation and improves both throughput and latency.

```mermaid
graph TD
    T[Tokens] --> R[KV Aware Router]

    R -.-> W1["Worker 1<br/>Cached: 2 blocks<br/>Prefill: 8 blks<br/>Decode: 10 blks"]
    R ==>|Selected| W2["Worker 2<br/>Cached: 5 blocks<br/>Prefill: 5 blks<br/>Decode: 5 blks"]
    R -.-> W3["Worker 3<br/>Cached: 8 blocks<br/>Prefill: 2 blks<br/>Decode: 9 blks"]

    style T fill:#fff3e0,stroke:#333,color:#333
    style R fill:#2e8b57,stroke:#333,color:#fff
    style W1 fill:#f3e5f5,stroke:#333,color:#333
    style W2 fill:#c8e6c9,stroke:#333,color:#333
    style W3 fill:#f3e5f5,stroke:#333,color:#333

    linkStyle 0,1,2,3 stroke:#8b4513,stroke-width:2px
```

KV cache reuse introduces complexity to LLM serving load balancing. While it can significantly reduce computation costs, routing strategies that ignore worker-specific KV states can lead to:
- Missed cache reuse opportunities due to suboptimal worker selection
- System throughput degradation from uneven request distribution across workers

The router uses a cost function that considers both the prefill cost (influenced by cached blocks) and the decode load to make optimal routing decisions.

## Cost Calculation

1. **Prefill blocks**: Calculated from active prompt-side token load plus the incoming request's input tokens, divided by the block size. The system updates active prompt load when the first output token signals prefill completion.
2. **Decode blocks**: Estimated from the request's input tokens and each worker's active sequences. The count updates when requests complete and their blocks are freed.
3. **Overlap credits**: Device-local, host, disk, and shared-cache hits reduce the prompt-side prefill load before the final prefill scale is applied.
4. **Cost formula**:

```text
adjusted_prefill_blocks = max(
    prefill_blocks
    - overlap_score_credit * device_overlap_blocks
    - host_cache_hit_weight * host_overlap_blocks
    - disk_cache_hit_weight * disk_overlap_blocks
    - shared_cache_multiplier * shared_beyond_blocks,
    0,
)
cost = prefill_load_scale * adjusted_prefill_blocks + decode_blocks
```

Lower costs indicate better routing choices.
`overlap_score_credit` is the device-local prefix-overlap credit multiplier, from 0.0 to 1.0.
Higher values favor cache reuse (improving TTFT), while lower values prioritize even load distribution (improving ITL). `prefill_load_scale` controls the weight of the adjusted prompt-side load relative to decode blocks.

### Active Load Modeling

The `prefill_blocks` and `decode_blocks` terms include projected load for the new request plus active load already assigned to each worker.

#### Prefill Load Modeling

For prefill load, the router first estimates the uncached prompt work for each candidate worker:

```text
effective_isl = input_tokens - cached_tokens
```

By default, that effective prefill load remains charged at full value until the first output token marks prefill complete. With `--router-prefill-load-model aic`, the router also asks AIC for an expected prefill duration using the effective ISL and cached prefix length. The active load tracker then decays the oldest active prefill request on each worker over that predicted duration. This only changes router-side prompt load accounting; it does not change backend execution.

#### Decode Load Modeling

For decode load, the router tracks active KV blocks assigned to each worker. By default, this covers the prompt-side blocks that are already assigned to active requests and frees them when each request finishes.

When `--router-track-output-blocks` is enabled, the router also adds placeholder output blocks as generation crosses block boundaries. If the request includes `nvext.agent_hints.osl`, those output blocks receive a fractional weight based on progress toward the expected output length. This expected OSL proxy lets requests near completion contribute less future decode load. Without an expected OSL, tracked output blocks count at full weight until the request finishes.

For the flags that enable these models, see [Configuration and Tuning](router-configuration.md).

## Worker Selection

The router selects the worker with the lowest cost. When `router_temperature` is set to a non-zero value, the router uses softmax sampling on the normalized cost logits to introduce randomness in the selection, which can help with load distribution.

Example calculation with `overlap_score_credit = 1.0`:
- Worker 1: raw prefill 10 blocks, device overlap 2 blocks, decode 10 blocks => cost = 8 + 10 = 18
- **Worker 2: raw prefill 10 blocks, device overlap 5 blocks, decode 5 blocks => cost = 5 + 5 = 10** (selected - lowest cost)
- Worker 3: raw prefill 10 blocks, device overlap 8 blocks, decode 9 blocks => cost = 2 + 9 = 11

## Using the KV Cache Router

To enable KV cache-aware routing, start the frontend node like this:

```bash
python -m dynamo.frontend --router-mode kv
```

When KV blocks are created or removed, the engine notifies the Dynamo router, which then identifies the worker with the best matching blocks and routes traffic accordingly.

To evaluate the benefits of KV-aware routing, compare your workload's performance using `--router-mode random|round-robin` against KV-aware routing.

For detailed CLI arguments and advanced configuration options, see [Configuration and Tuning](router-configuration.md).

## Basic Routing

Dynamo supports several routing strategies when sending requests from one component to another component's endpoint.

First, create a client tied to a component endpoint. Here we get a client tied to the `generate` endpoint of the `VllmWorker` component.

```python
client = runtime.endpoint("dynamo.VllmWorker.generate").client()
```

You can then use the default routing methods exposed by the client class to send requests to the `VllmWorker` component.

- **Random routing**: Default strategy, available via `client.generate()` or `client.random()`
- **Round-robin routing**: Cycles through available workers via `client.round_robin()`
- **Direct routing**: Explicitly targets a specific worker via `client.direct(input, component_id)`
- **Least-loaded routing**: Routes to the worker with fewest active connections via `--router-mode least-loaded`
- **Device-aware weighted routing**: Routes using CPU/non-CPU ratio budgeting plus least-loaded selection within the selected device group via `--router-mode device-aware-weighted`
KV cache routing uses direct routing with a special worker selection algorithm.

For benchmarking KV router performance, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).
For custom routing logic and advanced patterns, see [Routing Patterns](router-examples.md#routing-patterns).

## Device-Aware Weighted Routing

`device-aware-weighted` is designed for heterogeneous fleets where CPU and non-CPU workers share the same endpoint. Instead of comparing raw in-flight counts, the router compares a capability-normalized load across the CPU and non-CPU groups, then selects the least-loaded worker within the winning group.

```text
normalized_load = total_inflight(group) / (instance_count(group) x throughput_weight)
```

The throughput weight is `1` for CPU workers and `DYN_ENCODER_CUDA_TO_CPU_RATIO` for non-CPU workers. This lets the router route proportionally to device capability instead of permanently starving slower devices.

When only one device class is present, the behavior degenerates to standard least-loaded routing.
