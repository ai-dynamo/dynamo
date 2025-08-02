<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Router

## Overview

Dynamo's KV Router makes intelligent routing decisions by evaluating the computational cost of processing requests on different workers. The router considers both the decoding cost (active blocks) and prefill cost (new blocks that need to be computed). Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in a distributed inference setup.

## Quick Start

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8080
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8080 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers can register themselves using the `register_llm` API, and the KV Router will automatically include them in its routing decisions. The router will:
- Track the state of all registered workers
- Make intelligent routing decisions based on KV cache overlap
- Balance load across available workers

### Important Arguments

The KV Router supports several key configuration options:

- **`--kv-cache-block-size <size>`**: Sets the KV cache block size (default: backend-specific). Larger blocks reduce overlap detection granularity but improve memory efficiency. This should match your backend configuration.

- **`--router-temperature <float>`**: Controls routing randomness (default: 0.0)
  - `0.0`: Deterministic selection of the best worker
  - `> 0.0`: Probabilistic selection using softmax sampling
  - Higher values increase randomness, helping prevent worker saturation

- **`--kv-events` / `--no-kv-events`**: Controls how the router tracks cached blocks (default: `--kv-events`)
  - `--kv-events`: Uses real-time events from workers for accurate cache tracking
  - `--no-kv-events`: Uses approximation based on routing decisions (lower overhead, less accurate)

For a complete list of available options:
```bash
python -m dynamo.frontend --help
```

## KV Router Architecture

The KV Router tracks two key metrics for each worker:

1. **Potential Active Blocks**: The total number of blocks that would be actively used for decoding if a request were routed to that worker. This includes existing active blocks plus new blocks from the incoming request.

2. **Potential New Prefill Blocks**: The number of new tokens that would need to be prefilled (computed from scratch) on that worker, calculated as:
   - New prefill tokens = Total input tokens - (Overlap blocks × Block size)
   - Potential prefill blocks = New prefill tokens / Block size

### Block Tracking Mechanisms

The router maintains block information through two complementary systems:

- **Active Decoding Blocks**: Tracked locally by the router based on the request lifecycle:
  - Incremented when a new request is added
  - Updated as new tokens are generated
  - Decremented when a request completes

- **Cached Blocks**: Maintained globally by the KvIndexer, which builds a prefix tree from KV events reported by workers. This provides accurate overlap information for routing decisions.

## Cost Function

The KV Router's routing decision is based on a simple cost function:

```
logit = kv_overlap_score_weight × potential_prefill_blocks + potential_active_blocks
```

Where:
- Lower logit values are better (less computational cost)
- The router uses softmax sampling with optional temperature to select workers

### Key Parameter: kv-overlap-score-weight

The `kv-overlap-score-weight` parameter (default: 1.0) controls the balance between prefill and decode optimization:

- **Higher values (> 1.0)**: Emphasize reducing prefill cost
  - Prioritizes routing to workers with better cache hits
  - Optimizes for Time To First Token (TTFT)
  - Best for workloads where initial response latency is critical

- **Lower values (< 1.0)**: Emphasize decode performance
  - Distributes active decoding blocks more evenly
  - Optimizes for Inter-Token Latency (ITL)
  - Best for workloads with long generation sequences

## KV Events vs. Approximation Mode

By default, the router uses KV events from workers to maintain an accurate global view of cached blocks. However, you can disable this with the `--no-kv-events` flag:

- **With KV Events (default)**:
  - Accurate overlap calculation based on actual cached blocks
  - Higher accuracy but requires event processing overhead
  - Best for production deployments

- **Without KV Events (--no-kv-events)**:
  - Uses the ApproxKvIndexer to approximate cached blocks based on routing decisions
  - Assumes that recently routed requests will have their blocks cached
  - Lower overhead but potentially less accurate routing
  - Useful for testing or environments where event processing is a bottleneck

## Tuning Guidelines

### 1. Understand Your Workload Characteristics

- **Prefill-heavy workloads** (long prompts, short generations): Increase `kv-overlap-score-weight`
- **Decode-heavy workloads** (short prompts, long generations): Decrease `kv-overlap-score-weight`

### 2. Monitor Key Metrics

The router logs the cost calculation for each worker:
```
Formula for worker_1: 125.3 = 1.0 * 100.5 + 25.0 (cached_blocks: 15)
```

This shows:
- Total cost (125.3)
- Overlap weight × prefill blocks (1.0 × 100.5)
- Active blocks (25.0)
- Cached blocks that contribute to overlap (15)

### 3. Temperature-Based Routing

The `router_temperature` parameter controls routing randomness:
- **0.0 (default)**: Deterministic selection of the best worker
- **> 0.0**: Probabilistic selection, higher values increase randomness
- Useful for preventing worker saturation and improving load distribution

### 4. Iterative Optimization

1. Start with default settings
2. Monitor TTFT and ITL metrics
3. Adjust `kv-overlap-score-weight` based on your optimization goals:
   - If TTFT is too high: Increase the weight
   - If ITL is too high: Decrease the weight
4. Increase temperature if severe load imbalance occurs