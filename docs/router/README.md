<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Router

## Overview

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks). Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## Quick Start

### Python / CLI Deployment

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8000 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers register themselves using the `register_llm` API, after which the KV Router automatically:
- Tracks the state of all registered workers
- Makes routing decisions based on KV cache overlap
- Balances load across available workers

### Kubernetes Deployment

To enable the KV Router in a Kubernetes deployment, add the `DYN_ROUTER_MODE` environment variable to your frontend service:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    Frontend:
      dynamoNamespace: my-namespace
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv  # Enable KV Smart Router
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.0
    Worker:
      # ... worker configuration ...
```

**Key Points:**
- Set `DYN_ROUTER_MODE=kv` on the **Frontend** service only
- Workers automatically report KV cache events to the router
- No worker-side configuration changes needed

**Complete K8s Examples:**
- [TRT-LLM aggregated router example](../../examples/backends/trtllm/deploy/agg_router.yaml)
- [vLLM aggregated router example](../../examples/backends/vllm/deploy/agg_router.yaml)
- [SGLang aggregated router example](../../examples/backends/sglang/deploy/agg_router.yaml)
- [Distributed inference tutorial](../../examples/basics/kubernetes/Distributed_Inference/agg_router.yaml)

**For A/B Testing and Advanced K8s Setup:**
See the comprehensive [KV Router A/B Benchmarking Guide](../benchmarks/kv-router-ab-testing.md) for step-by-step instructions on deploying, configuring, and benchmarking the KV router in Kubernetes.

## Configuration Options

### CLI Arguments (Python Deployment)

The KV Router supports several key configuration options:

- **`--router-mode kv`**: Enable KV cache-aware routing (required)

- **`--kv-cache-block-size <size>`**: Sets the KV cache block size (default: backend-specific). Larger blocks reduce overlap detection granularity but improve memory efficiency. This should match your backend configuration.

- **`--router-temperature <float>`**: Controls routing randomness (default: 0.0)
  - `0.0`: Deterministic selection of the best worker
  - `> 0.0`: Probabilistic selection using softmax sampling
  - Higher values increase randomness, helping prevent worker saturation

- **`--kv-events` / `--no-kv-events`**: Controls how the router tracks cached blocks (default: `--kv-events`)
  - `--kv-events`: Uses real-time events from workers for accurate cache tracking
  - `--no-kv-events`: Uses approximation based on routing decisions (lower overhead, less accurate)

- **`--kv-overlap-score-weight <float>`**: Balance between prefill and decode optimization (default: 1.0)
  - Higher values (> 1.0): Prioritize reducing prefill cost (better TTFT)
  - Lower values (< 1.0): Prioritize decode performance (better ITL)

For a complete list of available options:
```bash
python -m dynamo.frontend --help
```

### Kubernetes Environment Variables

All CLI arguments can be configured via environment variables in Kubernetes deployments. Use the `DYN_` prefix with uppercase parameter names:

| CLI Argument | K8s Environment Variable | Default | Description |
|--------------|-------------------------|---------|-------------|
| `--router-mode kv` | `DYN_ROUTER_MODE=kv` | `round_robin` | Enable KV router |
| `--router-temperature <float>` | `DYN_ROUTER_TEMPERATURE=<float>` | `0.0` | Routing randomness |
| `--kv-cache-block-size <size>` | `DYN_KV_CACHE_BLOCK_SIZE=<size>` | Backend-specific | KV cache block size |
| `--no-kv-events` | `DYN_KV_EVENTS=false` | `true` | Disable KV event tracking |
| `--kv-overlap-score-weight <float>` | `DYN_KV_OVERLAP_SCORE_WEIGHT=<float>` | `1.0` | Prefill vs decode weight |
| `--http-port <port>` | `DYN_HTTP_PORT=<port>` | `8000` | HTTP server port |

### Example with Advanced Configuration

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    Frontend:
      dynamoNamespace: my-namespace
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv
        - name: DYN_ROUTER_TEMPERATURE
          value: "0.5"  # Add some randomness to prevent worker saturation
        - name: DYN_KV_OVERLAP_SCORE_WEIGHT
          value: "1.5"  # Prioritize TTFT over ITL
        - name: DYN_KV_CACHE_BLOCK_SIZE
          value: "16"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.0
```

### Alternative: Using Command Args in K8s

You can also pass CLI arguments directly in the container command:

```yaml
extraPodSpec:
  mainContainer:
    image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.0
    command:
      - /bin/sh
      - -c
    args:
      - "python3 -m dynamo.frontend --router-mode kv --router-temperature 0.5 --http-port 8000"
```

**Recommendation:** Use environment variables for easier configuration management and consistency with Dynamo's K8s patterns.

## KV Router Architecture

The KV Router tracks two key metrics for each worker:

1. **Potential Active Blocks**: The number of blocks that would be used for decoding if a request is routed to a worker. This includes both existing active blocks and new blocks from the incoming request.

2. **Potential New Prefill Blocks**: The number of tokens that need to be computed from scratch on a worker, calculated as:
   - New prefill tokens = Total input tokens - (Overlap blocks × Block size)
   - Potential prefill blocks = New prefill tokens / Block size

### Block Tracking Mechanisms

The router maintains block information through two complementary systems:

- **Active Decoding Blocks**: Tracked locally by the router throughout the request lifecycle:
  - Incremented when adding a new request
  - Updated during token generation
  - Decremented upon request completion

- **Cached Blocks**: Maintained globally by the KvIndexer using a prefix tree built from worker-reported KV events. This provides accurate overlap information for routing decisions.

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

The router uses KV events from workers by default to maintain an accurate global view of cached blocks. You can disable this with the `--no-kv-events` flag:

- **With KV Events (default)**:
  - Calculates overlap accurately using actual cached blocks
  - Provides higher accuracy with event processing overhead
  - Recommended for production deployments

- **Without KV Events (--no-kv-events)**:
  - Router predicts cache state based on routing decisions with TTL-based expiration and pruning
  - Tracks blocks from recent requests with configurable time-to-live
  - Reduces overhead at the cost of routing accuracy
  - **NATS is not needed** - suitable for simpler deployments without NATS infrastructure
  - Suitable for testing or when event processing becomes a bottleneck

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

1. Begin with default settings
2. Monitor TTFT and ITL metrics
3. Adjust `kv-overlap-score-weight` to meet your performance goals:
   - To reduce TTFT: Increase the weight
   - To reduce ITL: Decrease the weight
4. If you observe severe load imbalance, increase the temperature setting
