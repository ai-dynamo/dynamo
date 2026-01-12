# Dynamo

Dynamo is designed to be inference engine agnostic (supports TRT-LLM, vLLM, SGLang or others) and captures LLM-specific capabilities such as:

* Disaggregated prefill & decode inference – Maximizes GPU throughput and facilitates trade off between throughput and latency.
* Dynamic GPU scheduling – Optimizes performance based on fluctuating demand
* LLM-aware request routing – Eliminates unnecessary KV cache re-computation
* Accelerated data transfer – Reduces inference response time using NIXL.
* KV cache offloading – Leverages multiple memory hierarchies for higher system throughput

## KV Router

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks). Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.

## KV Router Architecture

The KV Router tracks two key metrics for each worker:

1. **Potential Active Blocks**: The number of blocks that would be used for decoding if a request is routed to a worker. This includes both existing active blocks and new blocks from the incoming request.

2. **Potential New Prefill Blocks**: The number of tokens that need to be computed from scratch on a worker, calculated as:
   - New prefill tokens = Total input tokens - (Overlap blocks × Block size)
   - Potential prefill blocks = New prefill tokens / Block size

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

### Temperature-Based Routing

The `router_temperature` parameter controls routing randomness:
- **0.0 (default)**: Deterministic selection of the best worker
- **> 0.0**: Probabilistic selection, higher values increase randomness
- Useful for preventing worker saturation and improving load distribution

## Objective

You task is to optimize the cost function to minimize overall latency and maximize throughput in a distributed inference setup. Consider the trade-offs between prefill and decode costs.
You're free to change any part of the code, add new functions, add more parameters or change the mechanism
of the cost function and temperature-based routing.
