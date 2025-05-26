<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# KV Router Performance Tuning

## Overview

Dynamo's KV Router listens to KV events from worker nodes to build a global prefix tree of KV caches. This enables the router to predict the KV hit rate per worker (overlap score) for incoming requests and make intelligent routing decisions. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in a distributed inference setup.

## KV Router Architecture

The KV Router maintains a global view of all KV caches across workers. When a new request arrives:

1. The router calculates an **overlap score** for each worker by finding matching blocks between the request and the prefix cache
2. It collects runtime metrics from each worker including **KV cache usage** and **waiting request count**
3. It applies a cost function to determine the optimal worker for the request

More details can be found in docs/kv_cache_routing.md

## Cost Function Tuning

The KV Router's decision-making is primarily controlled by its cost function, which can be customized in `kv_router.py`. The default cost function is:

```python
worker_logits[worker_id] = 2 * score - gpu_cache_usage - normalized_waiting
```

Where:
- `score`: Normalized overlap score (matching blocks × block_size / token_length)
- `gpu_cache_usage`: Percentage of GPU KV cache currently in use
- `normalized_waiting`: Number of waiting requests normalized by the max waiting requests across all workers

The router selects the worker with the highest logit value. In the event of a tie, it randomly chooses among the top-scoring workers.
Alternatively, applying a softmax to the logits and sampling based on the resulting probabilities can introduce stochasticity into the routing process.
This probabilistic approach helps prevent a failure mode where one worker receives a disproportionate number of requests, saturating its prefix cache.
Such saturation can create a feedback loop—where the cache-rich worker continues to be selected—making it difficult to break the cycle deterministically.

### Key Tuning Parameters

1. **Overlap Score Weight** (default: 2.0)
   - Higher values prioritize KV cache reuse
   - Lower values allow more even distribution of requests

2. **GPU Cache Usage Weight** (default: 1.0)
   - Higher values avoid workers with nearly full KV caches
   - Lower values ignore KV cache utilization

3. **Waiting Requests Weight** (default: 1.0)
   - Higher values avoid workers with queued requests
   - Lower values ignore queue lengths

## Tuning Guidelines

### 1. Consider Total KV Block Allocation

Check the total number of KV blocks allocated for your backend engine. For smaller models (e.g., 8B parameters), this can exceed one million blocks. In such cases:

- Reduce the weight on KV cache usage (`gpu_cache_usage`) since exhausting KV cache is less likely
- Focus more on overlap score and waiting requests

### 2. Analyze Your Dataset's Theoretical Hit Rate

Consider the expected theoretical hit rate of your dataset (assuming perfect caching):

- More formally, consider the depth of your core prefix tree (nodes visited at least twice)
- For lower hit rates, or if the core prefix tree depth is short compared to the ISL,
reduce the overlap score weight
- Alternatively, normalize the overlap score with the input sequence length (ISL)

### 3. Consider Prefix Tree Breadth

The breadth of your prefix tree can be proxied by how many unique context prompts you expect:

- If you can identify distinct "buckets" of similar prompts, each containing roughly equal number of prompts,
consider using multiple KV routers, one for each bucket
- Use a meta-router to direct requests to specialized KV routers for different prompt categories
- For very diverse context prompts, overlap scores should probably be prioritized

### 4. Balance Latency vs. Throughput

The weights directly impact your service level objectives:

- Higher weights on waiting requests improve latency but may reduce throughput
- Higher weights on overlap score may improve throughput but could increase tail latency

## Alternative Routing Strategies

The default strategy uses greedy selection (highest logit wins), but other approaches can be implemented:

- **Softmax Sampling**: Converts logits to probabilities and samples workers probabilistically
- **Temperature-Based Sampling**: Adds a temperature parameter to control sampling randomness
- **Two-Stage Routing**: For example, using a round-robin as a meta router, to route to multiple kv routers

## Monitoring and Refinement

To effectively tune your KV Router:

1. Monitor the router logs to see actual logit calculations for each worker
2. Track hit rates, latency, and throughput metrics
3. Iteratively adjust weights based on observed performance
4. Consider dynamically adjusting weights based on current load conditions