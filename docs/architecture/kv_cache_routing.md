<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# KV Cache Routing
This document explains how Dynamo's Key-Value (KV) cache routing optimizes large language model inference by intelligently directing requests to workers with the most relevant cached data, while maintaining load balance through worker utilization metrics.

To enable KV cache aware routing start the frontend node like this:
```
python -m dynamo.frontend --router-mode kv
```

When KV blocks are created or removed, the engine notifies the Dynamo router, which then identifies the worker with the best matching blocks and routes traffic accordingly.

To evaluate the benefits of KV-aware routing, compare your workload's performance using `--router-mode random|round-robin` against KV-aware routing.

The main KV-aware routing arguments:

- `--kv-overlap-score-weight`: Controls the importance of prefix cache overlaps in prefill cost calculations. Higher values improve Time To First Token (TTFT) at the cost of Inter-Token Latency (ITL). When set to 0, the router ignores prefix caches and uses pure load balancing. Defaults to 1.

- `--router-temperature`: Controls worker selection randomness through softmax sampling of router cost logits. A value of 0 (default) ensures deterministic selection of the lowest-cost worker, while higher values introduce more randomness.

- `--use-kv-events`/`--no-kv-events`: Determines how the router tracks cached blocks. When enabled (default), uses `KvIndexer` to monitor block creation and deletion events. When disabled, uses `ApproxKvIndexer`, which estimates cache hits based on a fixed time window (120s). Disable this if your backend doesn't support KV events.

- `--router-replica-sync`: Enables NATS-based synchronization of local routing decisions between router replicas. When enabled, routers share their active sequence information and local predictions of block usage, improving routing consistency across instances. Note that this does not sync the radix tree or cached KV block states themselves - those are synchronized through JetStream events. Disabled by default.

- `--router-reset-states`/`--router-persist-states`: Controls whether the router state is reset on startup. When `--router-reset-states` is used (default), the router clears both the JetStream event stream and NATs object store, starting with a fresh state. When `--router-persist-states` is used, the router retains existing state from previous runs, downloading any available snapshot from NATs object store and continuing to consume events from where it left off. This enables routers to maintain KV cache awareness across restarts. **Note**: State persistence is only available when `--use-kv-events` is enabled (default). When using `--no-kv-events` with `ApproxKvIndexer`, state persistence is not supported.

- `--router-snapshot-threshold`: Sets the number of messages in the JetStream before triggering a snapshot. When the message count exceeds this threshold, a router will attempt to purge acknowledged messages from the stream and create a snapshot of the current radix tree state in NATs object store. Defaults to 10000. This helps manage stream size and provides faster initialization for routers that restart.

## Architecture

Colloquially, we refer to a Dynamo component that serves an endpoint for LLM inference as a **worker**.

## Basic Routing
Dynamo supports several routing strategies when sending requests from one component to another component's endpoint.

First, we must create a client tied to a components endpoint, we can do this using the labels defined above. Here we are getting a client tied to the `generate` endpoint of the `VllmWorker` component.

```python
client = namespace('dynamo').component('VllmWorker').endpoint('generate').client()
```

We can then use the default routing methods exposed by the client class to send requests to the `VllmWorker` component.

- **Random routing**: Default strategy, available via `client.generate()` or `client.random()`
- **Round-robin routing**: Cycles through available workers via `client.round_robin()`
- **Direct routing**: Explicitly targets a specific worker via `client.direct(input, component_id)`

KV Cache routing uses direct routing with a special worker selection algorithm.

## Serving Two Router Replicas

For improved fault tolerance, you can launch two frontend + router replicas. Since the frontend and router are currently tied together, you'll need to use two different HTTP ports for each instance.

To enable state sharing between the router replicas (which provides more accurate routing decisions), use the `--router-replica-sync` flag when starting the frontend. Router replicas are currently tied to a component, and state syncing and sharing can only happen within the component group. Here's an example of running multiple router replicas:

```bash
# Router replica 1
python -m dynamo.frontend --router-mode kv --port 8000 --router-replica-sync

# Router replica 2 (can be started later, note the extra --router-persist-states arg)
python -m dynamo.frontend --router-mode kv --port 8001 --router-replica-sync --router-persist-states
```

After these two replicas are launched, they will share the same JetStream and snapshot state. The second replica can be started after the first has already been handling requests. As long as `--router-persist-states` is set, the new replica will sync its KV block indexer by consuming the JetStream events and/or downloading the latest snapshot, ensuring both replicas have the same view of cached blocks across workers. It's okay for one router to go down, or even both to go down - the state persistence ensures continuity (up to the message retention of an hour we set for the stream). When a third router starts (with `--router-persist-states`), the states will still persist:

```bash
# Router replica 3 (can be started even after replicas 1 and 2 have gone down)
python -m dynamo.frontend --router-mode kv --port 8002 --router-replica-sync --router-persist-states
```

> **Note:** If a router replica is launched without the `--router-persist-states` flag, the entire stream and radix snapshot will be purged. If you want to serve a separate router (targeting a different set of workers) independently without affecting the current state, consider using a new namespace/component (see [Distributed Runtime](distributed_runtime.md)) which will start a new stream and NATS object store path.

When `--router-replica-sync` is enabled, the router replicas will additionally share their local routing decisions and active sequence predictions via NATS. Active blocks information is communicated between routers in a fire-and-forget manner, but the routers will quickly become consistent as this information is tied to the request cycle. This helps maintain consistent load estimates across instances even when requests are distributed between routers.

## Understanding KV Cache
The leading Large Language Models (LLMs) today are auto-regressive and based off of the [transformer architecture](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf). One key inference optimization technique is to cache the already computed keys and values and to reuse them for the future tokens. This is called the [KV Cache](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/#key-value_caching).

### KV Cache Optimizations
Every inference framework will have a KV Cache for each worker. A popular inference framework library is [vLLM](https://github.com/vllm-project/vllm) where a key contribution was [PagedAttention](https://arxiv.org/abs/2309.06180), which allowed them to manage KV Cache in an efficient way by chunking requests into blocks.

Another popular inference framework, [SGLang](https://github.com/sgl-project/sglang), contributed [RadixAttention](https://arxiv.org/abs/2312.07104) which introduced a
prefix tree which allows for efficient matching, inserting and eviction of KV Cache blocks. The prefix tree structure popularized KV Cache reuse.

In Dynamo, we introduce a KVPublisher which emits KV Cache events that occur at each worker and a KVIndexer which keeps track of these events globally.

To get a feel for how KV Cache management works on a single worker with KV Cache reuse turned on and where the KVPublisher gets plugged in, we can walk through the KV Block management flow:
1. Request tokenization: The incoming prompt is converted into tokens
2. Block partitioning: The token sequence is divided into fixed-size blocks (e.g., 16 or 64 tokens per block)
3. Block hashing: Each block of tokens is hashed to create a unique identifier
4. Cache lookup:
    - For each block, the system checks if a matching block already exists in the KV cache
    - If a match is found, the existing KV cache block is reused
    - If no match is found, the system proceeds to the next step
5. Resource allocation:
    - For blocks without matches, the system attempts to allocate new memory space
    - If sufficient memory is available, allocate memory space and proceed to step 7
    - If memory is constrained, proceed to step 6
6. Cache eviction (when necessary):
    - The system applies an eviction policy (e.g., LRU, LFU) to identify blocks for removal
    - Selected blocks are evicted from the cache
    - **KVPublisher emits a KV removed event notifying KVIndexer about the removed block.**
    - Alternatively, some systems may offload less-frequently used blocks to CPU memory.
7. KV computation:
    - For new blocks, the model computes key and value tensors
    - These tensors are stored in the newly allocated cache blocks
    - **KVPublisher emits a kv stored event notifying KVIndexer about newly stored blocks**.

Further details can be found for: [TRT-LLM](https://developer.nvidia.com/blog/introducing-new-kv-cache-reuse-optimizations-in-nvidia-tensorrt-llm/), [vLLM](https://docs.vllm.ai/en/latest/design/automatic_prefix_caching.html#design-automatic-prefix-caching) and [SGLang](https://lmsys.org/blog/2024-01-17-sglang/).

## KV Cache Routing and Load Balancing
```text
+---------+          +------------------+           +---------+
|  Tokens |--------->| KV Aware Router  |---------> | Worker 2|
+---------+          +------------------+           +---------+
                             |
          +------------------+------------------+
          |                  |                  |
          | Cached: 2 blocks | Cached: 5 blocks | Cached: 8 blocks
          | Prefill: 8 blks  | Prefill: 5 blks  | Prefill: 2 blks
          | Decode: 10 blks  | Decode: 5 blks   | Decode: 9 blks
          v                  v                  v
   +----------------+  +----------------+  +----------------+
   |   Worker 1     |  |   Worker 2     |  |   Worker 3     |
   +----------------+  +----------------+  +----------------+
```

KV Cache reuse introduces complexity to LLM serving load balancing. While it can significantly reduce computation costs, routing strategies that ignore worker-specific KV states can lead to:
- Missed cache reuse opportunities due to suboptimal worker selection
- System throughput degradation from uneven request distribution across workers

The router uses a cost function that considers both the prefill cost (influenced by cached blocks) and the decode load to make optimal routing decisions:

### Cost Calculation

1. **Prefill blocks**: Calculated by dividing the number of tokens requiring prefill processing by the block size. The system predicts this based on input tokens and available cached blocks per worker, updating the count when the first output token signals prefill completion.

2. **Decode blocks**: Estimated from the request's input tokens and each worker's active sequences. The count updates when requests complete and their blocks are freed.

3. **Cost formula**: `cost = overlap_score_weight * prefill_blocks + decode_blocks`
   - Lower costs indicate better routing choices
   - `overlap_score_weight` balances cache hit optimization against load distribution
   - Higher weights favor cache reuse (improving TTFT), while lower weights prioritize even load distribution (improving ITL)

### Worker Selection

The router selects the worker with the lowest cost. When `router_temperature` is set to a non-zero value, the router uses softmax sampling on the normalized cost logits to introduce randomness in the selection, which can help with load distribution.

Example calculation with `overlap_score_weight = 1.0`:
- Worker 1: cost = 1.0 * 8 + 10 = 18
- **Worker 2: cost = 1.0 * 5 + 5 = 10** (selected - lowest cost)
- Worker 3: cost = 1.0 * 2 + 9 = 11

## Events

Dynamo supports KV Cache Routing across multiple backend implementations through a flexible event system. The KVPublisher component integrates with any framework to emit KV events, while the KVIndexer component maintains a global prefix tree of cached blocks by processing these events from all workers.

```text
+----------------+                         +-----------------+
|                |                         | KV Aware Router |
|     Worker     |                         |                 |
|                | create_kv_block()       | +-------------+ |
| +------------+ | remove_kv_block()       | |  KVIndexer  | |
| |KVPublisher | |------------------------>| +-------------+ |
| +------------+ |                         |                 |
|                |                         |                 |
+----------------+                         +-----------------+
```

### KVPublisher
The KVPublisher can be initialized and then called in the inference framework where blocks are allocated and removed.

The two types of events are:
- KV stored event
- KV removed event

The publisher can be initialized and used through C bindings or Python bindings.

### KVIndexer
The KVIndexer builds and maintains a global view of cached blocks in a prefix tree. We modify the original prefix tree by also storing the worker id on each node. This is so we can return the number of matched blocks for each worker.

The KVIndexer has a method `find_matches_for_request`, which takes in tokens and returns a dictionary with keys of worker id and values of the number of matched KV Blocks.

### Inter-Router Communication

In distributed deployments with multiple routers, each router maintains visibility over only a portion of the total requests. To ensure consistent routing decisions, routers synchronize their states through three event types:

1. **AddRequest**: Notifies other routers when a request is assigned to a worker. Includes request ID, worker ID, token sequence blocks, and overlap score to track block usage across the system.

2. **MarkPrefillCompleted**: Signals when a request moves from prefill to decode phase, allowing routers to update their worker load calculations by excluding completed prefill tokens.

3. **Free**: Indicates request completion and resource release, enabling accurate block reference counting across all routers.

Each event carries a unique router ID to prevent self-event processing. This asynchronous communication system ensures optimal routing decisions by maintaining consistent KV cache state across all routers, even as they handle different request streams.

### Event Persistence and Recovery

KV cache events are persisted in NATS JetStream, allowing router replicas to maintain their global view of KV blocks across restarts. When a router starts with `--router-persist-states`, it downloads any available snapshot from NATs object store and continues consuming events from its last acknowledged position in the stream.

To manage stream growth, when the message count exceeds `--router-snapshot-threshold`, a router acquires an etcd-based distributed lock, purges acknowledged messages from the stream, and uploads the current radix tree state to NATs object store. This snapshot serves as a checkpoint for faster initialization of future router instances.
