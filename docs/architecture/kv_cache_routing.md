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


# KV Cache Routing
This documentation explains how Key-Value (KV) cache routing works in Dynamo, providing optimized inference for large language models by intelligently directing requests to workers with the most relevant cached data while simultaneously load balancing based on utilization metrics sent by the workers.

## Architecture
Dynamo's architecture consists of three key concepts:

- **Namespace**: Groups related components (similar to directories in a file system). In our examples, we use the label `dynamo`. This avoids collisions between two different dynamo graphs.
- **Component**: The deployable unit in Dynamo. Components are self-contained and typically map to separate Docker containers. In our examples, we use labels like `VllmWorker `, `Router`, `Processor` for the components. Components can be created in Python or Rust.
- **Endpoint**: Functions attached to components that transform inputs into outputs. Endpoints are discoverable and callable by other components. In our examples we use the label `generate` for most of the endpoints.

A Dynamo graph is a collection of components that are linked together to form a graph. There are two paths through the graphs. The request path and the response path. For LLMs the request path is single-in (a single message) and the response path is many-out (streamed output).

A common pattern is to spin up multiple of the same components that serve the same endpoints, for example, when you want to duplicate models to serve more requests. Each endpoint will get a unique identifier and you will have to tell Dynamo how to route requests between these endpoints.

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
          | KV match: 15%    | KV match: 50%    | KV match: 75%
          v                  v                  v
   +----------------+  +----------------+  +----------------+
   |   Worker 1     |  |   Worker 2     |  |   Worker 3     |
   |  (Load: 30%)   |  |  (Load: 50%)   |  |  (Load: 80%)   |
   +----------------+  +----------------+  +----------------+
```

Load balancing in LLM serving becomes complex when enabling KV Cache reuse. While KV Cache reuse can save significant computation, if the routing strategy is not aware of the unique KV states of each worker we can:
- miss opportunities for KV Cache reuse if routing to the “wrong” node
- get into an imbalanced state where a few workers are processing many requests, lowering throughput of entire system

The best way to solve these issues is for the router to have a global view of KV Cache and load. With this view, the router can use a cost function to score the workers and make decisions to maximize cache hits while keeping the system balanced and throughput high.

In the above image, our cost function is (KV match - Load) so we select Worker 2 even though Worker 3 would offer the best KV match.
- Worker 1 = (0.15 - 0.30) = -0.15
- **Worker 2 = (0.50 - 0.50) = 0**
- Worker 3 = (0.75 - 0.80) = -0.05

## Events

In Dynamo, we want to support KV Cache Routing and load balancing for many backends that have different implementations of KV Cache and record different metrics. To that end, we built a KVPublisher that can be plugged into any framework to publish KV Events and a KvMetricsPublisher that can publish Metric Events.

On the receiving side we have a KVIndexer which accepts events from the KVPublisher and puts them into a global prefix tree and a KvMetricsAggregator which aggregates metric events by worker.

```text
+----------------+                         +-----------------+
|                |                         | KV Aware Router |
|     Worker     |                         |                 |
|                | create_kv_block()       | +-------------+ |
| +------------+ | remove_kv_block()       | |  KVIndexer  | |
| |KVPublisher | |------------------------>| +-------------+ |
| +------------+ |                         |                 |
|                | num_request_waiting     | +--------------+|
| +------------+ | gpu_cache_usage_perc    | |KvMetricsAggre||
| |KvMetrics   | |------------------------>| |    gator     ||
| |Publisher   | |        ...              | +--------------+|
| +------------+ |                         +-----------------+
+----------------+

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

Example:
```python
from dynamo.llm import KvIndexer
from dynamo.sdk import dynamo_context

runtime = dynamo_context["runtime"]
kv_listener = runtime.namespace("dynamo").component("VllmWorker")
await kv_listener.create_service()

indexer = KvIndexer(kv_listener, block_size=16)
indexer.find_matches_for_request([INPUT SEQUENCE OF TOKEN IDs])
```

Sample Output:
```
{
	123456789: 10,
	987654321: 3,
	543219876: 7,
}
```

```{note}
This example is designed to help you understand KV cache routing; it won't run outside of the context of dynamo serve. See the examples/ directory for runnable examples.
```

### KvMetricsPublisher
We added a KvMetrics Publisher which sends the following metrics to the KvMetricsAggregator:
- num_requests_waiting
- gpu_cache_usage_perc
- gpu_prefix_cache_hit_rate
- request_active_slots
- request_total_slots
- kv_active_blocks
- kv_total_blocks

Currently, the KvMetricsPublisher exists as a Python binding.

### KvMetricsAggregator
The KvMetricsAggregator receives these metrics and aggregates them. It has a method `get_metrics` which returns an object of `AggregatedMetrics`.

Example:
```python
from dynamo.llm import KvMetricsAggregator
from dynamo.sdk import dynamo_context

runtime = dynamo_context["runtime"]
kv_listener = runtime.namespace("dynamo").component("VllmWorker")
await kv_listener.create_service()
metrics_aggregator = KvMetricsAggregator(kv_listener)

for endpoint in metrics_aggregator.get_metrics().endpoints:
    print("Worker ID: ", endpoint.worker_id)
    print("GPU Cache Usage: ", endpoint.gpu_cache_usage_perc)
    print("Number of Requests Waiting: ", endpoint.num_requests_waiting)
    print("GPU Prefix Cache Hit Rate: ", endpoint.gpu_prefix_cache_hit_rate)
    print("***")
```

Sample Output:
```
Worker ID: 123456789
GPU Cache Usage: 0.5
Number of Requests Waiting: 2
GPU Prefix Cache Hit Rate: 0.1
***
Worker ID: 987654321
GPU Cache Usage: 0.5
Number of Requests Waiting: 1
GPU Prefix Cache Hit Rate: 0.1
***
```

```{note}
This example is for building understanding, it will not run outside of the context of dynamo serve. See the examples/ folder for runnable examples.
```

### [KV Router](https://github.com/ai-dynamo/dynamo/blob/main/examples/llm/components/kv_router.py)
The Router component makes intelligent worker selection decisions
1. Receives incoming requests as tokens
2. Queries the KVIndexer to find potential cache hits across workers
3. Collects performance metrics from workers (via KvMetricsAggregator)
4. Uses a cost function to determine the optimal worker for each request
5. Returns chosen worker

The processor manages tokenizing the request, sending it to the KV Router and then once it receives a response, directs the request to the selected worker using direct() routing.
