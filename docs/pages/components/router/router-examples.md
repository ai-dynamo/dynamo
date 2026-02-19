---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Examples
---

# Router Examples

For quick start instructions, see the [Router README](README.md). This document provides further examples for using the Dynamo Router, including Python API usage, Kubernetes deployments, and custom routing patterns.

## Table of Contents

- [Using KvRouter Python API](#using-kvrouter-python-api)
- [K8s Examples](#k8s-examples)
- [Routing Patterns](#routing-patterns)
- [Custom Routing Example: Minimizing TTFT](#custom-routing-example-minimizing-ttft)
- [KV Event Publishing for Custom Engines](#kv-event-publishing-for-custom-engines)
- [Global Router (Hierarchical Routing)](#global-router-hierarchical-routing)

## Using KvRouter Python API

Instead of launching the KV Router via command line, you can create a `KvRouter` object directly in Python. This allows per-request routing configuration overrides.

>[!Warning]
> **Multiple Routers in Same Process**: If you need to run multiple `KvRouter` instances for fault tolerance or load distribution, you must launch them in **separate processes** (e.g., using `python -m dynamo.frontend` with different ports). Creating multiple `KvRouter` objects in the same Python process is not supported - they share the same cancellation token from the component's primary lease, so dropping one router will cancel all routers in that process. For in-process routing, use a single `KvRouter` instance.

### Methods

The `KvRouter` provides the following methods:

- **`generate(token_ids, model, ...)`**: Route and execute a request, returning an async stream of responses. Automatically handles worker selection, state tracking, and lifecycle management.

- **`best_worker(token_ids, router_config_override=None, request_id=None)`**: Query which worker would be selected for given tokens. Returns `(worker_id, dp_rank, overlap_blocks)`.
  - Without `request_id`: Query-only, doesn't update router state
  - With `request_id`: Updates router state to track the request. **Note**: If used with `request_id`, you must call `mark_prefill_complete()` and `free()` at the appropriate lifecycle points to maintain accurate load tracking

- **`get_potential_loads(token_ids)`**: Get detailed load information for all workers, including potential prefill tokens and active decode blocks. Returns a list of load dictionaries.

- **`mark_prefill_complete(request_id)`**: Signal that a request has completed its prefill phase. Only used for [manual lifecycle management](#2-manual-state-management-advanced) when using `best_worker()` for manual routing instead of `generate()`.

- **`free(request_id)`**: Signal that a request has completed and its resources should be released. Only used for [manual lifecycle management](#2-manual-state-management-advanced) when using `best_worker()` for manual routing instead of `generate()`.

- **`dump_events()`**: Dump all KV cache events from the router's indexer as a JSON string. Useful for debugging and analysis.

### Setup

First, launch your backend engines:
```bash
python -m dynamo.vllm --model meta-llama/Llama-2-7b-hf
```

### Example Script

```python
import asyncio
from dynamollm import DistributedRuntime, KvRouter, KvRouterConfig

async def main():
    # Get runtime and create endpoint
    runtime = DistributedRuntime.detached()
    namespace = runtime.namespace("dynamo")
    component = namespace.component("backend")
    endpoint = component.endpoint("generate")

    # Create KV router
    kv_router_config = KvRouterConfig()
    router = KvRouter(
        endpoint=endpoint,
        block_size=16,
        kv_router_config=kv_router_config
    )

    # Your input tokens
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Generate with per-request routing override
    stream = await router.generate(
        token_ids=token_ids,
        model="meta-llama/Llama-2-7b-hf",
        stop_conditions={
            "max_tokens": 20,        # Generate exactly 20 tokens
            "ignore_eos": True,      # Don't stop at EOS token
        },
        sampling_options={
            "temperature": 0.7,
            "top_p": 0.9,
        },
        router_config_override={
            "overlap_score_weight": 2.0,    # Prioritize cache hits for this request
            "router_temperature": 0.5,       # Add routing randomness
        }
    )

    # Collect generated tokens
    generated_tokens = []
    async for response in stream:
        if isinstance(response, dict) and "token_ids" in response:
            generated_tokens.extend(response["token_ids"])

    print(f"Generated {len(generated_tokens)} tokens: {generated_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
```

## K8s Examples

For basic Kubernetes deployment with the KV Router, see the [Kubernetes Deployment section](README.md#kubernetes-deployment) in the Quick Start guide.

### Complete K8s Examples

- [TRT-LLM aggregated router example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/trtllm/deploy/agg_router.yaml)
- [vLLM aggregated router example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/vllm/deploy/agg_router.yaml)
- [SGLang aggregated router example](https://github.com/ai-dynamo/dynamo/tree/main/examples/backends/sglang/deploy/agg_router.yaml)
- [Distributed inference tutorial](https://github.com/ai-dynamo/dynamo/tree/main/examples/basics/kubernetes/Distributed_Inference/agg_router.yaml)

**For A/B Testing and Advanced K8s Setup:**
See the comprehensive [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md) for step-by-step instructions on deploying, configuring, and benchmarking the KV router in Kubernetes.

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
        - name: DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT
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

## Routing Patterns

The `KvRouter` supports multiple usage patterns depending on your control requirements:

### 1. Automatic Routing (Recommended)
Call `generate()` directly and let the router handle everything:
```python
stream = await router.generate(token_ids=tokens, model="model-name")
```
- **Best for**: Most use cases
- **Router automatically**: Selects best worker, updates state, routes request, tracks lifecycle

### 2. Manual State Management (Advanced)
Use `best_worker(request_id=...)` to select and track, then manage the request yourself:
```python
worker_id, _dp_rank, overlap = await router.best_worker(tokens, request_id="req-123")
response = await client.generate(tokens, request_id="req-123")
# await anext(response)  # Get first token
await router.mark_prefill_complete("req-123")  # After first token
# async for _ in response:  # Continue generating
#     ...
await router.free("req-123")  # After completion
```
- **Best for**: Custom request handling with router state tracking
- **Requires**: Calling `mark_prefill_complete()` and `free()` at correct lifecycle points
- **Caution**: Incorrect lifecycle management degrades load balancing accuracy

### 3. Hierarchical Router Probing
Query without state updates, then route through a chosen router:
```python
# Probe multiple routers without updating state
worker_id_1, dp_rank, overlap_1 = await router_1.best_worker(tokens)  # No request_id
worker_id_2, dp_rank, overlap_2 = await router_2.best_worker(tokens)

# Pick the best router based on results
chosen_router = router_1 if overlap_1 > overlap_2 else router_2
stream = await chosen_router.generate(tokens, model="model-name", worker_id=worker_id)
```
- **Best for**: Multi-tier deployments (e.g., Envoy Gateway routing to multiple router groups)
- **Advantage**: Query multiple routers before committing to one

### 4. Custom Load-Based Routing
Use `get_potential_loads()` to implement custom routing logic:
```python
loads = await router.get_potential_loads(tokens)
# Apply custom logic (e.g., weighted scoring, constraints)
best_worker = min(loads, key=lambda x: custom_cost_fn(x))
stream = await router.generate(tokens, model="model-name", worker_id=best_worker['worker_id'])
```
- **Best for**: Custom optimization strategies beyond the built-in cost function
- **Advantage**: Full control over worker selection logic
- **See also**: Detailed example below in "Custom Routing Example: Minimizing TTFT"

All patterns support `router_config_override` to adjust routing behavior per-request without recreating the router.

## Custom Routing Example: Minimizing TTFT

Here's an example of using `get_potential_loads()` to implement custom routing that minimizes Time To First Token (TTFT) by selecting the worker with the least prefill work:

```python
import asyncio
from dynamo.llm import DistributedRuntime, KvRouter, KvRouterConfig

async def minimize_ttft_routing():
    # Setup router
    runtime = DistributedRuntime.detached()
    namespace = runtime.namespace("dynamo")
    component = namespace.component("backend")
    endpoint = component.endpoint("generate")

    router = KvRouter(
        endpoint=endpoint,
        block_size=16,
        kv_router_config=KvRouterConfig()
    )

    # Your input tokens
    token_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Get potential loads for all workers
    potential_loads = await router.get_potential_loads(token_ids)

    # Find worker with minimum prefill tokens (best for TTFT)
    best_worker = min(potential_loads, key=lambda x: x['potential_prefill_tokens'])

    print(f"Worker loads: {potential_loads}")
    print(f"Selected worker {best_worker['worker_id']} with {best_worker['potential_prefill_tokens']} prefill tokens")

    # Route directly to the selected worker
    stream = await router.generate(
        token_ids=token_ids,
        model="meta-llama/Llama-2-7b-hf",
        worker_id=best_worker['worker_id'],  # Force routing to optimal worker
        stop_conditions={"max_tokens": 20}
    )

    # Process response
    async for response in stream:
        if isinstance(response, dict) and "token_ids" in response:
            print(f"Generated tokens: {response['token_ids']}")

if __name__ == "__main__":
    asyncio.run(minimize_ttft_routing())
```

This approach gives you complete control over routing decisions, allowing you to optimize for different metrics based on your specific requirements. As some examples:

- **Minimize TTFT**: Select worker with lowest `potential_prefill_tokens`
- **Maximize cache reuse**: Use `best_worker()` which considers both prefill and decode loads
- **Balance load**: Consider both `potential_prefill_tokens` and `potential_decode_blocks` together

See [Router Design](../../design-docs/router-design.md) for architecture details and the cost function algorithm.

## KV Event Publishing for Custom Engines

For full documentation on implementing KV event publishing for custom inference engines, see the dedicated [KV Event Publishing for Custom Engines](../../integrations/kv-events-custom-engines.md) guide. It covers:

- **Direct publishing**: Call `publish_stored()` / `publish_removed()` to push events over the Dynamo event plane
- **ZMQ relay**: For engines that emit raw KV events over ZMQ (like vLLM and SGLang), the same `KvEventPublisher` subscribes to the ZMQ socket and relays events automatically
- API reference, event structure, ZMQ wire format, and best practices

## Global Router (Hierarchical Routing)

For deployments with multiple worker pools, the **Global Router** enables hierarchical routing by sitting between the frontend and local routers. It selects the appropriate pool for each request based on configurable policies, supporting disaggregated topologies where pools are tuned for different workload characteristics.

- **Component details**: [`components/src/dynamo/global_router/`](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/global_router/)
- **Example**: [`examples/hierarchical_planner/`](https://github.com/ai-dynamo/dynamo/tree/main/examples/hierarchical_planner/)

## See Also

- **[Router README](README.md)**: Quick start guide for the KV Router
- **[Router Guide](router-guide.md)**: Configuration, tuning, and production setup
- **[Router Design](../../design-docs/router-design.md)**: Architecture details and event transport modes
