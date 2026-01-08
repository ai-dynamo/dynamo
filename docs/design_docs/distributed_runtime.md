<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Dynamo Distributed Runtime

## Overview

Dynamo's `DistributedRuntime` is the core infrastructure in the framework that enables distributed communication and coordination between different Dynamo components. It is implemented in rust (`/lib/runtime`) and exposed to other programming languages via bindings (i.e., python bindings can be found in `/lib/bindings/python`). `DistributedRuntime` follows a hierarchical structure:

- `DistributedRuntime`: This is the highest level object that exposes the distributed runtime interface. It maintains connection to external services (e.g., etcd for service discovery and NATS for messaging) and manages lifecycle with cancellation tokens.
- `Namespace`: A `Namespace` is a logical grouping of components that isolate between different model deployments.
- `Component`: A `Component` is a discoverable object within a `Namespace` that represents a logical unit of workers.
- `Endpoint`: An `Endpoint` is a network-accessible service that provides a specific service or function.

While theoretically each `DistributedRuntime` can have multiple `Namespace`s as long as their names are unique (similar logic also applies to `Component/Namespace` and `Endpoint/Component`), in practice, each dynamo components typically are deployed with its own process and thus has its own `DistributedRuntime` object. However, they share the same namespace to discover each other.

For example, a typical deployment configuration (like `examples/backends/vllm/deploy/agg.yaml` or `examples/backends/sglang/deploy/agg.yaml`) has multiple workers:

- `Frontend`: Starts an HTTP server and handles incoming requests. The HTTP server routes all requests to the `Processor`.
- `Processor`: When a new request arrives, `Processor` applies the chat template and performs the tokenization.
Then, it routes the request to the `Worker`.
- `Worker` components (e.g., `VllmDecodeWorker`, `SGLangDecodeWorker`, `TrtllmWorker`): Perform the actual computation using their respective engines (vLLM, SGLang, TensorRT-LLM).

Since the workers are deployed in different processes, each of them has its own `DistributedRuntime`. Within their own `DistributedRuntime`, they all share the same `Namespace` (e.g., `vllm-agg`, `sglang-agg`). Then, under their namespace, they have their own `Component`s: `Frontend` uses the `make_engine` function which handles HTTP serving and routing automatically, while worker components create components with names like `worker`, `decode`, or `prefill` and register endpoints like `generate`, `flush_cache`, or `clear_kv_blocks`. The `Frontend` component doesn't explicitly create endpoints - instead, the `make_engine` function handles the HTTP server and worker discovery. Worker components create their endpoints programmatically using the `component.endpoint()` method. Their `DistributedRuntime`s are initialized in their respective main functions, their `Namespace`s are configured in the deployment YAML, their `Component`s are created programmatically (e.g., `runtime.namespace("dynamo").component("worker")`), and their `Endpoint`s are created using the `component.endpoint()` method.

## Initialization

In this section, we explain what happens under the hood when `DistributedRuntime/Namespace/Component/Endpoint` objects are created. There are two modes for `DistributedRuntime` initialization: dynamic and static. In static mode, components and endpoints are defined using known addresses and do not change during runtime. In dynamic modes, components and endpoints are discovered through the network and can change during runtime. We focus on the dynamic mode in the rest of this document. Static mode is basically dynamic mode without registration and discovery and hence does not rely on etcd.

```{caution}
The hierarchy and naming in etcd and NATS may change over time, and this document might not reflect the latest changes. Regardless of such changes, the main concepts would remain the same.
```

- `DistributedRuntime`: When a `DistributedRuntime` object is created, it establishes connections to the following two services:
    - etcd (dynamic mode only): for service discovery. In static mode, `DistributedRuntime` can operate without etcd.
    - NATS (both static and dynamic mode): for messaging.

  where etcd and NATS are two global services (there could be multiple etcd and NATS services for high availability).

  For etcd, it also creates a primary lease and spin up a background task to keep the lease alive. All objects registered under this `DistributedRuntime` use this lease_id to maintain their life cycle. There is also a cancellation token that is tied to the primary lease. When the cancellation token is triggered or the background task failed, the primary lease is revoked or expired and the kv pairs stored with this lease_id is removed.
- `Namespace`: `Namespace`s are primarily a logical grouping mechanism and is not registered in etcd. It provides the root path for all components under this `Namespace`.
- `Component`: When a `Component` object is created, similar to `Namespace`, it isn't be registered in etcd. When `create_service` is called, it creates a NATS service group using `{namespace_name}.{service_name}` as the service identifier and registers a service in the registry of the `Component`, where the registry is an internal data structure that tracks all services and endpoints within the `DistributedRuntime`.
- `Endpoint`: When an Endpoint object is created and started, it performs two key registrations:
  - NATS Registration: The endpoint is registered with the NATS service group created during service creation. The endpoint is assigned a unique subject following the naming: `{namespace_name}.{service_name}.{endpoint_name}-{lease_id_hex}`.
  - etcd Registration: The endpoint information is stored in etcd at a path following the naming: `/services/{namespace}/{component}/{endpoint}-{lease_id}`. Note that the endpoints of different workers of the same type (i.e., two `VllmPrefillWorker`s in one deployment) share the same `Namespace`, `Component`, and `Endpoint` name. They are distinguished by their different primary `lease_id` of their `DistributedRuntime`.

## Calling Endpoints

Dynamo uses `Client` object to call an endpoint. When a `Client` objected is created, it is given the name of the `Namespace`, `Component`, and `Endpoint`. It then sets up an etcd watcher to monitor the prefix `/services/{namespace}/{component}/{endpoint}`. The etcd watcher continuously updates the `Client` with the information, including `lease_id` and NATS subject of the available `Endpoint`s.

The user can decide which load balancing strategy to use when calling the `Endpoint` from the `Client`, which is done in [push_router.rs](../../lib/runtime/src/pipeline/network/egress/push_router.rs). Dynamo supports three load balancing strategies:

- `random`: randomly select an endpoint to hit
- `round_robin`: select endpoints in round-robin order
- `direct`: direct the request to a specific endpoint by specifying the `lease_id` of the endpoint

After selecting which endpoint to hit, the `Client` sends the serialized request to the NATS subject of the selected `Endpoint`. The `Endpoint` receives the request and create a TCP response stream using the connection information from the request, which establishes a direct TCP connection to the `Client`. Then, as the worker generates the response, it serializes each response chunk and sends the serialized data over the TCP connection.

## Examples

We provide native rust and python (through binding) examples for basic usage of `DistributedRuntime`:

- Rust: `/lib/runtime/examples/`
- Python: We also provide complete examples of using `DistributedRuntime`. Please refer to the engines in `components/src/dynamo` for full implementation details.


