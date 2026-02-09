---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
---

# Dynamo Distributed Runtime

## Overview

Dynamo's `DistributedRuntime` is the core infrastructure in the framework that enables distributed communication and coordination between different Dynamo components. It is implemented in Rust (`/lib/runtime`) and exposed to other programming languages via bindings (i.e., Python bindings can be found in `/lib/bindings/python`). The runtime supports multiple discovery backends (Kubernetes-native or etcd) and request planes (TCP, HTTP, or NATS). `DistributedRuntime` follows a hierarchical structure:

- `DistributedRuntime`: This is the highest level object that exposes the distributed runtime interface. It manages connections to discovery backends (K8s API or etcd) and optional messaging (NATS for KV events), and handles lifecycle with cancellation tokens.
- `Namespace`: A `Namespace` is a logical grouping of components that isolate between different model deployments.
- `Component`: A `Component` is a discoverable object within a `Namespace` that represents a logical unit of workers.
- `Endpoint`: An `Endpoint` is a network-accessible service that provides a specific service or function.

While theoretically each `DistributedRuntime` can have multiple `Namespace`s as long as their names are unique (similar logic also applies to `Component/Namespace` and `Endpoint/Component`), in practice, each dynamo components typically are deployed with its own process and thus has its own `DistributedRuntime` object. However, they share the same namespace to discover each other.

For example, a typical deployment configuration (like `examples/backends/vllm/deploy/agg.yaml` or `examples/backends/sglang/deploy/agg.yaml`) has multiple components:

- `Frontend`: Starts an HTTP server (OpenAI-compatible API on port 8000), handles incoming requests, applies chat templates, performs tokenization, and routes requests to workers. The `make_engine` function encapsulates this functionality.
- `Worker` components (e.g., `VllmDecodeWorker`, `VllmPrefillWorker`, `SGLangDecodeWorker`, `TRTLLMWorker`): Perform the actual inference computation using their respective engines (vLLM, SGLang, TensorRT-LLM).

Since these components are deployed in different processes, each has its own `DistributedRuntime`. Within their own `DistributedRuntime`, they all share the same `Namespace` (e.g., `vllm-agg`, `sglang-disagg`). Under their namespace, each has its own `Component`:

- `Frontend` uses the `make_engine` function which handles HTTP serving, request preprocessing, and worker discovery automatically
- Worker components register with names like `backend`, `prefill`, `decode`, or `encoder` depending on their role
- Workers register endpoints like `generate`, `clear_kv_blocks`, or `load_metrics`

Their `DistributedRuntime`s are initialized in their respective main functions, their `Namespace`s are configured in the deployment YAML, their `Component`s are created programmatically (e.g., `runtime.namespace("dynamo").component("backend")`), and their `Endpoint`s are created using the `component.endpoint()` method.

## Initialization

In this section, we explain what happens under the hood when `DistributedRuntime/Namespace/Component/Endpoint` objects are created. There are multiple modes for `DistributedRuntime` initialization based on the deployment environment.

```{caution}
The hierarchy and naming may change over time, and this document might not reflect the latest changes. Regardless of such changes, the main concepts would remain the same.
```

### Service Discovery Backends

The `DistributedRuntime` supports two service discovery backends, configured via `DYN_DISCOVERY_BACKEND`:

- **KV Store Discovery** (`DYN_DISCOVERY_BACKEND=kv_store`): Uses etcd for service discovery. **This is the global default** for all deployments unless explicitly overridden.

- **Kubernetes Discovery** (`DYN_DISCOVERY_BACKEND=kubernetes`): Uses native Kubernetes resources (DynamoWorkerMetadata CRD, EndpointSlices) for service discovery. **Must be explicitly set.** The Dynamo operator automatically sets this environment variable for Kubernetes deployments. **No etcd required.**

> **Note:** There is no automatic detection of the deployment environment. The runtime always defaults to `kv_store`. For Kubernetes deployments, the operator injects `DYN_DISCOVERY_BACKEND=kubernetes` into pod environments.

When using Kubernetes discovery, the KV store backend automatically switches to in-memory storage since etcd is not needed.

### Runtime Initialization

- `DistributedRuntime`: When a `DistributedRuntime` object is created, it establishes connections based on the discovery backend:
    - **Kubernetes mode**: Uses K8s API for service registration via DynamoWorkerMetadata CRD. No external dependencies required.
    - **KV Store mode**: Connects to etcd for service discovery. Creates a primary lease with a background keep-alive task. All objects registered under this `DistributedRuntime` use this lease_id to maintain their lifecycle.
    - **NATS** (optional): Used for KV event messaging when using KV-aware routing. Can be disabled via `--no-kv-events` flag, which enables prediction-based routing without event persistence.
    - **Request Plane**: TCP by default. Can be configured to use HTTP or NATS via `DYN_REQUEST_PLANE` environment variable.
- `Namespace`: `Namespace`s are primarily a logical grouping mechanism. They provide the root path for all components under this `Namespace`.
- `Component`: When a `Component` object is created, it registers a service in the internal registry of the `DistributedRuntime`, which tracks all services and endpoints.
- `Endpoint`: When an Endpoint object is created and started, it performs registration based on the discovery backend:
  - **Kubernetes mode**: Endpoint information is stored in DynamoWorkerMetadata CRD resources, which are watched by other components for discovery.
  - **KV Store mode**: Endpoint information is stored in etcd at a path following the naming: `/services/{namespace}/{component}/{endpoint}-{lease_id}`. Note that endpoints of different workers of the same type (i.e., two `VllmPrefillWorker`s in one deployment) share the same `Namespace`, `Component`, and `Endpoint` name. They are distinguished by their different primary `lease_id`.

## Calling Endpoints

Dynamo uses a `Client` object to call an endpoint. When a `Client` is created, it is given the name of the `Namespace`, `Component`, and `Endpoint`. It then watches for endpoint changes:

- **Kubernetes mode**: Watches DynamoWorkerMetadata CRD resources for endpoint updates.
- **KV Store mode**: Sets up an etcd watcher to monitor the prefix `/services/{namespace}/{component}/{endpoint}`.

The watcher continuously updates the `Client` with information about available `Endpoint`s.

The user can decide which load balancing strategy to use when calling the `Endpoint` from the `Client`, which is done in [push_router.rs](https://github.com/ai-dynamo/dynamo/tree/main/lib/runtime/src/pipeline/network/egress/push_router.rs). Dynamo supports three load balancing strategies:

- `random`: randomly select an endpoint to hit
- `round_robin`: select endpoints in round-robin order
- `direct`: direct the request to a specific endpoint by specifying the instance ID

After selecting which endpoint to hit, the `Client` sends the request using the configured request plane (TCP by default). The request plane handles the actual transport:

- **TCP** (default): Direct TCP connection with connection pooling
- **HTTP**: HTTP/2-based transport
- **NATS**: Message broker-based transport (legacy)

## Examples

We provide native rust and python (through binding) examples for basic usage of `DistributedRuntime`:

- Rust: `/lib/runtime/examples/`
- Python: We also provide complete examples of using `DistributedRuntime`. Please refer to the engines in `components/src/dynamo` for full implementation details.


