# Dynamo Distributed Runtime

## Overview

Dynamo `DistributedRuntime` is the core infrasturcutre in dynamo that enables distributed communication and coordination between different dynamo components. It is implemented in rust (`/lib/runtime`) and exposed other programming languages via binding (i.e., python bindings can be found in `/lib/bindings/python`). `DistributedRuntime` follows a hierarchical structure:

- `DistributedRuntime`: This is the highest level object that exposes the distributed runtime interface. It maintains connection to external services (e.g., ETCD for service discovery and NATS for messaging) and manages lifecycle with cancellation tokens. 
- `Namespace`: A `Namespace` is a logical grouping of components that isolate between different model deployments. A `DistributedRuntime` can have multiple `Namespace`s.
- `Component`: A `Component` is a discoverable object within a `Namespace` that represents a logical unit of workers. A `Namespace` can have multiple `Component`s.
- `Endpoint`: An `Endpoint` is a network-accessible service that provides a specific service or function. A `Component` can have multiple `Endpoint`s.

For example, the deployment configuration `examples/llm/configs/disagg.yaml` contains four workers:

- `Frontend`: Start an HTTP server and register a `chat/completions` endpoint. The HTTP server route the request to the `Processor`.
- `Processor`: When a new request arrives, `Processor` applies the chat template and perform the tokenization. Then, it route the request to the `VllmWorker`.
- `VllmWorker` and `PrefillWorker`: Perform the actual decode and prefill computation.

Since the four workers are deployed in different processes, each of them have their own `DistributedRuntime`. Within their own `DistributedRuntime`, they all have their own `Namespace`s named `dynamo`. Then, under their own `dynamo` namespace, they have their own `Component`s named `Frontend/Processor/VllmWorker/PrefillWorker`. Lastly, for the `Endpoint`, `Frontend` has no `Endpoints`, `Processor` and `VllmWorker` each has a `generate` endpoint, and `PrefillWorker` has a placeholder `mock` endpoint. Their `DistributedRuntime`s and `Namespace`s are set in the `@service` decorators in `examples/llm/components/<frontend/processor/worker/prefill_worker>.py`. Their `Component`s are set by their name in `/deploy/dynamo/sdk/src/dynamo/sdk/cli/serve_dynamo.py`. Their `Endpoint`s are set by the `@dynamo_endpoint` decorators in `examples/llm/components/<frontend/processor/worker/prefill_worker>.py`.

## Initilization

In this section, we explain what happens under the hood when `DistributedRuntime/Namespace/Component/Endpoint` objects are created. There are two modes for `DistributedRuntime` initialization: dynamic and static. In static mode, componenets and endpoints are defined using known addresses and do not change during runtime. In dynamic modes, componenets and endpoints are discovered through the network and can change during runtime. We focus on the dynamic mode in the rest of this document. Static mode is simply dynamic mode without registration and discovery.

- `DistributedRuntime`: When a `DistributedRuntime` object is created, it will estiablish connections to the following two services:
    - ETCD (dynamic mode only): for service discovery,
    - NATS (both static and dynamic mode): for messaging,
  
  where ETCD and NATS are two global services (there could be multiple ETCD and NATS services for high availability).

  For ETCD, it also creates a primary lease and spin up a background task to keep the lease alive. All objects registered under this `DistributedRuntime` will use this lease_id to maintain their life cycle. There is also a cancellation token that is tied to the primary lease. When the cancellation token is fired or the background task failed, the primary lease will be revoked or expired and the kv pairs stored with this lease_id will be removed.
- `Namespace`: `Namespace`s are primarily a logical grouping mechanism and is not registered in ETCD. It provides the root path for all componenets under this `Namespace`.
- `Component`: When a `Component` object is created, similar to `Namespace`, it will not be registered in ETCD. When `create_service` is called, it creates a NATS service and register it in the registry so that later endpoints can be added to. 




