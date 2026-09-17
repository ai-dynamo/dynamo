---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Architecture
subtitle: Request flow, distributed runtime, service discovery, and communication planes
---

NVIDIA Dynamo separates request execution, service discovery, and event delivery. This page describes how those parts work together in local and Kubernetes deployments.

## Request Flow

The main request path is:

1. **Request (S1)**: HTTP client sends API request to Frontend (OpenAI-compatible server on port 8000)
2. **Preprocess (S2)**: Frontend preprocesses the request (applies chat template, tokenizes) and validates it
3. **Route to Prefill (S3)**: PrefillRouter selects a prefill worker using KV-aware routing or load balancing

### Prefill

4. **Prefill (S4)**: Prefill worker executes the prefill computation on the input tokens and generates KV cache
5. **Return Metadata (S5)**: Prefill worker returns `disaggregated_params` containing backend-specific transfer metadata

### Decode Routing

6. **Route to Decode (S6)**: PrefillRouter injects prefill result into decode request and routes to decode worker
7. **KV Transfer (S7)**: Decode worker coordinates with prefill worker for direct GPU-to-GPU KV cache transfer via NIXL

### Completion

8. **Decode (S8)**: Decode worker generates tokens using the transferred KV cache
9. **Response (S9)**: Generated tokens stream back through Frontend for post-processing (detokenization) and delivery to Client

## Distributed Runtime

The Rust `DistributedRuntime` in `lib/runtime` provides discovery, endpoint registration, request transport, and lifecycle management. Python components use the same runtime through the bindings in `lib/bindings/python`.

The runtime organizes services into four levels:

- `DistributedRuntime` owns connections, background tasks, and cancellation.
- `Namespace` isolates one logical deployment or model group.
- `Component` groups workers that perform the same role.
- `Endpoint` exposes a network service such as `generate`, `clear_kv_blocks`, or `load_metrics`.

Each process creates its own runtime. Components in one deployment use the same namespace so that frontends, routers, planners, and workers can discover each other. A client resolves an endpoint path such as `namespace.component.endpoint`, watches for membership changes, and selects an instance with random, round-robin, or direct dispatch.

### Local Worker Inhibition

After a routed request fails, the local runtime temporarily inhibits the failed worker while service discovery catches up. `DYN_RUNTIME_INHIBITED_DURATION_SECS` controls this interval and defaults to 5 seconds. Discovery remains authoritative and can restore or remove the worker before the timer expires.

## Communication Planes

Dynamo uses separate planes for discovery, requests, and events. The planes can use different transports.

### Discovery Plane

Workers register endpoints when they start. Clients watch the selected discovery backend for membership changes.

| Deployment | Discovery backend | Configuration |
| --- | --- | --- |
| Kubernetes with the Dynamo Operator | `DynamoWorkerMetadata` resources and `EndpointSlice` objects | The operator sets `DYN_DISCOVERY_BACKEND=kubernetes` |
| Local or bare metal | etcd by default | `DYN_DISCOVERY_BACKEND=etcd` and `ETCD_ENDPOINTS` |

The runtime also supports memory and file-backed discovery for development. In etcd mode, leases remove stale endpoints after a process stops sending keep-alive messages.

### Request Plane

The request plane carries RPC traffic between Dynamo components. `DYN_REQUEST_PLANE` selects the transport:

- `tcp` is the default and uses direct pooled connections.
- `nats` uses brokered request transport.

`DYN_REQUEST_PLANE_CODEC` selects `msgpack` or `json`. The destination endpoint advertises its codec, so one client can communicate with endpoints that use different codecs.

### Event Plane

The event plane carries KV cache updates, worker telemetry, and other asynchronous signals. `DYN_EVENT_PLANE` selects `zmq` or `nats`. ZMQ is the default and discovers publishers through the discovery plane. NATS uses subjects scoped by namespace and component.

The request and event planes are independent. For example, a deployment can use TCP for requests and ZMQ for KV events. To route without published KV events, start the frontend with `--no-router-kv-events`.

### Control Connections

- The frontend and workers expose signals that the Planner uses for scaling decisions.
- The Planner updates the desired worker counts.
- The Dynamo Operator reconciles those counts on Kubernetes.

## Technical Implementation Details

### PrefillRouter Orchestration
- The `PrefillRouter` sits between the Frontend and workers, orchestrating disaggregated serving
- Selects prefill workers using KV-aware routing (cache overlap scores + load) or simple load balancing
- Injects transfer metadata into decode requests for KV cache coordination

### NIXL
- Enables high-speed GPU-to-GPU data transfers using NVLink, InfiniBand/UCX, or PCIe
- Transfer metadata exchanged via `disaggregated_params` in prefill response
- Backend-specific coordination: SGLang uses bootstrap connections, TRTLLM uses opaque state, vLLM uses block IDs

### Disaggregated KV Cache
- Each worker maintains local KV cache in its GPU memory
- No shared storage bottlenecks—transfers are direct worker-to-worker via NIXL
- Non-blocking transfers allow GPU forward passes to continue during KV transfer

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'primaryColor': '#f4f4f4', 'primaryTextColor': '#333333', 'primaryBorderColor': '#888888', 'lineColor': '#4A90E2', 'sectionBkgColor': '#f9f9f9', 'altSectionBkgColor': '#eeeeee', 'tertiaryColor': '#f0f0f0', 'background': '#ffffff', 'mainBkg': '#f8f8f8', 'secondaryColor': '#f4f4f4', 'nodeTextColor': '#333333'}, 'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'fontFamily': 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', 'fontSize': '18px'}%%
graph TD
    %% Top Layer - Client & Frontend
    Client["<b>HTTP Client</b>"]
    Frontend["<b>Frontend</b><br/><i>OpenAI Compatible Server<br/>Port 8000</i>"]
    S1[["<b>1 REQUEST</b>"]]
    S2[["<b>2 PREPROCESS</b>"]]

    %% Router Layer
    PrefillRouter["<b>PrefillRouter</b><br/><i>Orchestrates Disaggregated Serving</i>"]
    S3[["<b>3 ROUTE TO PREFILL</b>"]]

    %% Infrastructure
    subgraph INF["<b>Infrastructure Layer</b>"]
        Discovery[("<b>Discovery</b><br/><i>Service Registry<br/>(ETCD or K8s)</i>")]
        NATS[("<b>NATS</b><br/><i>KV Events<br/>(Optional)</i>")]
        Planner["<b>Planner</b><br/><i>Auto-scaling</i>"]
    end

    %% Worker Layer
    subgraph WL["<b>Worker Layer</b>"]
        %% Prefill Worker
        PrefillWorker["<b>Prefill Worker</b><br/><i>Computes KV Cache</i>"]
        S4[["<b>4 PREFILL</b>"]]
        S5[["<b>5 RETURN METADATA</b>"]]

        %% Decode Worker
        DecodeWorker["<b>Decode Worker</b><br/><i>Token Generation</i>"]
        S6[["<b>6 ROUTE TO DECODE</b>"]]
        S7[["<b>7 KV TRANSFER</b>"]]
        S8[["<b>8 DECODE</b>"]]
        S9[["<b>9 RESPONSE</b>"]]

        %% KV Cache
        PrefillKVCache[("<b>Prefill KV Cache</b><br/><i>GPU VRAM</i>")]
        DecodeKVCache[("<b>Decode KV Cache</b><br/><i>GPU VRAM</i>")]
    end

    %% Main Request Flow (Blue)
    Client --> S1
    S1 -->|HTTP API Call| Frontend
    Frontend --> S2
    S2 -->|Tokenize & Validate| PrefillRouter
    PrefillRouter --> S3
    S3 -->|Select Prefill Worker| PrefillWorker

    %% Prefill Flow (Green)
    PrefillWorker --> S4
    S4 -->|Compute KV Cache| PrefillKVCache
    PrefillWorker --> S5
    S5 -->|disaggregated_params| PrefillRouter

    %% Decode Routing Flow (Orange)
    PrefillRouter --> S6
    S6 -->|Inject Transfer Metadata| DecodeWorker
    DecodeWorker --> S7
    S7 -->|NIXL GPU-to-GPU| PrefillKVCache
    PrefillKVCache -.->|Direct Transfer| DecodeKVCache

    %% Completion Flow (Purple)
    DecodeWorker --> S8
    S8 -->|Generate Tokens| DecodeKVCache
    DecodeWorker --> S9
    S9 -->|Stream Tokens| Frontend
    Frontend -->|HTTP Response| Client

    %% Infrastructure Connections
    Frontend -.->|Service Discovery| Discovery
    PrefillRouter -.->|Worker Discovery| Discovery
    PrefillWorker -.->|Register| Discovery
    DecodeWorker -.->|Register| Discovery
    Planner -.->|Service Discovery| Discovery

    %% NATS for KV events (optional)
    PrefillWorker -.->|KV Events| NATS
    DecodeWorker -.->|KV Events| NATS

    %% Planning Connections
    Frontend -.->|Metrics| Planner
    Planner -.->|Auto-scaling| PrefillWorker
    Planner -.->|Auto-scaling| DecodeWorker

    %% Styling
    classDef client fill:#e8f5e8,stroke:#2E7D32,stroke-width:3px
    classDef frontend fill:#fff3e0,stroke:#F57C00,stroke-width:3px
    classDef router fill:#f3e5f5,stroke:#7B1FA2,stroke-width:3px
    classDef worker fill:#e3f2fd,stroke:#1565C0,stroke-width:3px
    classDef prefillWorker fill:#e8f5e9,stroke:#388E3C,stroke-width:3px
    classDef planner fill:#f1f8e9,stroke:#558B2F,stroke-width:3px
    classDef storage fill:#e0f2f1,stroke:#00695C,stroke-width:3px
    classDef discovery fill:#fff9c4,stroke:#F9A825,stroke-width:3px
    classDef nats fill:#ede7f6,stroke:#5E35B1,stroke-width:3px
    classDef infraLayer fill:#fff9c4,stroke:#FFC107,stroke-width:3px
    classDef workerLayer fill:#e3f2fd,stroke:#2196F3,stroke-width:3px

    class Client client
    class Frontend frontend
    class PrefillRouter router
    class DecodeWorker worker
    class PrefillWorker prefillWorker
    class Planner planner
    class PrefillKVCache,DecodeKVCache storage
    class Discovery discovery
    class NATS nats
    class INF infraLayer
    class WL workerLayer

    %% Flow Colors
    %% Main Request Flow - Blue
    linkStyle 0,1,2,3,4,5 stroke:#1565C0,stroke-width:4px

    %% Prefill Flow - Green
    linkStyle 6,7,8,9 stroke:#2E7D32,stroke-width:4px

    %% Decode Routing Flow - Orange
    linkStyle 10,11,12,13,14 stroke:#E65100,stroke-width:4px

    %% Completion Flow - Purple
    linkStyle 15,16,17,18,19 stroke:#6A1B9A,stroke-width:4px

    %% Infrastructure - Gray dotted
    linkStyle 20,21,22,23,24,25,26,27,28,29 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
```
