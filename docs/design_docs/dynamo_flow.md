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

# Dynamo Architecture Flow

This diagram shows the NVIDIA Dynamo disaggregated inference system as implemented in [examples/backends/vllm](../../examples/backends/vllm). Color-coded flows indicate different types of operations.

> **Note**: The "Processor" shown in the diagram represents the request processing logic (tokenization, chat template application, routing) that runs within the Frontend component. It is not a separate deployment—the Frontend handles both HTTP serving and request preprocessing via the `make_engine` function.

## 🔵 Main Request Flow (Blue)
The primary user journey through the system:

1. **Discovery (S1)**: Client discovers the service endpoint
2. **Request (S2)**: HTTP client sends API request to Frontend (OpenAI-compatible server on port 8000)
3. **Validate (S3)**: Frontend preprocesses the request (applies chat template, tokenizes) and validates it
4. **Route (S3)**: Frontend routes the validated request to appropriate Decode Worker

## 🟠 Decision and Allocation Flow (Orange)
The system's intelligent routing and resource allocation:

4. **Query (S4)**: Decode Worker queries for prefix cache hits to optimize processing
5. **Disagg Decision (S5)**: Based on prefill length and queue size, the system decides whether it needs remote prefill
5a. **Allocate (S5a)**: Decode Worker pre-allocates KV cache blocks in its local GPU memory
6. **Queue (S6)**: If remote prefill is required, the system puts the RemotePrefillRequest with block IDs into the PrefillQueue

## 🟢 Prefill Worker Flow (Green)
The dedicated prefill processing pipeline:

7. **NATS Pull (S7)**: PrefillQueue uses a NATS consumer group to distribute work to available PrefillWorkers
8. **Load Metadata (S8)**: PrefillWorker loads NIXL metadata from ETCD to establish GPU communication
9. **Prefill (S9)**: Worker executes the prefill computation on the input tokens
10. **NIXL Transfer (S10)**: Direct GPU-to-GPU transfer writes the prefilled KV cache to the Decode Worker's pre-allocated blocks

## 🟣 Completion Flow (Purple)
The response generation and delivery:

11. **Notify (S11)**: PrefillWorker sends completion notification to Decode Worker
12. **Decode (S12)**: Decode Worker decodes from its local KV cache containing prefilled data
13. **Response (S13)**: The generated response flows back through the Frontend for post-processing (detokenization) and delivery to the Client

## 🔗 Infrastructure Connections (Dotted lines)
Coordination and messaging support:

### Service Discovery
- **On Kubernetes** (default): Uses native K8s resources (DynamoWorkerMetadata CRD, EndpointSlices). No etcd required.
- **On bare metal**: Uses etcd for service discovery and endpoint registration.

### Request Plane
- **TCP** (default): Direct TCP connections between Frontend and Workers for request/response transport.
- **HTTP/NATS**: Alternative transports configurable via `DYN_REQUEST_PLANE`.

### NATS Connections (Optional, for KV routing)
- **PrefillQueue**: JetStream consumer group for reliable work distribution in disaggregated serving
- **KV Events**: Cache state events for KV-aware routing (can be disabled with `--no-kv-events`)

### Planning Connections (Gold, dotted)
- **Frontend → Planner**: Metrics collection for auto-scaling decisions
- **Planner → Workers**: Resource scaling commands for both Decode Worker and PrefillWorker

## Technical Implementation Details

### NIXL (NVIDIA Interchange Library):
- Enables high-speed GPU-to-GPU data transfers using NVLink/PCIe
- Decode Worker publishes GPU metadata to ETCD for coordination
- PrefillWorker loads metadata to establish direct communication channels
- Block-based transfers (64–128 tokens per block) for efficient batching

### Disaggregated KV Cache:
- Each Decode Worker maintains local KV cache in its GPU memory
- No shared storage bottlenecks—all transfers are direct worker-to-worker
- Pre-allocated blocks ensure deterministic memory layout and performance

```mermaid
%%{init: {'theme':'dark', 'themeVariables': {'primaryColor': '#f4f4f4', 'primaryTextColor': '#333333', 'primaryBorderColor': '#888888', 'lineColor': '#4A90E2', 'sectionBkgColor': '#f9f9f9', 'altSectionBkgColor': '#eeeeee', 'tertiaryColor': '#f0f0f0', 'background': '#ffffff', 'mainBkg': '#f8f8f8', 'secondaryColor': '#f4f4f4', 'nodeTextColor': '#333333'}, 'flowchart': {'htmlLabels': true, 'curve': 'basis'}, 'fontFamily': 'Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif', 'fontSize': '18px'}%%
graph TD
    %% Top Layer - Client & Frontend
    Client["<b>HTTP Client</b>"]
    S1[["<b>1 DISCOVERY</b>"]]
    Frontend["<b>Frontend</b><br/><i>OpenAI Compatible Server<br/>Port 8000</i>"]
    S2[["<b>2 REQUEST</b>"]]

    %% Processing Layer
    Processor["<b>Processor</b><br/><i>Request Handler & Router</i>"]
    S3[["<b>3 VALIDATE</b>"]]

    %% Infrastructure - Positioned strategically to minimize crossings
    subgraph INF["<b>Infrastructure Layer</b>"]
        ETCD[("<b>ETCD</b><br/><i>Service Discovery &<br/>NIXL Metadata</i>")]
        NATS[("<b>NATS</b><br/><i>Message Broker</i>")]
        Planner["<b>Planner</b><br/><i>Resource Management<br/>Auto-scaling</i>"]
    end

    %% Worker Layer - Main processing
    subgraph WL["<b>Worker Layer</b>"]
        %% VllmWorker section
        VllmWorker["<b>Decode Worker</b><br/><i>Handles Decoding & Disagg Decisions</i>"]
        S4[["<b>4 QUERY</b>"]]
        S5[["<b>5 DISAGG DECISION</b>"]]
        S5a[["<b>5a ALLOCATE</b>"]]
        S12[["<b>12 DECODE</b>"]]
        S6[["<b>6 QUEUE</b>"]]
        S13[["<b>13 RESPONSE</b>"]]

        %% Storage positioned near workers
        LocalKVCache[("<b>Local KV Cache</b><br/><i>Pre-allocated Blocks</i>")]

        %% Prefill System - Right side to minimize crossings
        subgraph PS["<b>Prefill System</b>"]
            PrefillQueue["<b>Prefill Queue</b><br/><i>NATS JetStream<br/>Consumer Group</i>"]
            PrefillWorker["<b>Prefill Worker</b><br/><i>Dedicated Prefill Processing<br/>(Multiple Instances)</i>"]
            S7[["<b>7 NATS PULL</b>"]]
            S8[["<b>8 LOAD METADATA</b>"]]
            S9[["<b>9 PREFILL</b>"]]
            S10[["<b>10 NIXL TRANSFER</b>"]]
            S11[["<b>11 NOTIFY</b>"]]
        end
    end

    %% Main Request Flow (Blue) - Clean vertical flow
    Client -.-> S1
    S1 -->|HTTP API Call| Frontend
    Frontend -.-> S2
    S2 -->|Process & Validate| Processor
    Processor -.-> S3
    S3 -->|Route to Worker| VllmWorker

    %% VllmWorker Internal Flow (Orange)
    VllmWorker -.-> S4
    S4 -->|Query Prefix Cache Hit| S5
    S5 -->|Prefill Length & Queue Check| S5a
    S5a -->|Continue to Decode| S12

    %% Allocation & Queuing (Orange) - Minimize crossings
    S5a -->|Allocate KV Cache Blocks| LocalKVCache
    VllmWorker --> S6
    S6 -->|Put RemotePrefillRequest| PrefillQueue

    %% Prefill Worker Flow (Green) - Self-contained within PS
    PrefillQueue -.-> S7
    S7 -->|Consumer Group Pull| PrefillWorker
    PrefillWorker -.-> S8
    PrefillWorker -.-> S9
    S9 -->|Execute Prefill| S10
    S10 -->|Direct GPU Transfer| LocalKVCache
    PrefillWorker --> S11

    %% Return Flow (Purple) - Clean return path
    S11 -->|Completion Notification| S12
    S12 -->|Decode from KV Cache| S13
    S13 -->|Post-process Response| Processor
    Processor -->|HTTP Response| Frontend
    Frontend -->|Final Response| Client

    %% Infrastructure Connections - Organized to avoid crossings
    %% ETCD Connections - Grouped by proximity
    Frontend -.->|Service Discovery| ETCD
    Processor -.->|Service Discovery| ETCD
    VllmWorker -.->|NIXL Metadata| ETCD
    PrefillWorker -.->|NIXL Metadata| ETCD
    S8 -.->|Load NIXL Metadata| ETCD
    Planner -.->|Service Discovery| ETCD

    %% NATS Connections - Direct to queue system
    PrefillQueue -.->|JetStream| NATS
    Processor -.->|Load Balancing| NATS

    %% Planning Connections - Strategic positioning
    Frontend -.->|Metrics| Planner
    Planner -.->|Auto-scaling| VllmWorker
    Planner -.->|Auto-scaling| PrefillWorker

    %% Styling - Each component with unique colors
    classDef client fill:#e8f5e8,stroke:#2E7D32,stroke-width:3px
    classDef frontend fill:#fff3e0,stroke:#F57C00,stroke-width:3px
    classDef processor fill:#f3e5f5,stroke:#7B1FA2,stroke-width:3px
    classDef worker fill:#e3f2fd,stroke:#1565C0,stroke-width:3px
    classDef prefillQueue fill:#fff8e1,stroke:#E65100,stroke-width:3px
    classDef prefillWorker fill:#fce4ec,stroke:#C2185B,stroke-width:3px
    classDef prefillBox fill:#eceff1,stroke:#455A64,stroke-width:3px
    classDef planner fill:#f1f8e9,stroke:#558B2F,stroke-width:3px
    classDef storage fill:#e0f2f1,stroke:#00695C,stroke-width:3px
    classDef etcd fill:#fff9c4,stroke:#F9A825,stroke-width:3px
    classDef nats fill:#ede7f6,stroke:#5E35B1,stroke-width:3px
    classDef infraLayer fill:#fff9c4,stroke:#FFC107,stroke-width:3px
    classDef workerLayer fill:#e3f2fd,stroke:#2196F3,stroke-width:3px


    class Client client
    class Frontend frontend
    class Processor processor
    class VllmWorker worker
    class PrefillQueue prefillQueue
    class PrefillWorker prefillWorker
    class Planner planner
    class LocalKVCache storage
    class ETCD etcd
    class NATS nats
    class PS prefillBox
    class INF infraLayer
    class WL workerLayer



    %% Flow Colors - Different line styles to reduce visual clutter
    %% Main Request Flow - Blue (solid)
    linkStyle 0 stroke:#1565C0,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 1 stroke:#1565C0,stroke-width:4px
    linkStyle 2 stroke:#1565C0,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 3 stroke:#1565C0,stroke-width:4px
    linkStyle 4 stroke:#1565C0,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 5 stroke:#1565C0,stroke-width:4px

    %% Decision & Allocation Flow - Orange (mixed)
    linkStyle 6 stroke:#E65100,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 7 stroke:#E65100,stroke-width:4px
    linkStyle 8 stroke:#E65100,stroke-width:4px
    linkStyle 9 stroke:#E65100,stroke-width:3px,stroke-dasharray: 3 3

    %% KV Cache & Queue - Orange (solid)
    linkStyle 10 stroke:#E65100,stroke-width:4px
    linkStyle 11 stroke:#E65100,stroke-width:4px
    linkStyle 12 stroke:#E65100,stroke-width:4px

    %% Prefill Worker Flow - Green (mixed)
    linkStyle 13 stroke:#2E7D32,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 14 stroke:#2E7D32,stroke-width:4px
    linkStyle 15 stroke:#2E7D32,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 16 stroke:#2E7D32,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 17 stroke:#2E7D32,stroke-width:4px
    linkStyle 18 stroke:#2E7D32,stroke-width:4px
    linkStyle 19 stroke:#2E7D32,stroke-width:4px

    %% Completion Flow - Purple (mixed)
    linkStyle 20 stroke:#6A1B9A,stroke-width:4px
    linkStyle 21 stroke:#6A1B9A,stroke-width:3px,stroke-dasharray: 3 3
    linkStyle 22 stroke:#6A1B9A,stroke-width:4px
    linkStyle 23 stroke:#6A1B9A,stroke-width:4px
    linkStyle 24 stroke:#6A1B9A,stroke-width:4px

    %% Infrastructure Flows - Lighter and dotted to reduce visual noise
    %% ETCD Connections - Gray (dotted, thinner)
    linkStyle 25 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 26 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 27 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 28 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 29 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 30 stroke:#757575,stroke-width:2px,stroke-dasharray: 8 8

    %% NATS Connections - Teal (dotted, thinner)
    linkStyle 31 stroke:#26A69A,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 32 stroke:#26A69A,stroke-width:2px,stroke-dasharray: 8 8

    %% Planning Connections - Gold (dotted, thinner)
    linkStyle 33 stroke:#FFA726,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 34 stroke:#FFA726,stroke-width:2px,stroke-dasharray: 8 8
    linkStyle 35 stroke:#FFA726,stroke-width:2px,stroke-dasharray: 8 8
```
