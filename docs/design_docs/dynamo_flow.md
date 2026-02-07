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

This diagram shows the NVIDIA Dynamo disaggregated inference system. Color-coded flows indicate different types of operations.

## 🔵 Main Request Flow (Blue)
The primary user journey through the system:

1. **Request (S1)**: HTTP client sends API request to Frontend (OpenAI-compatible server on port 8000)
2. **Preprocess (S2)**: Frontend preprocesses the request (applies chat template, tokenizes) and validates it
3. **Route to Prefill (S3)**: PrefillRouter selects a prefill worker using KV-aware routing or load balancing

## 🟢 Prefill Flow (Green)
The prefill processing pipeline:

4. **Prefill (S4)**: Prefill worker executes the prefill computation on the input tokens and generates KV cache
5. **Return Metadata (S5)**: Prefill worker returns `disaggregated_params` containing backend-specific transfer metadata

## 🟠 Decode Routing Flow (Orange)
Router orchestration to decode phase:

6. **Route to Decode (S6)**: PrefillRouter injects prefill result into decode request and routes to decode worker
7. **KV Transfer (S7)**: Decode worker coordinates with prefill worker for direct GPU-to-GPU KV cache transfer via NIXL

## 🟣 Completion Flow (Purple)
The response generation and delivery:

8. **Decode (S8)**: Decode worker generates tokens using the transferred KV cache
9. **Response (S9)**: Generated tokens stream back through Frontend for post-processing (detokenization) and delivery to Client

## 🔗 Infrastructure Connections (Dotted lines)
Coordination and messaging support:

### Service Discovery
- **On Kubernetes** (default): Uses native K8s resources (DynamoWorkerMetadata CRD, EndpointSlices). No etcd required.
- **On bare metal**: Uses etcd or filesystem for service discovery and endpoint registration.

### Request Plane
- **TCP** (default): Direct TCP connections between Frontend and Workers for request/response transport.
- **HTTP/NATS**: Alternative transports configurable via `DYN_REQUEST_PLANE`.

### NATS Connections (Optional, for KV routing)
- **KV Events**: Cache state events for KV-aware routing (can be disabled with `--no-kv-events`)

### Planning Connections (Gold, dotted)
- **Frontend → Planner**: Metrics collection for auto-scaling decisions
- **Planner → Workers**: Resource scaling commands for workers

## Technical Implementation Details

### PrefillRouter Orchestration:
- The `PrefillRouter` sits between the Frontend and workers, orchestrating disaggregated serving
- Selects prefill workers using KV-aware routing (cache overlap scores + load) or simple load balancing
- Injects transfer metadata into decode requests for KV cache coordination

### NIXL (NVIDIA Interchange Library):
- Enables high-speed GPU-to-GPU data transfers using NVLink, InfiniBand/UCX, or PCIe
- Transfer metadata exchanged via `disaggregated_params` in prefill response
- Backend-specific coordination: SGLang uses bootstrap connections, vLLM uses block IDs, TRTLLM uses opaque state

### Disaggregated KV Cache:
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
