# Dynamo Kubernetes Stack Architecture Diagrams

This document provides visual diagrams to help new users understand the Dynamo Kubernetes stack.

---

## 1. High-Level Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KUBERNETES CLUSTER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    USER DEPLOYMENTS                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │ │
│  │  │   Frontend   │  │   Workers    │  │   Planner    │                  │ │
│  │  │   Service    │  │  (Prefill/   │  │  (Optional)  │                  │ │
│  │  │              │  │   Decode)    │  │              │                  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                         │
│                                    │ Creates & Manages                       │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    DYNAMO OPERATOR                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐    │ │
│  │  │                    Controller Manager                           │    │ │
│  │  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐        │    │ │
│  │  │  │     DGD       │ │     DCD       │ │    Model      │        │    │ │
│  │  │  │  Controller   │ │  Controller   │ │  Controller   │        │    │ │
│  │  │  └───────────────┘ └───────────────┘ └───────────────┘        │    │ │
│  │  └────────────────────────────────────────────────────────────────┘    │ │
│  │  ┌────────────────────────────────────────────────────────────────┐    │ │
│  │  │                    Admission Webhooks                          │    │ │
│  │  └────────────────────────────────────────────────────────────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                    ▲                                         │
│                                    │ Uses                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                    INFRASTRUCTURE SERVICES                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐           │ │
│  │  │   etcd   │  │   NATS   │  │  Grove   │  │ KAI Scheduler│           │ │
│  │  │  (State) │  │ (Messaging│  │(Optional)│  │  (Optional)  │           │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘           │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Custom Resource Definitions (CRDs) Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │   DynamoGraphDeploymentRequest      │
                    │   (High-level SLA-based config)     │
                    └─────────────────┬───────────────────┘
                                      │ generates
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      DynamoGraphDeployment (DGD)    │
                    │   (Top-level deployment resource)   │
                    │                                     │
                    │  spec.services:                     │
                    │    - Frontend                       │
                    │    - VllmDecodeWorker              │
                    │    - VllmPrefillWorker             │
                    └─────────────────┬───────────────────┘
                                      │ creates
           ┌──────────────────────────┼──────────────────────────┐
           │                          │                          │
           ▼                          ▼                          ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  DynamoComponent    │  │  DynamoComponent    │  │  DynamoComponent    │
│  Deployment (DCD)   │  │  Deployment (DCD)   │  │  Deployment (DCD)   │
│  [Frontend]         │  │  [DecodeWorker]     │  │  [PrefillWorker]    │
└─────────┬───────────┘  └─────────┬───────────┘  └─────────┬───────────┘
          │                        │                        │
          │ creates                │ creates                │ creates
          ▼                        ▼                        ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  K8s Deployment     │  │  Grove PodClique    │  │  Grove PodClique    │
│  or PodClique       │  │  or Deployment      │  │  or Deployment      │
│  + Service          │  │  + Service          │  │  + Service          │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘


        ┌─────────────────────────────────────┐
        │  DynamoGraphDeploymentScalingAdapter│
        │  (DGDSA)                            │
        │  - Manages replica counts           │
        │  - Integrates with HPA/KEDA/Planner │
        └─────────────────────────────────────┘

        ┌─────────────────────────────────────┐
        │           DynamoModel               │
        │  - Model lifecycle management       │
        │  - LoRA adapter support             │
        └─────────────────────────────────────┘
```

---

## 3. Controller Reconciliation Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         RECONCILIATION LOOP                                   │
└──────────────────────────────────────────────────────────────────────────────┘

     User applies DGD YAML
              │
              ▼
┌─────────────────────────────┐
│   Kubernetes API Server     │
│   (validates via webhooks)  │
└─────────────┬───────────────┘
              │ watch event
              ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────────┐
│  DynamoGraphDeployment      │     │  Step 1: Parse spec.services            │
│  Reconciler                 │────▶│  Step 2: Create DynamoComponentDeployment│
│                             │     │          for each service               │
└─────────────┬───────────────┘     │  Step 3: Update DGD status              │
              │                     └─────────────────────────────────────────┘
              │ creates DCDs
              ▼
┌─────────────────────────────┐     ┌─────────────────────────────────────────┐
│  DynamoComponentDeployment  │     │  Step 1: Determine deployment kind      │
│  Reconciler                 │────▶│          (Deployment vs PodClique)      │
│                             │     │  Step 2: Create K8s resources           │
└─────────────┬───────────────┘     │  Step 3: Create Services                │
              │                     │  Step 4: Create RBAC resources          │
              │ creates             │  Step 5: Update DCD status              │
              ▼                     └─────────────────────────────────────────┘
┌─────────────────────────────┐
│  Kubernetes Resources       │
│  - Deployment/PodClique     │
│  - Service (ClusterIP)      │
│  - ServiceAccount           │
│  - Ingress (optional)       │
└─────────────────────────────┘
```

---

## 4. Disaggregated Serving with KV-Routing

```
                              External Client Request
                                       │
                                       ▼
                           ┌───────────────────────┐
                           │       Frontend        │
                           │  (OpenAI-compatible   │
                           │       API)            │
                           │                       │
                           │  DYN_ROUTER_MODE=kv   │
                           └───────────┬───────────┘
                                       │
                                       ▼
                           ┌───────────────────────┐
                           │    KV-Cache Router    │
                           │                       │
                           │  Routes based on:     │
                           │  - Request type       │
                           │  - KV-cache location  │
                           │  - Worker load        │
                           └───────────┬───────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
                    ▼                                     ▼
        ┌───────────────────────┐           ┌───────────────────────┐
        │    Prefill Workers    │           │    Decode Workers     │
        │                       │           │                       │
        │  - Process prompts    │           │  - Generate tokens    │
        │  - Compute attention  │──────────▶│  - Use cached KV      │
        │  - Cache KV states    │  KV-cache │  - Stream responses   │
        │                       │  transfer │                       │
        │  Replicas: N          │           │  Replicas: M          │
        └───────────────────────┘           └───────────────────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │
                                       ▼
                              Generated Response


    ┌─────────────────────────────────────────────────────────────────────┐
    │                     KV-CACHE TRANSFER FLOW                          │
    │                                                                     │
    │   1. Client sends prompt to Frontend                                │
    │   2. Router selects Prefill Worker                                  │
    │   3. Prefill Worker computes attention, caches KV states            │
    │   4. KV states transferred to Decode Worker(s) via NATS/RDMA        │
    │   5. Decode Workers generate tokens using cached KV                 │
    │   6. Response streamed back through Frontend                        │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Infrastructure Components Detail

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE LAYER                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐     ┌─────────────────────────┐
│         etcd            │     │          NATS           │
│  (Distributed KV Store) │     │    (Message Broker)     │
├─────────────────────────┤     ├─────────────────────────┤
│                         │     │                         │
│  Purpose:               │     │  Purpose:               │
│  - Operator state       │     │  - Inter-component      │
│  - Service discovery    │     │    communication        │
│  - Configuration        │     │  - Pub/Sub messaging    │
│                         │     │  - KV-cache transfer    │
│  Port: 2379             │     │                         │
│                         │     │  Port: 4222 (client)    │
│  Persistence: 1Gi       │     │  JetStream: 10Gi        │
│                         │     │                         │
└─────────────────────────┘     └─────────────────────────┘

┌─────────────────────────┐     ┌─────────────────────────┐
│         Grove           │     │     KAI Scheduler       │
│  (Multi-node Orchestr.) │     │   (Advanced Scheduling) │
├─────────────────────────┤     ├─────────────────────────┤
│                         │     │                         │
│  CRDs:                  │     │  Purpose:               │
│  - PodClique            │     │  - Resource quotas      │
│  - PodCliqueSet         │     │  - Queue management     │
│  - PodCliqueScalingGroup│     │  - Intelligent          │
│  - PodGang              │     │    allocation           │
│                         │     │                         │
│  Features:              │     │  Integration:           │
│  - Gang scheduling      │     │  - Run.AI support       │
│  - Topology-aware       │     │  - Custom metrics       │
│  - Startup ordering     │     │                         │
│                         │     │                         │
│  Enable: grove.enabled  │     │  Enable:                │
│          = true         │     │  kai-scheduler.enabled  │
│                         │     │          = true         │
└─────────────────────────┘     └─────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                        SERVICE DISCOVERY OPTIONS                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Option 1: Kubernetes-native (Default)                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Components discover each other via K8s Service DNS                  │   │
│   │  e.g., vllmdecodeworker.namespace.svc.cluster.local                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   Option 2: etcd-based                                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  Components register with etcd for external service mesh support     │   │
│   │  Useful for hybrid cloud/edge deployments                            │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Deployment Architecture Patterns

### Pattern A: Aggregated Serving (Simple)

```
┌─────────────────────────────────────────────────────────────────┐
│                   AGGREGATED ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Client Requests                                             │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │    Frontend     │                                           │
│   │   (1 replica)   │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│   │     Worker      │  │     Worker      │  │     Worker      │ │
│   │  (Full Model)   │  │  (Full Model)   │  │  (Full Model)   │ │
│   │  Prefill+Decode │  │  Prefill+Decode │  │  Prefill+Decode │ │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│                                                                  │
│   Use Case: Simple deployments, single-GPU models                │
│   Scaling: Add more workers for higher throughput                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern B: Disaggregated with KV-Routing

```
┌─────────────────────────────────────────────────────────────────┐
│               DISAGGREGATED + KV-ROUTING ARCHITECTURE           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Client Requests                                             │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────────────────────────────┐                   │
│   │        Frontend + KV Router             │                   │
│   │        DYN_ROUTER_MODE=kv               │                   │
│   └────────────────────┬────────────────────┘                   │
│                        │                                         │
│         ┌──────────────┴──────────────┐                         │
│         │                             │                          │
│         ▼                             ▼                          │
│   ┌───────────────┐           ┌───────────────┐                 │
│   │Prefill Workers│           │Decode Workers │                 │
│   │  (1-2 GPU)    │──────────▶│  (2-4 GPU)    │                 │
│   │               │  KV-cache │               │                 │
│   └───────────────┘           └───────────────┘                 │
│                                                                  │
│   Use Case: Large models, high throughput requirements           │
│   Benefits: Specialized resources, KV-cache reuse                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern C: Disaggregated with Planner (SLA-driven)

```
┌─────────────────────────────────────────────────────────────────┐
│            DISAGGREGATED + PLANNER ARCHITECTURE                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Client Requests                                             │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐       ┌─────────────────────────────────┐ │
│   │    Frontend     │       │           Planner               │ │
│   │                 │◀─────▶│                                 │ │
│   └────────┬────────┘       │  - Monitors latency/QPS         │ │
│            │                │  - Scales workers to meet SLA    │ │
│            │                │  - Optimizes resource usage      │ │
│         ┌──┴──┐             │                                 │ │
│         │     │             └────────────┬────────────────────┘ │
│         ▼     ▼                          │ controls replicas    │
│   ┌─────────┐ ┌─────────┐                │                      │
│   │ Prefill │ │ Decode  │◀───────────────┘                      │
│   │ Workers │ │ Workers │                                        │
│   │ (auto)  │ │ (auto)  │                                        │
│   └─────────┘ └─────────┘                                        │
│                                                                  │
│   Use Case: Dynamic workloads with strict SLA requirements       │
│   Benefits: Automatic scaling based on latency targets           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Pattern D: Multi-node (Tensor Parallelism)

```
┌─────────────────────────────────────────────────────────────────┐
│                  MULTI-NODE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│      Client Requests                                             │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │    Frontend     │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              PodCliqueScalingGroup                       │   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              Worker Replica 0                    │   │   │
│   │   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐    │   │   │
│   │   │  │ GPU 0 │──│ GPU 1 │──│ GPU 2 │──│ GPU 3 │    │   │   │
│   │   │  │ Rank 0│  │ Rank 1│  │ Rank 2│  │ Rank 3│    │   │   │
│   │   │  └───────┘  └───────┘  └───────┘  └───────┘    │   │   │
│   │   │         Tensor Parallel Sharding                 │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   │   ┌─────────────────────────────────────────────────┐   │   │
│   │   │              Worker Replica 1                    │   │   │
│   │   │  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐    │   │   │
│   │   │  │ GPU 0 │──│ GPU 1 │──│ GPU 2 │──│ GPU 3 │    │   │   │
│   │   └─────────────────────────────────────────────────┘   │   │
│   │                                                          │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   Use Case: Very large models (70B+) requiring multiple GPUs     │
│   Grove: Handles gang scheduling and network topology            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         END-TO-END REQUEST FLOW                              │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────┐
 │  Client  │
 └────┬─────┘
      │ 1. HTTP POST /v1/chat/completions
      ▼
 ┌──────────────────────────────────────────────────────────┐
 │                    Kubernetes Ingress                    │
 │                    (or LoadBalancer)                     │
 └────────────────────────────┬─────────────────────────────┘
                              │
      2. Route to Frontend    │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │                       Frontend Pod                        │
 │  ┌────────────────────────────────────────────────────┐  │
 │  │              OpenAI-Compatible API                  │  │
 │  │                                                     │  │
 │  │  - Parse request                                    │  │
 │  │  - Validate input                                   │  │
 │  │  - Apply routing logic                              │  │
 │  └──────────────────────────┬──────────────────────────┘  │
 └─────────────────────────────┼──────────────────────────────┘
                               │
      3. Service Discovery     │  (K8s DNS or etcd)
         & Load Balancing      │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │                    Prefill Worker Pod                     │
 │  ┌────────────────────────────────────────────────────┐  │
 │  │                  vLLM/SGLang/TRT-LLM                │  │
 │  │                                                     │  │
 │  │  4. Process prompt                                  │  │
 │  │  5. Compute attention (prefill phase)              │  │
 │  │  6. Generate KV-cache states                       │  │
 │  │  7. Send KV-cache to Decode Worker                 │  │
 │  └──────────────────────────┬──────────────────────────┘  │
 └─────────────────────────────┼──────────────────────────────┘
                               │
      8. KV-cache transfer     │  (via NATS or RDMA)
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │                    Decode Worker Pod                      │
 │  ┌────────────────────────────────────────────────────┐  │
 │  │                  vLLM/SGLang/TRT-LLM                │  │
 │  │                                                     │  │
 │  │  9.  Receive KV-cache                              │  │
 │  │  10. Generate tokens (decode phase)                │  │
 │  │  11. Stream tokens back                            │  │
 │  └──────────────────────────┬──────────────────────────┘  │
 └─────────────────────────────┼──────────────────────────────┘
                               │
      12. Stream response      │
                               ▼
 ┌──────────────────────────────────────────────────────────┐
 │                       Frontend Pod                        │
 │                                                           │
 │  13. Aggregate response                                   │
 │  14. Stream SSE to client                                │
 └─────────────────────────────┬─────────────────────────────┘
                               │
      15. SSE tokens stream    │
                               ▼
                         ┌──────────┐
                         │  Client  │
                         └──────────┘
```

---

## 8. Helm Chart Structure

```
dynamo-platform/
├── Chart.yaml                    # Main chart metadata
├── values.yaml                   # Configuration values
│
├── templates/
│   ├── _helpers.tpl             # Template helpers
│   └── ...                       # Platform resources
│
└── components/
    └── operator/                 # Operator subchart
        ├── Chart.yaml
        ├── values.yaml
        └── templates/
            ├── deployment.yaml           # Operator deployment
            ├── manager-rbac.yaml         # Cluster-wide RBAC
            ├── leader-election-rbac.yaml # HA leader election
            ├── webhook-certificates.yaml # TLS certs
            ├── webhook-configuration.yaml# Validation webhooks
            ├── prometheus.yaml           # Metrics (optional)
            └── planner.yaml              # Planner config

dynamo-crds/
├── Chart.yaml
└── templates/
    ├── nvidia.com_dynamographdeployments.yaml
    ├── nvidia.com_dynamocomponentdeployments.yaml
    ├── nvidia.com_dynamographdeploymentrequests.yaml
    ├── nvidia.com_dynamographdeploymentscalingadapters.yaml
    ├── nvidia.com_dynamomodels.yaml
    └── nvidia.com_dynamoworkermetadatas.yaml
```

---

## 9. Scaling Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SCALING OPTIONS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Option 1: Manual Scaling                                                    │
│  ────────────────────────                                                    │
│                                                                              │
│    spec.services.MyService.replicas: 5                                       │
│                                                                              │
│    - Direct replica count in DGD spec                                        │
│    - User manually updates YAML to scale                                     │
│    - Simple but not dynamic                                                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Option 2: HPA/KEDA via ScalingAdapter                                       │
│  ─────────────────────────────────────                                       │
│                                                                              │
│    spec.services.MyService.scalingAdapter.enabled: true                      │
│                                                                              │
│                                                                              │
│    ┌────────────┐        ┌─────────────┐        ┌───────────────┐           │
│    │    HPA     │───────▶│   DGDSA     │───────▶│ Deployment/   │           │
│    │ or KEDA    │ scale  │             │updates │ PodClique     │           │
│    │            │ target │             │replicas│               │           │
│    └────────────┘        └─────────────┘        └───────────────┘           │
│                                                                              │
│    - DGDSA exposes /scale subresource                                        │
│    - HPA/KEDA controls replica count                                         │
│    - DGD spec.replicas ignored when adapter enabled                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  Option 3: Planner-driven Scaling                                            │
│  ───────────────────────────────                                             │
│                                                                              │
│    ┌─────────────────────────────────────────────────────────────┐          │
│    │                        Planner                               │          │
│    │                                                              │          │
│    │   Inputs:                    Outputs:                        │          │
│    │   - Latency metrics          - Optimal prefill replicas      │          │
│    │   - QPS metrics              - Optimal decode replicas       │          │
│    │   - SLA targets              - Resource recommendations      │          │
│    │   - Cost constraints                                         │          │
│    │                                                              │          │
│    └──────────────────────────────┬──────────────────────────────┘          │
│                                   │                                          │
│                                   ▼                                          │
│    ┌──────────────────────────────────────────────────────────────┐         │
│    │                       DGD Controller                         │         │
│    │                                                              │         │
│    │   Updates spec.services.*.replicas based on Planner output  │         │
│    └──────────────────────────────────────────────────────────────┘         │
│                                                                              │
│    - SLA-aware autoscaling                                                   │
│    - Considers model-specific characteristics                                │
│    - Optimizes cost vs performance                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Quick Reference: Key Environment Variables

```
┌────────────────────────────┬────────────────────────────────────────────────┐
│      Environment Var       │                 Description                     │
├────────────────────────────┼────────────────────────────────────────────────┤
│ DYN_ROUTER_MODE            │ Routing mode: "kv" for KV-routing              │
│ DYNAMO_NAMESPACE           │ Logical namespace for service discovery        │
│ ETCD_ENDPOINTS             │ etcd connection string                         │
│ NATS_URL                   │ NATS connection URL                            │
│ HF_TOKEN                   │ Hugging Face token for model downloads         │
│ PROMETHEUS_ENDPOINT        │ Prometheus metrics endpoint                    │
│ KAI_SCHEDULER_QUEUE        │ KAI scheduler queue name                       │
│ DYNAMO_SYSTEM_PORT         │ System port for internal communication (9090)  │
└────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 11. Troubleshooting Quick Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMMON ISSUES & SOLUTIONS                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┬───────────────────────────────────────────────┐
│ Issue                       │ Solution                                       │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ Pods not starting           │ kubectl describe pod <pod-name>               │
│                             │ Check GPU availability: kubectl get nodes     │
│                             │ -o jsonpath='{.items[*].status.allocatable}'  │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ Workers not discovered      │ kubectl get svc -n <namespace>                │
│                             │ Check DYNAMO_NAMESPACE matches                │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ KV-routing not working      │ Verify DYN_ROUTER_MODE=kv in Frontend         │
│                             │ Check NATS connectivity                       │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ DGD stuck in pending        │ kubectl describe dgd <name>                   │
│                             │ Check operator logs                           │
├─────────────────────────────┼───────────────────────────────────────────────┤
│ Grove pods not scheduling   │ Verify Grove CRDs installed                   │
│                             │ kubectl get crd | grep grove                  │
└─────────────────────────────┴───────────────────────────────────────────────┘
```

---

## Summary

The Dynamo Kubernetes stack consists of:

1. **CRDs**: DynamoGraphDeployment (top-level), DynamoComponentDeployment (per-service), and supporting resources
2. **Operator**: Watches CRDs and creates/manages underlying K8s resources
3. **Infrastructure**: etcd (state), NATS (messaging), Grove (multi-node), KAI (scheduling)
4. **Patterns**: Aggregated, Disaggregated with KV-routing, Planner-driven, Multi-node

The system enables deploying complex LLM inference architectures with a single YAML file while handling service discovery, scaling, and orchestration automatically.
