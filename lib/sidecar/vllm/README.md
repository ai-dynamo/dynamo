<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (intro, support matrix, launch example,
Kubernetes example, topologies). Do not edit it unless the user explicitly asks
you to.
-->

# vLLM sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC server
(`vllm-rs`). See the [sidecar overview](../README.md) for installation.

## Support matrix

| Feature | Supported |
|---------|-----------|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |

## Run locally

See [`launch/`](launch/) for all topologies. For example, aggregated serving on
one GPU:

```bash
export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
lib/sidecar/vllm/launch/agg.sh
```

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`deploy/`](deploy/) for all manifests. For example, aggregated serving:

```bash
kubectl apply -f lib/sidecar/vllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/vllm-sidecar-agg-frontend 8000:8000
```

## Topologies

Solid arrows carry inference requests; dotted arrows carry KV cache events.

### Single-node TP

One engine on one node, with one sidecar.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph P[Worker pod, node 0]
    S[Dynamo sidecar] -->|Native gRPC| E[vLLM: TP ranks]
    E -.->|KV events| S
  end
  F -->|Requests| S
  S -.->|KV events| F
```

### Multi-node TP

One engine spans two nodes. Only the leader pod has a sidecar; the follower
pod holds the remaining TP ranks.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph L[Leader pod, node 0]
    S[Dynamo sidecar] -->|Native gRPC| E0[vLLM: local TP ranks]
    E0 -.->|KV events| S
  end
  subgraph W[Follower pod, node 1]
    E1[vLLM: remote TP ranks]
  end
  F -->|Requests| S
  S -.->|KV events| F
  E0 <-->|TP collectives| E1
```

### Multi-node DP

Hybrid DP load balancing: each node runs vLLM for its local DP ranks plus a
sidecar that serves requests. The frontend routes to either pod.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph A[Worker pod A, node 0]
    SA[Dynamo sidecar: DP 0-1] -->|Native gRPC| EA[vLLM]
    EA -.->|KV events| SA
  end
  subgraph B[Worker pod B, node 1]
    SB[Dynamo sidecar: DP 2-3] -->|Native gRPC| EB[vLLM]
    EB -.->|KV events| SB
  end
  C{{DP/EP coordination}}
  F -->|Requests| SA
  F -->|Requests| SB
  SA -.->|KV events| F
  SB -.->|KV events| F
  EA <--> C
  EB <--> C
```
