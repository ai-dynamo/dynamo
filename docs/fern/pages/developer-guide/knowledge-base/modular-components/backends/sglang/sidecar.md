---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Note to AI agents: keep this page minimal (intro, support matrix, launch example,
# Kubernetes example, topologies). Do not edit it unless the user explicitly asks.
title: SGLang Sidecar
subtitle: Run Dynamo beside a stock SGLang engine through native gRPC.
---

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-sglang-sidecar` connects a Dynamo worker to SGLang's native gRPC
server. See [Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md)
for the architecture.

## Support Matrix

| Feature | Supported |
|---|---|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |

## Launch Locally

See [`lib/sidecar/sglang/launch/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/sglang/launch)
for all topologies. For example, aggregated serving on one GPU:

```bash
cargo build --release -p dynamo-sglang-sidecar
export PATH="$PWD/target/release:$PATH" DYN_DISCOVERY_BACKEND=file
./lib/sidecar/sglang/launch/agg.sh
```

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`lib/sidecar/sglang/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/sglang/deploy)
for all manifests. For example, aggregated serving:

```bash
kubectl apply -f lib/sidecar/sglang/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/sglang-sidecar-agg-frontend 8000:8000
```

## Topologies

Solid arrows carry inference requests; dotted arrows carry KV cache events.

### Single-node TP

One engine on one node, with one sidecar.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph P[Worker pod, node 0]
    S[Dynamo sidecar] -->|Native gRPC| E[SGLang: TP ranks]
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
    S[Dynamo sidecar] -->|Native gRPC| E0[SGLang: local TP ranks]
    E0 -.->|KV events| S
  end
  subgraph W[Follower pod, node 1]
    E1[SGLang: remote TP ranks]
  end
  F -->|Requests| S
  S -.->|KV events| F
  E0 <-->|TP collectives| E1
```

### Multi-node DP

The leader sidecar registers every DP rank and serves all requests; SGLang
dispatches each one to the right scheduler. The follower sidecar accepts no
requests and relays its node's KV events directly to the frontend.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph L[Leader pod, node 0]
    SL[Dynamo sidecar: DP 0-1] -->|Native gRPC| EL[SGLang]
    EL -.->|KV events| SL
  end
  subgraph W[Follower pod, node 1]
    SW[Dynamo sidecar: KV events only, DP 2-3] ~~~ EW[SGLang]
    EW -.->|KV events| SW
  end
  C{{Dispatch + collectives}}
  F -->|Requests| SL
  F ~~~ SW
  SL -.->|KV events| F
  SW -.->|KV events| F
  EL <--> C
  EW <--> C
```
