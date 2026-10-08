<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (intro, support matrix, launch example,
Kubernetes example, topologies). Do not edit it unless the user explicitly asks
you to.
-->

# TensorRT-LLM sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-trtllm-sidecar` connects a Dynamo worker to TensorRT-LLM's OpenEngine
gRPC server. See the [sidecar overview](../README.md) for installation.

## Support matrix

| Feature | Supported |
|---------|-----------|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | No |

## Run locally

See [`launch/`](launch/) for all topologies. For example, aggregated serving on
one GPU:

```bash
docker compose -f dev/docker-compose.yml up -d
lib/sidecar/trtllm/launch/agg.sh
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`deploy/`](deploy/) for all manifests. For example, aggregated serving:

```bash
kubectl apply -f lib/sidecar/trtllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/trtllm-sidecar-agg-frontend 8000:8000
```

## Topologies

Arrows carry inference requests. The TensorRT-LLM sidecar does not publish KV
cache events.

### Single-node TP

One engine on one node, with one sidecar.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph P[Worker pod, node 0]
    S[Dynamo sidecar] -->|Native gRPC| E[TensorRT-LLM: TP ranks]
  end
  F -->|Requests| S
```

### Multi-node TP

One engine spans two nodes. Only the leader pod has a sidecar; the follower
pod holds the remaining TP ranks.

> [!NOTE]
> This topology has not been validated with the TensorRT-LLM sidecar yet.

```mermaid
flowchart LR
  F[Dynamo frontend]
  subgraph L[Leader pod, node 0]
    S[Dynamo sidecar] -->|Native gRPC| E0[TensorRT-LLM: local TP ranks]
  end
  subgraph W[Follower pod, node 1]
    E1[TensorRT-LLM: remote TP ranks]
  end
  F -->|Requests| S
  E0 <-->|TP collectives| E1
```

### Multi-node DP

Not supported yet. The TensorRT-LLM sidecar does not target DP ranks or
publish KV events.
