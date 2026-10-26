---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Note to AI agents: keep this page minimal (intro, support matrix, launch example,
# Kubernetes example, topologies). Do not edit it unless the user explicitly asks.
title: TensorRT-LLM Sidecar
subtitle: Run Dynamo beside a TensorRT-LLM engine through its OpenEngine gRPC API.
---

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-trtllm-sidecar` connects a Dynamo worker to TensorRT-LLM's OpenEngine
gRPC server. See [Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md)
for the architecture.

> [!TIP]
> For the best and latest support, use the upstream TensorRT-LLM nightly image,
> which carries the latest gRPC server updates. Nightly builds are the
> `nvcr.io/nvidia/tensorrt-llm/release:<version>.dev<date>` tags on
> [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release/-).
> KV routing is available in
> `nvcr.io/nvidia/tensorrt-llm/release:1.4.0a0.dev202610180820`, which includes
> native KVCM2 event streaming and OpenEngine routing load support.

## Support Matrix

| Feature | Supported |
|---|---|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |

## Launch Locally

See [`lib/sidecar/trtllm/launch/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/trtllm/launch)
for all topologies. For example, aggregated serving on one GPU:

```bash
export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
./lib/sidecar/trtllm/launch/agg.sh
```

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`lib/sidecar/trtllm/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/trtllm/deploy)
for all manifests. For example, aggregated serving:

Before applying, replace `<your-registry>/dynamo-sidecar` in the manifest with a
[sidecar image](../../../concepts/system-architecture/sidecar-backends.md#container-packaging) and create the `hf-token-secret`
Secret that the manifest reads.

```bash
kubectl create secret generic hf-token-secret -n <namespace> \
  --from-literal=HF_TOKEN=<your-hf-token>
kubectl apply -f lib/sidecar/trtllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/trtllm-sidecar-agg-frontend 8000:8000
```

## Topologies

The frontend sends inference requests to each engine's leader sidecar, which forwards them to OpenEngine over gRPC and polls routing load for all attention-DP ranks. Each sidecar discovers its node's KV event sources through OpenEngine and relays local KV events through Dynamo's event plane. Follower sidecars do not serve inference requests.

### Single-Node TP

One engine on one node, with one sidecar.

![On one node, a request reaches the TensorRT-LLM tensor-parallel ranks through the Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar.](../../../../../../assets/img/sidecar-trtllm-single-node-tp.svg)

### Multi-Node TP

One engine spans two nodes. Only the leader node has a sidecar; the follower
node holds the remaining TP ranks.

![When one TensorRT-LLM engine spans two nodes with tensor parallelism, only the leader node runs a Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar on Node 0.](../../../../../../assets/img/sidecar-trtllm-multinode-tp.svg)

### Multi-Node DP

The sidecar forwards the selected attention-DP rank to OpenEngine for KV-aware routing when the engine has more than one attention-DP rank. For single-rank engines, it omits the redundant rank hint and does not synthesize a conversation ID for anonymous requests. Explicit session IDs are preserved.

> [!WARNING]
> For multi-rank engines with conversation affinity enabled, OpenEngine currently requires a conversation ID with a rank hint. The sidecar supplies a request-scoped ID when the client provides no session. With KVCM2 `block_reuse_config.policy: per_conversation`, those one-shot IDs can retain conversation state until cache reset or engine restart, causing host memory to grow with anonymous request count. Avoid this combination for sustained anonymous traffic until TensorRT-LLM accepts rank hints without conversation IDs or bounds retained conversation state.

![TensorRT-LLM data parallelism across two nodes. Only the leader sidecar serves requests: the Dynamo Frontend router picks a DP rank and sends requests over the request plane to the node 0 Dynamo Sidecar, which registers DP ranks 0-3 and calls its local TensorRT-LLM over native gRPC.](../../../../../../assets/img/sidecar-trtllm-multinode-dp.svg)
