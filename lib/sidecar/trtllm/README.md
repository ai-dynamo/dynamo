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

> [!TIP]
> For the best and latest support, use the upstream TensorRT-LLM nightly image,
> which carries the latest gRPC server updates. Nightly builds are the
> `nvcr.io/nvidia/tensorrt-llm/release:<version>.dev<date>` tags on
> [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/tensorrt-llm/containers/release/-).

## Support matrix

| Feature | Supported |
|---------|-----------|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes, with a compatible OpenEngine server and KVCM2 |

For child-agent context affinity, enable frontend session affinity and
TensorRT-LLM conversation affinity, then set the same private
`TRTLLM_SUBAGENT_AFFINITY_AUTH_KEY` on the prefill sidecar and OpenEngine server.
The sidecar requires the server to advertise signed affinity support. It sends
the child's own conversation ID and a signed parent hint for context requests;
decode requests retain the child's identity. The gateway must authorize client
parent-session headers before they reach the router.

KV routing examples require a TensorRT-LLM build with native KVCM2 event
streaming and a Dynamo frontend and sidecar built from this PR. The pinned
private engine image in the Kubernetes examples requires `ghcr-pull-secret`.

## Run locally

See [`launch/`](launch/) for all topologies. For example, aggregated serving on
one GPU:

```bash
export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
lib/sidecar/trtllm/launch/agg.sh
```

For KV-aware routing, use `launch/agg_kv_router.sh` (two GPUs) or
`launch/disagg_kv_router.sh` (four GPUs). Both enable OpenEngine load metrics
and publish one ZMQ event source per worker.

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

Before applying, replace `<your-registry>/dynamo-sidecar` in the manifest with a
[sidecar image](../README.md#docker) and create the `hf-token-secret`
Secret that the manifest reads.

See [`deploy/`](deploy/) for all manifests. For example, aggregated serving:

```bash
kubectl create secret generic hf-token-secret -n <namespace> \
  --from-literal=HF_TOKEN=<your-hf-token>
kubectl apply -f lib/sidecar/trtllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/trtllm-sidecar-agg-frontend 8000:8000
```

Use `deploy/agg_kv_router.yaml` or `deploy/disagg_kv_router.yaml` for KV
routing. Build the Dynamo images from this PR and replace their image references
in the manifests before applying.

## Topologies

The frontend reaches each sidecar over Dynamo's request, discovery, and event
planes; the sidecar reaches the engine over its native gRPC API. The sidecar
discovers per-rank KV event sources and routing load from OpenEngine.

### Single-Node TP

One engine on one node, with one sidecar.

![On one node, a request reaches the TensorRT-LLM tensor-parallel ranks through the Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar.](../../../docs/fern/assets/img/sidecar-trtllm-single-node-tp.svg)

### Multi-Node TP

One engine spans two nodes. Only the leader node has a sidecar; the follower
node holds the remaining TP ranks.

![When one TensorRT-LLM engine spans two nodes with tensor parallelism, only the leader node runs a Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar on Node 0.](../../../docs/fern/assets/img/sidecar-trtllm-multinode-tp.svg)

### Multi-Node DP

The sidecar forwards the selected attention-DP rank to OpenEngine when the
server advertises rank targeting. KV routing requires a TensorRT-LLM build
with native KVCM2 event streaming and routing load support.

![TensorRT-LLM data parallelism across two nodes. Only the leader sidecar serves requests: the Dynamo Frontend router picks a DP rank and sends requests over the request plane to the node 0 Dynamo Sidecar, which registers DP ranks 0-3 and calls its local TensorRT-LLM over native gRPC.](../../../docs/fern/assets/img/sidecar-trtllm-multinode-dp.svg)
