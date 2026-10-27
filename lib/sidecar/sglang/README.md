<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Note to AI agents: keep this README minimal (intro, support matrix, launch example,
Kubernetes example, topologies). Do not edit it unless the user explicitly asks
you to.
-->

# SGLang sidecar

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-sglang-sidecar` connects a Dynamo worker to SGLang's Rust native gRPC
server. See the [sidecar overview](../README.md) for installation.

> [!TIP]
> For the best and latest support, use the upstream SGLang nightly image,
> which carries the latest gRPC server updates: [`lmsysorg/sglang:dev`](https://hub.docker.com/r/lmsysorg/sglang/tags?name=dev).

## Support matrix

| Feature | Supported |
|---------|-----------|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |
| Data parallelism (DP>1) | Not yet |

## Run locally

See [`launch/`](launch/) for all topologies. For example, aggregated serving on
one GPU:

```bash
export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
lib/sidecar/sglang/launch/agg.sh
```

The scripts start SGLang with `SGLANG_RUST_SERVER=1`.

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

Before applying, replace `<your-registry>/dynamo-sidecar` in the manifest with a
[sidecar image](../README.md#docker). The `hf-token-secret` references are
optional; create the Secret only for gated or private models.

See [`deploy/`](deploy/) for all manifests. For example, aggregated serving:

```bash
# Optional for gated or private models:
kubectl create secret generic hf-token-secret -n <namespace> \
  --from-literal=HF_TOKEN=<your-hf-token>
kubectl apply -f lib/sidecar/sglang/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/sglang-sidecar-agg-frontend 8000:8000
```

### Kubernetes manifest notes

The examples use an engine-primary layout: SGLang runs as the `main` container
and `dynamo-sglang-sidecar` runs as a restartable `runtime` init container.
Those reserved names activate operator sidecar behavior. SGLang native gRPC
requires SGLang's Rust server (`SGLANG_RUST_SERVER=1`) from a build with
[sgl-project/sglang#45859](https://github.com/sgl-project/sglang/pull/45859) and
[sgl-project/sglang#45883](https://github.com/sgl-project/sglang/pull/45883).
SGLang can alternatively load the sidecar in-process with
`--sidecar dynamo.sglang.sidecar`, which under the Rust server needs
[sgl-project/sglang#46635](https://github.com/sgl-project/sglang/pull/46635).

Aggregated serving uses loopback between the engine and sidecar. Disaggregated
serving binds SGLang to the pod network so decode workers can reach prefill's
bootstrap server, which the Rust server serves on its HTTP port; the prefill
sidecar auto-detects and advertises its pod IP. For KV routing, the sidecar
subscribes to the ZMQ endpoints SGLang reports in its discovery metadata.
Restrict pod-network gRPC, HTTP, and ZMQ ports with NetworkPolicy.

Disaggregated serving requires an SGLang image with NIXL/UCX support, RDMA
devices, `IPC_LOCK`, and shared memory. Tune UCX for the deployment fabric.
Keep the model, served model name, and tensor parallelism consistent across
prefill and decode; KV routing also requires a consistent page size.

## Topologies

The frontend reaches each sidecar over Dynamo's request, discovery, and event
planes; the sidecar reaches the engine over its native gRPC API. Dashed arrows
carry KV events.

### Single-Node TP

One engine on one node, with one sidecar.

![On one node, a request reaches the SGLang tensor-parallel ranks through the Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar.](../../../docs/fern/assets/img/sidecar-sglang-single-node-tp.svg)

### Multi-Node TP

One engine spans two nodes. Only the leader node has a sidecar; the follower
node holds the remaining TP ranks.

![When one SGLang engine spans two nodes with tensor parallelism, only the leader node runs a Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar on Node 0.](../../../docs/fern/assets/img/sidecar-sglang-multinode-tp.svg)

### Data parallelism

Not supported yet; see the [support matrix](#support-matrix).
