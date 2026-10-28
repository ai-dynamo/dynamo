---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Note to AI agents: keep this page minimal (intro, support matrix, launch example,
# Kubernetes example, topologies). Do not edit it unless the user explicitly asks.
title: vLLM Sidecar
subtitle: Run Dynamo beside a stock vLLM engine through native gRPC.
---

> [!WARNING]
> **Experimental.** The sidecars and their deployment examples are
> experimental. Manifests, flags, and behavior may change without notice.

`dynamo-vllm-sidecar` connects a Dynamo worker to vLLM's native gRPC server
(`vllm-rs`). See [Sidecar Backends](../../../concepts/system-architecture/sidecar-backends.md)
for the architecture.

> [!TIP]
> For the best and latest support, use the upstream vLLM nightly image,
> which carries the latest gRPC server updates: [`vllm/vllm-openai:nightly`](https://hub.docker.com/r/vllm/vllm-openai/tags?name=nightly).

## Support Matrix

| Feature | Supported |
|---|---|
| Aggregated | Yes |
| Disaggregated | Yes |
| KV routing | Yes |
| Kubernetes multinode TP and hybrid-LB DP | Experimental; explicit leader and worker PodTemplates |

## Launch Locally

See [`lib/sidecar/vllm/launch/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/vllm/launch)
for all topologies. For example, aggregated serving on one GPU:

```bash
export DYN_DISCOVERY_BACKEND=file   # single host: no etcd or NATS needed
./lib/sidecar/vllm/launch/agg.sh
```

In a second terminal:

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

## Deploy on Kubernetes

See [`lib/sidecar/vllm/deploy/`](https://github.com/ai-dynamo/dynamo/tree/main/lib/sidecar/vllm/deploy)
for all manifests. For example, aggregated serving:

Before applying, replace `<your-registry>/dynamo-sidecar` in the manifest with a
[sidecar image](../../../concepts/system-architecture/sidecar-backends.md#container-packaging).

```bash
kubectl apply -f lib/sidecar/vllm/deploy/agg.yaml -n <namespace>
kubectl port-forward -n <namespace> svc/vllm-sidecar-agg-frontend 8000:8000
```

Multinode deployments require `backendFramework: vllm` and complete `leader` and
`worker` role PodTemplates. The templates own the native engine commands and
sidecar placement; the operator does not inject executor or coordination flags.
See the [TP example](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/multinode_tp.yaml),
which uses the multiprocessing executor with a sidecar on the leader only, and the
[hybrid-LB DP example](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/vllm/deploy/multinode_dp.yaml),
where every Pod runs a sidecar for its local data-parallel ranks. Both examples
launch `vllm serve --grpc-port` with `VLLM_USE_RUST_FRONTEND=1`, which requires
vLLM v0.32.0 or later; `vllm-rs serve` does not support `--data-parallel-hybrid-lb`.
Engine-failure KV recovery remains incomplete and is tracked in
[DEP #14897](https://github.com/ai-dynamo/dynamo/issues/14897).

### Migrate an existing DGD to multinode sidecars

The multinode examples use `DYNAMO_RANK` and `DYNAMO_LEADER_ADDRESS`. The operator
injects these variables only into DGDs originally created by operator 1.6.0 or
later. Upgrading the operator or switching to sidecar mode does not enable them
for older DGDs.

For a DGD created before operator 1.6.0, choose one of these options:

- Delete and recreate the DGD after upgrading the operator, then use the examples
  with their existing topology arguments.
- Keep the DGD and replace the topology variables in the `main` container of both
  the leader and worker role templates with the provider-specific values below.
  The TP example uses them in `--node-rank` and `--master-addr`; the DP example
  uses them in `--data-parallel-start-rank` and `--data-parallel-address`.

For Grove, use these values. The group-wide pod index requires Grove
`v0.1.0-alpha.15` or later and matching CRDs.

```yaml
- --node-rank
- $(GROVE_PCSG_POD_INDEX)
- --master-addr
- $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-vllmworker-ldr-0.$(GROVE_HEADLESS_SERVICE)
```

Here, `vllmworker` is the example component name, `VllmWorker`, in lowercase.
Replace it with your component name in lowercase if you rename the component.

For LeaderWorkerSet, use these values:

```yaml
- --node-rank
- $(LWS_WORKER_INDEX)
- --master-addr
- $(LWS_LEADER_ADDRESS)
```

Kubernetes expands these variable references in container arguments. The DP
worker computes its start rank in a shell, so replace `DYNAMO_RANK` there with
`GROVE_PCSG_POD_INDEX` or `LWS_WORKER_INDEX`.

## Topologies

The frontend reaches each sidecar over Dynamo's request, discovery, and event
planes; the sidecar reaches the engine over its native gRPC API. Dashed arrows
carry KV events.

### Single-Node TP

One engine on one node, with one sidecar.

![On one node, a request reaches the vLLM tensor-parallel ranks through the Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar.](../../../../../../assets/img/sidecar-vllm-single-node-tp.svg)

### Multi-Node TP

One engine spans two nodes. Only the leader node has a sidecar; the follower
node holds the remaining TP ranks.

![When one vLLM engine spans two nodes with tensor parallelism, only the leader node runs a Dynamo Sidecar. The Dynamo Frontend sends requests over the request plane to the sidecar on Node 0.](../../../../../../assets/img/sidecar-vllm-multinode-tp.svg)

### Multi-Node DP

Hybrid DP load balancing: each node runs vLLM for its local DP ranks plus a
sidecar that serves requests. The frontend routes to either node.

![vLLM hybrid data parallelism across two nodes. The Dynamo Frontend router picks a DP rank and sends requests over the request plane to the Dynamo Sidecar on the node that owns that rank: node 0 serves DP ranks 0-1 and node 1 serves DP ranks 2-3.](../../../../../../assets/img/sidecar-vllm-multinode-dp.svg)
