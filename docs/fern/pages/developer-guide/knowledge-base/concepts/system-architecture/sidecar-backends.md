---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Sidecar Backends
subtitle: Run Dynamo beside a stock inference engine through its native gRPC API.
---

> [!WARNING]
> **Experimental.** Sidecar packaging, launchers, and API coverage are still
> evolving. The sidecar path does not yet match every feature of the in-process
> backends.

A Dynamo sidecar runs beside the inference engine process. It registers the
engine with Dynamo discovery, forwards engine events into the Dynamo event
plane, and serves requests from the Dynamo frontend by calling the engine's
native gRPC API.

## Design Goals

- Keep the upstream engine's native serve path and argument surface.
- Move toward public, versioned gRPC contracts with explicit backward
  compatibility instead of importing private engine APIs.
- Isolate Dynamo and engine dependencies in separate processes.
- Attribute failures through engine-specific and Dynamo-specific logs and health
  checks.
- Reuse Dynamo's frontend, routing, planning, and disaggregated-serving
  orchestration.

## Architecture

![Dynamo Sidecar architecture. A client sends OpenAI-compatible HTTP to the Dynamo Frontend, which tokenizes, routes, and sends token IDs over the request plane to the Dynamo Sidecar.](../../../../../assets/img/sidecar-architecture.svg)

The frontend and router discover sidecars and send requests to them over the
Dynamo request plane. Each sidecar converts requests to the engine's native gRPC
API and streams responses back.

## Responsibilities

| Layer | Responsibility |
|---|---|
| Dynamo frontend and router | OpenAI-compatible API, preprocessing, and routing to sidecars |
| Dynamo sidecar | Engine registration and discovery, request forwarding over native gRPC, and event forwarding |
| Inference engine | Native gRPC request serving, scheduling, sampling, token generation, KV cache, and GPU execution |

## Container Packaging

The sidecar Dockerfile builds all four engine-specific sidecar executables into
one CPU-only image. Deployments select an engine by setting the container
`command`:

| Engine | Container command |
|---|---|
| vLLM | `dynamo-vllm-sidecar` |
| SGLang | `dynamo-sglang-sidecar` |
| TensorRT-LLM | `dynamo-trtllm-sidecar` |
| LPX | `dynamo-lpx-sidecar` |

The image's default entrypoint, `dynamo-sidecar`, maps the short names `vllm`,
`sglang`, `trtllm`, and `lpx` onto those executables. It is a convenience for ad-hoc
`docker run`; the deployment manifests override it with `command`. The inference
engine remains in a separate GPU container, so the sidecar image does not
include vLLM, SGLang, TensorRT-LLM, CUDA, or engine-specific Python
dependencies. The image runs as the non-root `dynamo` user with numeric user ID
`1000` and declares port `9090` for Dynamo system endpoints, so Kubernetes can
enforce `runAsNonRoot`.

The sidecar image is published to NGC for each release, starting with 1.6.0:

```bash
docker pull nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version>  # 1.6.0 or later
```

To build it from source instead, run from the repository root with the
[sidecar Dockerfile](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/Dockerfile):

```bash
docker build -f lib/sidecar/Dockerfile -t dynamo-sidecar:1.6.0-dev .
```

## Pod Layout

> [!NOTE]
> Kubernetes sidecar mode is a work in progress.

The engine and sidecar share one worker pod. The engine is the `main`
container; the sidecar is a
[native sidecar](https://kubernetes.io/docs/concepts/workloads/pods/sidecar-containers/):
an init container named `runtime` with `restartPolicy: Always`, which requires
Kubernetes 1.29 or later. The two connect over loopback.

```yaml
podTemplate:
  spec:
    initContainers:
    - name: runtime
      image: nvcr.io/nvidia/ai-dynamo/dynamo-sidecar:<version>
      command: [dynamo-vllm-sidecar]
      args: [--grpc-endpoint, 127.0.0.1:30001]
      restartPolicy: Always
    containers:
    - name: main
      image: vllm/vllm-openai:<version>
```

Kubernetes starts the sidecar before the engine and keeps it running for the
life of the pod. Declaring the `runtime` init container enables sidecar mode in
the Dynamo operator, which gives the sidecar `/live` and `/health` probes on
port `9090`. These probes do not track engine loading, so keep the engine's own
probes on `main`. Sidecar mode supports worker, prefill, and decode components;
multinode deployments are not yet supported.

## LPX Partition Topology

An LPX deployment executes compiler partitions on Agent pods. The leader of each
Agent allocation, Grove pod index 0, is the only authority for where that
allocation's partitions execute. Its Agent serves `TopologyService` from the
[`lpx.topology.v1` protocol](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/lpx/proto/lpx/topology/v1/topology.proto)
on a pod-local Unix socket. The LPX sidecar relays that service through Dynamo
discovery to the conductor pods, so that each conductor reads a live answer from
the authority when it starts. A sidecar's topology role is independent of
inference serving:

| Role | Runs beside | Behavior |
|---|---|---|
| `publisher` | The Agent allocation's leader | Waits for the Agent's first valid snapshot, then registers the endpoint `lpx-topology/snapshot-<allocation scope>-<build scope>` with no model card and answers each request on it with a live answer from the Agent |
| `consumer` | Each conductor | Serves `TopologyService` on a Unix socket. Each request lists the `lpx-topology` endpoints, asks every authority registered for the identity the request expects at the same time, and validates the one `OK` answer |

| Flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--topology-role` | `DYN_LPX_TOPOLOGY_ROLE` | Unset | `publisher` or `consumer`. Unset: no topology work |
| `--topology-agent-socket` | `DYN_LPX_TOPOLOGY_AGENT_SOCKET` | `/run/lpx-topology/agent.sock` | The publisher reads the Agent's service on this socket |
| `--topology-socket` | `DYN_LPX_TOPOLOGY_SOCKET` | `/run/lpx-topology/topology.sock` | The consumer serves on this socket |
| `--node-rank` | `DYN_SIDECAR_NODE_RANK` | `0` | The pod's rank in its engine or Agent allocation |

| Role | Rank | Behavior |
|---|---|---|
| Unset | 0 | Serves the engine |
| Unset | Above 0 | Stands by: answers its probes only |
| `publisher` | 0 | Publishes topology and serves no engine; `--grpc-endpoint` and `--model-path` are not required |
| `publisher` | Above 0 | Stands by and publishes nothing |
| `consumer` | 0 | Serves the engine and the topology API. The API starts before the sidecar waits for the engine to be ready, because a conductor needs topology to become ready |
| `consumer` | Above 0 | Serves the topology API only and registers nothing |

A sidecar that runs a topology role connects to Dynamo before anything else,
with `/live` and `/health` probes that check only Dynamo's own dependencies. A
topology failure never stops inference serving, and no hop caches a snapshot.

The endpoint name scopes discovery by the whole identity. The allocation scope
is the first 16 lowercase hex digits of the SHA-256 of the bytes
`lpx.topology.v1` NUL *deployment* NUL *allocation*. The build scope is the
same over `lpx.topology.v1` NUL *deployment* NUL *allocation* NUL *build*. All
authorities of one allocation share the prefix `snapshot-<allocation scope>-`,
so the consumer can tell an allocation that serves another build, which is a
configuration error, from one that is not published yet. Both sidecars use
their effective Dynamo namespace, `DYN_NAMESPACE` plus
`DYN_NAMESPACE_WORKER_SUFFIX`, so the publisher and its consumers must share it.

Exactly one authority must answer `OK`. A leader that is shutting down answers
`UNAVAILABLE`, and the registration of a publisher that crashed stays until
discovery expires it. Neither one hides a replacement that answers. A
conductor retries `UNAVAILABLE`, which covers these cases:

- No authority is registered for the allocation yet.
- More than one authority answers `OK`.
- No authority answers `OK` and none refuses the request. The message gives
  each authority's reason. If the publisher cannot reach the Agent, the reason
  names the Agent's socket.

A conductor fails at once on any other error. `FAILED_PRECONDITION` means
that the allocation's authority serves another build, that an authority
refuses the expected identity, or that the snapshot is invalid or has another
identity. `INVALID_ARGUMENT` means that the request has no complete expected
identity.

The consumer's socket has mode `0660`. Put it in a volume that the conductor
and sidecar containers share, such as an `emptyDir`, and run both containers
with the same user or group.

> [!NOTE]
> With Kubernetes discovery, use pod discovery mode. Container mode never treats
> a native sidecar as ready, so the publisher's endpoint would stay invisible.

## Topologies

Each engine page shows its single-node TP, multi-node TP, and multi-node DP
topologies:
[vLLM](../../modular-components/backends/vllm/sidecar.md#topologies),
[SGLang](../../modular-components/backends/sglang/sidecar.md#topologies),
[TensorRT-LLM](../../modular-components/backends/tensorrt-llm/sidecar.md#topologies).
