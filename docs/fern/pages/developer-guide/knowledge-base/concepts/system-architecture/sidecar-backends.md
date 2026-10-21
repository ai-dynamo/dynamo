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
| An LPX engine ([LPX](#lpx-engines)) | `dynamo-lpx-sidecar` |

The image's default entrypoint, `dynamo-sidecar`, maps the short names `vllm`,
`sglang`, `trtllm`, and `lpx` onto those executables. It is a convenience for
ad-hoc `docker run`; the deployment manifests override it with `command`. The
LPX sidecar ships only in this image; the other three are also in the
`ai-dynamo` Python package. The inference engine remains in a separate
container, so the sidecar image does not include vLLM, SGLang, TensorRT-LLM,
the LPX engines, CUDA, or engine-specific Python dependencies. The image runs
as the non-root `dynamo` user with numeric user ID `1000` and declares port
`9090` for Dynamo system endpoints, so Kubernetes can enforce `runAsNonRoot`.

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
Kubernetes 1.29 or later. The two connect over loopback, or, for an engine
that serves gRPC on a Unix socket, through a socket in a volume both containers
mount: `--grpc-endpoint unix:///ABSOLUTE/PATH`.

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
probes on `main`. The sidecar owns port `9090` and the port name `system`, so
the engine must not use port `9090`, and `main` must declare the port its
probes use. The LPX sidecar's `/health` also follows its engine once it has
identified it; see [Health](#health). Sidecar mode supports worker, prefill,
and decode components; multinode worker components are not yet supported. LPX
components declare the `runtime` init container on the role whose pods host
the engine, including an engine that spans several Agent pods; see
[Operator Placement](#operator-placement).

## LPX Engines

The LPX sidecar connects Dynamo to the OpenEngine gRPC server of two LPX
engines. It uses the `engine_name` the server reports only to name the engine
in logs and errors, and fails at startup on an OpenEngine client revision it
does not implement or on a role that does not match `--disaggregation-mode`.

| | Cyborg | LPU-only engine |
|---|---|---|
| Hardware | Hybrid (GPU and LPU) | LPU only |
| `--disaggregation-mode` | `agg`, or `decode` behind a vLLM prefill worker | `agg` |
| OpenEngine listener | Unix socket, with Cyborg's `--request-plane openengine` (`--openengine-socket`, default `/run/openengine/openengine.sock`) | Unix socket, `LPU_IE_OPENENGINE_SOCKET` (default `/run/openengine/openengine.sock`) |
| `--grpc-endpoint` for the default socket | `unix:///run/openengine/openengine.sock` | `unix:///run/openengine/openengine.sock` |
| Served model name set by | Cyborg's `--dynamo-model-name` | `LPU_IE_SERVED_MODEL_NAME` |

The sidecar registers the served model name and context window the engine
reports, and its metrics' `model` and `model_name` labels carry that name;
`--model-path` names only the tokenizer and chat template. Its own
`--served-model-name` (`DYN_SERVED_MODEL_NAME`) is optional and only checked
against the engine's name, failing at startup on a mismatch, so a `runtime`
container written for the TensorRT-LLM sidecar can switch to `lpx` and keep
it. The
[LPX sidecar README](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/lpx/README.md)
shows how to run it and how its errors reach the frontend.

### OpenEngine on a Unix Socket

Both LPX engines serve OpenEngine only on a Unix socket. Mount its directory
in both containers; these are the changes to the pod above:

```yaml
podTemplate:
  spec:
    securityContext:
      fsGroup: 1000
    initContainers:
    - name: runtime
      command: [dynamo-lpx-sidecar]
      args: [--grpc-endpoint, unix:///run/openengine/openengine.sock, --model-path, <model>]
      volumeMounts:
      - {name: openengine, mountPath: /run/openengine}
    containers:
    - name: main
      volumeMounts:
      - {name: openengine, mountPath: /run/openengine}
    volumes:
    - name: openengine
      emptyDir: {}
```

Within `--grpc-startup-deadline-secs`, the sidecar retries until the engine
creates the socket and then until it answers `GetModelInfo`, which both engines
do once they admit requests. It reconnects through the same path after the
engine restarts. Both engines create the socket with mode `0660`, so the
sidecar, user ID `1000`, needs the engine's user or group. The pod `fsGroup`
provides the group: the `emptyDir` passes it on to the socket, and every
container in the pod belongs to it.

### Request Shaping

| Request field | What the sidecar sends |
|---|---|
| `max_tokens` set by the client | The client's value |
| `max_tokens` absent | The rest of the context window after the prompt, at least 1. The LPU-only engine requires a positive value; Cyborg fills an absent one the same way |
| Stop strings | Nothing. The Dynamo frontend matches them in the detokenized text and stops the request, which ends the engine's stream; with `--dyn-chat-processor vllm` it only cuts the text, and the engine generates until EOS or `max_tokens`. The LPU-only engine rejects stop strings and Cyborg ignores them |
| Stop token IDs | The client's and the model's. Both engines stop only on their model's stop tokens; the frontend's default pipeline still stops on the others |
| Other options the client set | Forwarded; engine refusals return 4xx. The LPU-only engine accepts and ignores non-neutral frequency, presence and repetition penalties. The sidecar rejects unsupported options with a 4xx, including logprobs, guided decoding, any `allowed_token_ids` or `logprob_token_ids`, and nonempty `bad_words_token_ids`. It also rejects `no_stop_trim=true` when client stop token IDs are nonempty or `skip_special_tokens=false`. See [LPX limitations](https://github.com/ai-dynamo/dynamo/blob/main/lib/sidecar/lpx/README.md#limitations) for the full list |

### Cyborg Decode

With a vLLM prefill worker, Cyborg serves the decode half of disaggregated
serving. The decode sidecar sends the prefill worker's `kv_transfer_params`
unchanged in the request's KV session, and Cyborg pulls the KV cache over
NIXL.

| Setting | Requirement |
|---|---|
| Cyborg build | One with the OpenEngine decode role; a Cyborg without it refuses `--use-disagg-prefill` with `--request-plane openengine` |
| Cyborg `--dynamo-model-name` | Equals the vLLM prefill worker's `--served-model-name`. The frontend pairs workers by name, and with different names the model answers 503 without a startup error |
| vLLM `--max-model-len` | Cyborg's context window. The frontend can report the prefill worker's window for the model, and a mismatch fails no check |
| Cyborg `--disagg-prefill-k-transform` | Matches the prefill worker's KV layout. A wrong setting still returns text, so compare greedy output against an aggregated run. Cyborg's disaggregated-prefill documentation lists the supported combinations |
| Conditional disaggregation | Off. Every decode request needs a prefill handoff |
| Migration | A second decode worker, for `--migration-limit` above 0 to have a target |

A decode leg is always sent to Cyborg, even after the client disconnects. A
cancellation then ends its stream at once, and the Cyborg build that serves the
decode role still runs the cancelled request for one token, so the prefill
worker's KV blocks are pulled and released rather than held until their lease
expires.

### Health

Once the engine is identified, the LPX sidecar's `/health` also follows the
engine's `grpc.health.v1` status for the whole server (service `""`). Both LPX
engines report `SERVING` exactly while they admit requests.

| Engine state | Sidecar `/health` |
|---|---|
| `SERVING` | Ready, when the sidecar's runtime is |
| Starting, draining, or stopping | Not ready; checked again every `--grpc-retry-interval-secs` |
| Unreadable: killed, restarting, socket gone, or no health service (`UNIMPLEMENTED`) | Not ready; the watch is re-established with backoff, at most 10 s apart or the retry interval if that is longer |

Before the engine is identified, `/health` covers only the sidecar's runtime,
and the worker is not yet registered. `/live` and the worker's discovery
registration do not follow the engine's health.

### Metrics

On the serving rank, the sidecar polls `Control.GetLoad` to publish router load
and the worker's KV cache gauges, and planner forward-pass metrics. An engine
without `GetLoad` (`UNIMPLEMENTED`) publishes neither.

### Shutdown

The `runtime` container is a native sidecar, so when the pod is deleted,
Kubernetes stops the engine in `main` first and sends the sidecar `SIGTERM`
only after `main` exits. While the engine drains, the sidecar keeps relaying
its open streams; with Kubernetes discovery, the frontend stops routing new
requests to the pod once it is terminating. On its own `SIGTERM` the sidecar
unregisters and, after `DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS` (5 s by
default), ends any stream still open. It does not wait for streams, so
`DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT` does not extend the engine's drain.

The pod's `terminationGracePeriodSeconds`, 60 s by default, covers the engine's
drain and the sidecar's shutdown together. Set it in the role template to at
least the engine's drain timeout plus 15 s. Cyborg drains for its
`--graceful-shutdown-timeout`, 2 minutes by default, so 180 s fits it. When
the grace period runs out, Kubernetes kills both containers and cuts off the
streams still open.

### Standby Ranks

When one engine spans several pods, only rank 0 serves. Each LPX sidecar takes
its pod's rank with `--node-rank` or `DYN_SIDECAR_NODE_RANK`, passed by
whatever launches the pods (see [Operator Placement](#operator-placement) for a
DynamoGraphDeployment). The default is `0`, and an empty or non-numeric value
fails at startup rather than serving as a second rank 0. On any other rank the
sidecar stands by:

| | Standby behavior |
|---|---|
| System port | Answers the health and live paths (`DYN_SYSTEM_HEALTH_PATH` and `DYN_SYSTEM_LIVE_PATH`, `/health` and `/live` by default) with 200, `/metrics` with an empty Prometheus exposition, and every other path with 404, or only waits when the system server is disabled |
| Engine | Never connected, so its health is never watched |
| Dynamo | Joins no discovery, request plane, or event plane, so it never registers a second worker |
| Arguments | Parsed but not checked, so every rank takes the same container arguments |
| SIGTERM | Exits cleanly |

### Operator Placement

In a DynamoGraphDeployment, an LPX component declares the `runtime` init
container in the template of the role whose pods host the engine's OpenEngine
endpoint:

| Engine | Role template | Pods that serve | Sidecar rank |
|---|---|---|---|
| Cyborg, hybrid build | `conductor` | Every conductor pod | Not set |
| LPU-only engine, single model, no `conductor` role | `agent` | Agent pod 0 | Not set with one Agent pod; `--node-rank $(GROVE_PCLQ_POD_INDEX)` with more than one |

On the agent, the operator merges the sidecar environment and discovery
defaults into `runtime`, adds no default port, probes, or rank, and leaves port
and probe configuration to the template: Agent pods use host networking, where
every declared container port reserves a host port and the LPX scheduler
refuses any Agent pod that declares a port, so declare no `ports` on the Agent
containers, set `DYN_SYSTEM_PORT` on `runtime` to a free host port, and declare
the `runtime` readiness probe by that port number. When a build
places more than one Agent pod, also add `--node-rank $(GROVE_PCLQ_POD_INDEX)`
to the sidecar's `args`, without the `+ 1` that other backends' multinode
worker ranks add. Grove sets this zero-based index in every container, so pod
0 of each engine replica serves and the others stand by. A single Agent pod
needs no rank, and a conductor's sidecar never takes one: every conductor pod
serves. The
[roles reference](../../../../reference/kubernetes-api/dynamo-component-deployment.mdx)
lists every rule.

## Topologies

Each engine page shows its single-node TP, multi-node TP, and multi-node DP
topologies:
[vLLM](../../modular-components/backends/vllm/sidecar.md#topologies),
[SGLang](../../modular-components/backends/sglang/sidecar.md#topologies),
[TensorRT-LLM](../../modular-components/backends/tensorrt-llm/sidecar.md#topologies).
