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

The sidecar Dockerfile builds all three engine-specific sidecar executables into
one CPU-only image. Deployments select an engine by setting the container
`command`:

| Engine | Container command |
|---|---|
| vLLM | `dynamo-vllm-sidecar` |
| SGLang | `dynamo-sglang-sidecar` |
| TensorRT-LLM | `dynamo-trtllm-sidecar` |

The image's default entrypoint, `dynamo-sidecar`, maps the short names `vllm`,
`sglang`, and `trtllm` onto those executables. It is a convenience for ad-hoc
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
      args: [--grpc-endpoint, 127.0.0.1:50051]
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

## Metrics

The sidecar serves Dynamo's worker metrics on `DYN_SYSTEM_PORT` at `/metrics`,
as the in-process backends do. The engine's own Prometheus families stay with
the engine process, so the sidecar re-exports them: on every scrape of its
`/metrics`, it fetches the engine's `/metrics`, keeps the engine's families,
injects the Dynamo identity labels (`dynamo_namespace`, `dynamo_component`,
`dynamo_endpoint`, `worker_id`, `model`, `model_name`, with the engine's own
labels taking precedence), and appends them to the response. Nothing is polled
or cached between scrapes; the scraper sets the cadence, and the same families
are available to the OTLP exporter when it is configured.

| Engine | Engine endpoint | Requirement |
|---|---|---|
| vLLM | `vllm-rs serve` HTTP server, port `8000` by default | None. Set `--vllm-http-endpoint` on the sidecar when vLLM's `--port` differs. |
| SGLang | SGLang HTTP server on `--port` | Start SGLang with `--enable-metrics`. The sidecar reads the port from `GetServerInfo`. |
| TensorRT-LLM | Not available | `trtllm-serve --grpc` does not serve an HTTP metrics endpoint. |

The fetch is bounded to 3 seconds and 8 MiB. When it fails, the engine families
are absent from that scrape and Dynamo's own families are served as usual;
`dynamo_component_engine_metrics_scrape_success` reads `0` and
`dynamo_component_engine_metrics_scrape_duration_seconds` records the attempt.
The sidecar logs once when the engine becomes unreachable and once when it is
back.

To scrape the engine directly with your own PodMonitor instead, turn the
re-export off so the families are not collected twice. Set
`DYN_SIDECAR_ENGINE_METRICS_REEXPORT=false` on the `runtime` container, or pass
`--engine-metrics-reexport false`.

## Topologies

Each engine page shows its single-node TP, multi-node TP, and multi-node DP
topologies:
[vLLM](../../modular-components/backends/vllm/sidecar.md#topologies),
[SGLang](../../modular-components/backends/sglang/sidecar.md#topologies),
[TensorRT-LLM](../../modular-components/backends/tensorrt-llm/sidecar.md#topologies).
