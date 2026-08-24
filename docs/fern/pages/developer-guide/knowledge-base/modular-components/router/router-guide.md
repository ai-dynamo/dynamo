---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Guide
subtitle: Deployment modes, quick start, and page map for Dynamo routing docs
---

## Overview

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.
This guide helps you get started with using the Dynamo router and points to the pages that cover routing concepts, configuration, disaggregated serving, and operations in more detail.

## Quick Start

The router can be deployed using [Python / CLI](#python--cli-deployment), [Kubernetes](#kubernetes-deployment), or as a [standalone component](#standalone-router).

> [!IMPORTANT]
> Event-driven KV routing requires both `--router-mode kv` on the frontend and
> `--kv-events-config` on each event-producing backend worker. The frontend flag does not start the
> vLLM or SGLang event publisher. If workers do not publish events, use
> `--no-router-kv-events` for approximate cache prediction.

### Python / CLI Deployment

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8000 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers register with the `register_model` API. For event-driven prefix-cache state, add
the applicable worker flag:

```bash
# vLLM: enable_kv_cache_events is required
python -m dynamo.vllm --model MODEL_NAME \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080","enable_kv_cache_events":true}'

# SGLang: publisher and endpoint are required; topic is optional
python -m dynamo.sglang --model-path MODEL_NAME \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:5557"}'
```

For workers that share a network namespace, use a unique endpoint port. See the
[Router Quick Start](overview.md#quick-start) for the event flow and backend schemas.

The [Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router) is the
canonical list of embedded-router CLI arguments, environment variables, defaults,
and boolean forms. Use [Configuration and Tuning](configuration-and-tuning.md) for
workload-specific guidance, [Router Filtering](worker-filtering.md) for candidate
eligibility, and [Routing Concepts](routing-concepts.md#active-load-modeling) for the
prefill and decode cost model.

### Kubernetes Deployment

To enable the KV Router in Kubernetes, add the `DYN_ROUTER_MODE` environment variable to your frontend service:

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: my-deployment
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv  # Enable KV Smart Router
```

**Key Points:**
- Set `DYN_ROUTER_MODE=kv` on the **Frontend** service only
- Add the backend `--kv-events-config` to every event-producing worker
- Frontend KV mode does not enable worker event publication
- Use `--no-router-kv-events` for approximate cache-state prediction when workers are not publishing events

For exact environment-variable mappings, see the
[Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router). For complete
Kubernetes examples and tuning guidance, see
[Kubernetes Examples](router-examples.md#k8s-examples) and
[Configuration and Tuning](configuration-and-tuning.md).
For A/B testing and advanced K8s setup, see the [KV Router A/B Benchmarking Guide](../../../../recipes/feature-benchmarks/kv-router-a-b-testing.md).

### Standalone Router

You can also run the KV router as a standalone service without the Dynamo frontend for disaggregated serving, multi-tier architectures, or custom routing pipelines. See [Standalone Router](../../../../cli/kv-aware-routing/standalone-router.md) for the Fern guide.

#### Frontend-Embedded vs. Standalone Router

| Deployment | Process | Metrics Port | Use Case |
|------------|---------|--------------|----------|
| **Frontend-embedded** | `python -m dynamo.frontend --router-mode kv` | Frontend HTTP port (default 8000) | Standard deployment; router runs inside the frontend process |
| **Standalone** | `python -m dynamo.router` | `DYN_SYSTEM_PORT` (if set) | Multi-tier architectures, advanced disaggregated prefill routing, custom pipelines |

The standalone router does not include the HTTP frontend and does not expose `/v1/chat/completions`. It exposes routing endpoints through the Dynamo runtime and, when configured, router metrics through the system status server.

## Deployment Modes

The Dynamo router can be deployed in several configurations. The table below shows common combinations and when to use them:

| Mode | Command | Routing Logic | KV Events | Topology | Use Case |
|------|---------|---------------|-----------|----------|----------|
| **Frontend + Round-Robin** | `python -m dynamo.frontend --router-mode round-robin` | Cycles through workers | None | Aggregated | Simplest baseline; no KV awareness |
| **Frontend + Random** | `python -m dynamo.frontend --router-mode random` | Random worker selection | None | Aggregated | Stateless load balancing |
| **Frontend + Power of Two** | `python -m dynamo.frontend --router-mode power-of-two` | Samples two workers and chooses the less loaded one | None | Aggregated or disaggregated fallback | Low-overhead load balancing with better distribution than random selection |
| **Frontend + KV (Aggregated)** | `python -m dynamo.frontend --router-mode kv` | KV cache overlap + load | NATS Core / ZMQ / Approx | Aggregated | Production single-pool serving with cache reuse |
| **Frontend + KV (Disaggregated)** | `python -m dynamo.frontend --router-mode kv` with prefill + decode workers | KV cache overlap + load | NATS Core / ZMQ / Approx | Disaggregated (prefill + decode pools) | Separate prefill/decode for large-scale serving |
| **Frontend + Least-Loaded** | `python -m dynamo.frontend --router-mode least-loaded` | Fewest active connections | None | Aggregated or disaggregated fallback | Simple load-aware balancing without KV awareness |
| **Frontend + Device-Aware Weighted** | `python -m dynamo.frontend --router-mode device-aware-weighted` | Device-aware budget + least-loaded within selected device group | None | Aggregated or disaggregated fallback | Heterogeneous fleet balancing (CPU/non-CPU); degenerates to least-loaded when only one device class is present |
| **Frontend + Direct** | `python -m dynamo.frontend --router-mode direct` | Worker ID from request hints | None | Aggregated | External orchestrator (e.g., EPP/GAIE) selects workers |
| **Standalone Router** | `python -m dynamo.router` | KV cache overlap + load | NATS Core / ZMQ | Any | Routing without the HTTP frontend (multi-tier, custom pipelines) |

> [!IMPORTANT]
> With `DYN_LORA_ENABLED`, use KV, random, or round-robin routing. Direct,
> power-of-two, least-loaded, and device-aware-weighted modes are not LoRA-aware
> and fail startup. Session affinity with LoRA is supported only in KV mode;
> random and round-robin plus affinity are rejected.

### Routing Modes (`--router-mode`)

| Mode | Value | How Workers Are Selected |
|------|-------|-------------------------|
| **Round-Robin** | `round-robin` (default) | Cycles through available workers in order |
| **Random** | `random` | Selects a random worker for each request |
| **Power of Two** | `power-of-two` | Samples two workers and routes to the one with fewer in-flight requests; in disaggregated prefill paths it falls back to synchronous prefill |
| **KV** | `kv` | Evaluates KV cache overlap and decode load per worker; picks lowest cost |
| **Least-Loaded** | `least-loaded` | Routes to the worker with fewest active connections; in disaggregated prefill paths it skips bootstrap optimization and falls back to synchronous prefill |
| **Device-Aware Weighted** | `device-aware-weighted` | Partitions workers into CPU and non-CPU groups, applies capability-normalized ratio budgeting using `DYN_ENCODER_CUDA_TO_CPU_RATIO` to decide which group receives the request, then selects the least-loaded worker within that group |
| **Direct** | `direct` | Reads the target `worker_id` from the request's routing hints; no selection logic |

### Device-Aware Weighted Routing

`device-aware-weighted` is designed for heterogeneous fleets where workers of different compute capability, for example CPU embedding encoders alongside GPU embedding encoders, share the same endpoint.

Workers are split into CPU and non-CPU groups. The router compares a capability-normalized load across the two groups:

```text
normalized_load = total_inflight(group) / (instance_count(group) x throughput_weight)
```

The throughput weight is `1` for CPU workers and `DYN_ENCODER_CUDA_TO_CPU_RATIO` for non-CPU workers. The next request is routed to the group with the lower normalized load, then to the least-loaded worker inside that group.

For multimodal requests, a full embedding-cache hit on one or more workers bypasses the CPU-to-non-CPU ratio. The router selects the least-loaded worker among those that hold every distinct embedding-cache key in the request. Partial hits continue through the normal weighted group selection. See [Embedding Cache](../../../../use-cases/multimodal-serving/embedding-cache.md#configuration).

Use `DYN_ENCODER_CUDA_TO_CPU_RATIO` to approximate the throughput ratio of a non-CPU worker relative to one CPU worker. The default is `8`.

When only one device class is present, the policy degenerates to standard least-loaded routing.

### KV Cache State Modes (within `--router-mode kv`)

The KV router needs the cache state of each worker. Choose one of these state modes:

| State Mode | How to Enable | Description |
|------------|---------------|-------------|
| **Event-driven** | Router default plus backend `--kv-events-config` | The engine emits raw events to its Dynamo worker. The worker normalizes the events and updates its worker-local indexer. The worker republishes the updates over Dynamo's event plane. The router can query the worker-local indexer to recover worker-to-router event gaps. |
| **Approximate (no events)** | `--no-router-kv-events` | The router consumes no events. It predicts cache state from its routing decisions and uses time-based expiration. |

The backend `--kv-events-config` and Dynamo's event-plane configuration control different hops.
One configuration does not change the other. See [Router Design](router-design.md#event-flow-and-recovery)
for the data path. See [Configuration and Tuning](configuration-and-tuning.md) for event-plane
selection.

### Aggregated vs. Disaggregated Topology

| Topology | Workers | How It Works |
|----------|---------|--------------|
| **Aggregated** | Single pool (prefill + decode in one process) | All workers handle the full request lifecycle |
| **Disaggregated** | Separate prefill and decode pools | Frontend routes to a prefill worker first, then to a decode worker; requires workers registered with `WorkerType.Prefill` |

Disaggregated mode is activated automatically when prefill workers register alongside decode workers. See [Disaggregated Serving](disaggregated-serving.md) for details.

## Per-Worker Router Configuration

`--router-mode` on the frontend sets the default for every worker. A worker set can override it for itself by declaring its own routing in its model deployment card, which the frontend then uses in place of its own configuration when routing to that set. This lets one deployment serve worker sets that want different strategies — for example a heterogeneous CPU/GPU encoder pool on `device-aware-weighted` while everything else stays round-robin.

Workers accept the same flags as the frontend, on vLLM, SGLang, TensorRT-LLM, and the mocker:

```bash
# Frontend default for the deployment
python -m dynamo.frontend --router-mode round-robin --http-port 8000

# Worker set A -- overrides to KV. Every replica is launched with the same flags.
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --router-mode kv --router-kv-overlap-score-credit 2.0
python -m dynamo.vllm --model Qwen/Qwen3-0.6B --router-mode kv --router-kv-overlap-score-credit 2.0

# Worker set B -- a different model, no router flags, so it inherits round-robin
python -m dynamo.vllm --model meta-llama/Llama-3.1-8B-Instruct

# Worker set C -- a different model again, on its own strategy
python -m dynamo.vllm --model BAAI/bge-m3 --router-mode device-aware-weighted
```

Sets A, B, and C are distinct because a worker set is keyed on model name as well as endpoint and worker type. Each carries its own routing configuration and the three do not interact.

A worker that omits `--router-mode`, like set B, advertises nothing and inherits the frontend's configuration. That is the default, and it is what every deployment written before this option did.

### The override is per worker set, not per worker

Workers that share a namespace, component, endpoint, model type, and worker type form a single worker set, and a worker set has one routing configuration. Two replicas of the same model launched with different router flags are not two independently routed workers — they are one set whose members disagree.

> [!WARNING]
> Every worker in a set must be launched with identical router flags. The frontend fingerprints each worker's card together with its effective routing configuration, so mismatched flags split the set into two cohorts. A split set is treated as a conflict: **no instances are admitted and the set stops serving**. It does not fall back to one worker's configuration or to the frontend's.

This is not specific to router flags — any card difference splits a set the same way — but router flags are easy to apply to one replica by mistake. Two practical consequences:

- **Changing the routing of a running fleet is a card change.** Roll the whole set rather than mixing old and new replicas, the same as any other change that alters the card.
- **Applying the flag to only some replicas of a set splits it.** Whether by flag or by an exported `DYN_ROUTER_MODE` that reaches some processes and not others, every member of a set must end up with the same value.

> [!IMPORTANT]
> An advertised configuration **replaces** the frontend's for that worker set rather than merging with it. On a worker, a flag you do not pass means **the default**, not "inherit the frontend's value". If the frontend was tuned with `--router-kv-overlap-score-credit 2.5` and a worker advertises only `--router-mode kv`, that worker set routes with the default `1.0` — the tuning is not inherited, it is replaced. Restate on the worker every flag that matters for it.

Merging is not offered because it cannot be done correctly: the KV tuning is flat concrete values with no record of which ones were set explicitly, so a deliberate `1.0` is indistinguishable from a default `1.0`.

Workers and the frontend share defaults exactly — both build their configuration from the same argument groups — so the values only diverge when the frontend is tuned and the worker is not.

The flags also read the same environment variables, and this changes the answer depending on how you deploy. If the frontend's tuning comes from the environment and the worker shares that environment — a single shell starting both — the worker picks the tuning up and nothing is lost. Kubernetes scopes environment per service, so there the worker sees only its own flags. The same intent can therefore produce different routing in the two setups. `--router-mode` itself is the exception in the other direction: a shared `DYN_ROUTER_MODE` makes a worker advertise when you meant it to inherit.

The `Activating prefill router` log line reports the configuration each hop actually resolved — mode, block size, session TTL, and KV tuning — which is the quickest way to confirm what a worker set ended up with.

The frontend-only options (`--router-min-initial-workers`, `--enforce-disagg`, `--admission-control`) are not accepted by workers, since a model card does not carry them.

Advertised configuration applies to the worker sets the frontend routes to directly — aggregated and decode. See [Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router) for the full flag list.

## More Router Docs

- **[Routing Concepts](routing-concepts.md)**: Cost model, worker selection, and routing primitives
- **[Frontend Configuration Reference](../../../../reference/components/frontend-configuration.mdx#router)**: Canonical router flags, environment variables, defaults, and boolean forms
- **[Configuration and Tuning](configuration-and-tuning.md)**: Router behavior, transport modes, load tracking, and tuning guidance
- **[Disaggregated Serving](disaggregated-serving.md)**: Prefill and decode routing setups
- **[Topology-Aware KV Transfer](topology-aware-kv-transfer.md)**: Runtime metadata and decode routing constraints for topology-aware prefill/decode handoff
- **[Router Operations](router-operations.md)**: Replicas, remote indexers, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Recommended test layers for non-trivial router changes
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service
- **[Standalone Selection Service](standalone-selection.md)**: Select workers and account for reservations without forwarding requests
- **[Standalone Slot Tracker](standalone-slot-tracker.md)**: Run active-request accounting as a separate service
- **[KV Event Replay — Dynamo vs vLLM](kv-event-replay-comparison.md)**: Gap detection and replay behavior
