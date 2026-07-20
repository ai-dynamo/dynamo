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

### Python / CLI Deployment

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8000 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers register themselves using the `register_model` API. For accurate prefix-cache state, workers must also publish KV cache events with the backend-specific event flags; otherwise the router can run in approximate mode with `--no-router-kv-events`.

The [Frontend Configuration Reference](../frontend/configuration.md#router) is the
canonical list of embedded-router CLI arguments, environment variables, defaults,
and boolean forms. Use [Configuration and Tuning](router-configuration.md) for
workload-specific guidance, [Router Filtering](router-filtering.md) for candidate
eligibility, and [Routing Concepts](router-concepts.md#active-load-modeling) for the
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
- Configure worker-side KV event publishing when you want event-driven prefix-cache state
- Use `--no-router-kv-events` for approximate cache-state prediction when workers are not publishing events

For exact environment-variable mappings, see the
[Frontend Configuration Reference](../frontend/configuration.md#router). For complete
Kubernetes examples and tuning guidance, see
[Kubernetes Examples](router-examples.md#k8s-examples) and
[Configuration and Tuning](router-configuration.md).
For A/B testing and advanced K8s setup, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend) for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions. See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

#### Frontend-Embedded vs. Standalone Router

| Deployment | Process | Metrics Port | Use Case |
|------------|---------|--------------|----------|
| **Frontend-embedded** | `python -m dynamo.frontend --router-mode kv` | Frontend HTTP port (default 8000) | Standard deployment; router runs inside the frontend process |
| **Standalone** | `python -m dynamo.router` | `DYN_SYSTEM_PORT` (if set) | Multi-tier architectures, advanced disaggregated prefill routing, custom pipelines |

The standalone router does not include the HTTP frontend (no `/v1/chat/completions` endpoint). It exposes only the `RouterRequestMetrics` via the system status server. See the [Standalone Router README](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/router/README.md).

## Deployment Modes

The Dynamo router can be deployed in several configurations. The table below shows every combination and when to use it:

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

Use `DYN_ENCODER_CUDA_TO_CPU_RATIO` to approximate the throughput ratio of a non-CPU worker relative to one CPU worker. The default is `8`.

When only one device class is present, the policy degenerates to standard least-loaded routing.

### KV Event Transport Modes (within `--router-mode kv`)

When using KV routing, the router needs to know what each worker has cached. There are three ways to get this information:

| Event Mode | How to Enable | Description |
|------------|---------------|-------------|
| **ZMQ (local indexer)** | Router default (no router flag) | Workers maintain a local indexer and publish KV events via ZMQ PUB sockets; the router recovers state by querying live workers. This is the default event plane for all backends |
| **NATS Core (local indexer)** | `--event-plane nats` (or `DYN_EVENT_PLANE=nats`) | Same local-indexer model, but events flow over NATS Core instead of ZMQ. |
| **Approximate (no events)** | `--no-router-kv-events` | No events consumed; router predicts cache state from its own routing decisions with TTL-based expiration |

### Aggregated vs. Disaggregated Topology

| Topology | Workers | How It Works |
|----------|---------|--------------|
| **Aggregated** | Single pool (prefill + decode in one process) | All workers handle the full request lifecycle |
| **Disaggregated** | Separate prefill and decode pools | Frontend routes to a prefill worker first, then to a decode worker; requires workers registered with `WorkerType.Prefill` |

Disaggregated mode is activated automatically when prefill workers register alongside decode workers. See [Disaggregated Serving](router-disaggregated-serving.md) for details.

## More Router Docs

- **[Routing Concepts](router-concepts.md)**: Cost model, worker selection, and routing primitives
- **[Multi-DC KV Routing and the DC Relay](multi-dc-kv-routing.md)**: DC-local exact ownership, pool publication, and cross-DC CKF search
- **[Frontend Configuration Reference](../frontend/configuration.md#router)**: Canonical router flags, environment variables, defaults, and boolean forms
- **[Configuration and Tuning](router-configuration.md)**: Router behavior, transport modes, load tracking, and tuning guidance
- **[Disaggregated Serving](router-disaggregated-serving.md)**: Prefill and decode routing setups
- **[Topology-Aware KV Transfer](topology-aware-kv-transfer.md)**: Runtime metadata and decode routing constraints for topology-aware prefill/decode handoff
- **[Router Operations](router-operations.md)**: Replicas, remote indexers, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Recommended test layers for non-trivial router changes
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service
- **[Standalone Selection Service](standalone-selection.md)**: Select workers and account for reservations without forwarding requests
- **[Standalone Slot Tracker](standalone-slot-tracker.md)**: Run active-request accounting as a separate service
- **[KV Event Replay — Dynamo vs vLLM](kv-event-replay-comparison.md)**: Gap detection and replay behavior
