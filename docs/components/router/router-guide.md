---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Router Guide
subtitle: Deployment modes, quick start, and page map for Dynamo routing docs
---

## Overview

The Dynamo KV Router intelligently routes requests by evaluating their computational costs across different workers. It considers both decoding costs (from active blocks) and prefill costs (from newly computed blocks), using KV cache overlap to minimize redundant computation. Optimizing the KV Router is critical for achieving maximum throughput and minimum latency in distributed inference setups.
This guide helps you get started with using the Dynamo router and points to the pages that cover routing concepts, configuration, disaggregated serving, and operations in more detail.

## Deployment Modes

The Dynamo router can be deployed in several configurations. The table below shows every combination and when to use it:

| Mode | Command | Routing Logic | KV Events | Topology | Use Case |
|------|---------|---------------|-----------|----------|----------|
| **Frontend + Round-Robin** | `python -m dynamo.frontend --router-mode round-robin` | Cycles through workers | None | Aggregated | Simplest baseline; no KV awareness |
| **Frontend + Random** | `python -m dynamo.frontend --router-mode random` | Random worker selection | None | Aggregated | Stateless load balancing |
| **Frontend + KV (Aggregated)** | `python -m dynamo.frontend --router-mode kv` | KV cache overlap + load | NATS Core / JetStream / ZMQ / Approx | Aggregated | Production single-pool serving with cache reuse |
| **Frontend + KV (Disaggregated)** | `python -m dynamo.frontend --router-mode kv` with prefill + decode workers | KV cache overlap + load | NATS Core / JetStream / ZMQ / Approx | Disaggregated (prefill + decode pools) | Separate prefill/decode for large-scale serving |
| **Frontend + Least-Loaded** | `python -m dynamo.frontend --router-mode least-loaded` | Fewest active connections | None | Aggregated or disaggregated fallback | Simple load-aware balancing without KV awareness |
| **Frontend + Device-Aware Weighted** | `python -m dynamo.frontend --router-mode device-aware-weighted` | Device-aware budget + least-loaded within selected device group | None | Aggregated or disaggregated fallback | Heterogeneous fleet balancing (CPU/non-CPU); degenerates to least-loaded when only one device class is present |
| **Frontend + Direct** | `python -m dynamo.frontend --router-mode direct` | Worker ID from request hints | None | Aggregated | External orchestrator (e.g., EPP/GAIE) selects workers |
| **Standalone Router** | `python -m dynamo.router` | KV cache overlap + load | NATS Core / JetStream / ZMQ | Any | Routing without the HTTP frontend (multi-tier, custom pipelines) |

### Routing Modes (`--router-mode`)

| Mode | Value | How Workers Are Selected |
|------|-------|-------------------------|
| **Round-Robin** | `round-robin` (default) | Cycles through available workers in order |
| **Random** | `random` | Selects a random worker for each request |
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

When using KV routing, the router needs to know what each worker has cached. There are four ways to get this information:

| Event Mode | How to Enable | Description |
|------------|---------------|-------------|
| **NATS Core (local indexer)** | Default (no extra flags) | Workers maintain a local indexer; router queries workers on startup and receives events via NATS Core |
| **JetStream (durable)** | `--router-durable-kv-events` | Events persisted in NATS JetStream; supports snapshots and durable consumers. *Deprecated.* |
| **ZMQ** | `--event-plane zmq` | Workers publish via ZMQ PUB sockets; the standalone `dynamo.indexer` service aggregates events |
| **Approximate (no events)** | `--no-router-kv-events` | No events consumed; router predicts cache state from its own routing decisions with TTL-based expiration |

### Aggregated vs. Disaggregated Topology

| Topology | Workers | How It Works |
|----------|---------|--------------|
| **Aggregated** | Single pool (prefill + decode in one process) | All workers handle the full request lifecycle |
| **Disaggregated** | Separate prefill and decode pools | Frontend routes to a prefill worker first, then to a decode worker; requires workers registered with `ModelType.Prefill` |

Disaggregated mode is activated automatically when prefill workers register alongside decode workers. See [Disaggregated Serving](router-disaggregated-serving.md) for details.

### Frontend-Embedded vs. Standalone Router

| Deployment | Process | Metrics Port | Use Case |
|------------|---------|--------------|----------|
| **Frontend-embedded** | `python -m dynamo.frontend --router-mode kv` | Frontend HTTP port (default 8000) | Standard deployment; router runs inside the frontend process |
| **Standalone** | `python -m dynamo.router` | `DYN_SYSTEM_PORT` (if set) | Multi-tier architectures, SGLang disagg prefill routing, custom pipelines |

The standalone router does not include the HTTP frontend (no `/v1/chat/completions` endpoint). It exposes only the `RouterRequestMetrics` via the system status server. See the [Standalone Router README](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/router/README.md).

## Quick Start

### Python / CLI Deployment

To launch the Dynamo frontend with the KV Router:

```bash
python -m dynamo.frontend --router-mode kv --http-port 8000
```

This command:
- Launches the Dynamo frontend service with KV routing enabled
- Exposes the service on port 8000 (configurable)
- Automatically handles all backend workers registered to the Dynamo endpoint

Backend workers register themselves using the `register_model` API, after which the KV Router automatically tracks worker state and makes routing decisions based on KV cache overlap.

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--router-mode kv` | `round-robin` | Enable KV cache-aware routing |
| `--router-temperature <float>` | `0.0` | Controls routing randomness (0.0 = deterministic, higher = more random) |
| `--kv-cache-block-size <size>` | Backend-specific | KV cache block size (should match backend config) |
| `--router-kv-events` / `--no-router-kv-events` | `--router-kv-events` | Enable/disable real-time KV event tracking |
| `--router-kv-overlap-score-weight <float>` | `1.0` | Balance prefill vs decode optimization (higher = better TTFT) |
| `--router-track-prefill-tokens` / `--no-router-track-prefill-tokens` | `--router-track-prefill-tokens` | Include prompt-side load in active worker load accounting |
| `--router-prefill-load-model <none\|aic>` | `none` | Prompt-side load model. `aic` decays only the oldest active prefill using an AIC-predicted duration |
| `--router-queue-threshold <float>` | `4.0` | Queue threshold fraction; enables priority scheduling via `priority` |
| `--router-queue-policy <str>` | `fcfs` | Scheduling policy for the queue: `fcfs` (tail TTFT), `wspt` (avg TTFT), or `lcfs` (comparison-only reverse ordering) |
| `--serve-indexer` | `false` | Serve the Dynamo-native remote indexer from this frontend/router on the worker component |
| `--use-remote-indexer` | `false` | Query the worker component's served remote indexer instead of maintaining a local overlap indexer |

For all available options: `python -m dynamo.frontend --help`

For detailed configuration options and tuning parameters, see [Configuration and Tuning](router-configuration.md).

#### AIC Prefill Load Model

The KV router can use AIC to estimate the expected duration of the selected worker's prompt-side prefill work. When enabled, the router:

- computes `prefix = overlap_blocks * block_size` for the chosen worker
- computes `effective_isl = input_tokens - prefix`
- stores one prompt-load hint for the admitted request
- decays only the **oldest** active prefill request on each worker over time

This affects router-side prompt load accounting only. It does not change backend execution or decode-side accounting.

Enable it on the frontend like this:

```bash
python -m dynamo.frontend \
    --router-mode kv \
    --router-prefill-load-model aic \
    --aic-backend vllm \
    --aic-system h200_sxm \
    --aic-model-path nvidia/Llama-3.1-8B-Instruct-FP8
```

The standalone router uses the same AIC flags:

```bash
python -m dynamo.router \
    --endpoint dynamo.prefill.generate \
    --router-prefill-load-model aic \
    --aic-backend vllm \
    --aic-system h200_sxm \
    --aic-model-path nvidia/Llama-3.1-8B-Instruct-FP8
```

Required when `--router-prefill-load-model=aic` is enabled:

- `--router-mode kv` on the frontend
- `--router-track-prefill-tokens`
- `--aic-backend`
- `--aic-system`
- `--aic-model-path`

Optional AIC knobs:

- `--aic-backend-version`: pinned AIC database version; if omitted, Dynamo uses a backend-specific default
- `--aic-tp-size`: tensor-parallel size for the modeled backend; defaults to `1`

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
      dynamoNamespace: my-namespace
      componentType: frontend
      replicas: 1
      envs:
        - name: DYN_ROUTER_MODE
          value: kv  # Enable KV Smart Router
```

**Key Points:**
- Set `DYN_ROUTER_MODE=kv` on the **Frontend** service only
- Workers automatically report KV cache events to the router
- No worker-side configuration changes needed

#### Environment Variables

All CLI arguments can be configured via environment variables using the `DYN_` prefix:

| CLI Argument | Environment Variable | Default |
|--------------|---------------------|---------|
| `--router-mode kv` | `DYN_ROUTER_MODE=kv` | `round-robin` |
| `--router-temperature` | `DYN_ROUTER_TEMPERATURE` | `0.0` |
| `--kv-cache-block-size` | `DYN_KV_CACHE_BLOCK_SIZE` | Backend-specific |
| `--no-router-kv-events` | `DYN_ROUTER_USE_KV_EVENTS=false` | `true` |
| `--router-kv-overlap-score-weight` | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` | `1.0` |
| `--router-queue-policy` | `DYN_ROUTER_QUEUE_POLICY` | `fcfs` |
| `DYN_ENCODER_CUDA_TO_CPU_RATIO` | `8` | Throughput ratio of a non-CPU worker relative to one CPU worker for `device-aware-weighted` routing |

For complete K8s examples and advanced configuration, see [K8s Examples](router-examples.md#k8s-examples) and [Configuration and Tuning](router-configuration.md).
For A/B testing and advanced K8s setup, see the [KV Router A/B Benchmarking Guide](../../benchmarks/kv-router-ab-testing.md).

### Standalone Router

You can also run the KV router as a standalone service (without the Dynamo frontend) for disaggregated serving (e.g., routing to prefill workers), multi-tier architectures, or any scenario requiring intelligent KV cache-aware routing decisions. See the [Standalone Router component](https://github.com/ai-dynamo/dynamo/tree/main/components/src/dynamo/router/) for more details.

## More Router Docs

- **[Routing Concepts](router-concepts.md)**: Cost model, worker selection, and routing primitives
- **[Configuration and Tuning](router-configuration.md)**: Router flags, transport modes, load tracking, and metrics
- **[Disaggregated Serving](router-disaggregated-serving.md)**: Prefill and decode routing setups
- **[Router Operations](router-operations.md)**: Replicas, remote indexers, persistence, and recovery
- **[Router Examples](router-examples.md)**: Python API usage, K8s examples, and custom routing patterns
- **[Router Testing](router-testing.md)**: Recommended test layers for non-trivial router changes
- **[Standalone Indexer](standalone-indexer.md)**: Run the KV indexer as a separate service
- **[KV Event Replay — Dynamo vs vLLM](kv-event-replay-comparison.md)**: Gap detection and replay behavior
