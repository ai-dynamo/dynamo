---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Standalone KV Indexer
subtitle: Run the KV cache indexer as an independent HTTP service for querying block state
---

## Overview

The standalone KV indexer (`python -m dynamo.indexer`) is a lightweight service that maintains a radix tree of cached blocks and exposes HTTP endpoints for querying and managing workers.

- It subscribes to ZMQ KV event streams directly from workers.
- It exposes an HTTP API for registration, inspection, and overlap queries.
- It preserves P2P recovery and gap detection/replay for the standalone ZMQ path.

This is distinct from the [Standalone Router](../../../components/src/dynamo/router/README.md), which is a full routing service. The standalone indexer provides only the indexing and query layer without routing logic.

For Dynamo-native remote indexing, use `--serve-indexer` on `dynamo.frontend` or `dynamo.router` and `--use-remote-indexer` on consumers instead. That request-plane service reuses the router's existing event ingestion and recovery machinery; it is not implemented by `dynamo.indexer`.

The HTTP API follows the [Mooncake KV Indexer RFC](https://github.com/kvcache-ai/Mooncake/issues/1403) conventions.

`DYN_ROUTER_MIN_INITIAL_WORKERS` is also honored here. When set to a positive integer, the
standalone indexer waits for that many workers to register before opening its startup-ready
gate, matching the frontend/router startup behavior.

## Multi-Model and Multi-Tenant Support

The indexer maintains one radix tree per `(model_name, tenant_id)` pair. Workers registered with different model names or tenant IDs are isolated into separate indexers — queries against one model/tenant never return scores from another.

- **`model_name`** (required on `/register` and `/query`): Identifies the model. Workers serving different models get separate radix trees.
- **`tenant_id`** (optional, defaults to `"default"`): Enables multi-tenant isolation within the same model. Omit for single-tenant deployments.
- **`block_size`** is per-indexer: the first `/register` call for a given `(model_name, tenant_id)` sets the block size. Subsequent registrations for the same pair must use the same block size or the request will fail.

## Compatibility

The standalone indexer works with any engine that publishes KV cache events over ZMQ in the expected msgpack format. This includes bare vLLM and SGLang engines, which emit ZMQ KV events natively — no Dynamo-specific wrapper is required.

## Use Cases

- **Debugging**: Inspect the radix tree state to verify which blocks are cached on which workers.
- **State verification**: Confirm that the indexer's view of KV cache state matches the router's internal state (used in integration tests).
- **Custom routing**: Build external routing logic that queries the indexer for overlap scores and makes its own worker selection decisions.
- **Monitoring**: Observe KV cache distribution across workers without running a full router.
- **Standalone microservice**: Run an indexer independently of the router/frontend when you want direct HTTP inspection and ZMQ-based ingestion.

## P2P Recovery

Multiple indexer replicas can subscribe to the same ZMQ worker endpoints for fault tolerance. When a replica starts (or restarts after a crash), it bootstraps its radix tree state from a healthy peer before processing live events.

### How It Works

1. Workers are registered via `--workers` or `/register`. Each ZMQ listener enters `pending` state and begins its initial subscribe/connect attempt in the background.
2. A 1-second delay biases peer recovery past the slow-joiner window, so the dump covers events that may have occurred before a fresh listener can safely start draining.
3. The indexer fetches a `/dump` from the first reachable peer in `--peers`.
4. Dump events are applied to populate the radix tree.
5. After recovery completes, the ready gate opens. Any listener whose initial ZMQ connect has already succeeded transitions to `active` and begins draining buffered events; listeners for workers that are still down remain `pending` until they connect.

If no peers are reachable, the indexer starts with an empty state.

### Example: Two-Replica Setup

```bash
# Replica A (first instance, no peers)
python -m dynamo.indexer --port 8090 --block-size 16 \
  --workers "1=tcp://worker1:5557,2=tcp://worker2:5558"

# Replica B (recovers from A on startup)
python -m dynamo.indexer --port 8091 --block-size 16 \
  --workers "1=tcp://worker1:5557,2=tcp://worker2:5558" \
  --peers "http://localhost:8090"
```

Both replicas subscribe to the same workers. Replica B recovers A's tree state on startup, then both independently process live ZMQ events going forward.

### Consistency

The dump is a weakly consistent BFS snapshot of the radix tree — concurrent writes may race with the traversal. This is acceptable because:

- **Stale blocks** (partially removed branches): live `Remove` events will clean them up.
- **Missing blocks** (partially added branches): live `Stored` events will add them.
- The tree converges to the correct state after live events catch up.

### Peer Management

Peers can be registered at startup via `--peers` or dynamically via the HTTP API. The peer list is used for recovery only — peers do not synchronize state in real time.

## Building

The service is exposed through the Python bindings package and launched with `python -m dynamo.indexer` after building the bindings with maturin. Feature flags control which capabilities are compiled in:

| Feature | Description |
|---------|-------------|
| `kv-indexer` | Core standalone indexer service path (`python -m dynamo.indexer`: HTTP API, ZMQ listeners, P2P recovery) |
| `kv-indexer-metrics` | Optional `/metrics` endpoint |

### Standalone build

```bash
cd lib/bindings/python && VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --uv --features kv-indexer
```

After installation, launch the service with `python -m dynamo.indexer`.

### Standalone build with metrics

```bash
cd lib/bindings/python && VIRTUAL_ENV=../../.venv ../../.venv/bin/maturin develop --uv --features kv-indexer,kv-indexer-metrics
```

This keeps the default `kv-indexer` build lean while still allowing Prometheus metrics when needed.

## CLI

```bash
python -m dynamo.indexer --port 8090 [--threads 4] [--block-size 16 --model-name my-model --tenant-id default --workers "1=tcp://host:5557,2:1=tcp://host:5558"] [--peers "http://peer1:8090,http://peer2:8091"]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--block-size` | (none) | KV cache block size for initial `--workers` (required when `--workers` is set) |
| `--port` | `8090` | HTTP server listen port |
| `--threads` | `4` | Number of indexer threads (1 = single-threaded, >1 = thread pool) |
| `--workers` | (none) | Initial workers as `instance_id[:dp_rank]=zmq_address,...` pairs (dp_rank defaults to 0) |
| `--model-name` | `default` | Model name for initial `--workers` |
| `--tenant-id` | `default` | Tenant ID for initial `--workers` |
| `--peers` | (none) | Comma-separated peer indexer URLs for P2P recovery on startup |

### Shared Startup Gate

Set `DYN_ROUTER_MIN_INITIAL_WORKERS=<n>` to require at least `<n>` workers before the
standalone indexer, frontend push-router path, and KV router config-ready gate all proceed.
Leave it unset or set it to `0` to disable the startup wait.

## HTTP API

### `GET /health` — Liveness check

Returns `200 OK` unconditionally.

```bash
curl http://localhost:8090/health
```

### `GET /metrics` — Prometheus metrics

Returns metrics in Prometheus text exposition format. Available when the Python bindings are built with the `kv-indexer-metrics` feature.

```bash
curl http://localhost:8090/metrics
```

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `dynamo_kvindexer_request_duration_seconds` | Histogram | `endpoint` | HTTP request latency |
| `dynamo_kvindexer_requests_total` | Counter | `endpoint`, `method` | Total HTTP requests |
| `dynamo_kvindexer_errors_total` | Counter | `endpoint`, `status_class` | HTTP error responses (4xx/5xx) |
| `dynamo_kvindexer_models` | Gauge | — | Number of active model+tenant indexers |
| `dynamo_kvindexer_workers` | Gauge | — | Number of registered worker instances |
| `dynamo_kvindexer_listeners` | Gauge | `status` | Number of ZMQ listeners by status (`pending`, `active`, `paused`, `failed`) |

### `POST /register` — Register an endpoint

Register a ZMQ endpoint for an instance. Each call creates or reuses the indexer for the given `(model_name, tenant_id)` pair.
Registration is non-blocking: if the worker is not up yet, the listener is accepted in `pending` state and transitions to `active` once the initial ZMQ connection succeeds.

```bash
# Single model, default tenant
curl -X POST http://localhost:8090/register \
  -H 'Content-Type: application/json' \
  -d '{
    "instance_id": 1,
    "endpoint": "tcp://127.0.0.1:5557",
    "model_name": "llama-3-8b",
    "block_size": 16
  }'

# With tenant isolation
curl -X POST http://localhost:8090/register \
  -H 'Content-Type: application/json' \
  -d '{
    "instance_id": 2,
    "endpoint": "tcp://127.0.0.1:5558",
    "model_name": "llama-3-8b",
    "tenant_id": "customer-a",
    "block_size": 16,
    "dp_rank": 0
  }'
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `instance_id` | yes | — | Worker instance identifier |
| `endpoint` | yes | — | ZMQ PUB address to subscribe to |
| `model_name` | yes | — | Model name (used to select the indexer) |
| `block_size` | yes | — | KV cache block size (must match the engine) |
| `tenant_id` | no | `"default"` | Tenant identifier for isolation |
| `dp_rank` | no | `0` | Data parallel rank |
| `replay_endpoint` | no | — | ZMQ ROUTER address for gap replay (e.g. `tcp://host:5560`) |

### `POST /unregister` — Deregister an instance

Remove an instance. Omitting `tenant_id` removes the instance from **all** tenants for the given model; providing it targets only that tenant's indexer.

```bash
# Remove from all tenants
curl -X POST http://localhost:8090/unregister \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": 1, "model_name": "llama-3-8b"}'

# Remove from a specific tenant
curl -X POST http://localhost:8090/unregister \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": 1, "model_name": "llama-3-8b", "tenant_id": "customer-a"}'

# Remove a specific dp_rank
curl -X POST http://localhost:8090/unregister \
  -H 'Content-Type: application/json' \
  -d '{"instance_id": 1, "model_name": "llama-3-8b", "tenant_id": "default", "dp_rank": 0}'
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `instance_id` | yes | — | Worker instance to remove |
| `model_name` | yes | — | Model name (identifies the indexer) |
| `tenant_id` | no | — | Tenant identifier (omit to remove from all tenants) |
| `dp_rank` | no | — | Specific dp_rank to remove (omit to remove all) |

### `GET /workers` — List registered instances

```bash
curl http://localhost:8090/workers
```

Returns:
```json
[
  {
    "instance_id": 1,
    "source": "zmq",
    "status": "active",
    "endpoints": {
      "0": "tcp://127.0.0.1:5557",
      "1": "tcp://127.0.0.1:5558"
    },
    "listeners": {
      "0": {
        "endpoint": "tcp://127.0.0.1:5557",
        "status": "active"
      },
      "1": {
        "endpoint": "tcp://127.0.0.1:5558",
        "status": "active"
      }
    }
  },
  {
    "instance_id": 2,
    "source": "discovery",
    "status": "active",
    "endpoints": {},
    "listeners": {}
  }
]
```

For ZMQ-managed workers, `status` is aggregated across listeners with priority `failed > pending > active > paused`. Each listener entry may also expose a `last_error` field when the most recent startup or recv-loop attempt failed.

### `POST /query` — Query overlap for token IDs

Given raw token IDs, compute block hashes and return per-instance overlap scores (in matched tokens):

```bash
curl -X POST http://localhost:8090/query \
  -H 'Content-Type: application/json' \
  -d '{"token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], "model_name": "llama-3-8b"}'
```

Returns:
```json
{
  "scores": {"1": {"0": 32}, "2": {"1": 0}},
  "frequencies": [1, 1],
  "tree_sizes": {"1": {"0": 5}, "2": {"1": 3}}
}
```

Scores are in **matched tokens** (block overlap count × block size). Nested by `instance_id` then `dp_rank`.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `token_ids` | yes | — | Token sequence to query |
| `model_name` | yes | — | Model name (selects the indexer) |
| `tenant_id` | no | `"default"` | Tenant identifier |
| `lora_name` | no | — | LoRA adapter (overrides indexer-level lora_name for this query) |

### `POST /query_by_hash` — Query overlap for pre-computed hashes

```bash
curl -X POST http://localhost:8090/query_by_hash \
  -H 'Content-Type: application/json' \
  -d '{"block_hashes": [123456, 789012], "model_name": "llama-3-8b"}'
```

Same response format as `/query`. Scores are in matched tokens.

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `block_hashes` | yes | — | Pre-computed block hash array |
| `model_name` | yes | — | Model name (selects the indexer) |
| `tenant_id` | no | `"default"` | Tenant identifier |

### `GET /dump` — Dump all radix tree events

Returns the full radix tree state as a JSON object keyed by `model_name:tenant_id`:

```bash
curl http://localhost:8090/dump
```

Returns:
```json
{
  "llama-3-8b:default": {
    "block_size": 16,
    "events": [<RouterEvent>, ...]
  },
  "mistral-7b:customer-a": {
    "block_size": 16,
    "events": [<RouterEvent>, ...]
  }
}
```

Each indexer is dumped concurrently. The `block_size` field lets recovering peers create indexers with the correct block size without requiring `--block-size` on every replica.

### `POST /register_peer` — Register a peer indexer

```bash
curl -X POST http://localhost:8090/register_peer \
  -H 'Content-Type: application/json' \
  -d '{"url": "http://peer:8091"}'
```

### `POST /deregister_peer` — Remove a peer indexer

```bash
curl -X POST http://localhost:8090/deregister_peer \
  -H 'Content-Type: application/json' \
  -d '{"url": "http://peer:8091"}'
```

### `GET /peers` — List registered peers

```bash
curl http://localhost:8090/peers
```

Returns:
```json
["http://peer:8091"]
```

## DP Rank Handling

When a worker registers with the standalone KV indexer (`/register`), it provides an `instance_id`, a ZMQ `endpoint`, and an optional `dp_rank` (defaults to 0). The service spawns one ZMQ listener per registration.

Each incoming `KvEventBatch` may carry an optional `data_parallel_rank` field. If present, it **overrides** the statically-registered `dp_rank` for that batch. This allows a single ZMQ port to multiplex events from multiple DP ranks.

**Caveat**: the registry only tracks dp_ranks from explicit `/register` calls. If an engine dynamically emits batches with a dp_rank that was never registered, the indexer will store those blocks correctly (under the dynamic `WorkerWithDpRank` key), but per-dp_rank deregistration (`/unregister` with `dp_rank`) will not find them. Full-instance deregistration (`/unregister` without `dp_rank`) still cleans up all dp_ranks for a given `worker_id` in the tree via `remove_worker`.

## Gap Detection and Replay

ZMQ PUB/SUB is lossy — messages can be dropped under backpressure or brief disconnects. The indexer detects gaps by tracking the sequence number of each batch: if `seq > last_seq + 1`, a gap is detected.

When a `replay_endpoint` is provided during `/register`, the indexer connects a DEALER socket to the engine's ROUTER socket and requests the missing batches by sequence number. The engine streams back buffered `(seq, payload)` pairs from its ring buffer until an empty-payload sentinel.

If no `replay_endpoint` is configured, gaps are logged as warnings but not recovered.

The sequence counter (`last_seq`) persists across unregister/register cycles, so re-registering a worker after a gap will trigger replay on the first batch received by the new listener.

## Limitations

- **Standalone mode is ZMQ only**: Workers must publish KV events via ZMQ PUB sockets.
- **No routing logic**: The indexer only maintains the radix tree and answers queries. It does not track active blocks, manage request lifecycle, or perform worker selection.

## Architecture

### Standalone Mode

```mermaid
graph TD
    subgraph Workers
        W1[Worker 1<br/>ZMQ PUB]
        W2[Worker 2<br/>ZMQ PUB]
    end

    subgraph "Standalone Indexer (HTTP)"
        REG[Worker Registry]
        ZMQ[ZMQ SUB Listeners]
        IDX["Indexer Map<br/>(model, tenant) → Radix Tree"]
        HTTP[HTTP API<br/>/query /dump /register /health]
    end

    CLIENT[External Client]

    W1 -->|ZMQ events| ZMQ
    W2 -->|ZMQ events| ZMQ
    CLIENT -->|POST /register| REG
    REG -->|spawn listeners| ZMQ
    ZMQ -->|apply events| IDX
    CLIENT -->|POST /query, GET /dump| HTTP
    HTTP -->|query| IDX

    style W1 fill:#f3e5f5,stroke:#333,color:#333
    style W2 fill:#f3e5f5,stroke:#333,color:#333
    style IDX fill:#2e8b57,stroke:#333,color:#fff
    style ZMQ fill:#2e8b57,stroke:#333,color:#fff
    style REG fill:#2e8b57,stroke:#333,color:#fff
    style HTTP fill:#2e8b57,stroke:#333,color:#fff
    style CLIENT fill:#fff3e0,stroke:#333,color:#333
```

### P2P Recovery Flow

```mermaid
sequenceDiagram
    participant B as Replica B (new)
    participant A as Replica A (healthy)
    participant W as Workers (ZMQ PUB)

    B->>W: Connect ZMQ SUB sockets
    Note over B,W: 1s delay for peer tree to advance past connection point
    B->>A: GET /dump
    A-->>B: Radix tree snapshot + block sizes
    Note over B: Apply dump events
    Note over B: Unblock ZMQ listeners
    B->>W: Start draining buffered events
    Note over B: Ready to serve queries
```

## See Also

- **[Mooncake KV Indexer RFC](https://github.com/kvcache-ai/Mooncake/issues/1403)**: Community API standardization for KV cache indexers
- **[Router Guide](router-guide.md)**: Full KV router configuration and tuning
- **[Router Design](../../design-docs/router-design.md)**: Architecture and event transport modes
- **[Standalone Router](../../../components/src/dynamo/router/README.md)**: Full routing service (routes requests to workers)
