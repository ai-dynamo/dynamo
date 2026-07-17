---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Standalone Selection Service
subtitle: Select workers and account for reservations without forwarding inference requests
---

## Overview

The standalone selection service (`python -m dynamo.select_service`) exposes the
KV router's worker selection and active-load accounting over HTTP. It does not
forward model requests or own response streams. External runtimes such as Ray
register their worker catalog, request a selection, contact the selected worker,
and report the reservation lifecycle.

The service combines:

- KV overlap indexing from worker ZMQ events.
- KV-aware and load-aware worker selection.
- Explicit or atomic selection and reservation.
- Best-effort active-load synchronization between selector replicas.
- Startup KV index recovery from another selector or standalone indexer.

## Build And Launch

Build the Python bindings with the `select-service` feature:

```bash
cd lib/bindings/python
VIRTUAL_ENV=../../../.venv ../../../.venv/bin/maturin develop --uv --features select-service
```

Launch the service from the repository root:

```bash
.venv/bin/python -m dynamo.select_service --port 8092
```

The service binds to `0.0.0.0` and does not provide authentication. Run it on a
trusted internal network or place it behind an appropriate network policy.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8092` | HTTP server port. |
| `--threads` | `4` | KV indexer worker threads. |
| `--indexer-peers` | none | Comma-separated HTTP URLs used for startup KV recovery through `/dump`. |
| `--replica-sync-port` | none | Local ZMQ PUB port for active-load lifecycle events. The selector binds `tcp://*:<port>` internally. |
| `--replica-sync-peers` | none | Comma-separated ZMQ PUB endpoints for selector peers. Requires `--replica-sync-port`. |

Router scheduling behavior continues to use the standard Dynamo router
environment configuration.

## Worker Registration

Every selector replica must receive the same worker catalog before it serves
selection traffic. Replica traffic never creates workers.

```http
POST /workers
Content-Type: application/json

{
  "worker_id": 1,
  "model_name": "model",
  "tenant_id": "default",
  "endpoint": "http://worker:8000",
  "block_size": 16,
  "data_parallel_start_rank": 0,
  "data_parallel_size": 2,
  "kv_events_endpoints": {
    "0": "tcp://worker:5557",
    "1": "tcp://worker:5558"
  },
  "replay_endpoint": "tcp://worker:5560"
}
```

`POST /workers` returns `201`. `PATCH /workers/{worker_id}` updates supplied
fields, `DELETE /workers/{worker_id}` removes the worker, and `GET /workers`
lists catalog state. `model_name` and `tenant_id` scope all selection, indexer,
and load state; both default to `"default"` when omitted.

`GET /health` is process liveness. `GET /ready` returns `200` only after at
least one worker is schedulable, otherwise `503` with lifecycle details.

## Selection API

### `POST /select`

Select a worker without booking active load:

```json
{
  "selection_id": "select-123",
  "model_name": "model",
  "tenant_id": "default",
  "block_hashes": [11, 12, 13, 14, 15, 16, 17, 18],
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512
}
```

### `POST /select_and_reserve`

Select and atomically book load in the receiving selector process. Supply a
globally unique `reservation_id`, or allow the service to generate one:

```json
{
  "selection_id": "select-123",
  "reservation_id": "request-123",
  "model_name": "model",
  "tenant_id": "default",
  "block_hashes": [11, 12, 13, 14, 15, 16, 17, 18],
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512
}
```

Both endpoints return the same selection shape:

```json
{
  "selection_id": "select-123",
  "reservation_id": "request-123",
  "model_name": "model",
  "tenant_id": "default",
  "worker_id": 1,
  "dp_rank": 0,
  "endpoint": "http://worker:8000",
  "block_size": 16,
  "overlap": {
    "longest_matched": 128,
    "gpu": 64,
    "dp": {"0": 64, "1": 32},
    "cpu": 96,
    "disk": 128
  },
  "effective_prefill_tokens": 384
}
```

`selection_id` and `reservation_id` are omitted when absent. All `overlap`
values are matched token counts. `gpu`, `cpu`, and `disk` use the cumulative
Mooncake tier semantics documented in the standalone indexer's
[per-instance tier breakdown](standalone-indexer.md#per-instance-tier-breakdown).
A zero-overlap response includes the selected `dp_rank` with value `0`.

The overlap summary is raw observability. `effective_prefill_tokens` is the
authoritative weighted prefill-load value computed by the same cache-credit
formula used for scheduler booking. It is not derived from `longest_matched`.

The previous public fields `cached_tokens` and `effective_overlap_blocks` have
been removed. Their values remain internal scheduler inputs.

## Ray Select-Then-Reserve Flow

Ray can keep model invocation separate from selector admission:

1. Call `POST /select`.
2. Send the request to the returned `endpoint` and `dp_rank`.
3. Call `POST /reservations` with a globally unique reservation ID, selected
   worker identity, the same prompt representation, and the returned
   `effective_prefill_tokens`.
4. Report prefill completion and request completion through the lifecycle API.

```http
POST /reservations
Content-Type: application/json

{
  "reservation_id": "request-123",
  "model_name": "model",
  "tenant_id": "default",
  "worker_id": 1,
  "dp_rank": 0,
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512,
  "effective_prefill_tokens": 384
}
```

When supplied, `effective_prefill_tokens` is authoritative and directly enables
prefill-load tracking. It must not exceed the normalized input sequence length.
When omitted, existing router configuration controls prefill tracking. The
reservation API does not accept or derive accounting from overlap fields.

## Reservation Lifecycle

```http
POST /reservations/{reservation_id}/prefill_complete
POST /reservations/{reservation_id}/output_block
DELETE /reservations/{reservation_id}
```

`prefill_complete` clears active prefill load. `output_block` updates only the
receiving selector's local decode-block accounting and accepts an optional
`decay_fraction` in `[0.0, 1.0]`. `DELETE` frees the reservation.

**NOTE:** Output-block updates are intentionally not replica-synchronized.
They can occur at high frequency, and broadcasting them would consume
disproportionate network bandwidth.

## Peer Planes

The selector has two independent peer configurations:

| Plane | Transport | Flags | Purpose |
|-------|-----------|-------|---------|
| Indexer recovery | HTTP | `--indexer-peers` | Fetch a compatible `/dump` during startup and replay KV events into the local indexer. |
| Replica synchronization | ZMQ | `--replica-sync-port`, `--replica-sync-peers` | Share admission, prefill-complete, and free events by model and tenant. |

Example:

```bash
.venv/bin/python -m dynamo.select_service \
  --port 8092 \
  --indexer-peers http://selector-b:8092 \
  --replica-sync-port 9092 \
  --replica-sync-peers 'tcp://selector-b:9092'
```

Configure the reverse peer direction on selector B for bidirectional lifecycle
synchronization. `GET /dump` exposes the selector's current indexer snapshot in
the same recovery format as the standalone indexer.

Replica-sync peers may also be changed without restarting the selector:

```http
POST /replica_sync/register_peer
Content-Type: application/json

{"endpoint":"tcp://selector-b:9092"}
```

The same body is accepted by `POST /replica_sync/deregister_peer`.
`GET /replica_sync/peers` returns the sorted configured endpoints. Dynamic
membership is in-memory; after restart, only peers supplied through
`--replica-sync-peers` are restored. These routes only manage live ZMQ
replica-sync peers. They do not alter the HTTP indexer-recovery peers.

## Consistency Invariants

- Replica synchronization is bounded and best-effort. Delays, reordering,
  dropped events, and temporary active-load divergence are accepted.
- There is no sequencing, acknowledgement, replay, backpressure, or
  resynchronization for replica lifecycle events.
- Unknown worker, model, tenant, DP-rank, and block-size events are dropped.
  Register the same worker catalog on every selector before routing traffic.
- Admission, prefill-complete, and free are synchronized. Output-block growth
  remains local to avoid excessive network bandwidth.
- Startup recovery waits for recovered events to be submitted to the indexer,
  not for complete processing. Early selections may temporarily miss recovered
  KV state.
- `/select` followed by `/reservations` provides eventual, not atomic,
  cross-replica admission. Use `/select_and_reserve` for atomic local booking.
- Reservation IDs must be globally unique. Existing conflict and retry
  semantics are unchanged; no idempotency ledger is added.

## Inspection APIs

- `GET /loads` returns active-load snapshots, optionally filtered by
  `model_name` and `tenant_id`.
- `POST /potential_loads` estimates worker load for a prompt without selection.
- `POST /overlap_scores` returns per-worker/per-rank tiered overlap rows.
- `GET /dump` returns the compatible indexer recovery snapshot.
