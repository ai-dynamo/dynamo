---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Standalone Slot Tracker
subtitle: Run active-request load accounting as an independent HTTP service
---

## Overview

The standalone slot tracker (`python -m dynamo.slot_tracker`) exposes the KV router's
active-request accounting as a small HTTP service. It is runtime-independent: consumers
register workers manually, submit request lifecycle events, and read advisory load
snapshots for their own routing decisions.

The service accepts ordered final chained sequence hashes, one hash per prompt block.
Hashes are serialized as signed 64-bit JSON integers and reinterpreted bit-for-bit as internal
unsigned hashes. Send hashes rather than prompt tokens.

The service optionally replicates lifecycle events directly between slot-tracker processes
over ZMQ. It still excludes metrics, discovery-based worker registration, output block
updates, persistence, and peer recovery.

## Build And Launch

Build the Python bindings with the `slot-tracker` feature:

```bash
cd lib/bindings/python
VIRTUAL_ENV=../../../.venv ../../../.venv/bin/maturin develop --uv --features slot-tracker
```

Launch the service:

```bash
.venv/bin/python -m dynamo.slot_tracker --port 8091
```

Enable replica synchronization:

```bash
.venv/bin/python -m dynamo.slot_tracker \
  --port 8091 \
  --replica-sync-port 8092 \
  --replica-sync-peers 'tcp://slot-tracker-b:8092'
```

The default port is `8091`. `GET /health` returns `200 OK` with an empty body as soon as
the HTTP listener is ready. This endpoint is liveness-only. After a restart the registry
is empty; consumers must re-register workers and replay active requests if they need
restored accounting.

The service binds to `0.0.0.0` and does not provide authentication. Run it on a trusted
internal network or place it behind an appropriate network policy.

`tenant_id` partitions load state but does not authenticate the caller.

## Replica Synchronization

`--replica-sync-port` enables a ZMQ PUB endpoint and replica-event consumption. The
service binds `tcp://*:<port>` internally and generates an ephemeral process identity;
neither the bind expression nor process identity is a configuration parameter.
`--replica-sync-peers` accepts a comma-separated list of peer PUB endpoints and requires
`--replica-sync-port`.

Peer connections are directional. For bidirectional synchronization, configure each
replica with the other replica's reachable ZMQ endpoint. All replicas must independently
receive the same worker registrations before lifecycle traffic begins. A replica event
for an unknown `(model_name, routing_group)`, block size, worker ID, or DP rank is dropped.
Replica traffic never creates workers.

The transport is bounded and best-effort. A full queue drops new events rather than
blocking HTTP operations, and ZMQ subscription setup may lose events sent immediately
after peer registration. There is no acknowledgement, replay, snapshot, or gap recovery.
`/add`, `/prefill_complete`, and `/free` should normally remain sticky to the replica
that accepted `/add`; replica synchronization provides advisory peer state rather than
cross-replica lifecycle ownership.

> [!WARNING]
> The replica envelope uses `routing_group` on the existing v1 topic and is not compatible
> with binaries that send `tenant_id`. Drain lifecycle traffic, stop all connected replicas,
> upgrade them together, re-register worker catalogs, and then resume traffic. Active advisory
> state is not migrated.

Peers may also be managed dynamically:

```http
POST /replica_sync/register_peer
Content-Type: application/json

{"endpoint":"tcp://slot-tracker-b:8092"}
```

The same body is accepted by `POST /replica_sync/deregister_peer`.
`GET /replica_sync/peers` returns the sorted configured endpoints. Registration confirms
that the endpoint was accepted by the local SUB socket, not that the asynchronous ZMQ
subscription handshake has completed. Dynamic membership is in-memory; after restart,
only peers supplied through `--replica-sync-peers` are restored.

## Common Responses

Successful topology and lifecycle writes return:

```json
{"status": "ok"}
```

Errors, including malformed JSON, oversized JSON bodies, unknown routes, and unsupported
methods, return:

```json
{"error": "concise description"}
```

`routing_group` defaults to `"default"` when omitted. Request bodies use Axum's default
bounded JSON handling.

## Topology API

### `POST /register`

Register one contiguous data-parallel range:

```json
{
  "worker_id": 7,
  "model_name": "llama-3-8b",
  "routing_group": "default",
  "block_size": 16,
  "dp_start": 0,
  "dp_size": 2
}
```

Returns `201`. `block_size` and `dp_size` must be positive, and the DP range must not
overflow. Workers in the same `(model_name, routing_group)` tracker must use the same block
size. Worker IDs are scoped by `(model_name, routing_group)`.

### `POST /unregister`

Remove a worker's full DP range and active requests immediately:

```json
{
  "worker_id": 7,
  "model_name": "llama-3-8b",
  "routing_group": "default"
}
```

Returns `200`, or `404` if the registration does not exist.

### `GET /workers`

List workers with independent optional `model_name` and `routing_group` filters:

```json
[
  {
    "worker_id": 7,
    "model_name": "llama-3-8b",
    "routing_group": "default",
    "block_size": 16,
    "dp_start": 0,
    "dp_size": 2
  }
]
```

The response is sorted for stable inspection.

## Lifecycle API

### `POST /add`

Record prompt blocks on a registered worker rank:

```json
{
  "model_name": "llama-3-8b",
  "routing_group": "default",
  "request_id": "req-123",
  "worker_id": 7,
  "dp_rank": 0,
  "sequence_hashes": [101, -22, 303],
  "new_isl_tokens": 48
}
```

Returns `201`. `sequence_hashes` is required and may be empty. `new_isl_tokens` defaults
to `0`; positive values enable prefill-token accounting. Duplicate request IDs return
`409`. Unknown trackers or worker ranks return `404`.

### `POST /prefill_complete`

Mark prompt processing complete:

```json
{
  "model_name": "llama-3-8b",
  "routing_group": "default",
  "request_id": "req-123"
}
```

Returns `200` for an active request. Repeated completion is a no-op. Unknown requests
return `404`.

### `POST /free`

Release prompt blocks and any remaining prefill state:

```json
{
  "model_name": "llama-3-8b",
  "routing_group": "default",
  "request_id": "req-123"
}
```

Returns `200`. Free is idempotent while the model/routing-group tracker exists, including for
an unknown request. Unknown trackers return `404`.

Lifecycle writes preserve the core slot tracker's arrival ordering. Consumers should
normally wait for `/add` success before sending later lifecycle writes. The service does
not repair reordered delivery: an early unknown `/free` or `/prefill_complete` is
forgotten, so a later `/add` may remain accounted until a later free or expiry. A request
older than 300 seconds may be removed by inherited stale-request cleanup.

## Load API

### `GET /loads`

Read current load snapshots with independent optional `model_name` and `routing_group`
filters:

```json
[
  {
    "model_name": "llama-3-8b",
    "routing_group": "default",
    "worker_id": 7,
    "dp_rank": 0,
    "active_prefill_tokens": 48,
    "active_decode_blocks": 3
  }
]
```

The response is sorted for stable inspection.

### `POST /potential_loads`

Project the loads for a new request:

```json
{
  "model_name": "llama-3-8b",
  "routing_group": "default",
  "sequence_hashes": [101, -22, 303, 404],
  "new_isl_tokens": 48
}
```

Returns:

```json
[
  {
    "worker_id": 7,
    "dp_rank": 0,
    "potential_prefill_tokens": 96,
    "potential_decode_blocks": 4,
    "active_requests": 2
  }
]
```

Projection response order is unspecified to keep the routing read path lean. `/loads`
and `/potential_loads` are advisory snapshots, not reservations. A selected worker may
disappear before `/add`; recompute after `/add` returns `404`. An ambiguous `/add`
timeout is also consumer-owned: automatically retrying the same request is not
guaranteed safe because duplicate adds return `409`.
