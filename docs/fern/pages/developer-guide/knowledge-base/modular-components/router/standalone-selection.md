---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

## Embedded Rust API

Use `SelectionServiceBuilder` to embed selection without the HTTP server. The
resulting `SelectionService` owns worker registration, KV-event listeners,
indexer recovery, replica synchronization, and shutdown. It exposes the same
worker, selection, bookkeeping, inspection, peer-membership, and recovery
operations as the standalone HTTP service.

`SelectionCore::try_new_local` creates an intentionally unsynchronized core for
tests and local-only use while reporting invalid tracking-hash configuration.
Production integrations should use `SelectionServiceBuilder` so startup recovery,
readiness, and background-task lifecycle remain consistent with the standalone service.

To inject native Rust scorers and a picker while retaining those service-owned capabilities, see [Write Custom Routing Strategies](custom-worker-selection.mdx).

The C and Go bindings do not currently expose `SelectionService`. An EPP
integration requires separate FFI lifecycle, error-mapping, worker, and peer
APIs. Those bindings should wrap `SelectionService` rather than construct
`SelectionCore` directly.

### CLI

| Setting | Default | Description |
|------|---------|-------------|
| `--port` | `8092` | HTTP server port. |
| `--threads` | `4` | KV indexer worker threads. |
| `--indexer-peers` | none | Comma-separated HTTP URLs used for startup KV recovery through `/dump`. Ignored with `--remote-indexer-url`. |
| `--remote-indexer-url` | none | Base URL of a [standalone indexer](standalone-indexer.md) that serves the primary KV index. The selector then does not subscribe to worker KV events; workers publish to the indexer instead. Requires `use_kv_events=true`. |
| `--replica-sync-port` | none | Local ZMQ PUB port for active-load lifecycle events. The selector binds `tcp://*:<port>` internally. |
| `--replica-sync-peers` | none | Comma-separated ZMQ PUB endpoints for selector peers. Requires `--replica-sync-port`. |
| `--session-affinity-ttl-secs` | none | Pin each request `session_id` to the worker that served it for this long after its last request (1 to 31,536,000 seconds). When `--replica-sync-port` is configured, bindings replicate to peers over the replica mesh (`dynamo.session-affinity.v1` topic). |
| `--selection-cache-ttl-secs` | `120` | Seconds an unclaimed pending selection lives before eviction. |
| `--selection-cache-max-entries` | `4096` | Maximum resident pending selections, evicting oldest first. |
| `--selection-cache-max-bytes` | `268435456` | Approximate byte budget across resident pending selections. |
| `--router-tracking-hash` | environment/default | Override the tracking algorithm with `public-xxh3-v1` or experimental `keyed-xxh3-v1`. |
| `--router-tracking-key-file` | environment/none | Override the path to the 32-byte provider key file. |
| `--router-tracking-key-id` | environment/none | Override the provider-managed key epoch. |
| `DYN_ROUTER_ACTIVE_REQUEST_EXPIRY_SECS` | `300` | Override the absolute request age at which the standalone slot tracker may reclaim stale active state. |

Router scheduling behavior continues to use the standard Dynamo router
environment configuration.

### KV Indexing Modes

The service resolves its indexer shape from the same router configuration the
frontend uses, so the two index the same way:

| Configuration | Primary indexer | Routing decisions |
|---|---|---|
| `use_kv_events=true` (default) | Event-driven from worker ZMQ events | Not recorded |
| `use_kv_events=true`, `DYN_ROUTER_PREDICTED_TTL_SECS` set | Event-driven | Recorded into a short-TTL side indexer merged into device scores by per-worker max |
| `use_kv_events=false` | Approximate: no events, entries expire after `router_ttl_secs` | Recorded into the primary |
| `--remote-indexer-url` (or `SelectionServiceBuilder::kv_index(KvIndexSource::Remote(..))`) | Served by a standalone indexer, queried through `POST /query_tiered_by_hash` | Recorded into the side indexer when `DYN_ROUTER_PREDICTED_TTL_SECS` is set, else not recorded |

With a remote primary, worker records do not need `kv_events_endpoint(s)` to
become schedulable, no ZMQ listener is started in the selector, and `/dump`
returns no primary events (peers recover from the indexer instead). A remote
lookup failure is reported as an offline indexer for that selection.

A routing decision is recorded when a reservation is booked (`select_and_reserve`,
or `select` followed by `POST /reservations`), never for a query-only `select`.
Hash-only reservations that supply `sequence_hashes` without `block_hashes` or
`token_ids` are not recorded. `router_approximate_cache_policy=lru` is rejected at startup when
`use_kv_events=true`; with `use_kv_events=false` it falls back to TTL retention
with a warning.

The standalone expiry guard measures absolute age from admission; output progress does not refresh
it. Periodic cleanup therefore reclaims stale state approximately five to six minutes after
admission by default. The embedded `KvRouter` uses the same `300`-second value as its shared
request-liveness CLOCK scan interval and reclaims an idle lease approximately five to ten minutes
after the last progress touch.

## Worker Registration

Every selector replica must receive the same worker catalog before it serves
selection traffic. Replica traffic never creates workers. When the catalog comes
from a discovery source (Kubernetes, a workers file), a snapshot whose apply
fails is retried after 5 seconds unless a newer snapshot arrives first, so a
stale worker can persist for that long after a failed pass.

```http
POST /workers
Content-Type: application/json

{
  "worker_id": 1,
  "model_name": "model",
  "routing_group": "default",
  "endpoint": "http://worker:8000",
  "block_size": 16,
  "data_parallel_start_rank": 0,
  "data_parallel_size": 2,
  "kv_events_endpoints": {
    "0": "tcp://worker:5557",
    "1": "tcp://worker:5558"
  }
}
```

`replay_endpoint` (KV event gap recovery) is accepted only when
`data_parallel_size` is 1; a multi-rank worker that sets it stays `incomplete`
because the replay protocol carries no rank.

`worker_id` is service-wide, not scoped by model or routing group. `POST /workers`
is an upsert and returns `201`: reusing an existing ID replaces its catalog
record. If the model or routing group changes, the worker is removed from the
previous partition and moved to the new one, which can leave the previous
partition not ready. Assign a unique ID to every live worker across the entire
service.

A worker that can consume router hints registers `router_hint_worker_type`
(its backend role, used to match hint sources to targets) and, when it can also
serve as a source, `router_hint_source_control_endpoints` keyed by global DP
rank:

```json
{
  "worker_id": 1,
  "router_hint_worker_type": "decode",
  "router_hint_source_control_endpoints": { "0": "tcp://worker:5600", "1": "tcp://worker:5601" }
}
```

A worker whose KV events come from a state agent sets
`"kv_event_source_mode": "state_agent_v2"`; it can receive hints but is never
chosen as a hint source.

`PATCH /workers/{worker_id}` updates supplied fields, `DELETE
/workers/{worker_id}` removes the worker, and `GET /workers` lists catalog
state. `model_name` and `routing_group` scope selection, indexer, and load state;
both default to `"default"` when omitted.

`GET /health` is process liveness. `GET /ready` returns `200` only after at
least one worker is schedulable, otherwise `503` with lifecycle details.

### Worker Restarts

When a worker or its cache-event publisher restarts, update its registration on every
selector replica. A reconnect at the same address does not reset the existing event
sequence watermark or invalidate old cache ownership.

With KV events enabled, `POST /workers` or `PATCH /workers/{worker_id}` for an
existing schedulable worker reconciles its indexer registration rank by rank: a rank
whose event endpoint and replay endpoint are unchanged keeps its listener, its
sequence watermark, and its indexed blocks, so capacity or topology updates do not
drain the cache view. A rank whose endpoint changed gets a new listener with a fresh
watermark. Because an unchanged update keeps the old watermark, it does not recover a
publisher that restarted at the same address: for that, call
`DELETE /workers/{worker_id}`, wait for the response, then `POST /workers` with the
current record, or register the restarted publisher under a new `kv_events_endpoint`.
Supply the complete current worker record to `POST /workers` because it replaces the
catalog record. Serialize lifecycle operations for the same worker.

This differs from the standalone indexer's `/register`, which rejects duplicate
registrations and requires `/unregister` first. See
[Indexer Worker Restarts](standalone-indexer.md#worker-restarts) for cleanup and
replay limits. Neither a successful catalog update nor `/ready` certifies a complete
cache view: missed startup events must still be recovered, and expired replay
history requires another source of complete state.

## Selection API

### `POST /select`

Select a worker without booking active load:

```json
{
  "selection_id": "select-123",
  "model_name": "model",
  "routing_group": "default",
  "block_hashes": [11, 12, 13, 14, 15, 16, 17, 18],
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512,
  "session_id": "session-abc"
}
```

### `POST /select_and_reserve`

Select and atomically book load in the receiving selector process. Supply a
globally unique `selection_id`, or allow the service to generate one:

```json
{
  "selection_id": "select-123",
  "model_name": "model",
  "routing_group": "default",
  "block_hashes": [11, 12, 13, 14, 15, 16, 17, 18],
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512,
  "session_id": "session-abc"
}
```

`select_and_reserve` returns the selected worker and the normalized booking inputs:

```json
{
  "selection_id": "select-123",
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512,
  "track_prefill_tokens": true,
  "model_name": "model",
  "routing_group": "default",
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
  "effective_prefill_tokens": 384,
  "potential_decode_blocks": 212,
  "decode_busy": false
}
```

`select` returns the same selection fields but omits `sequence_hashes`, `isl_tokens`, and
`track_prefill_tokens`. `selection_id` is omitted when absent.

`potential_decode_blocks` is the scheduler's projection of KV blocks on the
chosen worker once this request is decoding, including the request's own
blocks. `decode_busy` compares it against the worker's `total_kv_blocks` at
`conditional_disagg_decode_busy_threshold` and is omitted when either the
threshold or the capacity is unknown.

A booking response may also carry a `kv_hint` when another worker of the same
`router_hint_worker_type` (the worker capability key is `router_hint`) holds a
longer cached prefix than the chosen worker and advertises a source control
endpoint for the matching DP rank. The hint is a versioned action envelope
whose `message_id` is the `selection_id`:

```json
{
  "kv_hint": {
    "protocol_version": "0.1",
    "message_id": "select-123",
    "actions": [
      {
        "action_id": "a1",
        "action_type": "kv.fetch",
        "action_version": "1.0",
        "payload": {
          "source_control_endpoint": "tcp://worker-1:5600",
          "block_hashes": [8713492873, 1928374650]
        }
      }
    ]
  }
}
```

`block_hashes` are root-aligned external sequence hashes; entry `i` is request
block `i`, and the target decides which suffix to fetch from the source. Hints
require a local event-driven primary indexer (no approximate or remote primary)
and are never attached to query-only `select` responses.

### Advisory selection

`POST /select` accepts `"advisory": true` to select from current scheduler
state without queue admission. The request never waits in the router queue,
and the response adds the chosen worker's projected load:

```json
{
  "worker_id": 1,
  "dp_rank": 0,
  "potential_decode_blocks": 212,
  "decode_busy": false,
  "worker_load": {
    "active_prefill_tokens": 2048,
    "prefill_token_capacity": 8192,
    "total_kv_blocks": 4096,
    "prefill_busy": false
  }
}
```

`prefill_busy` compares `active_prefill_tokens` against
`prefill_token_capacity` at `conditional_disagg_prefill_busy_threshold` and is
omitted when that threshold is unset. This is the probe a disaggregation
coordinator uses to decide whether to bypass remote prefill: one advisory
`select` against the prefill pool answers "would the prefill worker I'd get be
busy?" without a separate load read. `advisory` is ignored by
`select_and_reserve`, which always books. All `overlap`
values are matched token counts. `gpu`, `cpu`, and `disk` use the cumulative
Mooncake tier semantics documented in the standalone indexer's
[per-instance tier breakdown](standalone-indexer.md#per-instance-tier-breakdown).
A zero-overlap response includes the selected `dp_rank` with value `0`.

The overlap summary is raw observability. `effective_prefill_tokens` is the
authoritative weighted prefill-load value computed by the same cache-credit
formula used for scheduler booking. It is not derived from `longest_matched`.
When a request waits in the scheduler queue, both fields reflect the final
overlap inputs after any dequeue-time refresh rather than the enqueue-time
snapshot.

The previous public fields `cached_tokens` and `effective_overlap_blocks` have
been removed. Their values remain internal scheduler inputs.

When a request supplies `token_ids`, the selector derives tracking hashes with
its configured tracking-hash context while retaining public hashes for indexer
lookups. Requests that instead supply `block_hashes`, `sequence_hashes`, and
`isl_tokens` remain trusted precomputed inputs. The service does not reject,
rewrite, or label those identities in keyed mode. Configure every precomputed
hash producer with the same algorithm, key, and key ID as the selector.

### `session_id`

Both `POST /select` and `POST /select_and_reserve` accept an optional `session_id` string. With `--session-affinity-ttl-secs` enabled, each model and routing group has its own binding table. `/select_and_reserve` binds a new session to the selected worker and holds its lease until the reservation is released or expires. The idle TTL starts when the last lease is released. `/select` can use an existing binding but does not create one.

Bindings replicate over the configured replica mesh. An explicit `affinity_target` or `pinned_worker` takes precedence over the session binding. Without a configured TTL, `session_id` is only policy input: custom policies can read it through `WorkerSelectionContext::session_id()`, while the built-in selector ignores it. See [Write Custom Routing Strategies](custom-worker-selection.mdx).

The pending-selection cache keeps the session metadata, so a later `POST /reservations` for that selection binds the session to the booked worker, the same as `/select_and_reserve` does. The frontend uses the same table implementation for request-header affinity; see [Configuration and Tuning](configuration-and-tuning.md).

### `session_context`

Both endpoints also accept an optional `session_context` object that carries
the full session metadata the frontend hands to worker selection. When it is
present, the flat `session_id` field is ignored.

```json
{
  "token_ids": [1, 2, 3, 4],
  "session_context": {
    "session_id": "child-session",
    "parent_session_id": "root-session",
    "session_final": false,
    "input_trigger": "tool_result"
  }
}
```

Only `session_id` is required inside the object. `input_trigger` is one of `user_message`, `tool_result`, or `other`. The affinity table uses this session ID when a TTL is configured. Custom policies can read the remaining values through `WorkerSelectionContext::session_context()`.

## Ray Select-Then-Reserve Flow

Ray can keep model invocation separate from selector admission:

1. Call `POST /select` with a `selection_id`.
2. Send the request to the returned `endpoint` and `dp_rank`.
3. Call `POST /reservations` using the cached selection replay form, or the
   explicit form with the full worker identity and prompt.
4. Report prefill completion and request completion through the lifecycle API.

### Cached selection replay form

A `/select` that carries a `selection_id` caches its booking inputs (the
chosen worker, the normalized prompt, `effective_prefill_tokens`,
`expected_output_tokens`, and the prefill-tracking decision) on the selector
that served it. A reservation that passes the same `selection_id`,
`model_name`, and `routing_group` replays the cached selection, booked under
that `selection_id`, without re-sending the prompt:

```http
POST /reservations
Content-Type: application/json

{
  "selection_id": "select-123",
  "model_name": "model",
  "routing_group": "default"
}
```

- **Id namespace**: `selection_id` is client-chosen and scoped per
  `(model_name, routing_group)`; use a distinct id per in-flight select. A new
  `select` reusing a pending id replaces it (latest wins), and an explicit
  booking discards the cached selection for its `selection_id`.
- **Required id**: `selection_id` is always required (the replay key and the
  booking id); a request without it is rejected.
- **Single-use**: The first successful booking consumes the entry; a repeat
  replay returns `404` (`no pending selection`). Concurrent replays of the same
  id collide at the scheduler, so only one books.
- **Retryable on failure**: A booking that fails before landing (worker no
  longer schedulable, service not ready) leaves the entry in place, so the same
  call can be retried once the condition clears.
- **Bounded window**: By default, entries expire after 120 seconds and each
  selector retains at most 4096 pending selections within a 256 MiB budget,
  evicting oldest first. All three limits are configurable.
- **Replica-local**: The cache lives in the selector process that served the
  `/select`. With multiple selector replicas, route the reservation to the
  same replica or use the explicit form.
- **Pure replay**: The booking uses exactly what `select` captured; other
  request fields are ignored. Supplying `worker_id` switches to the explicit
  form.

On any miss (expired, already consumed, wrong model or routing group, or a
different replica) the call returns `404`; fall back to the explicit form.

### Explicit form

The self-contained form carries the worker identity and prompt and needs no
cached selection; it wins whenever `worker_id` is present. It discards the
cached selection for its `selection_id`, so a later replay of the same id
cannot book stale state:

```http
POST /reservations
Content-Type: application/json

{
  "selection_id": "request-123",
  "model_name": "model",
  "routing_group": "default",
  "worker_id": 1,
  "dp_rank": 0,
  "sequence_hashes": [21, 22, 23, 24, 25, 26, 27, 28],
  "isl_tokens": 512,
  "effective_prefill_tokens": 384,
  "track_prefill_tokens": true
}
```

When supplied, `effective_prefill_tokens` is authoritative and directly enables
prefill-load tracking unless `track_prefill_tokens` is `false`. It must not exceed the normalized
input sequence length. When `track_prefill_tokens` is omitted, existing behavior applies:
`effective_prefill_tokens` enables tracking, otherwise router configuration controls it. The
reservation API does not accept or derive accounting from overlap fields.

## Reservation Lifecycle

```http
POST /reservations/{selection_id}/prefill_complete
POST /reservations/{selection_id}/output_block
DELETE /reservations/{selection_id}
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
| Replica synchronization | ZMQ | `--replica-sync-port`, `--replica-sync-peers` | Share admission, prefill-complete, and free events by model and routing group. |

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
- Unknown worker, model, routing-group, DP-rank, and block-size events are dropped.
  Register the same worker catalog on every selector before routing traffic.
- Replica messages carry opaque tracking hashes and do not negotiate the
  tracking algorithm or key epoch. Configure all hash producers consistently.
  For key rotation, stop traffic, change the key and key ID together, restart
  all producers, and recreate derived tracker state before resuming traffic.
- The v1 replica envelope uses `routing_group` and is incompatible with binaries that
  send `tenant_id`. Drain active reservations and lifecycle traffic, upgrade all connected
  selectors together, re-register worker catalogs, and then resume traffic. Active advisory
  state is not migrated.
- Admission, prefill-complete, and free are synchronized. Output-block growth
  remains local to avoid excessive network bandwidth.
- Startup recovery waits for recovered events to be submitted to the indexer,
  not for complete processing. Early selections may temporarily miss recovered
  KV state.
- `/select` followed by `/reservations` provides eventual, not atomic,
  cross-replica admission, and the pending-selection cache behind the minimal
  reservation form is local to the selector that served the `/select`. Use
  `/select_and_reserve` for atomic local booking.
- Reservation IDs must be globally unique. Duplicate bookings for the same ID
  conflict (`409`), regardless of the target worker. An explicit booking that
  carries a `selection_id` also discards that cached selection.

## Inspection APIs

- `GET /loads` returns active-load snapshots, optionally filtered by
  `model_name` and `routing_group`.
- `POST /potential_loads` estimates worker load for a prompt without selection.
- `POST /overlap_scores` returns per-worker/per-rank tiered overlap rows.
- `GET /dump` returns the compatible indexer recovery snapshot.
