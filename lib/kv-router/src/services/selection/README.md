<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Selection, admission, and reservation ownership

The public deployment and HTTP API contract is documented in
[Standalone Selection Service](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md).

The selection service chooses a worker and tracks reserved load. It does not run the model or observe the backend response stream. Its caller dispatches work and reports progress and completion.

The frontend and standalone HTTP service share `SelectionCore`. `SelectionService` is a concrete service and state owner, not a trait that `EmbeddedSelection` implements. The frontend's `EmbeddedSelection` owns an `Arc<SelectionService>` and calls its core directly. The standalone HTTP handlers call service methods that reach the same core. Their discovery, transport, and reservation-lifetime adapters differ.

Read [EmbeddedSelection ownership](../../../../llm/src/kv_router/embedded.rs) and [its call into SelectionCore](../../../../llm/src/kv_router/embedded.rs).

## Four selection modes

Admission decides whether work can proceed under scheduler policy. Selection chooses a worker and rank. Booking records the load associated with that work. These are separate effects, even when one API performs all three.

| `SelectionAdmission` | Enters queue admission | Selects a worker | Books load | Cleanup owner |
|---|---|---|---|---|
| `Advisory` | No | Yes | No | No booking |
| `Query` | Yes | Yes | No | No booking |
| `Lease` | Yes | Yes | Yes | Caller receives an armed booking handle |
| `Book` | Yes | Yes | Yes | Core registers the booking under `selection_id` |

The scheduler may queue a request entering admission until policy permits it to proceed. Capacity and queue limits, priority, cancellation, and worker eligibility can affect the result. Admission is part of the selection implementation, but an admitted query does not retain capacity after it returns. A later explicit booking is a separate operation.

An advisory request avoids the admission queue; it still does validation, cache lookup, filtering, and policy selection. It can fail and is not a zero-latency or capacity guarantee. A selected worker is the policy's choice, not necessarily the worker with the highest raw cache hit.

Sources: [mode definitions](core/operation.rs) and [scheduler dispatch](core/run.rs).

## HTTP API and lifecycle

| Operation | Behavior |
|---|---|
| `POST /select` | Uses `Query` by default; `advisory: true` uses `Advisory`. Neither books. |
| `POST /select_and_reserve` | Uses `Book`: selects and books within the receiving selector process. |
| `POST /reservations` | Books an explicit worker or replays a cached selection. Does not rerun ordinary queue admission and worker selection. |
| `POST /reservations/{selection_id}/prefill_complete` | Clears prefill load and advances the phase while retaining the reservation. |
| `POST /reservations/{selection_id}/output_block` | Adds one output block to local accounting; accepts optional `decay_fraction`. |
| `DELETE /reservations/{selection_id}` | Releases the reservation. |

There is no endpoint named `/reserve` in this route table. Use the exact names above.

Sources: [HTTP routes](server.rs), [select modes](core/run.rs), and [reservation lifecycle implementation](core/reservations.rs).

A caller can report prefill completion, then output growth, then request completion. The service cannot infer these transitions from model execution because it does not own that execution. The lifecycle API resolves the ID to the exact scheduler booking. Unknown or expired reservations can return not-found errors.

Output-block updates stay local. Admission, prefill-completion, and free events have best-effort replica synchronization. This does not provide globally atomic admission across service replicas.

## Why both Book and Lease exist

Both modes create scheduler bookings. Their difference is who retains the cleanup responsibility:

- `Lease` returns an armed handle. Dropping it releases the booking unless responsibility was explicitly transferred. A frontend route plan can carry it into request-stream cleanup.
- `Book` keeps the booking in the core's reservation registry. An HTTP caller receives an ID and later sends lifecycle requests. The reservation must survive the short-lived HTTP handler that created it.

Thus `Book` is conceptually service-managed lifetime over the same booking mechanism. It is not literally a wrapper that calls `Lease`, and it does not eliminate the external caller's obligation to report completion. Expiry provides recovery for abandoned state.

The embedded frontend uses its request lease manager for expiry. Standalone selection uses periodic age-based expiry. Progress and expiry semantics are not interchangeable between those adapters. See [expiry behavior](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md).

## Two-step selection and booking

`POST /select` followed by `POST /reservations` is a supported external-scheduler workflow. The public Ray example selects, invokes the returned worker, then reports the reservation and its lifecycle. The example is an integration recipe, not a Ray runtime hook that automatically releases reservations. The integration must send `DELETE /reservations/{selection_id}` when work finishes, including its error/cancellation cleanup path.

When `/select` carries `selection_id`, the service can cache inputs for later replay. That cache belongs to the selector that handled the query. The gap between query and booking permits intervening work and membership changes; it is not a held reservation. Prefer `/select_and_reserve` when the caller wants an immediate local commitment before dispatch.

A query is not contractually required to be followed by a booking. A second `/select` also does not complete a prior query: the committing follow-up is `/reservations`, or a new `/select_and_reserve` operation.

Sources: [Ray select-then-reserve flow](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md), [cached-selection implementation](pending.rs), and [lifecycle contract](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md).

## Where Query is reached

In the current implementation there are two production constructors of `SelectionAdmission::Query`: the public service's non-advisory `select` path and the embedded `KvRouter` mapping for `WithAdmission` with `update_states=false`. Tests and enum dispatch arms are not additional public workflows.

| Caller or entry point | How it reaches Query | Follow-up booking |
|---|---|---|
| Standalone `/select`, default mode | Public core wrapper | Optional; external callers may use `/reservations` |
| KV `query_instance_id` annotation | `RoutingHost::generate` selects with admission and query-only state | No booking or dispatch in that path |
| Python `best_worker` without `request_id` | Sets `update_states=false` and uses admitted lookup | No automatic booking; `update_indexer` is a separate concern |
| Low-level `find_best_match_details_with_policy_class*` with `update_states=false` | Embedded mapping | Caller-dependent |
| Runtime EPP | Query followed by `add_request` | Yes, separate registration |

`RouterRequest::New` uses `update_states=true`; it is a tracked path, not another query-only case. `preview_kv_route` and the prefill busy probe use `WithoutAdmission` and therefore `Advisory`.

Sources: [embedded mode mapping](../../../../llm/src/kv_router.rs), [Python best_worker](../../../../bindings/python/rust/llm/kv.rs), [query route](../../../../llm/src/kv_router/routing_host.rs), and [RouterRequest::New](../../../../llm/src/kv_router.rs).

## Recommended direction, not the current universal contract

Use `Advisory` for inspection and admitted booking for committed work. Under that proposed rule, the KV query annotation and no-ID Python inspection path are candidates for conversion to `Advisory`. The public `/select` default needs compatibility review rather than an unannounced behavior change. Existing two-step external integrations must also retain an explicit commitment step.

The route-host comment describes `query_instance_id` as advisory, but its KV implementation still uses `WithAdmission`. Treat the implementation above as the observed behavior and resolve the comment/API mismatch before documenting a stronger guarantee.

## Existing module invariants

This module composes the existing worker catalog, KV indexer, scheduler queue,
and active-sequence accounting. Keep these implementation invariants explicit:

- `/select` is query-only; `/select_and_reserve` books before returning.
- `/reservations` accepts `effective_prefill_tokens` as a direct
  `PrefillLoadHint` and rejects a value greater than normalized ISL.
- Mooncake overlap fields are raw matched-token observability. Effective
  prefill tokens use the scheduler's weighted cache credit and are not derived
  from `longest_matched`.
- Discovery-driven worker membership goes through `WorkerCatalogSource` and
  `CatalogReconciler` (`membership.rs`), not `upsert_worker`/`delete_worker`.
- A partition's KV index comes from `HostCache.index: KvIndexSource::Owned`. Its `KvEventIngress` builds and feeds it: `ZmqDirectIngress` here, `RuntimeIngress` in the frontend. The frontend ingress also supports Dynamo-native remote indexing. The `Indexer` type itself is shared with the frontend (`services::indexer::backend`).
- Each partition owns its `SessionAffinity` table, including versioned
  bindings, idle TTL, and lease lifecycle. A reservation owns its affinity
  lease, so every reservation removal releases both. Frontend routing hosts
  share the partition table and one coordinator for stream leases and runtime
  replication.
- Valid worker metadata updates preserve live bookings on surviving ranks and
  KV state from unchanged event sources. Catalog commits and ingress changes
  are serialized; partition policy factories can initialize independently.
- The frontend request lease manager owns expiry for embedded partitions.
  Standalone partitions use periodic request expiry.
- Selector replicas synchronize admission, prefill-complete, and free events.
- **NOTE:** Output-block updates remain local. They are deliberately excluded
  from replica sync because their frequency would consume disproportionate
  network bandwidth.
- Replica sync is best-effort and may delay, reorder, or drop events. Unknown
  catalog entries are dropped under `ReplicaWorkerPolicy::RequireRegistered`.
- Startup indexer recovery waits for replay submission, not a full processing
  barrier.
- Reservation IDs must be globally unique. Retry and idempotency behavior is
  the existing active-sequence behavior.
