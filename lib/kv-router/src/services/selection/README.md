# Dynamo Selection Service

`dynamo-select-service` is a runtime-free HTTP service built on `dynamo-kv-router`.
It selects workers and tracks reservations, but it never forwards model requests
and never owns model responses.

## Readiness

- `GET /health` returns process liveness.
- `GET /ready` returns `200` only after at least one worker is schedulable.
- `GET /ready` returns `503` with worker lifecycle details while no workers are schedulable.
- `/select`, `/select_and_reserve`, `/reservations`, `/potential_loads`, and
  `/overlap_scores` return `503` while the target worker pool is not ready.

## Worker Lifecycle

Workers are accepted into a single `WorkerCatalog` and become schedulable only
after the reconciler has published consistent derived state to the scheduler,
slot tracker, and indexer/listener view.

Lifecycle values:

- `incomplete`: accepted, but missing metadata required for valid booking.
- `schedulable`: metadata is complete and the scheduler can safely book it.
- `draining`: removed from the scheduler view before cleanup.
- `unschedulable`: cleanup has completed and the worker is no longer selectable.

When queueing is enabled, `max_num_batched_tokens` is required before a worker
can become schedulable. When KV events are enabled, KV event listener endpoints
are also required.

## Selection APIs

### `POST /select`

Selects a worker through the existing `SchedulerQueue` admission path with
`update_states=false`.

This endpoint is query-only. It does not book active-load state. If
`selection_id` is provided, the service accepts and echoes it for observability,
but it does not dedupe, replay, or otherwise make retries idempotent. Retrying
`/select` may produce a fresh selection.

### `POST /select_and_reserve`

Selects a worker through the same `SchedulerQueue` admission path and books
active-load state before returning.

`reservation_id` is used as the scheduler and slot-tracker request ID. If it is
omitted, the service generates one. Duplicate IDs use the existing slot-tracker
behavior and may return `409 Conflict`; V1 does not provide retry idempotency,
distributed atomicity, or exactly-once booking semantics.

### `POST /reservations`

Books an explicit post-select reservation on the supplied worker and dp-rank.
`reservation_id` is the booking request ID. Duplicate IDs use existing
slot-tracker behavior and may return `409 Conflict`.

Clients must treat reservation creation as at-most-once unless they can tolerate
duplicate/conflict handling.

### Lifecycle

- `POST /reservations/{reservation_id}/prefill_complete`
- `DELETE /reservations/{reservation_id}`

Lifecycle endpoints operate on `reservation_id`.

## Request Inputs

For `/select`, `/select_and_reserve`, `/potential_loads`, and `/overlap_scores`:

- If `token_ids` is present, the service computes block hashes and sequence
  hashes internally and ignores supplied hashes. `isl_tokens` is derived from
  `token_ids.len()`.
- Otherwise, callers must provide `block_hashes`, `sequence_hashes`, and
  `isl_tokens`.

For `POST /reservations`:

- If `token_ids` is present, the service computes sequence hashes internally.
- Otherwise, callers must provide `sequence_hashes` and `isl_tokens`.

Signed hash values are reinterpreted bit-for-bit as unsigned router hashes,
matching the standalone indexer and slot-tracker APIs.

## Retry Semantics

V1 intentionally has no internal idempotency ledger.

- `/select` retries may return a different worker.
- `/select_and_reserve` retries with the same `reservation_id` may return
  `409 Conflict` if the first booking succeeded.
- `POST /reservations` retries with the same `reservation_id` may return
  `409 Conflict` if the first booking succeeded.
- Clients that need replayable retry semantics should keep their own request
  ledger. The service core is structured so a core idempotency ledger can be
  added later without changing the scheduler queue contract.
