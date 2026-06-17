# Dynamo Selection Service

The public deployment and HTTP API contract is documented in
[Standalone Selection Service](../../../../../docs/components/router/standalone-selection.md).

This module composes the existing worker catalog, KV indexer, scheduler queue,
and active-sequence accounting. Keep these implementation invariants explicit:

- `/select` is query-only; `/select_and_reserve` books before returning.
- `/reservations` accepts `effective_prefill_tokens` as a direct
  `PrefillLoadHint` and rejects a value greater than normalized ISL.
- Mooncake overlap fields are raw matched-token observability. Effective
  prefill tokens use the scheduler's weighted cache credit and are not derived
  from `longest_matched`.
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
