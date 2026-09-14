<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Dynamo Selection Service

The public deployment and HTTP API contract is documented in
[Standalone Selection Service](../../../../../docs/fern/pages/developer-guide/knowledge-base/modular-components/router/standalone-selection.md).

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
- A partition's KV index comes from `HostCache.index: KvIndexSource`: `Owned`
  (a `KvEventIngress` builds and feeds it: `ZmqDirectIngress` here, the
  runtime event plane in the frontend) or `Remote` (a standalone indexer
  serves it). The `Indexer` type itself is shared with the frontend
  (`services::indexer::backend`).
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

## Plugin-controlled prefill execution

A linked decode strategy can opt into path planning by implementing
`WorkerPicker::path_planning()` and returning `Some(PathPlanningRequirements)`.
Dynamo then runs an advisory decode preview even when built-in conditional
disaggregation is disabled. Existing pickers return `None` by default and do
not incur an extra preview.

During that preview, `pick_route(context, input)` returns a `RouteChoice`:

- `Default` uses the configured conditional-disaggregation policy, including
  its decode-busy guard and unavailable-signal behavior.
- `LocalOnDecode` requests local prefill and generation on the selected worker.
- `Remote` continues normal P/D routing. The preview's decode choice is not
  reserved or guaranteed to be the later decode destination.

Ordinary P selection, decode-after-prefill, and external queries call `pick`,
not `pick_route`. A picker must support both. Filtering, scoring, explicit
worker pins, and row validation still apply. The preview is not a dispatch
notification and plugins must tolerate repeated invocations.

`context.is_path_planning()` identifies the preview. Pickers requesting
`WorkerInputs::CACHE` receive raw `device_overlap_blocks()` separately from
`effective_overlap_blocks()`, the weighted cache estimate. The standard cache
estimator converts weighted overlap to tokens with
`(cache.effective_overlap_blocks() * context.block_size() as f64).round().max(0.0) as usize`.
Multiply raw device
coverage by `context.block_size()` and clamp it to `context.prompt_tokens()`
before computing uncached tokens. `candidate.can_prefill_locally()` reports
`Some(true)`, `Some(false)`, or `None` for unreported backend support.
`WorkerInputs::LOAD` also exposes optional `total_kv_blocks()` during path
planning; missing capacity must not be interpreted as zero load.

`PathPlanningRequirements::prefill_load` requests the host's selected-P load
probe. `context.prefill_worker_busy()` remains `None` when no threshold or load
is available; it must not be interpreted as idle. No P probe is added unless
the strategy requests it or the configured `Default` decision requires it.
The pure `conditional_disagg::evaluate_conditional_disagg` helper lets plugins
reuse built-in thresholds with supplied facts and no routing side effects.

Explicit actions replace heuristic path preferences. They do not bypass
backend support, eligibility, explicit P pins, or admission. A local request
must target a worker advertising `local_prefill=true`; vLLM and TensorRT-LLM
publish this runtime metadata for supported workers. Unknown support fails
closed for explicit local actions. `Default` retains the existing backend
compatibility behavior.

The frontend validates local actions, admits the previewed worker and DP rank,
charges local prefill, and emits `x-bypass-remote-prefill`. Unsupported or
conflicting explicit local actions and failed local admission return an error;
they do not silently switch workers or fall back to remote prefill. The shared
selection core transports the choice; this does not add a disaggregation
coordinator or frontend orchestration to the standalone service or EPP.
