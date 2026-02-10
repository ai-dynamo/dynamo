# kvbm v2 Planning

## Goals
- Real-time visibility into KV cache slot/request lifecycles, akin to `tokio-console` for async tasks.
- Historical timeline of every slot transition, transfer, and completion for post-mortem analysis.
- Breakpoint-style interception so operators can pause/resume scheduler interactions and inspect state.

## Components
- **Event stream:** Leverage `SlotActions`/`StateNotification` from `slot_v2` to emit structured events per `request_id`.
- **Recorder:** Persist events to an in-memory ring buffer and optional durable log for replay/testing.
- **Executor hooks:** Wrap slot transitions with interceptors that can `await` external signals (enable pausing).
- **Broadcast channel:** Provide subscription API (likely `tokio::sync::broadcast`) for the console/UI.
- **Console UI:** Build a TUI that connects to the broadcast stream, visualizes state timelines, and exposes pause/resume controls.

## Design
- **Leader as orchestration hub:** `leader.rs` drives slot lifecycle (`get_num_new_matched_tokens`, `update_state_after_alloc`, `build_connector_metadata`). Instrument these entry points to publish state events and queue transfer candidates so the console has a single source-of-truth.
- **Asynchronous transfer pipeline:** Slot transitions produce `PlannedTransfer` intents; instead of dispatching immediately, enqueue them as `TransferCandidate`s processed by offload/onboard evaluators. Evaluators finalize sources/destinations (e.g. match host blocks or skip if already cached) and decide whether to promote to actual transfers or mark as no-op.
- **Deferred materialization:** At intent time we only know the desired movement (e.g. “offload these device blocks”); concrete src/dst resources are discovered in the pipeline. Any additional metadata required for worker scheduling is materialized just before promotion and encoded into scheduler outputs for batched worker acknowledgements.
- **Manager feedback loop:** Evaluator results flow back to the slot manager, which applies `PromotedTransfer` plans to the owning slot, queues `WorkerTransferRequest`s, and emits concrete engine commands. The next `build_connector_metadata` pass drains pending worker operations, keeping behaviour identical to v1 while the transfer engine executes the promoted plans asynchronously.
- **Operation artifacts:** Each transfer candidate carries boxed, type-aware artifacts that can (a) emit a worker-facing `OperationEntry` (layout handles + block ids) when promoted, and (b) execute a completion callback once the transfer finishes. This keeps resource management localized to the storage-specific implementation while the manager/worker interact through a uniform payload.
- **Executor/slot split:** `crates/kvbm/src/v2/logical/executor/mod.rs` now owns the transfer pipeline plumbing (`SlotExecutor`, `TransferPipelineRuntime`, dispatch/broadcast traits). The shared slot state machine, evaluator, and `TransferSlotHandle` live in `v2/integrations/connector/slot.rs`, isolating domain logic from the core executor.
- **Candidate tracking:** Extend `InFlightTransfers` with per-UUID status (`PendingEvaluation`, `Promoted`, `Skipped`, `Completed`). Demote candidates when the pipeline skips work; promote when dispatching to the transfer engine; mark complete once worker signals.
- **Batch completion semantics:** Offload pipelines may acknowledge completion only after a group of operations finishes. Maintain counters so grouped candidates stay “in-flight” until the batch completes.
- **Console integration:** Use the broadcaster to emit events for candidate lifecycle changes and transfer completion. Recorder consumes those events to build per-request timelines and power the TUI. Breakpoint hooks sit between transition and candidate enqueue.
  - When console support is enabled we will wrap the executor/leader entry points with hook traits that expose `will_call`/`finished_call` interception without touching the core logic. The default build wires no-op hooks.

## v1 logic
- **State inventory:** Slots move through `Initialized → (Optional OnboardStaged → Onboarding) → Prefilling → Decoding → Finishing → Finished`, with detours for `SkippedPrefill`, `SkippedDecode`, and `Preempted`.
  - `Initialized`: fresh slot created by `ConnectorSlotManager::create_slot`.
  - `OnboardStaged(num_tokens)`: `acquire_local_matches` finds host/disk blocks while in `Initialized`/`Preempted`.
  - `Onboarding(num_tokens)`: `trigger_onboarding` pairs staged blocks with device ids and issues load operations.
  - `Prefilling` / `Decoding`: `apply_scheduler_output(_with_computed_position)` marks state based on scheduler tokens.
  - `Skipped*`: `mark_as_skipped_prefill/decode` invoked when scheduler omits request within an iteration.
  - `Preempted`: `reset_after_preemption` clears device ownership while preserving sequence tokens.
  - `Finishing`: `mark_as_finished` when leader receives `request_finished`; `Finished` reached after outstanding operations drain (worker side, not explicit in code).
- **Transition triggers & side effects:**
  - `acquire_local_matches` (leader `get_num_new_matched_tokens`) scans host/disk caches, records cached token counts, stages blocks, and transitions to `OnboardStaged`.
  - `update_state_after_alloc` (leader) appends device blocks, records cached device tokens, advances computed position, and calls `trigger_onboarding` which:
    - Builds `LocalTransferRequest::Onboard` + matching `WorkerTransferRequest` via `onboard_blocks`.
    - Moves slot to `Onboarding(num_external_tokens)` and increments evaluated blocks past cached content.
  - `build_connector_metadata` (leader) drives the steady-state loop:
    - For onboarding slots: `md.create_slot` and flush `pending_operations` (load requests) via `take_pending_operations`.
    - For `new_requests`: ensures slot in `Initialized|Onboarding`, calls `record_start_iteration`, then `apply_scheduler_output` with zero tokens to pre-seed policy.
    - For `cached_requests`: `apply_scheduler_output` with new decode tokens and device block ids; method may extend token sequence, push device ids, and run eviction policy.
    - `apply_scheduler_output` (prefill/decode phases) decides tokens vs state (`Prefilling` if no tokens, else `Decoding`), appends device blocks, validates capacity, and executes eviction policy: collects next candidate blocks, clones `TokenBlock`s, and calls `offload_blocks`.
      - `offload_blocks` emits `LocalTransferRequest::Offload` and mirrors a `WorkerTransferRequest { transfer_type=Store, request_type=Scheduled }`, storing it in `pending_operations`.
  - `mark_as_skipped` handles unscheduled requests: transitions from `Prefilling/Decoding` to the corresponding skipped state without issuing transfers.
  - `take_pending_operations` hands accumulated `WorkerTransferRequest`s to the leader so they are serialized into the metadata and handed to the scheduler.
  - `record_cached_*_tokens` persist metrics used for logging/finish telemetry.
- **Leader bookkeeping:**
  - `inflight_requests` tracks active request_ids; removal happens during `build_connector_metadata` when scheduler reports new/cached requests or via `request_finished`.
  - `onboarding_slots` temporarily holds request_ids between `trigger_onboarding` and the next metadata build, ensuring worker recreates request slots for load operations.
  - Metrics: `matched_tokens.inc_by` records number of external tokens staged; cached token counts logged at finish.
- **Transfer completion semantics:** Worker side consolidates completion notifications—scheduled offloads only report back once all outstanding operations in the batch finish, so slots keep `pending_operations` until the aggregated signal arrives.
- **Finish path:**
  - `request_finished` locks slot, calls `mark_as_finished` (→ `Finishing`), removes slot from manager, and returns whether outstanding operations remain (`Finishing` ⇒ return `true`, `Finished` ⇒ `false` once worker signals completion).

## Milestones
1. Formalize event schema (protobuf/flatbuffers or serde-friendly structs) and integrate recorder with executor.
2. Expose management API (gRPC/HTTP or Unix socket) for console clients to subscribe and send control signals.
3. Prototype TUI showing live slot list, per-request timeline, and outstanding transfers.
4. Add breakpoint controls (pause on specific states, manual resume).
5. Support replay mode: feed recorded events into tests or offline debugging sessions.
