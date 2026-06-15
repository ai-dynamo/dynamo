# Mocker vLLM disagg KV stranding via `disaggregated_params` (NIXL model)

**Date:** 2026-06-15  **Branch:** `neelays/mocker-dis2147-stranding`  **PR:** #10557

## Intent

Model disaggregated prefill→decode KV **stranding** in the mocker the way real
vLLM does it — via the engine-opaque `disaggregated_params` (NIXL
`kv_transfer_params`) — instead of the sglang-shaped bootstrap rendezvous the
mocker currently borrows for all engines.

### Why

- Real vLLM disagg uses `NixlConnector` + `kv_transfer_params`; the prefill
  **emits** `disaggregated_params` in its output and Dynamo plumbs them opaquely
  to the decode request (`lib/llm/src/kv_router/prefill_router/execution.rs:306`
  → `PrefillResult`). sglang's bootstrap triple is just *one* possible
  `disaggregated_params` payload (`llm_backend.rs:120` documents the field as
  engine-owned: "vLLM `kv_transfer_params`, SGLang bootstrap triple, …").
- The mocker instead emits `disaggregated_params: "dummy"` and coordinates over
  a TCP `BootstrapServer` (room/host/port) — sglang's mechanism — for **all**
  engines.
- That bootstrap path is **opt-in** (`--bootstrap-ports`) and was **never
  engaged** in any actual mocker cascade/E2 run (no `*disagg__mocker*` scenario
  sets it). So there is no installed base to preserve, and the cascades observed
  in those runs were decode-side, not prefill-KV stranding.

The strand duration must be **event-driven and load-dependent** — the time from
prefill-complete until *that request's* decode is admitted and pulls the KV — not
a fixed time (a constant erases the cascade).

## Scope

- **Engine:** vLLM. sglang keeps its bootstrap-triple `disaggregated_params`
  format and its current behavior **unchanged** (engine-gated, additive).
- **Targets:** both **live** (separate prefill/decode worker processes) and
  **replay** (offline single-process trace replay).

## Approach (A — Evolve)

Reuse the verified scheduler pin/release core and the existing cross-process
channel; change only the **keying and release trigger** so the handoff is
driven by `disaggregated_params` rather than a frontend-computed bootstrap room.

### Data flow

**vLLM `disaggregated_params` payload (mocker):** emitted by the prefill in its
first output, replacing `"dummy"`:
```json
{ "transfer_id": <u64>, "prefill_host": <str>, "prefill_port": <u16> }
```
`transfer_id` is the correlation key; host/port let the decode connect for the
live pull. Emitted **post-prefill-compute** (sourcing flips from the
frontend-pre-computed bootstrap room).

**Live (two processes):**
1. Frontend routes the prefill request (no pre-computed room).
2. Prefill **runs** → allocates KV → on prefill-complete the scheduler **pins**
   that KV (keyed by request `uuid`); prefill emits
   `disaggregated_params = {transfer_id, prefill_host, prefill_port}`.
3. Dynamo's existing machinery attaches those `disaggregated_params` to the
   **decode** request.
4. Decode reads `disaggregated_params` → connects to `prefill_host:prefill_port`
   over the reframed channel keyed by `transfer_id` → models the pull → prefill
   **releases the pin** (frees KV). Decode no-show → `kv_transfer_abort_timeout_ms`
   frees it.

**Replay (one process):** the trace's decode request carries the prefill's
`disaggregated_params`; an in-process correlator `transfer_id → pinned prefill
uuid` **releases** the pin when the matching decode is **scheduled in-process**.
Event-driven, load-dependent — **replaces** the PR's interim time-based release.

### Naming / aliases

- The channel uses a single internal key name **`transfer_id`** (engine-neutral:
  it correlates a KV transfer for both engines).
- Type aliases, both `= u64`, let each engine read in its native vocabulary with
  no conversion:
  ```rust
  pub type RoomId = u64;      // sglang vocabulary (bootstrap triple)
  pub type TransferId = u64;  // vLLM vocabulary (kv_transfer_params)
  ```
- The shared map is keyed by the one `u64`; sglang (`RoomId`) and vLLM
  (`TransferId`) hit the same entry for free (no dual-accessor sugar).

### Reused unchanged

- Scheduler pin/release core (`pin_completed` / `release_pinned` / `take_pinned`
  / `pinned_block_footprint`) in both vLLM and sglang cores.
- Pinned KV **counts toward pool capacity** (the cascade) but is **discounted
  from the router-facing load metric** (`active_decode_blocks`) — the fix that
  keeps stranding from mis-diverting unrelated overlap routing.
- The 758-line channel's connect/wait/ACK/ABORT machinery (only the key name
  generalizes `room_id`→`transfer_id`; behavior identical).
- Prefill-runs-first ordering — now **structural**: prefill must run to emit the
  handle, so the "submit-after-wait" bug cannot recur.

## Per-file components

| File | Change |
|---|---|
| `lib/llm/src/mocker.rs` | Engine-gate the disagg block. **vLLM:** prefill pins + emits real `disaggregated_params`; decode parses it → live channel connect (by `transfer_id`) or replay in-process correlator. **sglang:** existing bootstrap-room path, unchanged. |
| `lib/mocker/src/services/bootstrap.rs` | Generalize key `room_id`→`transfer_id` (one `u64`); add `RoomId`/`TransferId` aliases + equivalence doc. connect/wait/abort machinery unchanged. |
| `lib/mocker/src/scheduler/vllm/core.rs`, `sglang/core.rs` | Pin/release core **reused as-is**; release just triggered by the transfer correlation. |
| `lib/mocker/src/common/protocols.rs` | Keep `bootstrap_room` (sglang) + `release_pin`. vLLM transfer handle rides on the request's existing `disaggregated_params: Option<Value>` (parsed in mocker.rs); pin keyed by `uuid`, correlation by `transfer_id`. |
| `lib/mocker/src/replay/offline/{core,single}.rs` | **Remove** time-based release; add `transfer_id → pinned uuid` in-process correlator releasing on the matching decode's scheduling. |
| `lib/llm/src/kv_router/prefill_router/` | **Key integration point:** ensure the vLLM mocker prefill reaches decode via the **output-`disaggregated_params`** path (`execution.rs:306`), not `compute_bootstrap_room`. Verify the bootstrap `Resolved`/`NoBootstrapEndpoint` branch is bypassed for vLLM. |

## Error / edge cases

- **Decode no-show (live):** `abort_timeout` → ABORT → release pin + surface abort.
- **Decode never appears (replay):** release on teardown; `log` unmatched pins
  (silent leak would skew occupancy); optional virtual-time abort bound.
- **Prefill cancelled mid-strand:** release on the cancellation path.
- **Pool full while pinned = the cascade (intended):** pinned KV in
  `num_active_blocks` → new prefills rejected/queued; router-metric discount
  prevents mis-diverting overlap routing.
- **Duplicate/late release:** `release_pinned(uuid)` is idempotent (no-op if gone).
- **vLLM request with no `disaggregated_params`** (aggregated): no pin, normal free.

## Testing

1. **Cascade via disagg_params** — vLLM prefill emits real `disaggregated_params`,
   pins; pool occupancy stays elevated; 2nd prefill rejected/queued; decode with
   that `transfer_id` releases → occupancy drops.
2. **Router metric not skewed** — `active_decode_blocks` discounted while pinned.
3. **Replay event-driven release** — pin released when matching decode scheduled;
   no time-based path.
4. **Abort/no-show** — decode never pulls → abort_timeout (live) / teardown
   (replay) frees; double-release no-op.
5. **No disagg_params → no strand** — aggregated vLLM frees immediately.
6. **sglang regression guard** — existing sglang stranding/bootstrap tests pass
   **unchanged**.
7. **Integration** — `test_disagg_background_prefill_sticky` stays green.

Verification: `cargo test -p dynamo-mocker --lib` + `dynamo-llm kv_router` +
`cargo fmt --check` + `clippy -p dynamo-mocker -p dynamo-llm --all-targets -D
warnings`, in `dynamo:latest-vllm-dev` (install rustfmt+clippy; media-ffmpeg
unbuildable locally).

## Non-goals

- Replacing/altering the sglang disagg path or its bootstrap format.
- Real NIXL agent-metadata fidelity (engine_id, block-id lists) — the mocker
  models the *timing/holding* semantics, not the wire format.
- Removing the cross-process channel for live (inherent to multi-process disagg).
