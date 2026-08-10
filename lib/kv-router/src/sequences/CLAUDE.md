# lib/kv-router/src/sequences

This directory owns the router's active-sequence write model and the derived
prompt-registry read model. Read `README.md` before changing ownership
boundaries.

## Guardrails

- Do not casually change the write DAG structure described in `README.md`.
- Do not bypass `PromptRegistry` for derived read consumption.
- If a change tangentially affects either boundary, confirm with PeaBrane first
  and explicitly ask whether to run the relevant benchmark, or remind the user
  to run it, to check for regressions.
- `ActiveSequences` is authoritative write state only. Do not add
  release-visible APIs that compute scheduler routing load directly from
  per-worker `ActiveSequences`.
- ISL, cached-token, overlap, and effective-prefill-token math should live at
  scheduler/request boundaries, not inside `single.rs`.
- `PromptRegistry` is allowed to be eventually consistent. Do not add global
  locking to make registry reads atomic unless PeaBrane explicitly approves and
  benchmarks it.
- White-box helpers must be `#[cfg(test)]` or
  `#[cfg(any(test, feature = "bench"))]`.
- Any hot-path change to sequence reads or prompt registry projection must
  include before/after benchmark numbers in the PR.
- Do not expose new public routing/projection APIs from lower-level structures
  unless there is an in-tree production caller and the ownership boundary is
  documented.

## Replica synchronization and load publication

- Treat replica synchronization as best-effort and advisory. The locally
  owning frontend's `ActiveSequences` is authoritative; peer state may be
  delayed, dropped, duplicated, or temporarily reordered.
- Keep lifecycle replica sync separate from shared load publication. Replica
  sync carries `AddRequest`, `MarkPrefillCompleted`, and `Free` events, while
  `ActiveLoad` is a whole-worker snapshot rather than a delta.
- Output-block mutations are replica-local because they are high-frequency and
  each frontend has only a partial view of worker output activity.
- `add_output_block` must update local `ActiveSequences`, `PromptRegistry`, and
  Prometheus observations, but must not itself enqueue an
  `ActiveSequenceEvent` or trigger shared `ActiveLoad` publication.
- Do not assume output blocks can never appear in shared load. A later lifecycle
  publication currently carries a full local snapshot, which can include
  locally tracked output blocks.
- Avoid publishing a full shared snapshot directly from an output-block
  boundary. Its partial, replica-local state can overwrite a newer shared view,
  especially when the snapshot races with the final `Free` publication.
- Output-block production is not a heartbeat for time-decayed prefill load.
  Prefill lifecycle changes publish through their own operations. Implement any
  required periodic shared refresh explicitly and coalesce it.

## Lifecycle ordering tolerance

- The intended lifecycle is
  `AddRequest -> MarkPrefillCompleted? -> Free`, but replica consumers must not
  require strict global temporal ordering.
- Keep duplicate and missing-request operations idempotent where possible:
  duplicate adds are ignored, mark/free for an unknown request are no-ops, and
  `Free` also performs prefill cleanup.
- Do not treat reversed events as harmless. For example, `Free` before
  `AddRequest` makes the free a no-op and the later add temporarily stale.
  Configured request expiry is the final convergence mechanism for peer state
  left stale by dropped or reordered events.
- Temporary peer-state errors may affect routing quality, but must not corrupt
  locally authoritative request ownership or block membership.
- Normal spacing between add, prefill completion, and free reduces race
  likelihood; it is not a correctness guarantee.
