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
