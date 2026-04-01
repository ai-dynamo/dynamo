# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 12:38:04 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - live detached `HEAD`: `78c762331d90`
- Current implementation direction:
  - `G3PB` remains the peer-cache replacement for the unlanded `G4`
    disk-identity surface
  - peer ownership remains rendezvous-hash based
  - remote identity remains keyed by `sequence_hash` only
  - peer-local persistence stays hidden behind `G3pbPeerStorage`
- Validated non-docs implementation baseline still present in the tree:
  - `abfc85ffd0a4` (`llm: stabilize g3pb cache storage test`)
- Already-landed follow-on implementation commits still present in the tree:
  - `8ddc2f2e1` (`llm: reclaim g3pb backend staging`)
  - `c231d60fb` (`build: patch local nixl-sys invalidation`)
- Scope status:
  - no open implementation work remains for the active first-pass `G3PB` slice
  - this run is a live-head validation and handoff compaction refresh only

## Current run (2026-04-01 12:38:04 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Re-audited the live repo state from detached `HEAD` `78c762331d90`
- Re-confirmed the live tree still contains the seams required by the handoff
  and design doc:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Re-ran the focused `G3PB` and bindings validation stack from the live
  detached `HEAD`, and it is green
- Re-ran the slice audit search for `G3PB`/`TODO`/`FIXME` markers across the
  handoff, design doc, and implementation areas
- Re-confirmed that the plan still has no open implementation work for the
  active `G3PB` slice
- Confirmed again that no stronger standalone runnable `G3PB` e2e target is
  present in-tree beyond the existing smoke binary and focused validation stack
- Refreshed the on-disk handoff for this validation pass
- Re-read `PLANS.md` after the edit and re-ran the post-edit `g3pb::` spot
  check; both passed

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on detached
  `HEAD` `78c762331d90`
- no new `G3PB` implementation gap was identified by the audit or validation
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- no code changes are pending for the active `G3PB` slice
- this run has refreshed the on-disk handoff and completed its post-edit
  validation; the only remaining question is whether to land another signed
  docs-only handoff commit

### Validation completed in this run

- `git status --short`
  - pass (`M PLANS.md` at start of run)
- `sed -n '1,220p' Agents.md`
  - pass
- `sed -n '1,260p' PLANS.md`
  - pass before refresh
- `sed -n '1,260p' docs/design-docs/kvbm-g3pb-plan.md`
  - pass
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty)
- `git rev-parse --short=12 HEAD`
  - pass (`f54c6d1aed1d`)
- `git log --oneline -8`
  - pass
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 12:35:58 UTC`)
- `rg -n "G3pbPeerStorage|delete_blocks|g3pb_admission|G3PB_OFFLOAD_ALL|patch\\.crates-io|nixl-sys|kvbm_g3pb_backend|kvbm_g3pb_worker_smoke" Cargo.toml lib/llm lib/bindings/kvbm third_party/nixl-sys`
  - pass as a seam audit
  - result: the workspace `nixl-sys` patch, backend-side delete/reclaim path,
    bindings-side admission-config adoption, and smoke/backend binaries remain
    present in the live tree
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: no new active `G3PB` implementation gap surfaced in code or the
    design doc; unrelated repo-wide `TODO` and `FIXME` markers still exist
    outside this slice
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `rg -n "g3pb.*(smoke|e2e)|G3PB.*(smoke|e2e)|worker_smoke|end-to-end|end to end" lib/llm lib/bindings tests docs -g '!target'`
  - pass as an end-to-end test inventory search
  - result: no standalone runnable `G3PB` e2e test target surfaced beyond the
    existing `kvbm_g3pb_worker_smoke` binary and the focused validation stack
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty)
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 12:36:52 UTC`)
- `sed -n '1,260p' PLANS.md`
  - pass as the post-edit reread
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) as the post-edit spot check with only `PLANS.md` dirty

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- land a signed docs-only handoff commit from the still green tree
- re-read repo state from the new detached `HEAD`
- re-run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` as the
  final post-commit spot check
- re-read `PLANS.md` one more time before stopping

### Exact next step

- run `git commit --signoff -m "docs: refresh g3pb handoff"`

### Handoff for next run

- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming this docs-only
  tip remains current
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- this run has validated the active slice from detached `HEAD`
  `78c762331d90` and completed the post-edit reread plus post-edit `g3pb::`
  spot check; the remaining work is a final optional docs-only commit and
  post-commit verification
- the active `G3PB` slice still has no pending code changes
- `PLANS.md` is intentionally left dirty only with this in-progress handoff
  refresh; complete the post-edit validation before deciding whether another
  docs-only commit is useful
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
