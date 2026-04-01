# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 12:53:57 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - live detached `HEAD`: `1f92d47cb45e`
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

## Current run (2026-04-01 12:53:57 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Re-read and revalidated the live detached `HEAD` `1f92d47cb45e` instead of
  assuming the prior docs-only handoff tip was still current
- Re-confirmed the live tree still contains the seams required by the handoff
  and design doc:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Re-ran the focused `G3PB` audit and validation stack from live detached
  `HEAD` `1f92d47cb45e`, and it is green
- Re-confirmed that the plan still has no open implementation work for the
  active first-pass `G3PB` slice
- Confirmed again that no stronger standalone runnable `G3PB` e2e target is
  present in-tree beyond the existing smoke binary and focused validation stack
- Refreshed the on-disk handoff for this validation pass
- Confirmed before the refresh that `PLANS.md` remained the only dirty file
- Next in this run: re-read the refreshed `PLANS.md`, re-run the post-edit
  `g3pb::` spot check, and land a signed docs-only handoff commit if the tree
  still only differs in `PLANS.md`

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on detached
  `HEAD` `1f92d47cb45e`
- no new `G3PB` implementation gap was identified by the audit or validation
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- no code changes are pending for the active `G3PB` slice
- this run revalidated the current docs-only handoff tip and refreshed the
  on-disk handoff for the next validation/commit step

### Validation completed in this run

- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty)
- `sed -n '1,220p' Agents.md`
  - pass
- `sed -n '1,260p' PLANS.md`
  - pass before refresh
- `sed -n '1,260p' docs/design-docs/kvbm-g3pb-plan.md`
  - pass
- `git rev-parse --short=12 HEAD`
  - pass (`1f92d47cb45e`)
- `git log --oneline -8`
  - pass
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
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 12:53:57 UTC`)

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- re-read the refreshed `PLANS.md`
- re-run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` as the
  post-edit spot check
- if `PLANS.md` is still the only dirty file, land a signed docs-only handoff
  commit

### Exact next step

- `sed -n '1,260p' PLANS.md`, then
  `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`, then
  `git status --short --branch`; if only `PLANS.md` is dirty, run
  `git add PLANS.md && git commit --signoff -m "docs: refresh g3pb handoff"`

### Handoff for next run

- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming this docs-only
  tip remains current
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- this run validated detached `HEAD` `1f92d47cb45e`
- detached `HEAD` `1f92d47cb45e` has green focused `G3PB` validation:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
- the active `G3PB` slice still has no pending code changes
- the intended close-out for this run is a docs-only handoff commit after the
  post-edit `g3pb::` spot check, but if this run stops before that commit then
  `PLANS.md` should be the only dirty file
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
