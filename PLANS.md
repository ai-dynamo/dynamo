# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 10:52:11 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - current live tip is the docs-only handoff refresh commit from this run on
    top of validated detached `HEAD` `05c3cbd70b8e`
- Current implementation direction:
  - `G3PB` is the peer-cache replacement for the unlanded `G4` disk-identity
    surface
  - peer ownership remains rendezvous-hash based
  - remote identity is keyed by `sequence_hash` only
  - peer-local persistence stays hidden behind `G3pbPeerStorage`
- Validated non-docs implementation baseline still present in the tree:
  - `abfc85ffd0a4` (`llm: stabilize g3pb cache storage test`)
- Already-landed follow-on implementation commits still present in the tree:
  - `8ddc2f2e1` (`llm: reclaim g3pb backend staging`)
  - `c231d60fb` (`build: patch local nixl-sys invalidation`)
- Current scope status:
  - no open implementation work remains for the active `G3PB` slice
  - this run is a fresh validation plus handoff compaction refresh on the live
    detached `HEAD`
  - keep `PLANS.md` as the compact execution log and handoff document

## Current run (2026-04-01 10:49:57 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Re-audited the live repo state from detached `HEAD` `05c3cbd70b8e`
- Re-confirmed the active tree still contains the seams required by the
  handoff and design doc:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Re-ran the focused `G3PB` and bindings validation stack from the live
  detached `HEAD`, and it is green
- Re-ran the slice audit search for `G3PB`/`TODO`/`FIXME` markers across the
  handoff, design doc, and implementation areas
- Re-confirmed that the plan still has no open implementation work for the
  active `G3PB` slice; this run only refreshes validation and handoff state
- Compacted `PLANS.md` so this file reflects the fresh validation state from
  `05c3cbd70b8e`
- Re-ran post-edit `g3pb::` spot checks as the handoff was compacted, and they
  passed
- Landed signed docs-only handoff commits after the post-edit validation passed

### Current findings in this run

- the detached `HEAD` validated in this run is `05c3cbd70b8e`
- the current live detached `HEAD` after this run is newer than the validated
  detached `HEAD` only by the docs-only handoff refresh commit from this run
- the live tree still contains the same validated non-docs `G3PB`
  implementation baseline and follow-on code changes
- the active `G3PB` implementation slice still appears complete on the live
  tree
- no new `G3PB` implementation gap was identified by the audit or validation
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- no code changes are pending for the active `G3PB` slice

### Validation completed in this run

- `git rev-parse --short=12 HEAD`
  - pass (`05c3cbd70b8e`)
- `git log --oneline -8`
  - pass
- `git status --short --branch`
  - pass (`## HEAD (no branch)`)
- `sed -n '1,220p' Agents.md`
  - pass
- `sed -n '1,260p' PLANS.md`
  - pass before refresh
- `sed -n '1,260p' docs/design-docs/kvbm-g3pb-plan.md`
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
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) as the post-edit spot check with only `PLANS.md` dirty
- `git add PLANS.md`
  - pass
- `git commit --signoff -m "docs: refresh g3pb handoff state"`
  - pass
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) after the final handoff compaction edit
- `git commit --signoff -m "docs: finalize g3pb handoff state"`
  - pass

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- none

### Exact next step

- run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`, then
  `git rev-parse --short=12 HEAD`, `git status --short --branch`, and `sed -n
  '1,260p' PLANS.md`, then leave the active `G3PB` slice closed unless a fresh
  regression or explicitly new scope appears

### Handoff for next run

- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming this docs-only
  tip remains current
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- this run fully revalidated the active slice from detached `HEAD`
  `05c3cbd70b8e`, then landed docs-only handoff commits on top
- no code changes are pending for the active `G3PB` slice
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
