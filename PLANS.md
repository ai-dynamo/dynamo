# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 09:56:05 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - current live tip after the signed docs-only handoff compaction commit:
    `9bf6bb8ab454`
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
  - this run is a fresh validation plus handoff compaction refresh
  - keep `PLANS.md` as the compact execution log and handoff document

## Current run (2026-04-01 09:56:05 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Re-audited the live repo state from detached `HEAD` `9bf6bb8ab454`
- Confirmed the active tree still contains the seams the handoff and design
  doc require:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Re-ran the focused `G3PB` and bindings validation stack from the live
  detached `HEAD`, and it is green
- Re-confirmed that the plan still has no open implementation work for the
  active `G3PB` slice; only docs/handoff compaction remains in this run
- Started compacting `PLANS.md` again so this file reflects the fresh
  validation state from `9bf6bb8ab454`

### Current findings in this run

- the live detached `HEAD` validated in this run is `9bf6bb8ab454`
- the live detached `HEAD` still contains the same validated non-docs `G3PB`
  implementation baseline and follow-on code changes
- the active `G3PB` implementation slice still appears complete on the live
  tree
- no new `G3PB` implementation gap was identified by the audit or validation
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- the only repo change currently in the worktree is this in-progress
  `PLANS.md` handoff refresh

### Validation completed in this run

- `git rev-parse --short=12 HEAD`
  - pass (`9bf6bb8ab454`)
- `git log --oneline -8`
  - pass
- `sed -n '1,220p' Agents.md`
  - pass
- `sed -n '1,260p' PLANS.md`
  - pass
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

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- run a focused post-edit `g3pb::` spot check after this `PLANS.md` refresh
- write a signed docs-only handoff compaction commit for this refreshed
  validation state
- re-run the focused `g3pb::` library spot check after that commit

### Exact next step

- run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`, then
  commit this compacted handoff with `git commit --signoff -m "docs: refresh
  g3pb handoff state"` and re-run the same spot check on the new detached
  `HEAD`

### Handoff for next run

- this file is the compact current handoff; keep it compact and update it from
  the live tree rather than appending repeated docs-only history
- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming any previous
  docs-only tip remains current
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- this run fully revalidated the active slice from detached `HEAD`
  `9bf6bb8ab454`; if the current `PLANS.md` refresh is committed, re-check the
  new detached `HEAD` rather than assuming the next docs-only tip in advance
- before stopping, either finish the pending docs-only commit plus post-commit
  `g3pb::` spot check or record that exact unfinished state here
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
