# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 09:37:26 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - always capture the live tip with `git rev-parse --short=12 HEAD` before
    trusting any prior handoff note
- Current implementation direction:
  - `G3PB` is the peer-cache replacement for the unlanded `G4` disk-identity
    surface
  - peer ownership remains rendezvous-hash based
  - remote identity is keyed by `sequence_hash` only
  - peer-local persistence stays hidden behind `G3pbPeerStorage`
- Validated non-docs implementation baseline:
  - `abfc85ffd0a4` (`llm: stabilize g3pb cache storage test`)
- Already-landed follow-on implementation commits still present in the tree:
  - `8ddc2f2e1` (`llm: reclaim g3pb backend staging`)
  - `c231d60fb` (`build: patch local nixl-sys invalidation`)
- Current scope status:
  - no open implementation work remains for the active `G3PB` slice
  - this run is a fresh validation plus handoff-compaction refresh
  - keep `PLANS.md` as the compact execution log and handoff document

## Current run (2026-04-01 09:37:26 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff and design context:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-audited the live repo state from detached `HEAD`
- ✅ Confirmed the active tree still contains the seams the handoff and design
  doc claim are landed:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- ✅ Re-ran the focused `G3PB` and bindings validation stack from current
  detached `HEAD`, and it is green
- ✅ Compacted `PLANS.md` so the next run starts from the actual live validated
  tip instead of a stale prior snapshot

### Current findings in this run

- the current detached `HEAD` at validation start is `113347556c34`
- `PLANS.md` was still anchored to the previous run snapshot; the live tree now
  includes newer docs-only refresh commits:
  - `113347556` (`docs: stabilize g3pb handoff state`)
  - `ea8f7e7a0` (`docs: finalize g3pb handoff refresh`)
  - `e0f1a8bf4` (`docs: refresh g3pb handoff state`)
- the live detached `HEAD` still contains the same validated non-docs `G3PB`
  implementation baseline and follow-on code changes
- the active `G3PB` implementation slice still appears complete on the live
  tree
- no new `G3PB` implementation gap was identified by the audit or validation
- the only repo work performed in this run is handoff compaction and refresh

### Validation completed in this run

- `git rev-parse --short=12 HEAD`
  - pass (`113347556c34`)
- `git log --oneline -8`
  - pass
  - current recent history:
    - `113347556 docs: stabilize g3pb handoff state`
    - `ea8f7e7a0 docs: finalize g3pb handoff refresh`
    - `e0f1a8bf4 docs: refresh g3pb handoff state`
    - `d34e673fe docs: finalize g3pb handoff refresh`
    - `eae3ce029 docs: refresh g3pb handoff state`
    - `a5bd8ead0 docs: remove self-referential g3pb tip`
    - `c2a53e97d docs: finalize g3pb handoff refresh`
    - `280530069 docs: refresh g3pb handoff state`
- `rg -n "G3pbPeerStorage|delete_blocks|g3pb_admission|G3PB_OFFLOAD_ALL|patch\\.crates-io|nixl-sys|kvbm_g3pb_backend|kvbm_g3pb_worker_smoke" Cargo.toml lib/llm lib/bindings/kvbm third_party/nixl-sys`
  - pass as a seam audit
  - result: the workspace `nixl-sys` patch, backend-side delete/reclaim path,
    bindings-side admission-config adoption, and smoke/backend binaries remain
    present in the live tree
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: no new active `G3PB` implementation gap surfaced in code or the
    design doc; unrelated repo-wide TODO/FIXME markers still exist outside this
    slice
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

- rerun the focused `g3pb::` library test after this `PLANS.md` refresh
- create a signed docs-only handoff refresh commit once the post-edit spot
  check is green

### Exact next step

- run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`, then if it
  passes commit this `PLANS.md` refresh with `git commit --signoff`

### Handoff for next run

- this file is the compact current handoff; do not re-expand it with repeated
  per-run history unless genuinely new scope or a regression appears
- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming any prior
  docs-only tip remains current
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- the most recent full green validation stack in this file was run from
  detached `HEAD` `113347556c34`
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
