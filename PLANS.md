# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 09:29:09 UTC

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

## Current run (2026-04-01 09:29:09 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff and design context:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-audited the live repo state from detached `HEAD`
- ✅ Confirmed the active tree still contains the seams the handoff claims are
  landed:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- ✅ Re-ran the focused `G3PB` and bindings validation stack from current
  detached `HEAD`, and it is green
- ✅ Compacted `PLANS.md` so the next run starts from the actual live validated
  tip instead of the prior run snapshot
- ✅ Re-ran the focused `g3pb::` library test after editing `PLANS.md`
- ⏳ Pending in this run after that spot check:
  - write the signed docs-only handoff refresh commit
  - re-run the focused `g3pb::` library test after that commit
  - re-read `PLANS.md` once more and leave it in a completed no-open-work
    state

### Current findings in this run

- the current detached `HEAD` at validation start is `d34e673fe72b`
- `PLANS.md` was still anchored to the previous run snapshot; the live tree now
  includes:
  - `d34e673fe` (`docs: finalize g3pb handoff refresh`)
  - `eae3ce029` (`docs: refresh g3pb handoff state`)
  - `a5bd8ead0` (`docs: remove self-referential g3pb tip`)
  above the older previously recorded docs-only refreshes
- the live detached `HEAD` still contains the same validated non-docs `G3PB`
  implementation baseline and follow-on code changes
- the active `G3PB` implementation slice still appears complete on the live
  tree
- no new `G3PB` implementation gap was identified by the audit or validation
- the only repo work performed in this run is handoff compaction and refresh

### Validation completed in this run

- `git rev-parse --short=12 HEAD`
  - pass (`d34e673fe72b`)
- `git log --oneline -8`
  - pass
  - current recent history:
    - `d34e673fe docs: finalize g3pb handoff refresh`
    - `eae3ce029 docs: refresh g3pb handoff state`
    - `a5bd8ead0 docs: remove self-referential g3pb tip`
    - `c2a53e97d docs: finalize g3pb handoff refresh`
    - `280530069 docs: refresh g3pb handoff state`
    - `969730570 docs: refresh g3pb handoff state`
    - `2427306c0 docs: refresh g3pb handoff state`
    - `a4fbdf66c docs: stabilize g3pb handoff instructions`
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
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) as a post-edit spot check before the docs-only commit

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- write a signed docs-only handoff refresh commit if the post-edit spot check
  passes
- re-run the same focused `g3pb::` test after that commit
- re-read `PLANS.md` and keep the final handoff compact if no regression
  appears

### Exact next step

- run `git rev-parse --short=12 HEAD && sed -n '1,220p' Agents.md && sed -n
  '1,260p' PLANS.md && sed -n '1,260p' docs/design-docs/kvbm-g3pb-plan.md`,
  then leave the active `G3PB` slice closed unless a new regression or
  explicitly new scope appears

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
  detached `HEAD` `d34e673fe72b`
- if this run stops before the post-edit spot check and signed docs-only
  refresh commit, resume with:
  1. `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  2. `git add PLANS.md && git commit --signoff -m "docs: refresh g3pb handoff state"`
  3. `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  4. `sed -n '1,260p' PLANS.md`
- if future work is needed, treat it as separate follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
