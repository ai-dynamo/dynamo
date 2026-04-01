# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 15:21:06 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - live detached `HEAD`: `a43320b0ba4d`
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
  - this run has completed the required doc reread, a fresh live-tree seam
    audit, and the focused validation stack from the live tree
  - `PLANS.md` is intentionally left dirty during this run only for the active
    handoff refresh; the required post-edit spot check, signed docs-only commit,
    and post-commit reread/spot check still remain to be executed

## Current run (2026-04-01 15:21:06 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Reconfirmed the live tree started on detached `HEAD` `a43320b0ba4d`
- Re-ran the live-tree seam audit instead of assuming the previous handoff
  remained current
- Reconfirmed the live tree still contains the seams required by the handoff
  and design doc:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Reconfirmed again that no stronger standalone runnable `G3PB` e2e target is
  present in-tree beyond the existing smoke binary and focused validation stack
- Re-ran the focused `G3PB` validation stack from live detached
  `HEAD` `a43320b0ba4d`, and it is green
- Refreshed `PLANS.md` on disk with the new audit and validation state

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on detached
  `HEAD` `a43320b0ba4d`
- no new `G3PB` implementation gap was identified by the audit or validation
- the design doc's listed next milestones remain follow-on scope rather than
  unfinished execution for the active first-pass slice
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- no code changes are pending for the active `G3PB` slice
- before exit, this run still needs the required post-edit `g3pb::` spot check,
  a signed docs-only handoff commit, a post-commit reread of `PLANS.md`, and a
  final post-commit `g3pb::` spot check

### Validation completed in this run

- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty) at run start
- `sed -n '1,220p' Agents.md`
  - pass
- `sed -n '1,260p' PLANS.md`
  - pass before refresh
- `sed -n '1,260p' docs/design-docs/kvbm-g3pb-plan.md`
  - pass
- `git rev-parse --short=12 HEAD`
  - pass (`a43320b0ba4d`)
- `git log --oneline -8`
  - pass (recent history still remains docs-only handoff refreshes)
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
- `rg -n "g3pb.*(smoke|e2e)|G3PB.*(smoke|e2e)|worker_smoke|end-to-end|end to end" lib/llm lib/bindings tests docs -g '!target'`
  - pass as an end-to-end test inventory search
  - result: no standalone runnable `G3PB` e2e target surfaced beyond the
    existing `kvbm_g3pb_worker_smoke` binary and the focused validation stack
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) in the focused validation stack on detached
    `HEAD` `a43320b0ba4d`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 15:21:06 UTC`) after the focused validation stack
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty) after the on-disk
    handoff refresh

### Decisions confirmed in this run

- keep treating the active `G3PB` slice as complete unless a fresh regression
  appears
- keep peer-local disk details behind `G3pbPeerStorage`; do not reopen the
  unlanded `G4` identity surface
- treat broader admission-policy adoption, `nixl-sys` upstreaming, and any
  future retention tuning as separate follow-on scope rather than unfinished
  plan execution

### Remaining work in this run

- run the required post-edit `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` spot check
- create a signed docs-only handoff commit if the post-edit spot check stays green
- re-read the committed `PLANS.md`, rerun a post-commit `g3pb::` spot check,
  and leave a final precise handoff on disk if another run starts from the
  resulting worktree state

### Exact next step

- run `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` as the
  required post-edit spot check before attempting a docs-only handoff commit

### Handoff for next run

- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming this handoff
  still matches the checkout
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- detached `HEAD` `a43320b0ba4d` had green focused `G3PB` validation in this
  run:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
- no stronger standalone runnable `G3PB` e2e target is currently present in the
  tree beyond the smoke binary and focused validation stack
- the active `G3PB` slice still has no pending code changes
- this on-disk refresh is intentionally dirty until the required post-edit and
  post-commit `g3pb::` spot checks and the signed docs-only handoff commit are
  completed
- if future work is needed after the handoff is sealed, treat it as separate
  follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
