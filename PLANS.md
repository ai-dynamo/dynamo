# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-03 20:25:38 UTC

## Active state

- Mandatory read order:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - live detached `HEAD`: `634e7127124f`
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
  - the current live worktree includes uncommitted local edits in:
    - `PLANS.md`
    - `lib/llm/src/bin/kvbm_g3pb_backend.rs`
    - `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
    - `lib/llm/src/block_manager/distributed/g3pb.rs`
  - this run landed a new signed docs-only handoff commit:
    - `634e7127124f` (`docs: refresh g3pb handoff`)
  - the live tree was re-audited again from detached `HEAD` `3f3059523b64`
    in this run instead of trusting the prior handoff
  - the focused `G3PB` validation stack is green on detached
    `HEAD` `3f3059523b64`
  - detached `HEAD` `634e7127124f` also has a green post-commit
    `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` spot check
  - the etcd-backed `kvbm_g3pb_backend` + `kvbm_g3pb_worker_smoke` flow is
    now confirmed working from this container when `ETCD_ENDPOINTS` points at
    `http://172.17.0.1:2379`
  - no code changes or new `G3PB` implementation gaps surfaced in this run
  - `PLANS.md` is intentionally left dirty only with this refreshed on-disk
    handoff for the next run

## Current run (2026-04-01 16:03:41 UTC)

### Summary of accomplishments in this run

- Re-read the required handoff and design context from the live tree:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Reconfirmed the live tree is on detached `HEAD` `3f3059523b64`
- Re-audited the live tree seams rather than assuming the previous handoff was
  still current
- Reconfirmed the live tree still contains the seams required by the handoff
  and design doc:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
  - `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` binaries
- Reconfirmed again that no stronger standalone runnable `G3PB` e2e target is
  present in-tree beyond the existing smoke binary and focused validation stack
- Re-ran the focused `G3PB` validation stack from live detached
  `HEAD` `3f3059523b64`, and it is green
- Refreshed `PLANS.md` on disk with the new audit and validation state
- Re-ran the required post-edit `cargo test --manifest-path
  lib/llm/Cargo.toml g3pb:: --lib` spot check after the handoff refresh, and
  it stayed green
- Landed a signed docs-only handoff commit:
  - `634e7127124f` (`docs: refresh g3pb handoff`)
- Re-read the committed handoff from detached `HEAD` `634e7127124f`,
  confirmed the clean committed state, and re-ran a post-commit `g3pb::` spot
  check before this final on-disk refresh
- No code changes or newly discovered `G3PB` implementation gaps surfaced in
  this run

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on detached
  `HEAD` `3f3059523b64`
- no new `G3PB` implementation gap was identified by the audit or validation
- the design doc's listed next milestones remain follow-on scope rather than
  unfinished execution for the active first-pass slice
- unrelated repo-wide `TODO` and `FIXME` markers still exist, but none surfaced
  as unfinished work for this `G3PB` slice
- no code changes are pending for the active `G3PB` slice
- `PLANS.md` is intentionally left dirty only with this refreshed on-disk
  handoff for the next run

### Validation completed in this run

- `git status --short`
  - pass (`M PLANS.md`) before the initial reread
- `sed -n '1,240p' Agents.md`
  - pass
- `sed -n '1,320p' PLANS.md`
  - pass before refresh
- `sed -n '1,320p' docs/design-docs/kvbm-g3pb-plan.md`
  - pass
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty) during the audit
- `git rev-parse --short=12 HEAD`
  - pass (`3f3059523b64`)
- `git log --oneline -8`
  - pass (recent history remains docs-only handoff refreshes)
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
    `HEAD` `3f3059523b64`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 16:03:41 UTC`) after the focused validation stack
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) as the required post-edit spot check on detached
    `HEAD` `3f3059523b64`
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty) after the on-disk
    handoff refresh
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 16:05:16 UTC`) after the post-edit spot check
- `git add PLANS.md && git commit --signoff -m "docs: refresh g3pb handoff"`
  - pass (`634e7127124f`)
- `git status --short --branch`
  - pass (`## HEAD (no branch)`) immediately after the docs-only commit
- `git rev-parse --short=12 HEAD`
  - pass (`634e7127124f`) after the docs-only commit
- `sed -n '1,260p' PLANS.md`
  - pass after the docs-only commit
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) as the post-commit spot check on detached
    `HEAD` `634e7127124f`
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 16:05:58 UTC`) after the post-commit spot check
- `sed -n '1,280p' PLANS.md`
  - pass as the final reread; remaining work still reads `none`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`) after the final reread of `PLANS.md`
- `git status --short --branch`
  - pass (`## HEAD (no branch)` with only `PLANS.md` dirty) before exit
- `date -u '+%Y-%m-%d %H:%M:%S UTC'`
  - pass (`2026-04-01 16:07:00 UTC`) before exit
- `curl -fsS http://172.17.0.1:2379/health`
  - pass (`{"health":"true","reason":""}`) from inside the current container
- `ETCD_ENDPOINTS=http://172.17.0.1:2379 cargo run --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - pass
  - result: backend registered successfully via etcd-backed discovery
- `ETCD_ENDPOINTS=http://172.17.0.1:2379 cargo run --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_worker_smoke`
  - pass
  - result: worker discovered one healthy backend, completed staged NIXL put,
    query, fetch, local host registration, and device onboard
  - transferred `4 blocks / 32768 bytes` via staged `G3PB` NIXL descriptors

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

- leave the active `G3PB` slice closed unless a fresh regression or explicitly
  new scope appears

### Handoff for next run

- start by re-reading:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- validate from the live detached `HEAD` rather than assuming this handoff
  still matches the checkout
- the validated non-docs implementation baseline remains `abfc85ffd0a4`
- detached `HEAD` `3f3059523b64` had green focused `G3PB` validation in this
  run:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
- this run landed the latest signed docs-only handoff tip `634e7127124f`
- detached `HEAD` `634e7127124f` also had a green post-commit
  `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` spot check
- the current worktree is intentionally dirty only with this refreshed on-disk
  `PLANS.md` handoff
- no stronger standalone runnable `G3PB` e2e target is currently present in the
  tree beyond the smoke binary and focused validation stack
- the active `G3PB` slice still has no pending code changes
- from inside this container, `localhost:2379` is not the reachable etcd
  endpoint; use `ETCD_ENDPOINTS=http://172.17.0.1:2379` for local etcd-backed
  smoke validation
- the etcd-backed smoke path was confirmed green from this container with:
  - `ETCD_ENDPOINTS=http://172.17.0.1:2379 cargo run --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - `ETCD_ENDPOINTS=http://172.17.0.1:2379 cargo run --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_worker_smoke`
- if future work is needed after the handoff is sealed, treat it as separate
  follow-on scope:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption only when a
     real additional caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer or `foyer` retention knobs as a separate
     slice
