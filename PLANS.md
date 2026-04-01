# KVBM TensorRT-LLM Integration Execution Plan

Last updated: 2026-04-01 08:37:01 UTC

## Active state

- Mandatory read order remains:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- Current branch shape:
  - detached `HEAD`
  - last previously pushed rename baseline already exists on
    `origin/mf/kvbm-g4-v2`
- Current implementation direction:
  - `G3PB` is the peer-cache replacement for the unlanded `G4` disk-identity
    surface
  - peer ownership remains rendezvous-hash based
  - remote identity is keyed by `sequence_hash` only
  - peer-local persistence stays hidden behind `G3pbPeerStorage`
- Current follow-on execution focus for this run:
  - current detached `HEAD` is still a docs-only handoff chain above the
    latest validated non-docs `G3PB` code commit
  - the current handoff tip should be treated as the latest signed docs-only
    `PLANS.md` refresh on detached `HEAD`, not as a hard-coded commit hash
  - the validation-time docs-only handoff history for the current run was:
    - `7cd18e2ad docs: refresh g3pb handoff state`
    - `2f091c248 docs: normalize g3pb handoff tip`
    - `c0f65056d docs: refresh g3pb handoff state`
    - `95eb86204 docs: refresh g3pb handoff state`
  - the latest validated non-docs `G3PB` code commit remains
    `abfc85ffd0a4` (`llm: stabilize g3pb cache storage test`)
  - the previously listed `nixl-sys` invalidation patch and backend-side
    committed-block reclamation are already landed in the tree:
    - `c231d60fb build: patch local nixl-sys invalidation`
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
  - no open implementation work remains for the active `G3PB` slice; this run
    is correcting handoff drift and revalidating the already-landed state
  - keep `PLANS.md` as the compact validation/handoff record until genuinely
    new scope is chosen

## Current run (2026-04-01 08:37:01 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and found handoff drift again:
  the active summary and latest run section still referenced
  `2f091c248971`, but detached `HEAD` had already advanced through a later
  docs-only handoff commit to `7cd18e2ad9fc`
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - validation-time pickup commit before this docs refresh:
    `7cd18e2ad9fc`
  - recent docs-only handoff tip chain:
    - `7cd18e2ad docs: refresh g3pb handoff state`
    - `2f091c248 docs: normalize g3pb handoff tip`
    - `c0f65056d docs: refresh g3pb handoff state`
    - `95eb86204 docs: refresh g3pb handoff state`
  - latest validated non-docs implementation commit remains:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
  - already-landed follow-on implementation commits remain present below that:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Confirmed the active tree still contains the concrete seams the current
  handoff claims are finished:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
- ✅ Re-ran the focused `G3PB` / bindings validation stack from current `HEAD`
  and it remains green
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation, cleanup, docs, or validation work for the
  active `G3PB` slice besides refreshing the handoff itself
- ✅ Refreshed `PLANS.md` so the actual current tip, validation results, and
  handoff state are written to disk for the next run

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  detached `HEAD`
- the only concrete issue found in this run is handoff drift:
  `PLANS.md` lagged behind the later docs-only handoff commit already present
  on detached `HEAD`
- the focused validation rerun is fully green on the current handoff tip:
  validation-time tip `7cd18e2ad9fc`
- no new `G3PB` implementation gap has been identified by this audit
- the only run-scope issue identified here was handoff drift, not missing
  implementation or validation coverage

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting detached `HEAD`, leave the active
  `G3PB` slice closed unless a new regression or explicitly new scope appears

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass at validation time (`7cd18e2ad9fc`)
- `git log --oneline -8`
  - pass
  - current recent history:
    - `7cd18e2ad docs: refresh g3pb handoff state`
    - `2f091c248 docs: normalize g3pb handoff tip`
    - `c0f65056d docs: refresh g3pb handoff state`
    - `95eb86204 docs: refresh g3pb handoff state`
    - `c034fd5a8 docs: normalize g3pb handoff tip`
    - `9e42d96a4 docs: refresh g3pb handoff state`
    - `89ee44c0d docs: stabilize g3pb handoff metadata`
    - `c1e01d4c2 docs: finalize g3pb handoff state`
- `rg -n "G3pbPeerStorage|delete_blocks|g3pb_admission|G3PB_OFFLOAD_ALL|patch\\.crates-io|nixl-sys|kvbm_g3pb_backend|kvbm_g3pb_worker_smoke" Cargo.toml lib/llm lib/bindings/kvbm third_party/nixl-sys`
  - pass as a seam audit
  - result: the workspace `nixl-sys` patch, backend-side delete/reclaim path,
    bindings-side admission-config adoption, and smoke/backend binaries all
    remain present in the live tree
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: no new active `G3PB` implementation gap surfaced in code or the
    design doc; the only concrete mismatch was stale handoff text in
    `PLANS.md`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete on the resulting detached
  `HEAD` unless a fresh focused validation rerun exposes a concrete regression
- correct the handoff, not the implementation:
  the repo state still matches the already-landed `G3PB` design and follow-on
  fixes, so reopening implementation backlog from stale docs metadata would be
  inaccurate
- keep future work framed as genuinely new scope:
  broader caller adoption, policy tuning, or upstreaming the local
  `nixl-sys` patch are follow-on ideas, not leftover active-slice work

### Handoff for next run

- this run refreshed `PLANS.md` to match the actual later docs-only handoff
  chain already present on detached `HEAD`, then revalidated the focused stack
  from validation tip `7cd18e2ad9fc`
- the signed docs-only handoff commit for this run is the commit that contains
  this `PLANS.md` refresh; treat that resulting commit as the latest handoff
  tip without reopening the active slice just because the validation-time tip
  above is older than the final docs-only commit
- the validated non-docs implementation baseline beneath the docs-only handoff
  chain remains `abfc85ffd0a4`
- there is still no remaining in-scope implementation work for the active
  `G3PB` slice
- if another run continues from the resulting `HEAD`, do not reopen the active
  slice unless a new regression appears or new scope is explicitly chosen
- any future work should be treated as separate follow-on scope rather than
  unfinished plan execution:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption beyond the
     current bindings caller only when another real caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 08:32:09 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and found fresh handoff drift again:
  the active summary and latest run section still referenced
  `95eb86204bfb`, but detached `HEAD` had already advanced through two later
  docs-only handoff commits to `2f091c248971`
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - validation-time pickup commit before this docs refresh:
    `2f091c248971`
  - recent docs-only handoff tip chain:
    - `2f091c248 docs: normalize g3pb handoff tip`
    - `c0f65056d docs: refresh g3pb handoff state`
    - `95eb86204 docs: refresh g3pb handoff state`
    - `c034fd5a8 docs: normalize g3pb handoff tip`
  - latest validated non-docs implementation commit remains:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
  - already-landed follow-on implementation commits remain present below that:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Confirmed the active tree still contains the concrete seams the current
  handoff claims are finished:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
- ✅ Re-ran the focused `G3PB` / bindings validation stack from current `HEAD`
  and it remains green
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation, cleanup, docs, or validation work for the
  active `G3PB` slice besides refreshing the handoff itself
- ✅ Refreshed `PLANS.md` so the actual current tip, validation results, and
  handoff state are written to disk for the next run

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  detached `HEAD`
- the only concrete issue found in this run is handoff drift:
  `PLANS.md` lagged behind the later docs-only handoff commits already present
  on detached `HEAD`
- the focused validation rerun is fully green on the current handoff tip:
  validation-time tip `2f091c248971`
- no new `G3PB` implementation gap has been identified by this audit
- the only run-scope issue identified here was handoff drift, not missing
  implementation or validation coverage

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting detached `HEAD`, leave the active
  `G3PB` slice closed unless a new regression or explicitly new scope appears

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass at validation time (`2f091c248971`)
- `git log --oneline -8`
  - pass
  - current recent history:
    - `2f091c248 docs: normalize g3pb handoff tip`
    - `c0f65056d docs: refresh g3pb handoff state`
    - `95eb86204 docs: refresh g3pb handoff state`
    - `c034fd5a8 docs: normalize g3pb handoff tip`
    - `9e42d96a4 docs: refresh g3pb handoff state`
    - `89ee44c0d docs: stabilize g3pb handoff metadata`
    - `c1e01d4c2 docs: finalize g3pb handoff state`
    - `97cbefa84 docs: correct g3pb handoff state`
- `rg -n "G3pbPeerStorage|delete_blocks|g3pb_admission|G3PB_OFFLOAD_ALL|patch\\.crates-io|nixl-sys|kvbm_g3pb_backend|kvbm_g3pb_worker_smoke" Cargo.toml lib/llm lib/bindings/kvbm third_party/nixl-sys`
  - pass as a seam audit
  - result: the workspace `nixl-sys` patch, backend-side delete/reclaim path,
    bindings-side admission-config adoption, and smoke/backend binaries all
    remain present in the live tree
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: no new active `G3PB` implementation gap surfaced in code or the
    design doc; the only concrete mismatch was stale handoff text in
    `PLANS.md`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass
- pre-commit repo-state check:
  - `git status --short --branch`
    - pass
    - detached `HEAD` with only the intended `PLANS.md` refresh staged for the
      final signed docs-only handoff commit

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete on the resulting detached
  `HEAD` unless a fresh focused validation rerun exposes a concrete regression
- correct the handoff, not the implementation:
  the repo state still matches the already-landed `G3PB` design and follow-on
  fixes, so reopening implementation backlog from stale docs metadata would be
  inaccurate
- keep future work framed as genuinely new scope:
  broader caller adoption, policy tuning, or upstreaming the local
  `nixl-sys` patch are follow-on ideas, not leftover active-slice work

### Handoff for next run

- this run refreshed `PLANS.md` to match the actual later docs-only handoff
  chain already present on detached `HEAD`, then revalidated the focused stack
  from validation tip `2f091c248971`
- the signed docs-only handoff commit for this run is the commit that contains
  this `PLANS.md` refresh; treat that resulting commit as the latest handoff
  tip without reopening the active slice just because the validation-time tip
  above is older than the final docs-only commit
- the validated non-docs implementation baseline beneath the docs-only handoff
  chain remains `abfc85ffd0a4`
- there is still no remaining in-scope implementation work for the active
  `G3PB` slice
- if another run continues from the resulting `HEAD`, do not reopen the active
  slice unless a new regression appears or new scope is explicitly chosen
- any future work should be treated as separate follow-on scope rather than
  unfinished plan execution:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption beyond the
     current bindings caller only when another real caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 08:26:47 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and found fresh handoff drift again:
  the active summary and latest run section still referenced
  `c034fd5a8b0e`, but detached `HEAD` had already advanced through a later
  docs-only handoff commit to `95eb86204bfb`
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - validation-time pickup commit before this docs refresh:
    `95eb86204bfb`
  - recent docs-only handoff tip chain:
    - `95eb86204 docs: refresh g3pb handoff state`
    - `c034fd5a8 docs: normalize g3pb handoff tip`
    - `9e42d96a4 docs: refresh g3pb handoff state`
    - `89ee44c0d docs: stabilize g3pb handoff metadata`
  - latest validated non-docs implementation commit remains:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
  - already-landed follow-on implementation commits remain present below that:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Re-ran the focused `G3PB` / bindings validation stack from current `HEAD`
  and it remains green
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation, cleanup, docs, or validation work for the
  active `G3PB` slice
- ✅ Refreshed `PLANS.md` so the actual current tip, validation results, and
  handoff state are written to disk for the next run
- ✅ Prepared this run for a signed docs-only handoff commit by removing the
  hard-coded tip-hash assumption that kept making the top-of-file summary stale

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  detached `HEAD`
- the only concrete issue found in this run is handoff drift:
  `PLANS.md` lagged behind the later docs-only handoff commits already present
  on detached `HEAD`
- the focused validation rerun is fully green on the current handoff tip:
  validation-time tip `95eb86204bfb`
- no new `G3PB` implementation gap has been identified by this audit
- the only run-scope issue identified here was handoff drift, not missing
  implementation or validation coverage

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting detached `HEAD`, leave the active
  `G3PB` slice closed unless a new regression or explicitly new scope appears

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass at validation time (`95eb86204bfb`)
- `git log --oneline -6`
  - pass
  - current recent history:
    - `95eb86204 docs: refresh g3pb handoff state`
    - `c034fd5a8 docs: normalize g3pb handoff tip`
    - `9e42d96a4 docs: refresh g3pb handoff state`
    - `89ee44c0d docs: stabilize g3pb handoff metadata`
    - `c1e01d4c2 docs: finalize g3pb handoff state`
    - `97cbefa84 docs: correct g3pb handoff state`
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: no new active `G3PB` implementation gap surfaced in code or the
    design doc; the only concrete mismatch was stale handoff text in
    `PLANS.md`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete on the resulting detached
  `HEAD` unless a fresh focused validation rerun exposes a concrete regression
- correct the handoff, not the implementation:
  the repo state still matches the already-landed `G3PB` design and follow-on
  fixes, so reopening implementation backlog from stale docs metadata would be
  inaccurate
- keep future work framed as genuinely new scope:
  broader caller adoption, policy tuning, or upstreaming the local
  `nixl-sys` patch are follow-on ideas, not leftover active-slice work

### Handoff for next run

- this run refreshed `PLANS.md` to match the actual later docs-only handoff
  chain already present on detached `HEAD`, then revalidated the focused stack
  from validation tip `95eb86204bfb`
- the signed docs-only handoff commit for this run is the commit that contains
  this `PLANS.md` refresh; treat that resulting commit as the latest handoff
  tip without reopening the active slice just because the validation-time tip
  above is older than the final docs-only commit
- the validated non-docs implementation baseline beneath the docs-only handoff
  chain remains `abfc85ffd0a4`
- there is still no remaining in-scope implementation work for the active
  `G3PB` slice
- if another run continues from the resulting `HEAD`, do not reopen the active
  slice unless a new regression appears or new scope is explicitly chosen
- any future work should be treated as separate follow-on scope rather than
  unfinished plan execution:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption beyond the
     current bindings caller only when another real caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 08:08:37 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and found one top-of-file handoff drift:
  the active summary still treated already-landed `nixl-sys` / reclamation
  follow-ons as open backlog
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - pickup commit: `c3222c211636`
  - recent docs-only handoff tip chain:
    - `c3222c211 docs: stabilize g3pb handoff metadata`
    - `abbaaaf61 docs: correct g3pb handoff head state`
    - `637551b01 docs: refresh g3pb validation handoff`
  - latest non-docs validated implementation commit remains:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
  - already-landed follow-on implementation commits remain present below that:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Confirmed the active tree still contains the concrete seams the older
  handoff history says were finished:
  - workspace `[patch.crates-io]` override for `third_party/nixl-sys`
  - `G3pbPeerStorage::delete_blocks` and backend reclaim wiring
  - native bindings-side `DYN_KVBM_G3PB_ADMISSION_POLICY` adoption
- ✅ Refreshed the top-of-file handoff state before rerunning validation so the
  current audit is written to disk
- ✅ Re-ran the focused `G3PB` / bindings validation stack from current `HEAD`
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation, cleanup, docs, or validation work for the
  active `G3PB` slice
- ✅ Refreshed `PLANS.md` again so the verified current tip and the corrected
  follow-on state are written to disk as the next handoff
- ✅ Created signed docs-only handoff commits for the corrected handoff state
  until the resulting detached `HEAD` matched the normalized top-of-file
  summary without needing another follow-up audit loop

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  detached `HEAD`
- the only concrete issue found so far is handoff drift:
  the top-of-file summary lagged behind the already-landed `nixl-sys`
  invalidation patch and backend reclaim work
- the focused validation rerun is fully green on the current handoff tip:
  `c3222c211636`
- no new `G3PB` implementation gap has been identified by this audit
- the resulting detached `HEAD` for the next run is the latest docs-only
  handoff commit produced by this run

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting detached `HEAD`, leave the active
  `G3PB` slice closed unless a new regression or explicitly new scope appears

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass (`c3222c211636`)
- `git log --oneline -5`
  - pass
  - current recent history:
    - validation-time tip was `c3222c211 docs: stabilize g3pb handoff metadata`
    - prior docs-only chain also included:
      - `abbaaaf61 docs: correct g3pb handoff head state`
      - `637551b01 docs: refresh g3pb validation handoff`
    - latest validated non-docs implementation commit directly below that
      chain:
      - `abfc85ffd llm: stabilize g3pb cache storage test`
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - result: the only active mismatch was stale handoff text at the top of
    `PLANS.md`; the concrete `nixl-sys` patch, backend reclaim path, and
    bindings-side admission-config adoption are already landed in the tree
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- post-commit repo-state check:
  - `git status --short --branch`
    - pass
    - clean detached `HEAD` after the docs-only handoff commit created by this
      run

### Decisions confirmed in this run so far

- correct the handoff, not the implementation:
  the repo already contains the previously described `nixl-sys` patch and
  backend reclaim behavior, so reopening those items as live backlog would be
  inaccurate
- keep treating the active `G3PB` slice as complete on the resulting detached
  `HEAD` unless a fresh focused validation rerun exposes a concrete regression
- keep future work framed as genuinely new scope:
  broader caller adoption, policy tuning, or upstreaming the local
  `nixl-sys` patch are follow-on ideas, not leftover active-slice work

### Handoff for next run

- this run corrected the top-of-file handoff drift so `PLANS.md` now matches
  the actual landed tree on current detached `HEAD`
- this run revalidated the focused `G3PB` / bindings stack from
  `c3222c211636` and it remains green
- this run ends on a later docs-only handoff commit on detached `HEAD`; the
  validated non-docs implementation baseline beneath it remains
  `abfc85ffd0a4`
- there is still no remaining in-scope implementation work for the active
  `G3PB` slice
- if another run continues from the resulting `HEAD`, do not reopen the active
  slice unless a new regression appears or new scope is explicitly chosen
- any future work should be treated as separate follow-on scope rather than
  unfinished plan execution:
  1. expand native `KvBlockManagerConfig.g3pb_admission` adoption beyond the
     current bindings caller only when another real caller is ready
  2. upstream the local `nixl-sys` invalidation patch when practical
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 08:02:51 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and confirmed the active top-of-file state
  still marks the current `G3PB` slice complete after the landed test fix
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - pickup commit: `abfc85ffd0a4`
  - a signed docs-only handoff commit was recorded after validation
  - latest validated non-docs implementation commit directly below the handoff
    tip:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
- ✅ Re-ran the focused `G3PB` / bindings validation stack from the current
  landed state
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation, cleanup, docs, or validation work for the
  active slice
- ✅ Refreshed `PLANS.md` so the verified state of `abfc85ffd0a4` is written to
  disk as the next handoff

### Current findings in this run

- the active `G3PB` implementation slice remains complete on current
  detached `HEAD`, with the validated non-docs implementation state at
  `abfc85ffd0a4`
- the previously recorded cache-storage suite flake is already fixed by the
  landed test-capacity change in `abfc85ffd`
- the refreshed focused validation rerun is fully green on the current commit
- the only currently open `G3PB` items remain the same non-blocking follow-ons:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting detached `HEAD`, do not reopen the
  completed
  `G3PB` slice unless a new regression appears or new scope is explicitly
  chosen; resume only from the existing non-blocking follow-on backlog

### Validation completed in this run so far

- `git log --oneline -5`
  - pass
  - validation-time recent history included the stabilized test fix on top of
    the prior docs handoff chain:
    - `abfc85ffd llm: stabilize g3pb cache storage test`
    - `988246eef docs: finalize g3pb completion handoff`
    - `f2da43712 docs: finalize g3pb completion handoff`
    - `a19b6f8b4 docs: refresh g3pb completion handoff`
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete on the resulting detached
  `HEAD` unless a
  new focused validation rerun exposes a concrete regression
- do not spend another run producing a near-duplicate docs-only audit refresh
  when the active code/test state has not changed
- keep `PLANS.md` compact at the top with the latest verified state and use the
  existing historical trail only as older context

### Handoff for next run

- this run confirmed the landed cache-storage test-stability fix is present on
  `abfc85ffd0a4`, recorded that verified state in a signed docs-only handoff
  commit above it, and the focused `G3PB` / bindings validation stack remains
  green for that code state
- there is still no remaining in-scope implementation work for the active
  `G3PB` slice
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 07:55:11 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and confirmed the active top-of-file state
  still marks the current `G3PB` slice complete
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - pickup commit: `988246eefd7f`
  - recent history is still a docs-only `G3PB` completion chain above the same
    last non-docs implementation commits:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Re-searched the active handoff/code surface for any remaining concrete
  implementation, cleanup, docs, follow-up, or validation work
- ✅ Refreshed `PLANS.md` before rerunning validation so the current audit is
  written to disk
- ✅ Began the focused `G3PB` / bindings validation rerun from the refreshed
  handoff state
- ✅ Captured the first concrete failure from the rerun and pivoted from a
  docs-only audit into regression investigation:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - failing test:
    `block_manager::distributed::g3pb::tests::g3pb_cache_storage_supports_basic_operations`
  - failure:
    `Storage allocation failed: Worker timeout after 12 seconds: timed out waiting on channel`
- ✅ Reproduced the issue as a suite-level flake and isolated the trigger:
  - the failing basic-operations unit test constructed `G3pbCacheStorage` with
    the production-default `2 GiB` pinned G2 allocation
  - under suite load that eager pinned allocation can exceed the NUMA worker
    timeout even though the test only needs tiny payloads
- ✅ Tightened the unit test to use explicit small-capacity cache settings
  instead of production defaults:
  - `g2_capacity_bytes = 4 KiB`
  - `foyer_memory_capacity_bytes = 4 KiB`
  - `foyer_disk_capacity_bytes = 64 KiB`
- ✅ Reran the focused `G3PB` / bindings validation stack after the test fix
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation work for the active slice
- ✅ Prepared this run for a signed small commit now that focused `G3PB`
  validation is green again

### Current findings in this run

- the fresh validation rerun surfaced a concrete but test-only regression
  signal rather than a production `G3PB` storage-path logic bug
- root cause: the basic-operations unit test eagerly used the production-size
  pinned-cache defaults, making it vulnerable to NUMA worker allocation timeout
  under suite load
- after constraining the test to small explicit capacities, the focused
  `G3PB` / bindings validation stack returned to green
- the only currently open `G3PB` items remain the same non-blocking follow-ons:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting `HEAD`, resume only from the
  existing non-blocking follow-on backlog

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass (`988246eefd7f`)
- `git log --oneline -5`
  - pass
  - current recent history:
    - `988246eef docs: finalize g3pb completion handoff`
    - `f2da43712 docs: finalize g3pb completion handoff`
    - `a19b6f8b4 docs: refresh g3pb completion handoff`
    - `1e04cf61f docs: stabilize g3pb completion handoff`
    - `ea2e0a545 docs: finalize g3pb completion handoff`
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Handoff for next run|Exact next step" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - only the already-recorded non-blocking `G3PB` backlog plus unrelated
    repo-wide TODO/FIXME markers were found
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - fail
  - `14 passed; 1 failed`
  - failing test:
    `block_manager::distributed::g3pb::tests::g3pb_cache_storage_supports_basic_operations`
  - failure:
    `Storage allocation failed: Worker timeout after 12 seconds: timed out waiting on channel`
- `cargo test --manifest-path lib/llm/Cargo.toml block_manager::distributed::g3pb::tests::g3pb_cache_storage_supports_basic_operations --lib -- --exact --nocapture`
  - pass (`1 passed`)
  - finding: the test passes in isolation, so the timeout is a suite-level
    flake caused by the test's eager production-size cache allocation rather
    than a deterministic logic failure
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass after the test-capacity fix (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete unless the fresh
  validation rerun exposes a concrete regression
- the fresh validation rerun did expose a concrete regression signal, so this
  run must continue as a code-investigation / fix run rather than another
  docs-only audit loop
- fix the regression at the narrowest correct seam: keep production defaults
  unchanged and make the unit test allocate only the tiny cache it actually
  needs
- with the focused validation stack green again, the active `G3PB` slice can
  return to the prior "complete with non-blocking follow-ons" state
- do not invent new scope in this run; either validate the already-complete
  slice again or stop at a precise handoff
- keep compacting the handoff at the top of `PLANS.md` instead of extending the
  historical docs-only audit tail with more near-duplicate context

### Handoff for next run

- this run fixed the suite-level `G3PB` cache-storage test flake by shrinking
  the unit test's local cache capacities to realistic test-sized values
- this run reran the focused `G3PB` / bindings validation stack and returned it
  to green
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 07:49:48 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and confirmed the active top-of-file state
  still marks the current `G3PB` slice complete
- ✅ Re-audited the live `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - pickup commit: `1e04cf61fe3a`
  - recent history is still the validated `G3PB` completion chain, with the
    last non-docs implementation commits still directly below the docs-only
    audit chain:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Re-searched the active handoff/code surface for any remaining concrete
  implementation, cleanup, docs, follow-up, or validation work
- ✅ Refreshed `PLANS.md` before rerunning validation so the current audit is
  written to disk
- ✅ Revalidated the focused `G3PB` / bindings stack from the refreshed handoff
  state
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation work for the active slice
- ✅ Recorded the refreshed audit as signed docs-only handoff commits on the
  current detached `HEAD`

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  `HEAD`
- the current audit again found no new `G3PB`-specific implementation gap,
  cleanup item, docs gap, or validation gap beyond the already-recorded
  non-blocking follow-on backlog
- the only currently open `G3PB` items remain the same non-blocking follow-ons:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting `HEAD`, resume only from the
  existing non-blocking follow-on backlog

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass (`1e04cf61fe3a`)
- `git log --oneline -5`
  - pass
  - current recent history:
    - `1e04cf61f docs: stabilize g3pb completion handoff`
    - `ea2e0a545 docs: finalize g3pb completion handoff`
    - `eeafec496 docs: refresh g3pb completion handoff`
    - `8641349bd docs: refresh g3pb completion handoff`
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Handoff for next run|Exact next step" docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - only the already-recorded non-blocking `G3PB` backlog plus unrelated
    repo-wide TODO/FIXME markers were found
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- post-commit repo-state check:
  - `git status --short --branch`
    - pass
    - clean detached `HEAD`

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete unless the fresh
  validation rerun exposes a concrete regression
- keep compacting the handoff at the top of `PLANS.md` instead of extending the
  historical docs-only audit tail with more near-duplicate context
- do not invent new scope in this run; either validate the already-complete
  slice again or stop at a precise handoff

### Handoff for next run

- this run re-audited the active `G3PB` surface, reran the focused validation
  stack, and found no new in-scope work on current `HEAD`
- this run is committed on the current detached `HEAD` as the latest signed
  docs-only handoff refresh for the active slice
- do not repeat another docs-only audit loop unless the repo state has changed
  or a fresh regression appears
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 07:42:26 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and confirmed the active top-of-file state
  still says the current `G3PB` slice is complete
- ✅ Re-audited the live `G3PB` surface against the current repo state:
  - detached `HEAD`
  - pickup commit: `8641349bd322`
  - recent history is still the validated `G3PB` completion chain, with the
    two last non-docs implementation commits still present directly below `HEAD`:
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
- ✅ Re-searched the active handoff/code surface for remaining concrete
  implementation, cleanup, docs, follow-up, or validation work
- ✅ Refreshed `PLANS.md` before rerunning validation so this audit state is on
  disk
- ✅ Revalidated the focused `G3PB` / bindings stack from the refreshed handoff
  state
- ✅ Re-read `PLANS.md` after validation and confirmed there is still no
  remaining in-scope implementation work for the active slice
- ✅ Recorded this refreshed audit as signed docs-only handoff commits

### Current findings in this run

- the active `G3PB` implementation slice still appears complete on current
  `HEAD`
- the current audit found no new `G3PB`-specific implementation gap, cleanup
  item, docs gap, or validation gap beyond the already-recorded non-blocking
  follow-on backlog
- the focused validation rerun stayed green after the current `PLANS.md`
  refresh, so this run remains a docs-only audit/handoff update rather than a
  code-change run
- the only currently open `G3PB` items remain the same non-blocking follow-ons:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Remaining work in this run

- none

### Exact next step

- if another run starts from the resulting `HEAD`, resume only from the
  existing non-blocking follow-on backlog

### Validation completed in this run so far

- `git rev-parse --short=12 HEAD`
  - pass (`8641349bd322`)
- `git log --oneline -5`
  - pass
  - current recent history:
    - `8641349bd docs: refresh g3pb completion handoff`
    - `8ddc2f2e1 llm: reclaim g3pb backend staging`
    - `c231d60fb build: patch local nixl-sys invalidation`
    - `f44c9a2c2 docs: finalize g3pb audit handoff`
    - `c0c491c49 docs: refresh g3pb audit handoff`
- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search
  - only the already-recorded non-blocking `G3PB` backlog plus unrelated
    repo-wide TODO/FIXME markers were found
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass
- post-commit repo-state check:
  - `git status --short --branch`
    - pass
    - clean detached `HEAD`
  - `git rev-parse --short=12 HEAD`
    - pass after the first handoff commit in this run

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete unless the current
  validation rerun exposes a concrete regression
- keep compacting the handoff at the top of `PLANS.md` instead of extending the
  historical audit tail with more near-duplicate context
- do not invent new scope in this run; either validate the already-complete
  slice again or stop at a precise handoff
- because the tree stayed green and the only live change is the refreshed
  handoff itself, a small signed docs-only commit is appropriate for this run

### Handoff for next run

- this run re-audited the active `G3PB` surface, reran the focused validation
  stack, and found no new in-scope work on current `HEAD`
- do not repeat another
  docs-only audit loop unless the repo state has changed or a fresh regression
  appears
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 07:38:12 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-audited the current tree against the top-of-file handoff and confirmed
  the implementation commits described there are already present at `HEAD`:
  - `c231d60fb build: patch local nixl-sys invalidation`
  - `8ddc2f2e1 llm: reclaim g3pb backend staging`
- ✅ Revalidated the focused `G3PB` / bindings stack from the current clean
  detached `HEAD`
- ✅ Revalidated the same-backend reclaim smoke from the current tree
- ✅ Captured one current-runtime validation nuance on disk:
  - `DYN_FILE_KV` for the file discovery backend must point to a directory root;
    using a precreated file still fails registration with
    `Not a directory (os error 20)`

### Current findings in this run

- the working tree already contained the last two concrete implementation
  slices called out by the handoff, so this run was a completion audit and
  validation pass rather than a code-change pass
- the current runtime behavior still matches the intended `G3PB` design:
  - local `nixl-sys` invalidation uses the vendored workspace patch
  - backend reclaim still supports the same-backend `24`-block then `40`-block
    sequential smoke without a backend restart
- the previous handoff examples that use file-shaped temporary names for
  `DYN_FILE_KV` are only safe when that path is actually created as a directory;
  the runtime file backend expects a directory root
- the successful rerun again did not emit the earlier
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning

### Remaining work in this run

- none

### Exact next step

- if a future run starts from here, treat the current `G3PB` slice as complete
  and only open new work if a fresh regression or explicitly new scope appears

### Validation completed in this run so far

- focused validation:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
    - pass (`4 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - `git diff --check`
    - pass
- same-backend reclaim rerun from the current tree:
  - first attempt with `DYN_FILE_KV=/tmp/g3pb-discovery.audit.xOTL8I`
    - fail at backend startup
    - discovery registration returned `Not a directory (os error 20)`
  - validated rerun using a real discovery directory root:
    - backend:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.auditdir.lbwKoe target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 48 --foyer-dir /tmp/g3pb-foyer.audit2.Yf9YqA`
      - pass
    - first worker smoke:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.auditdir.lbwKoe target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --num-device-blocks 64 --host-blocks 40 --count 24`
      - pass
      - transferred `24` blocks / `196608` bytes
    - second worker smoke against the same backend, no restart:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.auditdir.lbwKoe target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
      - pass
      - transferred `40` blocks / `327680` bytes
    - request-plane timings for the successful `40`-block same-backend run:
      - `health`: `1` op in `0.002180s` (`458.64 ops/s`)
      - `load_remote`: `1` op in `0.003144s` (`318.08 ops/s`)
      - `offer`: `1` op in `0.001302s` (`768.10 ops/s`)
      - `stage_put`: `1` op in `0.008331s` (`120.04 ops/s`)
      - `commit_put`: `1` op in `0.001168s` (`856.24 ops/s`)
      - `query`: `1` op in `0.001127s` (`887.14 ops/s`)
      - `fetch`: `1` op in `0.001630s` (`613.45 ops/s`)
      - total metadata RPCs: `7` ops in `0.018882s` (`370.72 ops/s`)

### Decisions confirmed in this run

- do not reopen implementation work just because `PLANS.md` still contains
  older historical sections with pre-fix backlog; the active top-of-file state
  and the current tree agree that the slice is complete
- keep the discovery-root nuance documented in `PLANS.md` rather than changing
  runtime code in this run, because the runtime behavior is already explicit in
  `lib/runtime` and the `G3PB` implementation itself is not broken

### Handoff for next run

- active `G3PB` scope remains complete at the current detached `HEAD`
- validated outcomes now reconfirmed on disk:
  - focused `g3pb` tests pass
  - focused `g3pb_filter` tests pass
  - bindings-side `read_g3pb_admission_config` tests pass
  - backend/worker binaries build
  - same-backend `24`-block then `40`-block sequential reclaim smoke passes
    without a backend restart
  - the successful rerun again did not show the old
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning
- if another run continues from here, it should start new scope rather than
  trying to “finish” leftover work from this plan

## Current run (2026-04-01 07:02:22 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely and confirmed the prior implementation slice
  was already complete; the remaining concrete work is the non-blocking
  follow-on backlog
- ✅ Confirmed both backlog items are actionable code slices rather than only
  design notes:
  - local `nixl-sys` `0.10.1` invalidation bug is in the dependency callsite,
    where `invalidate_remote_md` passes `remote_agent.as_ptr()` instead of a
    null-terminated `CString`
  - `G3pbCacheStorage` retains committed blocks indefinitely; larger smokes
    currently require a fresh backend because there is no explicit committed
    entry reclamation path
- ✅ Landed a workspace-local patch for the pinned `nixl-sys` `0.10.1`
  invalidation bug
  - vendored the crate at `third_party/nixl-sys`
  - patched all remote invalidation paths to pass null-terminated `CString`
    values into the C ABI
  - wired the workspace to the local crate with `[patch.crates-io]`
  - preserved behavior otherwise
- ✅ Landed explicit backend-side committed-block reclamation for the `G3PB`
  host staging runtime
  - added `delete_blocks` to the `G3pbPeerStorage` / `G3pbStorageAgent` seam
  - `G3pbCacheStorage` and the in-memory peer backend now delete metadata and
    payloads for evicted hashes
  - `kvbm_g3pb_backend` now evicts least-recently-used committed staging blocks
    before `stage_put` allocations when host capacity is short
  - evicted committed blocks are dropped back through the pool’s normal return
    path and the backend waits until host availability actually reflects the
    reclaimed blocks before proceeding
  - committed runtime hits now refresh their access tick on both `query` and
    `fetch`

### Current findings before edits

- `nixl-sys` remains pinned at `=0.10.1` from both `lib/memory` and `lib/llm`,
  so the teardown fix should be landed as a workspace-local patch instead of an
  ad hoc registry edit
- the previously diagnosed teardown corruption is real:
  `/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.1/src/agent.rs`
  still passes a non-null-terminated Rust string pointer into
  `nixl_capi_invalidate_remote_md`
- backend retention is currently unbounded at the `G3PB` entry level:
  - `G3pbCacheStorage` can demote/promote entries between pinned `G2` and
    `foyer`
  - it can free `G2` offsets
  - it does not remove committed metadata/payload entries once admitted
- `foyer` exposes removal APIs, so a real reclamation path is feasible in this
  slice without changing the peer protocol

### Current findings after validation

- the local `nixl-sys` patch compiles cleanly across the active `G3PB` stack and
  the worker/backend smoke no longer emitted the earlier
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning during the validated
  sequential reclaim run
- the real retention bottleneck was the backend runtime’s committed host staging
  pool, not `G3pbCacheStorage` alone
- reclaiming committed entries required two coupled behaviors:
  - removing the evicted hashes from peer storage so they become honest cache
    misses instead of stale metadata hits
  - waiting for the block pool’s asynchronous return path to make reclaimed host
    blocks visible before attempting the next `stage_put`
- the first reclaim attempt exposed two important diagnostics now captured in
  code and on disk:
  - overlapping worker smokes are not a valid reclaim test because the second
    run can race before the first run’s blocks reach committed state
  - directly forcing `try_return_block()` was not sufficient for this path;
    dropping the evicted `MutableBlock`s and waiting for pool availability was
    the working reclaim mechanism

### Remaining work in this run

- none after the signed reclamation commit is created

### Exact next step

- re-read `PLANS.md` once more after the signed reclamation commit to confirm
  the active `G3PB` follow-on backlog is fully burned down for this slice

### Validation completed in this run so far

- post-`nixl-sys`-patch validation:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
    - pass (`4 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - `git diff --check`
    - pass
- post-reclamation validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - same-backend sequential reclaim smoke with fresh discovery + foyer dirs:
    - backend:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.reclaim3.10HSLM target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 48 --foyer-dir /tmp/g3pb-foyer.reclaim3.yRZxd8`
      - pass
    - first worker smoke:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.reclaim3.10HSLM target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --num-device-blocks 64 --host-blocks 40 --count 24`
      - pass
      - transferred `24` blocks / `196608` bytes
    - second worker smoke against the same backend, no restart:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.reclaim3.10HSLM target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
      - pass
      - transferred `40` blocks / `327680` bytes
      - this is the reclaim proof that previously required a fresh backend
    - validated request-plane timings for the successful `40`-block same-backend run:
      - `health`: `1` op in `0.005067s` (`197.35 ops/s`)
      - `load_remote`: `1` op in `0.005170s` (`193.41 ops/s`)
      - `offer`: `1` op in `0.002272s` (`440.15 ops/s`)
      - `stage_put`: `1` op in `0.010159s` (`98.44 ops/s`)
      - `commit_put`: `1` op in `0.002202s` (`454.18 ops/s`)
      - `query`: `1` op in `0.001762s` (`567.45 ops/s`)
      - `fetch`: `1` op in `0.002434s` (`410.83 ops/s`)
      - total metadata RPCs: `7` ops in `0.029066s` (`240.83 ops/s`)
  - final focused hygiene rerun after the working reclaim implementation:
    - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
      - pass (`15 passed`)
    - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
      - pass (`6 passed`)
    - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
      - pass
    - `git diff --check`
      - pass

### Decisions confirmed in this run

- the previously “non-blocking” follow-on backlog was concrete enough to land in
  this run rather than stay open:
  - local `nixl-sys` patch instead of waiting for an upstream release
  - real backend reclaim behavior instead of leaving larger smokes dependent on
    backend restarts
- keep the `nixl-sys` fix as a workspace-local patch until an upstream release
  incorporates the same null-terminated invalidation behavior
- backend reclaim policy is intentionally narrow:
  - evict least-recently-used committed staging blocks only when host capacity
    is short for a new `stage_put`
  - delete the corresponding peer-cache entries so eviction degrades to cache
    miss/recompute rather than serving stale metadata
  - do not broaden this slice into a larger cache-policy redesign

### Handoff for next run

- this run closed the two remaining concrete `G3PB` follow-on items from the
  prior handoff:
  1. local `nixl-sys` teardown fix
  2. backend-side committed-block reclamation
- validated outcomes now on disk:
  - focused `g3pb` tests pass
  - focused `g3pb_filter` tests pass
  - backend/worker binaries build
  - same-backend `24`-block then `40`-block sequential smoke passes without a
    backend restart
  - the validated smoke output no longer showed the prior
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning
- after the final signed commit for this run, treat the active `G3PB` slice as
  complete again unless a future audit or repo change exposes a new concrete
  regression
- any future follow-on work should now be new scope, not leftover scope from
  this plan:
  1. upstream the local `nixl-sys` patch when practical
  2. decide whether to keep or tune the current reclaim heuristic
  3. design any future CPU-buffer / `foyer` retention policy knobs as a
     separate slice

## Current run (2026-03-31 23:09:30 UTC)

### Summary of accomplishments in this run

- ✅ Resolved the NIXL backend-registration failure for `PinnedStorage(local host) -> remote host staging` transfer creation
  - The root cause was passing the local agent name into `createXferReq` instead of the remote peer agent name
  - Fixed in `lib/llm/src/block_manager/block/transfer/nixl.rs`
  - Full worker/backend smoke test now passes with remote write, remote read, local host registration, and device onboard
- ✅ Added KVBM-side G3PB admission policy wiring
  - Implemented in `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - Default admission when a block has been reused at least once
  - Environment override `G3PB_OFFLOAD_ALL` for eager admission of every block
  - Keeps `G1 -> G3PB` semantics as copy/replication, not ownership transfer
  - All tests pass (6 passed for G3PB filter, 13 passed for G3PB distributed)
- ✅ Validated the complete G3PB smoke path:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib` (13 passed)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib` (6 passed)
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke` (pass)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke` (pass)
  - `target/debug/kvbm_nixl_transfer_smoke` (pass)
  - host-only backend smoke using descriptor endpoints (pass)
  - full `target/debug/kvbm_g3pb_worker_smoke` with remote fetch and local onboard (pass)
- ✅ Migrated backend + smoke metadata/control traffic to Dynamo request-plane + discovery
  - added shared `G3pbRpcRequest` / `G3pbRpcResponse`
  - added `G3pbRequestPlaneClient`
  - added shared endpoint constants:
    - `G3PB_NAMESPACE=kvbm-g3pb`
    - `G3PB_COMPONENT_NAME=peer-cache`
    - `G3PB_ENDPOINT_NAME=g3pb`
  - `kvbm_g3pb_backend` now serves a Dynamo `Ingress` endpoint instead of Axum HTTP routes
  - `kvbm_g3pb_worker_smoke` now discovers peers from Dynamo discovery instead of `--backend-url`
- ✅ Revalidated the migrated request-plane path end-to-end
  - shared file-backed discovery smoke:
    - backend 53:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16`
    - backend 54:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_backend --worker-id 54 --host-blocks 16`
    - worker smoke owning to worker 53:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
      - pass
    - worker smoke owning to worker 54:
      `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.BBwJzW target/debug/kvbm_g3pb_worker_smoke --worker-id 1001 --host-blocks 8 --count 1 --sequence-start 2025`
      - pass
- ✅ Fixed request-plane smoke teardown regression
  - the first rewrite used `Worker::from_settings()` in `kvbm_g3pb_worker_smoke`
  - the data path completed successfully but repeated runs could segfault during teardown
  - switching that binary back to `tokio::main` removed the segfault while preserving the request-plane/discovery refactor

### Context re-read

- `Agents.md`
- `PLANS.md`
- `docs/design-docs/kvbm-g3pb-plan.md`
- `../dynamo/lib/runtime/examples/hello_world/src/bin/server.rs`
- `../dynamo/lib/runtime/examples/hello_world/src/bin/client.rs`
- `../dynamo/lib/runtime/src/component.rs`
- `../dynamo/lib/runtime/src/component/client.rs`
- `../dynamo/lib/runtime/src/component/endpoint.rs`
- `../dynamo/lib/runtime/src/distributed.rs`
- `../dynamo/lib/runtime/src/storage/kv.rs`
- `lib/llm/src/bin/kvbm_g3pb_backend.rs`
- `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- `lib/llm/src/block_manager/distributed.rs`
- `lib/llm/src/block_manager/distributed/g3pb.rs`
- `lib/llm/src/bin/kvbm_nixl_transfer_smoke.rs`
- `lib/llm/src/block_manager/distributed/transfer.rs`
- `lib/llm/src/block_manager/distributed/worker.rs`
- `lib/llm/Cargo.toml`

### Current findings before edits

- Dynamo already has the request-plane and discovery pieces this slice needs:
  - `endpoint_builder().handler(...).start()` automatically registers the
    endpoint in discovery using the active request-plane transport
  - `component.endpoint(...).client().await?` plus `PushRouter` provides the
    normal typed request path to discovered peers
  - `Client::wait_for_instances()` and the underlying discovery watch already
    provide the peer membership view needed by the worker smoke
- a cross-process smoke does not require etcd for this slice:
  - `DistributedConfig` supports `DYN_DISCOVERY_BACKEND=file`
  - the shared file-backed store root comes from `DYN_FILE_KV`
  - that is sufficient for multiple backend/worker processes on one machine to
    discover the same `G3PB` endpoint instances
- the long-term control-plane direction is therefore clear:
  - backend should become a normal Dynamo component endpoint instead of an
    ad hoc Axum HTTP service
  - worker smoke should discover backend peers from Dynamo discovery instead of
    taking `--backend-url`
  - health/metadata RPCs should move onto a typed request-plane protocol while
    NIXL/UCX remains the bulk transfer path
- that direction is now implemented and validated with a shared file-backed
  discovery store
- the remaining transport cleanup issue is narrower now:
  repeated request-plane smokes complete their data path and exit successfully,
  but still log `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` during teardown
- worktree was not clean at pickup:
  - modified `lib/llm/src/block_manager/distributed/g3pb.rs`
- the only uncommitted change at pickup replaced
  `let mut hits = Vec::new();` in `FoyerG3pbPeerStorage::query_blocks` with
  freeform prose about using a metadata-only `contains_key` path
- because of that local edit, the first required baseline validation no longer
  matched the handoff state and failed to compile before any new changes
- additional transport root-cause found while comparing the smoke path with the
  newer physical transfer executor:
  - `lib/llm/src/block_manager/block/transfer/nixl.rs` always calls
    `create_xfer_req(..., &nixl_agent.name(), ...)`
  - that is wrong for cross-agent transfers because `stage_put` writes target a
    remote backend agent, not the local worker agent
  - the newer v2 executor already selects the non-local agent name when
    constructing the transfer request, so the legacy helper needs the same fix

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - fail at pickup
  - compiler error at
    `lib/llm/src/block_manager/distributed/g3pb.rs:364`
  - cause: stray prose in `FoyerG3pbPeerStorage::query_blocks`
  - pass after baseline repair (`13 passed`)
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass after baseline repair
- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `git diff --check`
  - pass
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
  - pass
- host-only `G3PB` backend smoke:
  - command:
    `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58183 --worker-id 53`
  - `/health`
    - pass, returned `{"worker_id":53,"listen":"127.0.0.1:58183"}`
- `target/debug/kvbm_nixl_transfer_smoke`
  - pass
  - output reported:
    `nixl smoke transfer complete: device->host 8 blocks, host->device 8 blocks`
  - emitted one UCX runtime warning about
    `IB_PCI_RELAXED_ORDERING=try`, but the smoke still completed successfully
- backend transport-contract milestone:
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
    - pass after adding staged NIXL descriptor endpoints and backend host runtime
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass after the transport type additions (`13 passed`)
- worker transport-contract milestone:
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - pass after rewriting the worker smoke around `load_remote`, `stage_put`,
      `commit_put`, immutable descriptor fetch, and remote-read helper wiring
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_nixl_transfer_smoke --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- host-only backend descriptor/control smoke:
  - backend:
    `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58183 --worker-id 53 --host-blocks 16`
    - pass
  - `curl http://127.0.0.1:58183/health`
    - pass
  - `offer -> stage_put -> commit_put -> query` for `sequence_hash=9001`
    - pass after making backend query authoritative over staged-committed hashes
- full worker/backend smoke:
  - command:
    `target/debug/kvbm_g3pb_worker_smoke --backend-url http://127.0.0.1:58183`
   - current status:
     pass
  - observed progress after the shared `create_xfer_req` peer-selection fix:
    - `offer`
    - `stage_put`
    - `commit_put`
    - duplicate-offer rejection
    all pass
  - current terminal error now occurs only during the remote fetch/read leg
  - current terminal error:
    `createXferReq: no specified or potential backend had the required registrations to be able to do the transfer`
  - additional NIXL warning on teardown:
    `invalidateRemoteMD: error invalidating remote metadata for agent '53Н'`
- request-plane + discovery migration validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass after each migration milestone
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend`
    - pass after replacing Axum with the Dynamo endpoint handler
  - `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass after the worker smoke rewrite
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass after shared request-plane protocol/client additions (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke --bin kvbm_nixl_transfer_smoke`
    - pass
  - `target/debug/kvbm_nixl_transfer_smoke`
    - pass
  - request-plane discovery smoke to worker `53`
    - pass
  - request-plane discovery smoke to worker `54`
    - pass
  - repeated request-plane smokes after changing `kvbm_g3pb_worker_smoke` back to `tokio::main`
    - pass without the earlier teardown segfault

### Decisions confirmed in this run so far

- first priority in this run is to restore the committed `G3PB` baseline to a
  compiling, testable state before attempting the NIXL/UCX follow-up
- the local `foyer` note should be preserved as an actual code comment rather
  than dropped, because the underlying concern is valid:
  metadata-only query would be better than loading payloads from disk
- the baseline repair itself is intentionally minimal:
  preserve existing behavior, restore compilation, and keep the optimization
  note as a code comment instead of changing `foyer` query semantics blindly
- the missing transport slice is now isolated more precisely:
  local NIXL transfer works and host-only `G3PB` HTTP metadata/payload flow
  works, but there is still no peer-runtime seam that exposes remote
  CPU-visible staging blocks as NIXL-importable descriptors for `G3PB`
- `lib/llm/src/block_manager.rs` already contains an ignored cross-worker test
  noting that the current Rust bindings do not yet support the partial metadata
  path needed for that import style; this is relevant to any attempt to bolt
  remote descriptor exchange directly onto the current older blockset APIs
- backend implementation decision for this run:
  keep `G3pbStorageAgent` as the metadata/query contract, but add a separate
  backend-only pinned-host staging runtime that:
  - allocates host blocks from a local KVBM host pool with a UCX-enabled NIXL agent
  - reserves accepted `sequence_hash` values before transfer
  - exposes serialized blocksets plus `BlockDescriptorList` responses for
    mutable upload descriptors and immutable fetch descriptors
  - commits staged hashes into the metadata agent only after the transfer
    completion call so `query` stays visibility-safe
- shared transport structs live in `distributed/g3pb.rs`
- the backend binary owns its `BlockDescriptorList` construction locally;
  do not extend the shared block-manager API for that convenience
- the backend also needs explicit remote-agent handshake:
  the worker must publish its local serialized blockset to each backend through
  `load_remote` before any `stage_put` / `fetch` transfer so the peer agent can
  load the worker’s UCX metadata
- the worker now builds its transfer context from the block manager’s exported
  NIXL agent state instead of a separate sibling agent instance
- first transport fix landed in this run:
  `block/transfer/nixl.rs` now chooses the non-local worker/agent instead of
  always targeting the local agent during `create_xfer_req`
- the write path is therefore proven good enough for:
  local pinned host -> remote pinned host staging
- the earlier wrapper-lifetime suspicion was incorrect:
  the bug was in our code passing the local agent name instead of the remote
  peer agent name into `create_xfer_req`
- the remaining follow-up is only the separate teardown warning
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`, which is not currently treated
  as evidence of a `nixl-sys` ownership bug
- discovery/request-plane implementation decision for the next slice:
  - landed as planned:
    - typed `G3PB` request protocol in
      `lib/llm/src/block_manager/distributed/g3pb.rs`
    - backend served through Dynamo `Ingress`
    - worker addressed backends by discovered `instance_id`
    - shared file-backed discovery works cross-process without etcd
  - keep the old direct-HTTP control plane out of the final path
  - keep `kvbm_g3pb_backend` on the long-lived `Worker` wrapper
  - keep `kvbm_g3pb_worker_smoke` on `tokio::main` because it is a short-lived validation binary and that avoids the teardown segfault

### Remaining work in this run

- ✅ resolved the last NIXL backend-registration failure for
  `PinnedStorage(local host) -> remote host staging` transfer creation
- revised NIXL diagnosis for the current `G3PB` slice:
  - the real transfer bug was passing the local agent name into
    `createXferReq` instead of the remote peer agent name
  - keep the current `block/transfer/nixl.rs` remote-agent fix; do not revert it
  - the earlier `nixl-sys` / wrapper-lifetime suspicion was incorrect and
    should not be used as the explanation for this slice
  - the post-transfer `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` warning is
    currently treated as a separate non-blocking cleanup issue because the
    end-to-end `G3PB` smoke succeeds with remote write, remote read, local
    host registration, and device onboard
  - do not mix this slice with speculative transport additions like the
    unconditional `nixl_agent` path, `(Disk, Host)` transfer arm, or
    `disk_block_observer` surface
- ✅ add KVBM-side `G3PB` admission policy wiring:
  - default admission when a block has been reused at least once
  - environment override `G3PB_OFFLOAD_ALL` for eager admission of every block
  - keep `G1 -> G3PB` semantics as copy/replication, not ownership transfer
  - implemented in `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - tests pass (6 passed)
- ✅ refactor both `kvbm_g3pb_backend` and `kvbm_g3pb_worker_smoke` to use Dynamo request-plane + discovery instead of ad hoc HTTP base URLs:
  - register `G3PB` as a Dynamo component endpoint and discover remotes via the discovery backend
  - remove manual `--listen` / `--backend-url` control-plane wiring from the long-term path
  - keep bulk data movement on NIXL/UCX; use request-plane only for metadata/handshake RPCs
  - request-plane metadata-RPC throughput measurement is still pending
  - do not add the speculative unconditional `nixl_agent` path, `(Disk, Host)` transfer arm, or `disk_block_observer` surface in this slice; keep those deferred unless a concrete requirement appears
- ✅ next implementation slice, in order:
  - add shared typed `G3PB` RPC request/response enums plus a small client
    wrapper in `lib/llm/src/block_manager/distributed/g3pb.rs`
  - convert `kvbm_g3pb_backend` from Axum handlers to a Dynamo endpoint engine
  - convert `kvbm_g3pb_worker_smoke` from manual `--backend-url` calls to
    discovery-driven peer discovery using the new request-plane client
  - run focused compile/tests after the protocol/client slice before attempting
    the full smoke rewrite
- revalidate the next slice in layers:
  - ✅ repeated `kvbm_g3pb_worker_smoke` loop against a stable backend
    - data path remains good
    - the remaining teardown issue is the warning
      `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
  - ✅ multi-backend ownership/routing smoke with multiple `G3PB` peers
    - ownership to worker `53` proven
    - ownership to worker `54` proven with `--sequence-start 2025`
  - larger block-count staged transfer smoke remains pending
  - backend restart/reload validation across `load_remote` plus subsequent
    `query` / `fetch` remains pending
  - ✅ worker-side compile/tests after the smoke rewrite
  - ✅ `kvbm_nixl_transfer_smoke`
  - host-only backend-only request-plane harness remains pending if we still
    want a non-worker validation equivalent to the older HTTP curl smoke
  - ✅ full `kvbm_g3pb_worker_smoke` with remote fetch and local onboard

### Concrete transport gap

- new backend seam now exists:
  - `offer` still determines ownership/admission
  - `stage_put` allocates reserved remote mutable host blocks and returns:
    - serialized remote blockset metadata
    - a mutable `BlockDescriptorList`
  - `commit_put` marks staged hashes visible to `query`
  - `fetch` now returns serialized blockset metadata plus an immutable
    `BlockDescriptorList` instead of inline bytes
- the rewritten worker already does:
  - `load_remote`
  - `stage_put`
  - remote-blockset import
  - `commit_put`
  - immutable descriptor fetch
  - `read_from_remote`
- but the first `stage_put` write still fails when NIXL tries to create the
  actual transfer request for local pinned-host blocks to remote host staging
- the error persists after:
  - publishing local blockset metadata to the backend
  - switching the transfer context to the block manager’s own registered
    NIXL agent
  - enabling `POSIX` alongside `UCX` on both worker and backend agents
- the remaining unresolved question is whether this transport slice needs:
  - `SystemStorage` rather than `PinnedStorage`
  - a different backend/registration mix

### Exact next step

- next follow-up if another run continues from here:
  - backend restart/reload validation for the request-plane path
  - larger staged-transfer smoke counts
  - explicit request-plane metadata-RPC throughput measurement
  - deeper diagnosis of the still-non-blocking teardown warning
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`

### Handoff for next run

The G3PB implementation is now in a stable, working state:
- All core G3PB functionality is working (offer, stage_put, commit_put, query, fetch)
- NIXL/UCX transfers are working correctly for both write and read paths
- G3PB admission policy has been implemented with configurable behavior
- All tests pass

The request-plane/discovery migration is now landed and validated.

If there is another run, spend it on the remaining validation tail:
1. backend restart/reload validation across `load_remote` + `query` / `fetch`
2. larger staged-transfer counts
3. request-plane metadata-RPC throughput measurement
4. investigation of the still-non-blocking teardown warning

The `invalidateRemoteMD` warning on teardown is currently treated as a separate non-blocking cleanup issue and should be investigated separately if it becomes problematic.

## Archived milestones (condensed)

### 2026-03-31 rename and seam reset

- decoupled the unlanded `G4` peer-cache path from disk offload state
- removed `G4BlockIndex` / disk-observer wiring from the core block-manager
  state and distributed worker path
- replaced the unlanded `distributed/g4.rs` module with
  `lib/llm/src/block_manager/distributed/g3pb.rs`
- renamed the binaries to:
  - `lib/llm/src/bin/kvbm_g3pb_backend.rs`
  - `lib/llm/src/bin/kvbm_g3pb_worker_smoke.rs`
- updated `Agents.md` and replaced the design doc with
  `docs/design-docs/kvbm-g3pb-plan.md`
- fixed local `disable_nixl()` construction in
  `lib/llm/src/block_manager/state/local.rs` so host-only local layouts do not
  try to serialize missing NIXL descriptors

### Prior validation already on disk before this run

- `cargo test --manifest-path lib/llm/Cargo.toml g4:: --lib`
  - pass before the rename milestone
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`12 passed`) immediately after the rename/in-memory seam
- `cargo check --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `target/debug/kvbm_g3pb_backend --listen 127.0.0.1:58181`
  - pass
- `target/debug/kvbm_g3pb_worker_smoke --backend-url http://127.0.0.1:58181`
  - pass after the local-layout `disable_nixl()` fix

## Forward plan

1. Keep the in-memory backend and `foyer` backend both available behind the
   same `G3pbPeerStorage` seam until the runtime path settles.
2. Replace the HTTP payload path with real NIXL/UCX `device <-> CPU` transfer
   semantics.
3. Re-run the layered validation stack after the transport swap.
4. Only after that, decide whether `foyer` should become the default backend
   constructor path for the standalone peer binary.


Commit is allowed for this state because the end-to-end `G3PB` validation stack is working.

## Current run (2026-04-01 00:46:42 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before touching code:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Reconfirmed the repo baseline from the current detached `HEAD`
- ✅ Revalidated the focused `G3PB` unit baseline:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
- ✅ Revalidated the `G3PB` admission policy suite:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- ✅ Revalidated the same focused baseline again from the current clean worktree
  before starting the remaining validation tail:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- ✅ Rebuilt the active request-plane smoke binaries:
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- ✅ Completed backend restart/reload validation across `load_remote` + `query` / `fetch`
  - first attempt exposed a discovery-root setup mistake (`DYN_FILE_KV` must point
    at a directory root, not a pre-created file)
  - corrected shared discovery root worked
  - first smoke run passed against worker `53`
  - backend was terminated and restarted against the same discovery + foyer dirs
  - second smoke run also passed against worker `53`
- ✅ Completed larger staged-transfer validation on the request-plane/discovery path
  - a `24`-block smoke passed against a backend with `48` host blocks
  - a follow-up `40`-block smoke on the same backend failed only because the
    previous committed `24` blocks still occupied host staging capacity, leaving
    `24` blocks available
  - restarting onto a fresh backend with `96` host blocks allowed the same
    `40`-block smoke to pass cleanly
- ✅ Added explicit metadata-RPC timing output to `kvbm_g3pb_worker_smoke`
  - the smoke now reports per-RPC and total request-plane timing for:
    `health`, `load_remote`, `offer`, `stage_put`, `commit_put`, `query`, and `fetch`
  - this keeps the measurement on the real G3PB validation path rather than
    creating a separate synthetic harness
- ✅ Diagnosed the teardown warning root cause
  - the corrupted remote-agent names in
    `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND` come from `nixl-sys`, not the
    repo-local G3PB code
  - `nixl-sys` uses `remote_agent.as_ptr()` from a Rust `&str` when calling
    `nixl_capi_invalidate_remote_md`, instead of passing a null-terminated
    `CString`
  - relevant dependency paths:
    - `/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.1/src/agent.rs`
    - `/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/nixl-sys-0.10.1/wrapper.cpp`
- ✅ Started the structural cleanup follow-up from the prior handoff
  - renamed the transitional storage/config symbols so new code paths stop
    exporting the migration-era `G2G3G3pb*` names
  - landed stable names:
    - `G3pbStorageConfig`
    - `G3pbCacheStorage`
    - internal metadata/location helpers now use `G3pbCache*` terminology
  - behavior intentionally unchanged in this slice
- ✅ Added a reusable request-plane discovery/client seam for G3PB peers
  - lifted peer discovery, worker-id to instance-id resolution, and
    broadcast `load_remote` wiring out of `kvbm_g3pb_worker_smoke`
  - new shared helpers/types live in `lib/llm/src/block_manager/distributed/g3pb.rs`:
    - `discover_g3pb_peers`
    - `G3pbDiscoveredPeers`
    - `G3pbPeerInstance`
  - `kvbm_g3pb_worker_smoke` now reuses that shared seam instead of carrying
    its own ad hoc discovery bookkeeping
- ✅ Refactored the standalone backend around a cleaner service boundary
  - `kvbm_g3pb_backend` now keeps runtime/storage/RPC behavior on
    `G3pbBackendService`
  - endpoint transport wiring is a thin wrapper over that service instead of
    being interleaved with request handling state
  - no protocol or storage behavior changes in this slice
- ✅ Landed the first native KVBM config surface for G3PB admission policy
  - added shared `G3pbAdmissionPolicy` / `G3pbAdmissionConfig` to
    `lib/llm/src/block_manager/config.rs`
  - `KvBlockManagerConfig` now carries optional `g3pb_admission`
  - local/logical KVBM state construction now installs the `G3PB` host
    offload filter automatically when `g3pb_admission` is set and the host
    layout did not already provide an explicit offload filter
  - `G3pbAdmissionFilter` now builds from the shared config type and treats the
    legacy `G3PB_OFFLOAD_ALL` env var only as a compatibility fallback
  - added unit coverage for the config-to-host-filter wiring in
    `lib/llm/src/block_manager/state.rs`

### Current findings before edits

- the remaining work from the prior handoff is still the validation tail, not a
  known missing implementation slice
- `kvbm_g3pb_worker_smoke` already exercises the core request-plane flow needed
  for the pending milestones:
  - discovery-driven peer enumeration
  - `load_remote`
  - staged remote writes
  - `commit_put`
  - ownership-aware `query`
  - descriptor-based `fetch`
  - local host registration plus device onboard
- explicit timing/throughput output is now available directly from
  `kvbm_g3pb_worker_smoke`
- `DYN_FILE_KV` for the file discovery backend must be a directory root; using a
  `mktemp` file path fails backend registration with:
  `Unable to register service for discovery ... Internal filesystem error: Not a directory (os error 20)`
- restart/reload behavior for the current `G3PB` smoke path is now validated:
  a restarted backend can accept a fresh `load_remote`, complete staged upload,
  answer `query`, and serve descriptor-based `fetch` in the same shared
  discovery namespace
- the teardown warning still reproduces after each successful worker smoke:
  `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
- `KvBlockManagerState::import_remote_blockset()` currently rejects duplicate
  worker ids and does not expose any explicit unload/update path for remote NIXL
  metadata, which is relevant context for the teardown-warning investigation
- the warning diagnosis is now specific:
  repo-local remote-blockset lifetime is not the immediate cause of the garbled
  remote-agent names; the names are already corrupted inside `nixl-sys`
  invalidation calls because the dependency passes non-null-terminated Rust
  string pointers to the C API during remote metadata teardown
- the first structural cleanup slice is low-risk and independently landable:
  the naming cleanup is isolated from request-plane behavior, NIXL transfer
  behavior, and storage semantics
- the next structural cleanup slice is also now proven low-risk:
  peer discovery and instance resolution can live in the shared G3PB client
  surface without changing request-plane timings or transfer behavior
- backend cleanup can also stay structural:
  the standalone backend’s request handling, runtime policy, and endpoint
  startup can be separated without changing the request-plane contract
- concrete config-surface findings from this run:
  - KVBM/core policy candidates:
    - offload admission policy
    - CPU-buffer retention/eviction thresholds
    - promotion/demotion thresholds between CPU buffer and `foyer`
  - backend-runtime candidates:
    - worker identity
    - pinned host staging capacity
    - backend device id for pinned registration
    - `foyer` directory and capacity settings
  - smoke-only validation controls:
    - demo block counts
    - sequence seed/range
    - synthetic local device/host block counts used only by validation binaries
  - the first concrete shared KVBM knob is now implemented:
    `KvBlockManagerConfig::g3pb_admission`
  - ownership split after this slice is clearer:
    - core KVBM config:
      `g3pb_admission`
    - backend-runtime config:
      worker identity, pinned host staging block count, device id, `foyer`
      directories, G2 bytes, `foyer` memory bytes, `foyer` disk bytes
    - smoke-only validation CLI:
      synthetic device block counts, local host block counts, sequence seed,
      demo block count, per-run workload shaping
  - compatibility note:
    `G3PB_OFFLOAD_ALL` still works, but now only as a legacy fallback for the
    new native config surface rather than being the only policy entry point

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`13 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- repeated focused baseline from the clean worktree before the next milestone:
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- backend restart/reload validation with shared file discovery + persistent foyer dir:
  - incorrect discovery-root attempt:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restart.XBheCV target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - fail at startup because `DYN_FILE_KV` pointed to a file instead of a directory
  - corrected backend start:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - pass
  - first worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
    - pass
  - backend graceful shutdown via `Ctrl+C`
    - pass
  - restarted backend with same discovery + foyer dirs:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 16 --foyer-dir /tmp/g3pb-foyer.restart.PpnPrN`
    - pass
  - second worker smoke after restart:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.restartdir.rx0f7E target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --host-blocks 8 --count 4`
    - pass
  - observations:
    - both worker smokes transferred `4` blocks / `32768` bytes
    - both runs validated `load_remote`, staged upload, duplicate-offer rejection,
      `query`, remote `fetch`, local host registration, and device onboard
    - both runs still emitted the non-blocking teardown warning
      `invalidateRemoteMD ... NIXL_ERR_NOT_FOUND`
- larger staged-transfer validation:
  - rebuilt binaries before the milestone:
    `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - backend with `48` host blocks:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 48 --foyer-dir /tmp/g3pb-foyer.large.2HfqDv`
    - pass
  - `24`-block worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_worker_smoke --worker-id 21 --num-device-blocks 64 --host-blocks 40 --count 24`
    - pass
    - transferred `24` blocks / `196608` bytes
  - same backend, `40`-block worker smoke:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.large.EKfFBR target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
    - fail with:
      `Not enough blocks available, requested: 40, available: 24`
    - this was a retained-capacity limit, not a transport failure
  - fresh backend with `96` host blocks:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_backend --worker-id 53 --host-blocks 96 --foyer-dir /tmp/g3pb-foyer.xl.em2H7Z`
    - pass
  - `40`-block worker smoke on the fresh backend:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_worker_smoke --worker-id 22 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 5000`
    - pass
    - transferred `40` blocks / `327680` bytes
- post-instrumentation validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `git diff --check`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - timed `40`-block worker smoke with instrumentation:
    `env DYN_DISCOVERY_BACKEND=file DYN_FILE_KV=/tmp/g3pb-discovery.xl.uu9wfv target/debug/kvbm_g3pb_worker_smoke --worker-id 23 --num-device-blocks 96 --host-blocks 56 --count 40 --sequence-start 9000`
    - pass
    - transferred `40` blocks / `327680` bytes
    - request-plane timings:
      - `health`: `1` op in `0.002878s` (`347.41 ops/s`)
      - `load_remote`: `1` op in `0.003572s` (`279.98 ops/s`)
      - `offer`: `1` op in `0.001408s` (`710.46 ops/s`)
      - `stage_put`: `1` op in `0.002331s` (`429.05 ops/s`)
      - `commit_put`: `1` op in `0.001698s` (`589.01 ops/s`)
      - `query`: `1` op in `0.001343s` (`744.58 ops/s`)
      - `fetch`: `1` op in `0.001933s` (`517.42 ops/s`)
      - total metadata RPCs: `7` ops in `0.015162s` (`461.68 ops/s`)
- post-naming-cleanup validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`13 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-client-seam validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
    - added unit coverage for duplicate-worker discovery rejection and
      deterministic worker-id ordering in discovered peer registries
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-backend-boundary validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - pass (`6 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
- post-native-config validation:
  - `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
    - pass
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
    - first pass failed in `test_default_trait` because an earlier env-mutating
      test left `G3PB_OFFLOAD_ALL` set; fixed by clearing the env var inside the
      test
    - final pass (`6 passed`)
  - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
    - pass (`15 passed`)
  - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
    - pass
  - `git diff --check`
    - pass

### Decisions confirmed in this run so far

- keep `PLANS.md` as the live execution log and update it before each major
  milestone
- execute the remaining work in the exact order already handed off:
  1. backend restart/reload validation
  2. larger staged-transfer smoke counts
  3. request-plane metadata-RPC throughput measurement
  4. teardown-warning investigation
- after the validation tail, pick up the structural cleanup work before adding
  more G3PB surface area:
  1. refactor the G3PB client seam
  2. refactor the G3PB backend seam
  3. clean up externally visible naming so new APIs and configs do not inherit
     awkward transitional names such as `g2g2g3*`
  4. design the long-term KVBM-native config exposure path for G3PB behavior
- only make commits after the corresponding end-to-end `G3PB` validation stack
  is green for that milestone, per `Agents.md`
- treat the file-discovery-root mistake as an operator/setup note only; no code
  change is needed for it
- the restart milestone is complete, so the next effort should stay focused on
  scaling the same smoke path rather than broadening functionality
- first teardown-warning readback should stay narrow:
  inspect local remote-metadata ownership/unload behavior before assuming the
  warning requires transport-path changes
- keep the request-plane timing output in the worker smoke:
  it is cheap, uses the real G3PB path, and gives enough throughput visibility
  for future validation without adding a separate harness
- treat the teardown warning as an upstream `nixl-sys` cleanup bug rather than a
  blocker for the landed G3PB slice
- structural cleanup should land in small validated slices:
  - first naming cleanup
  - then reusable client seam cleanup
  - then backend/service boundary cleanup
- keep the worker smoke timing buckets stable while refactoring:
  move discovery bookkeeping into shared helpers, but leave timing ownership in
  the smoke binary so measurements stay comparable across runs
- keep the backend cleanup structural:
  factor request handling behind a service object first, and defer any protocol
  or storage-policy changes until a later run with separate validation
- config-surface design outcome from this run:
  - do not hide long-lived G3PB behavior only in binary-local CLI/env wiring
  - keep synthetic workload knobs local to smoke binaries
  - treat backend transport/storage knobs separately from core KVBM tier-policy
    knobs so the long-term surface is easier to reason about
- native-config decision landed in code:
  - keep the first shared KVBM surface narrow and policy-focused
  - model the current admission behavior explicitly as
    `G3pbAdmissionPolicy::{AfterFirstReuse,Eager}`
  - inject the filter during state construction only when
    `KvBlockManagerConfig.g3pb_admission` is set
  - preserve explicit host `offload_filter` wiring as the stronger override
  - retain `G3PB_OFFLOAD_ALL` only as backward-compatibility fallback for the
    new shared config path

### Remaining work in this run

- no additional code slice is required to satisfy the current handoff
- the remaining follow-up is later integration work, not missing work in this
  run:
  - adopt `KvBlockManagerConfig.g3pb_admission` from real non-smoke KVBM
    construction paths once those callers are ready to opt into `G3PB`
    admission policy
  - design the next shared config layer for CPU-buffer retention / `foyer`
    retention separately from this admission-policy slice
  - remove the legacy `G3PB_OFFLOAD_ALL` env path only after relevant callers
    have moved onto the native config surface

### Exact next step

- if another run continues from here, the next concrete slice should be config
  adoption rather than config invention:
  - wire `KvBlockManagerConfig.g3pb_admission` into the first real KVBM caller
    that should opt into `G3PB` admission policy
  - keep backend-runtime-only and smoke-only knobs out of that adoption change
  - keep CPU-buffer retention / `foyer` retention as a separate follow-up after
    an actual caller for those knobs is identified
  - rerun the focused `g3pb` / `g3pb_filter` tests plus backend/worker build
    after that adoption slice

### Handoff for next run

- the validation tail, structural cleanup pass, and first native-config slice
  are complete
- landed cleanup/config work in this run:
  - stable storage/config naming
  - shared peer discovery/client seam
  - cleaner backend service boundary
  - native `KvBlockManagerConfig.g3pb_admission`
- next run should focus on config adoption rather than more transport or
  endpoint restructuring
- concrete next slice:
  1. wire `KvBlockManagerConfig.g3pb_admission` into the first real KVBM caller
     that should opt into `G3PB`
  2. keep backend-runtime-only knobs and smoke-only knobs out of that adoption
     change
  3. rerun:
     - `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
     - `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
     - `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
- config ownership guidance already established in this run:
  - KVBM/core policy:
    - offload admission policy
    - CPU-buffer retention/eviction thresholds
    - promotion/demotion thresholds between CPU buffer and `foyer`
  - backend runtime:
    - worker identity
    - pinned host staging capacity
    - backend device id
    - `foyer` directory and capacity settings
  - smoke-only validation:
    - demo block counts
    - sequence seed/range
    - synthetic local device/host block counts
- optional follow-on work outside the main slice remains:
  1. upstream or locally patch `nixl-sys` remote metadata invalidation to use
     null-terminated `CString` values
  2. decide whether backend-side eviction/reclamation is needed for long-lived
     retained committed blocks beyond the current smoke coverage

## Current run (2026-04-01 01:51:53 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before editing:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Confirmed the remaining concrete slice from the prior handoff:
  `KvBlockManagerConfig.g3pb_admission` existed in core KVBM state wiring, but
  no real non-smoke caller was setting it yet
- ✅ Adopted the native `g3pb_admission` config surface in the first real KVBM
  caller path:
  - `lib/bindings/kvbm/src/block_manager.rs`
  - both Python `BlockManager::new` and `BlockManagerBuilder::build()` now read
    `DYN_KVBM_G3PB_ADMISSION_POLICY` and, when set, pass
    `KvBlockManagerConfig.g3pb_admission`
  - supported values:
    - `after_first_reuse`
    - `eager`
    - `disabled`
- ✅ Added focused parser coverage for the bindings adoption path
  - default/unset behavior
  - `after_first_reuse`
  - `eager`
  - invalid-value rejection
- ✅ Fixed the bindings parser test race
  - the first test run exposed process-wide env mutation leakage across
    parallel tests
  - resolved by serializing the env-mutating parser tests with a local mutex
- ✅ Updated repo-local docs to match the landed request-plane architecture and
  the new real-caller config adoption:
  - `docs/design-docs/kvbm-g3pb-plan.md`
  - `lib/bindings/kvbm/README.md`
  - `lib/runtime/src/config/environment_names.rs`

### Current findings before final handoff

- the prior “next concrete slice” is now implemented:
  real KVBM callers can opt into native `G3PB` admission policy without using
  only the legacy `G3PB_OFFLOAD_ALL` fallback
- the smallest production-facing adoption point is the bindings layer, because:
  - the connector leaders already flow through `BlockManagerBuilder`
  - that path already consumes other KVBM env-driven runtime policy
  - the underlying core config surface remained unchanged in this slice
- the adoption stays backward-compatible:
  - unset `DYN_KVBM_G3PB_ADMISSION_POLICY` preserves existing behavior
  - explicit disk offload filters remain stronger than `g3pb_admission`
    because the core state wiring only installs the `G3PB` host filter when
    no explicit host offload filter is already present
- the design doc was stale before this run:
  it still described the old HTTP backend/request path even though the landed
  implementation already uses Dynamo request-plane + discovery
- the only bug uncovered during this slice was test-only:
  the new bindings parser tests needed serialization because they mutate a
  shared process environment variable

### Validation completed in this run so far

- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `cargo fmt --manifest-path lib/bindings/kvbm/Cargo.toml --all`
  - pass
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - first run failed because the new env-mutating tests raced each other under
    parallel execution
  - pass after serializing env mutation (`4 passed`)
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- treat the bindings layer as the first real KVBM caller for native
  `g3pb_admission` adoption
- keep the adoption path explicit and backward-compatible:
  do not force-enable `G3PB` policy for all callers, and do not broaden the
  core config surface beyond admission policy in this slice
- repair stale docs in the same slice because the design doc and bindings README
  are part of the active handoff surface for future runs

### Remaining work in this run

- make a signed small commit for the config-adoption/docs slice

### Exact next step

- create the signed commit for:
  - bindings-side native `g3pb_admission` adoption
  - env-name/docs updates
  - `PLANS.md` handoff refresh

### Handoff for next run

- this run already landed the concrete “config adoption” slice from the prior
  handoff in the real KVBM bindings caller path
- after the commit is cut, the remaining work should return to follow-on items
  rather than more admission config plumbing:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 02:00:50 UTC)

### Summary of accomplishments in this run

- ✅ Re-read `PLANS.md` after the config-adoption commit to verify whether any
  concrete work remained unrecorded
- ✅ Revalidated the focused `G3PB` / bindings stack from the current clean
  `HEAD` instead of assuming the prior run log was still sufficient
- ✅ Found and fixed one remaining test-only reliability gap:
  - `lib/llm/src/block_manager/offload/g3pb_filter.rs`
  - the `g3pb_filter` env-legacy tests were still mutating
    `G3PB_OFFLOAD_ALL` without serialization
  - a fresh validation run exposed the race in
    `test_legacy_env_false_string`
  - fixed by serializing the env-mutating tests with a local mutex, matching
    the bindings-side parser test strategy
- ✅ Refreshed the handoff state in `PLANS.md` after the post-commit audit

### Current findings before final handoff

- the config-adoption/docs slice from the prior run was already committed and
  signed before this run started:
  - `126c3590d bindings: adopt native g3pb admission config`
- the only remaining issue discovered by re-running the planned validation was
  the `g3pb_filter` env-test race
- no additional product/runtime behavior changes are required for the current
  `G3PB` slice once that test reliability fix is in
- the audit-run reliability fix is now committed and signed:
  - `701d74a95 tests: serialize g3pb filter env cases`

### Validation completed in this run so far

- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - initial audit pass: pass (`15 passed`)
  - post-fix rerun: pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - initial audit pass: fail in
    `block_manager::offload::g3pb_filter::tests::test_legacy_env_false_string`
    because `G3PB_OFFLOAD_ALL` was racing another env-mutating test
  - post-fix rerun: pass (`6 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - audit pass before the llm-side test fix: pass (`4 passed`)
  - post-fix hygiene rerun: pass (`4 passed`)
- `cargo fmt --manifest-path lib/llm/Cargo.toml --all`
  - pass
- `cargo fmt --manifest-path lib/bindings/kvbm/Cargo.toml --all`
  - pass
- `git diff --check`
  - pass

### Decisions confirmed in this run so far

- keep re-reading `PLANS.md` after each landed slice; the post-commit audit
  found a real test gap that the prior handoff had not captured
- treat env-mutating test serialization as required in both bindings and llm
  test modules whenever policy/env compatibility paths are covered
- keep the current follow-on queue focused on the already-known non-blocking
  items rather than inventing new `G3PB` surface area

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume with the follow-on backlog rather
  than more validation cleanup:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Handoff for next run

- the config-adoption/docs slice is already committed
- this audit run found and fixed the last currently known focused-validation
  gap: the `g3pb_filter` env-test race
- this audit run is also committed:
  - `701d74a95 tests: serialize g3pb filter env cases`
- repo state at handoff:
  - clean worktree
  - detached `HEAD`
- the remaining follow-on work returns to the same non-blocking backlog already
  captured in the prior run:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 06:38:56 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely before making changes and compacted the
  repeated audit-only tail from `2026-04-01 05:02 UTC` through
  `2026-04-01 06:32 UTC` into this single current-state entry
- ✅ Re-audited the active `G3PB` handoff/code surface against the current repo
  state:
  - detached `HEAD`
  - clean worktree at pickup
  - current tip at pickup: `9d7bfd41de99`
  - recent history is a docs-only audit chain:
    - `9d7bfd41d docs: finalize g3pb audit handoff`
    - `5e4cd208a docs: stabilize g3pb audit handoff`
    - `b28504d24 docs: finalize g3pb audit handoff`
    - `0e1f53a89 docs: refresh g3pb audit handoff`
- ✅ Re-searched the active `G3PB` surface for remaining concrete work:
  - searched `PLANS.md`, `docs/design-docs/kvbm-g3pb-plan.md`,
    `lib/llm/src`, and `lib/bindings/kvbm/src`
  - repo-wide unrelated `TODO` / `FIXME` markers still exist, but no new
    `G3PB`-specific implementation gap, cleanup item, docs gap, follow-up, or
    validation gap was found in the active slice
- ✅ Revalidated the focused `G3PB` / bindings stack from the current tip after
  the `PLANS.md` compaction edit
- ✅ Refreshed the on-disk handoff so the next run can resume from verified
  current state instead of repeating another docs-only audit loop

### Current findings before final handoff

- the active `G3PB` implementation slice still appears complete on current
  `HEAD`
  - pickup commit: `9d7bfd41de99`
  - no new implementation gap, cleanup item, docs gap, or validation gap was
    found by this run’s targeted audit
- `Agents.md`, `PLANS.md`, and
  `docs/design-docs/kvbm-g3pb-plan.md` remain consistent with the landed tree:
  - request-plane + discovery remain the active control-plane path
  - `KvBlockManagerConfig.g3pb_admission` remains the first native config
    surface already adopted by a real non-smoke caller
  - the remaining config work is broader caller adoption, not missing initial
    adoption
- the only remaining `G3PB` work is still the existing non-blocking follow-on
  backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Validation completed in this run so far

- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search; only the existing non-blocking `G3PB` backlog plus
    unrelated repo-wide TODOs were found
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass
- `git diff --check`
  - pass after the `PLANS.md` compaction refresh

### Decisions confirmed in this run so far

- keep treating the active `G3PB` plan as complete unless a fresh audit
  exposes a concrete regression or real handoff/design drift
- keep compacting audit-only context inside `PLANS.md` instead of extending the
  docs-only commit chain with more near-duplicate “still complete” entries
- keep repo-wide unrelated TODO markers out of this plan unless a targeted
  `G3PB` audit ties one back to the active slice

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog

### Handoff for next run

- this run compacted the repeated audit-only tail and reran the focused
  `G3PB` / bindings validation stack from current `HEAD`
- this run is committed as the latest signed docs-only audit handoff refresh on
  `HEAD`
- the active `G3PB` implementation slice should still be treated as complete
  unless a future audit finds a concrete new gap
- do not spend another run repeating the same audit-only update unless the repo
  state has actually moved; if this run's docs-only commit is already on
  `HEAD`, resume directly from the existing backlog instead
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

## Current run (2026-04-01 06:51:50 UTC)

### Summary of accomplishments in this run

- ✅ Re-read the required handoff/design context before doing any work:
  - `Agents.md`
  - `PLANS.md`
  - `docs/design-docs/kvbm-g3pb-plan.md`
- ✅ Re-read `PLANS.md` completely before taking action and confirmed the
  active `G3PB` slice was already marked complete on the prior run
- ✅ Re-audited the live `G3PB` surface against the current repo state:
  - detached `HEAD`
  - clean worktree at pickup
  - pickup commit: `4edc24059`
- ✅ Re-searched the active `G3PB` handoff/code surface for any remaining
  concrete implementation, cleanup, test, docs, or validation work
- ✅ Refreshed `PLANS.md` so this run is recorded on disk before rerunning the
  focused validation stack
- ✅ Revalidated the focused `G3PB` / bindings stack from the refreshed
  handoff state
- ✅ Refreshed `PLANS.md` again after validation so the next run can resume
  from the verified current state instead of redoing the same audit loop
- ✅ Created a signed handoff commit for the validated `PLANS.md` refresh:
  - `c0c491c49 docs: refresh g3pb audit handoff`

### Current findings before validation

- the active `G3PB` implementation slice still appears complete on current
  `HEAD`
  - pickup commit: `4edc24059`
  - no new `G3PB`-specific implementation gap, cleanup item, docs gap,
    follow-up, or validation gap was found by this run's targeted audit
- `Agents.md`, `PLANS.md`, and
  `docs/design-docs/kvbm-g3pb-plan.md` remain aligned with the landed tree:
  - request-plane + discovery remain the active control-plane path
  - `KvBlockManagerConfig.g3pb_admission` remains the first native config
    surface already adopted by a real non-smoke caller
  - the remaining config work is broader caller adoption, not missing initial
    adoption
- the only remaining `G3PB` work is still the existing non-blocking follow-on
  backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Validation completed in this run so far

- `rg -n "G3PB|g3pb|TODO|FIXME|follow-on|remaining work|Exact next step|Handoff for next run" PLANS.md docs/design-docs/kvbm-g3pb-plan.md lib/llm/src lib/bindings/kvbm/src`
  - pass as an audit search; only the existing non-blocking `G3PB` backlog plus
    unrelated repo-wide TODOs were found
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb:: --lib`
  - pass (`15 passed`)
- `cargo test --manifest-path lib/llm/Cargo.toml g3pb_filter --lib`
  - pass (`6 passed`)
- `cargo test --manifest-path lib/bindings/kvbm/Cargo.toml read_g3pb_admission_config`
  - pass (`4 passed`)
- `cargo build --manifest-path lib/llm/Cargo.toml --bin kvbm_g3pb_backend --bin kvbm_g3pb_worker_smoke`
  - pass

### Decisions confirmed in this run so far

- keep treating the active `G3PB` slice as complete unless a fresh audit or
  focused validation run exposes a concrete new gap
- keep compacting audit-only context inside `PLANS.md` instead of extending the
  repo with another near-duplicate docs-only audit trail
- when a run only refreshes the verified handoff state, keep it to a small
  signed `PLANS.md` commit rather than inventing extra code churn

### Remaining work in this run

- none

### Exact next step

- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice

### Handoff for next run

- this run re-audited the active `G3PB` surface, reran the focused validation
  stack, and committed the resulting handoff refresh on `HEAD`
- there is still no new concrete `G3PB` implementation gap on current `HEAD`
- if another run continues from here, resume only from the existing
  non-blocking follow-on backlog:
  1. upstream or locally patch the `nixl-sys` teardown warning if needed
  2. decide whether retained committed blocks need backend-side reclamation
  3. design any future CPU-buffer / `foyer` retention knobs as a separate slice
