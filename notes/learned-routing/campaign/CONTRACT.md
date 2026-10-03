# Learned-routing campaign: shared contract

This file is authoritative for every agent in this campaign. Read it, together with
`<repo>/notes/learned-routing/PLAN.md` (the campaign plan; operator decisions are at its
bottom), before you act. If you have to deviate, record why in `facts/DEVIATIONS.md` and in your
returned summary.

## Paths

| Name | Path |
|---|---|
| CR (campaign root) | `<campaign-root>` |
| WT (worktree) | `<worktree>`, branch `rupei/learned-routing-public`, created from `origin/rupei/router-policy-aic-ttft` |
| PY | `WT/.venv/bin/python`. Install with `uv pip install --python WT/.venv/bin/python ...`. Never use bare pip or another worktree's venv. |
| Harness code | `WT/benchmarks/learned_routing/` (package `learned_routing`, installed editable into PY) |
| Engine config | `CR/config/engine.json`, written by setup. It is the single source of truth for model, hardware, backend, TP, `max_model_len`, AIS config and `MockEngineArgs`. |
| Traces | `CR/traces/<family>/...`, plus `CR/traces/MANIFEST.json` with each file's SHA-256, row count, format and block size |
| Cells | `CR/cells/{train,val,test}.jsonl`, plus `CR/cells/SPLIT_MANIFEST.json`. `test.jsonl` is frozen at calibration; its SHA-256 is recorded in `CR/facts/test_freeze.json`. |
| Runs | `CR/runs/<stage>/<run_id>/`: checkpoints, logs, `results.jsonl` |
| Result cache | `CR/runs/cache/`, keyed by `sha256(canonical policy YAML) + cell_id + harness_version` |
| Facts | `CR/facts/*.json` and `CR/facts/STATE.md`. STATE.md is append-only: one dated line per stage result. |
| Audits | `CR/audits/<checkpoint>/<lens>.md` |
| Literature | `CR/literature/LESSONS.md`, written by a sidecar and possibly appearing mid-run. Stages from calibration onward read it if present and record which lessons they applied or rejected. |
| Report | `CR/report/REPORT.md`, with figures in `CR/report/fig/` |
| Cleanup ledger | `CR/CLEANUP.md` |

## Hard rules

- **No deletion.** No `rm -rf`, no `git clean`, no deletion of any file you didn't create in this run.
  List bulky scratch (path and size) in `CR/CLEANUP.md` instead. Overwriting your own outputs is fine.
- **Don't touch the main checkout.** Leave `<repo>` and its branch
  `rupei/recovery-snapshot-cache` alone. Only the final report stage copies `REPORT.md` into
  `<repo>/notes/learned-routing/`.
- **No pushing and no PRs.** Commit in WT only:
  - sign off with `git commit -s`;
  - add **no** `Co-Authored-By` trailer;
  - use a Conventional Commit subject.

  During the parallel build stage, builders do not commit; the integrator commits.
- **Processes.** Stop processes by exact PID only, never by pattern.
- **Long jobs.**
  - Every job longer than about 9 minutes is a resumable, checkpointed CLI that takes
    `--max-wall-seconds`. Call it repeatedly in foreground chunks of 540 s or less, or start it with
    `nohup` and its PID recorded under `CR/runs/...`, then wait with
    `timeout 540 tail --pid=<pid> -f /dev/null` and re-check.
  - Never wait on wall-clock with a bare `sleep`.
- **CPU discipline.** All local replay processes take a slot from a shared pool of 20 file-lock slots in
  `CR/slots/` via the harness. No agent may run replays outside the harness pool, except single smoke
  runs.
- **Python.** Follow the user's CLAUDE.md Python rules. Set `PYTHONDONTWRITEBYTECODE=1` for ad hoc runs.
  Local unit tests are encouraged.
- **Dynamo/AISim bugs or missing plumbing.** Patch them on the WT branch, and record each in
  `CR/facts/UPSTREAM_FOLLOWUPS.md` (symptom, fix, file:line, upstream-worthy y/n). If the aisimulate-core
  crate (cargo registry) needs a patch, use a `[patch.crates-io]` path override that points at a copy
  under `CR/vendor/`. Never edit the registry in place.
- **Integrity.** Never fabricate or extrapolate a number. Every claimed metric must trace to a file under
  `CR/runs/`. Label every hypothesis as one.

## Deployment (operator-fixed)

- **Model and backend:** Qwen/Qwen3-32B on vLLM 0.24.0, h100_sxm, TP2, aggregated.
- **Timing:** AIS, via `ais_perf_config` in `MockEngineArgs`. The bindings must be built with the
  `ais-forward-pass` feature (use the `build` skill's rule for this mode).
- **`max_model_len`:** 131072 (YaRN). If AIS or replay can't run 128K, token-scale AgentX plays to fit
  32768 (divide lengths and hash blocks by a common factor) and record this in `facts/DEVIATIONS.md`.
- **Worker counts:** N ∈ {2, 4, 6, 8, 16, 32}. Train on {4, 8}; validation uses {4, 8} plus 6; test uses
  {2, 6, 16, 32} and the held-out cells at {4, 8}.

## Policy YAML (what the harness writes and replay consumes)

Every policy, baseline or learned, is a single router policy YAML passed to replay as
`KvRouterConfig(router_policy_config=<path>)`, or through the bench binary's `--router-policy-config`.
The harness writes one canonical file per policy evaluation, with keys sorted and floats in repr form,
under `CR/runs/policies/<sha256>.yaml`. Round-robin uses `router_mode="round_robin"`, with no YAML.

```yaml
worker_selection:
  aggregated: candidate
  instances:
    - name: candidate
      type: <policy type>          # e.g. dynamo-default-cost-fn, lmetric, learned-choice, sticky-session
      parameters: { ... }          # policy-specific
```

Default-cost knobs that live in `KvRouterConfig` rather than in parameters (for example
`router_temperature` and `router_queue_threshold`) travel in a sidecar `router_config` object in the
harness policy spec. The harness passes them through.

## `learned-choice` policy (Rust, builtin catalog, type string `learned-choice`)

- **Where:** implement it as a `WorkerPicker`, using the provider pattern of
  `lib/router-plugins/builtin/src/two_tier_cost_fn.rs`. It declares the inputs `CACHE | LOAD`, plus
  whatever `worker_capacity()` needs.
- **Purpose:** it is the candidate production artifact. It must not depend on AIS.

**Utility of candidate i in set S:**

```
x_i = features(i)                     (feature_set "v1" below, fixed order)
u_i = θ·x_i + Σ_k (p_k·x_i)(q_k·x̄_S)  (low-rank context term; x̄_S = mean of x over candidates)
```

**Decision:**
- With `temperature == 0`, take the argmax of u, breaking ties with a seeded RNG (`seed` parameter).
- Otherwise sample from `softmax(u / temperature)` with the seeded RNG.
- Hard pins and the host's eligibility always win.

**Parameters:**

```yaml
parameters:
  feature_set: v1
  theta: [..d floats..]
  context: { p: [[..d..], ...], q: [[..d..], ...] }   # rank r = len(p) = len(q); empty = plain MNL
  temperature: 0.0
  seed: 1
```

The policy rejects unknown keys and wrong lengths at startup.

**Feature set v1, in order** (normalized, dimensionless where possible; computed only from the
public plugin API):

| # | Feature | Definition |
|---|---|---|
| 0 | `default_logit_scaled` | the default cost-fn logit for this row at default config, divided by `max(request_blocks, 1)`. With θ = −e₀ (−1 here, 0 elsewhere) and no context, the policy reproduces default argmin routing. This anchors the parity test. |
| 1 | `overlap_frac` | effective overlap blocks / request_blocks (`accounting_cache_estimate`, since replay leaves device-tier counts at 0) |
| 2 | `new_prefill_tokens_k` | (prompt_tokens − estimated cached tokens) / 8192 |
| 3 | `active_prefill_tokens_k` | active_prefill_tokens / 8192 |
| 4 | `kv_load_frac` | decode_cost_blocks / total_kv_blocks from `worker_capacity()` (fallback: / 65536; record it in facts) |
| 5 | `active_requests_s` | active_requests / 32 |
| 6 | `session_affinity` | 1.0 if this worker served the session's previous request (policy-internal map keyed by `session_context().session_id`, bounded LRU), else 0 |
| 7 | `isl_x_prefill_load` | (prompt_tokens / 8192) × feature 3 |

**Never used:** `expected_output_tokens`, anything from AIS (including `modeled_prefill_backlog_ms`), and
the true OSL.

**Optional feature_set v2:** additive features such as within-set rank of load and z-scores, documented
in `WT/lib/router-plugins/builtin/src/learned_choice/FEATURES.md`. v1 must stay stable once
calibration starts.

**Tests:**
- Unit tests cover the feature math, the context term and determinism.
- A **parity test** shows that θ = −e₀, rank 0, temperature 0 picks the same row as the default cost
  fn's argmin on randomized candidate tables with no ties.
- Replay parity: the same cells under `learned-choice@θ0` and `dynamo-default-cost-fn` give goodput
  equal within tie-noise. Measured at the integration stage.

## `sticky-session` policy (Rust, type string `sticky-session`)

- **Parameters:** `mode: hard | bounded`, `load_factor: 1.25` (bounded only), `max_sessions: 65536`.
- **Behavior:**
  - First request of a session, or no session ID: choose with the default cost fn.
  - Later requests with `hard`: the session's bound worker if it's eligible, otherwise the default cost
    fn (then rebind).
  - Later requests with `bounded`: the bound worker unless its `active_requests` is greater than
    `load_factor` × the candidate mean, in which case the default cost fn (then rebind).
- This mirrors live `SessionAffinityMode` intent. Replay never sets `affinity_target`, so stickiness must
  live in policy state.
- If replay does not pass `session_context` to policies, patch the replay request construction on WT and
  record it as a follow-up.

## Cell spec (one JSON object per line in `CR/cells/*.jsonl`)

```json
{"cell_id": "mooncake-w03-isl1.0-n8-open-L2", "split": "train|val|test", "family": "mooncake|toolagent|agentx|synthetic_sessions",
 "trace_files": ["CR/traces/..."], "trace_format": "mooncake|weka|...", "trace_block_size": 512,
 "transform": {"window": [t0_ms, t1_ms], "isl_unique_mult": 1.0, "isl_prefix_mult": 1.0, "osl_mult": 1.0, "prefix_root_mult": 1, "think_mult": 1.0, "seed": 0},
 "load": {"mode": "open_speedup|open_rate|closed_concurrency|agentic_lanes", "value": 1.0, "level": "L1|L2|L3"},
 "num_workers": 8, "sla": {"ttft_ms": null, "itl_ms": null}, "holdout_axis": null,
 "engine_ref": "CR/config/engine.json", "trace_sha256": ["..."]}
```

- Calibration fills in `sla` and `load.value`, then freezes the cells.
- Transformed traces are materialized once into `CR/traces/derived/<sha>.jsonl` and referenced by path.

## Harness API (`learned_routing` package)

**`lr-eval`.** Usage: `lr-eval --policy-spec spec.json --cells cells.jsonl --out results.jsonl
[--repeats K] [--slots 20] [--max-wall-seconds S]`. It evaluates every (policy, cell, repeat) through the
slot pool and the cache, and appends one record per evaluation:

```json
{"policy_sha", "policy_type", "cell_id", "repeat", "num_requests", "completed",
 "goodput_rps", "goodput_rps_report", "good_frac", "duration_ms", "ttft_p50", "ttft_p90", "itl_p50",
 "itl_p90", "throughput_tok_s", "prefix_reuse", "wall_s", "replay_entrypoint", "harness_version",
 "error"}
```

- `goodput_rps` is recomputed per request from `per_request` (TTFT ≤ T and mean ITL ≤ I, with ITL skipped
  when the output has ≤ 1 token). `goodput_rps_report` is the native field. A disagreement above 1e-6
  relative is an error.

**`lr-train`.** Usage: `lr-train --space space.yaml --cells train.jsonl --val val.jsonl --run-dir DIR
--budget-evals N [--popsize λ] [--seed s] --max-wall-seconds S`.

- **Method:** CMA-ES via the `cma` package.
- **Objective:** maximize the mean over cells of `goodput_rps(policy) / goodput_rps(default@defaults)`.
  The default reference comes from the cache.
- **Resumable:** checkpoints after every generation and resumes from `DIR/checkpoint.pkl`.
- **Writes:** `DIR/best.json`, `DIR/history.jsonl` and `DIR/best_policy.yaml`.
- **Bounds and transforms:** `space.yaml` declares the free parameters, their bounds or transforms, and
  the fixed ones. The same tool tunes baselines (their catalog parameters plus `router_config` knobs)
  and the learned models (`theta`, `context.p`, `context.q`, `temperature`).

**`lr-report`.** Utilities that build tables and figures from `results.jsonl` files, with bootstrap
confidence intervals over cells and repeats.

**Remote shard mode.** `lr-eval` must also run on a CPU-cluster node from a self-contained bundle:
- a built wheel,
- the `learned_routing` sdist,
- the needed traces,
- `cells.jsonl`,
- the policy specs.

Results sync back into `CR/runs/`.

## Noise, determinism, gates

- **Noise floor.** Setup measures replay determinism: the same policy, cell and repeat run twice. If
  results are nondeterministic (for example from default-picker tie RNG), `lr-eval` uses `--repeats 3`
  for gate and test decisions and reports mean ± sd.
- **"Differs beyond noise".** |Δ mean normalized goodput| > 3 × the pooled sd of repeats on at least one
  validation cell, **and** the mean Δ over validation cells > 2 × its standard error.
- **Pilot gate.** Escalate only if all three hold:
  1. Fewer than 2% of replays error.
  2. The projected full-run cost is feasible, using CPU-cluster if needed. Use the cluster runbook
     to request CPU nodes when the projected local wall-clock for the full stage exceeds 8 hours.
  3. Some policy differs from default beyond noise on validation.

  If (3) fails: recalibrate once, then re-check. A second failure stops the campaign with a diagnostic.

- **Amendment A1 (2026-10-02, setup fixer r0, audit F1; binding over the two bullets above).**
  Identical re-runs are deterministic for every deterministic policy, so they never measure noise.
  - `--repeats K` means CRN workload replicates k = 0..K-1 (protocol `crn-order-v1`): replicate k of a
    cell is `learned_routing.replicates.materialize_replicate(trace, format, k)`, a seeded permutation of
    the arbitrary order of simultaneous arrivals shared by every policy, plus policy `seed` k + 1.
    Synthetic sessions use `arrival_seed = synthetic_arrival_seed(spec_sha, k)`.
  - The cache key adds the protocol id and k. The policy seed is not part of `policy_sha`.
  - The pooled sd is that of paired per-replicate ratios against default on the same replicate
    (`learned_routing.noise`). The 2 × SE clause runs over independent workload segments (≥ 3).
  - Replicates: ≥ 3 for validation, test and the gate; ≥ 2 per CMA-ES evaluation; 8 for noise
    calibration per family and N.
  - Specification and measured values: `facts/noise.json`. Rationale: `facts/DEVIATIONS.md`.

## Amendment A2: operator decisions (2026-10-02 ~14:30 PDT)

A2 binds over the SLA and goodput definitions above, PLAN.md's Objective section, and any
`literature/LESSONS.md` advice that conflicts with it (in particular LR-04's paper-faithful baselines and
LR-08's length-scaled **TTFT** SLO).

1. **Headline bar: best baseline on the same footing.** Every policy, learned or baseline, is ONE config
   tuned on the pooled train split with the equal CMA-ES budget. Validation selects each policy's
   config and the "best baseline". The headline is the learned model against that val-selected best
   baseline on the frozen test set. Report the per-family best baseline and the per-cell oracle as
   secondary context only.
2. **No TTFT SLO.** The operator cares about ITL and total request latency, not TTFT. A request is
   "good" iff both hold:
   - **ITL:** mean ITL, (e2e − ttft)/(output_length − 1), is ≤ I. Skip this check when output_length ≤ 1.
   - **E2E slowdown:** e2e_latency ≤ S × E0(ISL, OSL). E0 is the request's **policy-independent
     uncontended latency**: the AIS-timed latency of that request alone on an idle worker with **no
     prefix reuse** (full prefill of ISL, then OSL decode steps at batch 1).
     - Compute E0 once per (ISL, OSL), from the AIS estimator or from a single-request replay, cache
       it, and record the method in `facts/calibration.json`.
     - This is an evaluation-only use of AIS. Policies must never see E0.
     - Because E0 assumes no reuse, a policy that gets more cache hits is never penalized by a tighter
       reference.

   Calibration chooses I and S per family by the band rule (default's good fraction about 0.95, 0.85,
   0.65 at L1, L2, L3, with SLAs fixed from L2), records the rule, and applies it identically to every
   split.
   - TTFT is still reported (p50/p90, and by ISL bucket) but is not part of "good".
   - Per LR-08's rescoring idea, re-score every evaluation at SLO scales {0.5, 0.75, 1, 1.5, 2, 3} × (I, S)
     from `per_request` without extra replays. REPORT states where the headline flips.
3. **Windowed goodput (LR-01 adopted).**
   - **Open loop:** good fraction among requests that ARRIVE inside the measurement window, which
     excludes warmup and post-window arrivals. In-window arrivals that finish late still count, and
     incomplete ones count as not good. Report good in-window requests divided by window length.
   - **Closed loop:** good completions per second inside the steady-state window.
   - Calibration fixes and records the window rule.
   - The training objective's per-cell ratio against the default reference may use LR-01's clipped
     log-ratio. Record the choice.
4. **Native-report cross-check.** The native `SlaThresholds` can't express E2E slowdown, so the
   "mismatch > 1e-6 is an error" check applies only to an extra ITL-only scoring pass, where the two are
   comparable. The A2 goodput itself is verified by the harness's unit tests and by an independent
   auditor's recomputation.
5. **Baselines are evaluated as implemented on this branch. Do not reproduce paper numbers.**
   - Do not build paper-faithful variants of the ported policies (lmetric, ramjet, dualmap, chwbl,
     llm-d-*), and don't make paper reproduction a goal.
   - Do only a brief sanity check that each port isn't wildly off, e.g. its decisions respond to its
     inputs in the documented direction. Record any gross bug as a follow-up.
   - Each baseline is still tuned over its shipped parameters with the equal budget, and also reported
     at its shipped defaults.

## Amendment A3: AgentX via our own Agentic Mooncake v2 lowering (operator-approved 2026-10-02 ~16:20 PDT)

**Why.** Native Weka replay can't hold a steady load at N = 16/32:
- In timestamp mode every play's root starts at t = 0 (`weka.rs` sets `not_before_ms = t − root_time`).
- In lanes mode plays are dealt round-robin to lanes in one pass, with no recycling.

So AgentX was limited to N ≤ 8.

**What changes.** A sidecar workflow builds `learned_routing.workloads.agentx_lowered` and reports in
`facts/agentx_lowered.json`. When that file says `"status": "ok"`, AgentX cells for EVERY N in the
contract grid use the lowered traces instead of native Weka:

- **Format.** Agentic Mooncake v2 (`trace_format="agentic_mooncake"`). Dependency edges (sequence,
  spawn, join, triggers, `delay_ms`), tool gaps and request lengths are preserved exactly as AISim's Weka
  lowering produces them.
- **Recycling.** Plays are drawn per split, with replacement, from that split's own play pool, using
  the same disjoint train/val/test play pools as B2. Every copy gets unique `play_id`, `session_id` and
  `request_id` values, plus a disjoint hash-ID range, so copies never share KV prefixes.
- **Open loop.** Play roots arrive as a seeded Poisson process. `not_before_ms` is set on root rows,
  and descendants keep their relative timing and dependency semantics. The rate is a per-worker play
  rate × N.
- **Closed loop.** Native `agentic_lanes` = per-worker lanes × N over a lowered file that holds many
  more copies than lanes, so lanes recycle and reach a steady state.
- **Windowed goodput (A2).** Applies as usual, with a warmup long enough for in-flight plays to fill.

**Calibration ordering.** Calibrate Mooncake, FAST25 and the session families first. Reach AgentX last.
If `facts/agentx_lowered.json` is still missing then, wait for it in bounded chunks of 540 s or less,
up to 120 minutes total, re-checking the file. If it's still missing, or its status isn't ok, fall back
to the native-Weka AgentX cells (N ≤ 8), record that in `DEVIATIONS.md`, and continue.

## Amendment A4: synthetic trace extension by duplication (operator-approved 2026-10-02 ~16:35 PDT)

Any stage may lengthen a workload by duplicating its source units. This covers AgentX plays (per A3),
Mooncake or FAST25 segments, and session traces. Use it when a cell needs a longer replay, for example a
long enough steady-state window after warmup at large N, or enough independent segments for LR-11's
headline test.

**Rules:**
- **Isolation.** Each duplicate gets a disjoint hash-ID range and unique request, session and play IDs,
  so duplicates never share KV prefixes with each other or with the original.
  - The one exception is a cross-copy prefix-sharing structure that is part of the cell's documented
    transform, such as `prefix_root_mult`.
- **Split purity.** Duplicate only within a split's own source pool. Never copy a val or test segment or
  play into train, or the reverse.
- **Arrival timing.**
  - Open loop: duplicated segments are concatenated or interleaved in time with the original
    inter-arrival structure.
  - AgentX: arrivals come from A3's play-arrival process.
  - Document the method in the cell's transform metadata.
- **Provenance.** The derived trace's metadata and `traces/MANIFEST.json` record the duplication
  factor, the method and the seed.
- **Statistics.** Treat duplicates as NOT independent for LR-11-style significance. Segments remain
  the unit of independence, and duplication only lengthens the measurement.

## Amendment A5: staleness robustness and worker-set scope (operator, 2026-10-02 ~16:45 PDT)

1. **Staleness, as a test-stage robustness check only.**
   - **Plumbing.** Replay normally gives the router perfectly fresh overlap and load state. Before the
     test stage, patch replay on the WT branch to add a configurable router-state lag (e.g. delay the KV
     overlap/indexer updates and the load updates the policies see by `lag_ms`). Record it in
     `UPSTREAM_FOLLOWUPS.md`.
   - **Proof the knob works:**
     - lag = 0 is byte-identical to the unpatched replay;
     - lag > 0 actually changes what policies see.
   - **Evaluation.** Re-evaluate the finalists on the frozen test cells at lag ∈ {0, 50, 200} ms. REPORT
     states whether the headline ranking holds.
   - **Training is unchanged:** fresh state, no lag randomization.
2. **Worker-set extrapolation means worker COUNTS only.** All workers are identical (Qwen3-32B TP2
   H100). Heterogeneous worker sets are out of scope.

## Amendment A6: straight-main lane for production-ready generic fixes (operator, 2026-10-02 ~16:55 PDT)

The operator authorizes pushing production-ready, generic, permanent fixes, patches and ergonomics
improvements found during this campaign directly to `main`, through the `straight-main` skill, for
ai-dynamo/dynamo and ai-dynamo/aisimulate.

**Campaign agents:**
- Never push or run `straight-main` yourselves.
- When you find a fix that is generic beyond this campaign, add it to `facts/UPSTREAM_FOLLOWUPS.md` with
  "straight-main candidate: y" and a one-line rationale. The top-level session harvests candidates.

**Bar:**
- It's a real bug fix, or a small backward-compatible option or ergonomics improvement.
- It's not campaign-specific, and it doesn't touch the operator's open PR stack (#15450/#15453). Those
  fixes are recorded as follow-ups for that stack instead.
- It comes with tests, and passes fmt, clippy and tests.
- It follows repo conventions, including docs for user-facing parameters.
- If default behavior must not change, byte-level canonical replay parity is proven (`offline_replay_bench`,
  4 canonical rows).

**Where the work happens:**
- Every port goes into a dedicated clean worktree off `origin/main`:
  `<other-worktree>`, and `<aisimulate-checkout>`, a fresh worktree of
  <aisimulate-checkout>.
- Never use the main checkout, the campaign WT, or `<aisimulate-checkout>`'s current branch, because
  `straight-main` commits everything in its worktree.
- Cargo builds for this lane run with `CARGO_BUILD_JOBS=6` under `nice -n 10`, to spare calibration CPU.

### A6 update (2026-10-02 ~17:35 PDT)

- **Direct pushes to `ai-dynamo/aisimulate` main are blocked.** A repository ruleset requires 3 status
  checks, and pushes are declined. Dynamo main almost certainly has the same protection.
- **The lane becomes "prepare a clean branch, push the branch, open a draft PR".** Opening PRs is
  blocked until the operator re-authenticates `gh`.
- **Campaign agents:** keep flagging candidates in `UPSTREAM_FOLLOWUPS.md`; nothing else changes for
  you.

## Amendment A7: replay device-tier overlap (operator, 2026-10-02 ~19:40 PDT)

- **Pending fix.** If the stack fix that populates device-tier KV overlap for catalog policies in replay
  is confirmed, the top-level session will cherry-pick it onto the campaign branch before phase 2 tunes
  baselines. The fix must keep default replay byte-identical. Without it, two-tier and other policies
  that read GPU-tier overlap degrade to load-only routing in replay.
- **Until then:** phase-1 stages must not build conclusions on two-tier's replay behavior.
- **Bindings rebuild.** If the fix lands mid-calibration, the bindings `build_id` changes. Cache keys
  already include `build_id`, so stale results stay unreachable.

### A7 correction (2026-10-02 ~19:55 PDT)

- **The replay device-tier overlap fix is already in the campaign base.** The stack head 2be8d9c43a fills
  `tier_overlap_blocks.device` from the overlap scores in both replay hosts (offline
  `kv_router/mod.rs:297-306`, online `router.rs:305-312`) and has the regression test
  `replay_passes_device_overlap_to_catalog_policies`.
- **No cherry-pick is needed.** Two-tier's replay behavior is valid.
- **Stale notes.** Earlier notes saying "replay leaves device-tier counts at 0" described main, not this
  branch. Feature 1 still uses `accounting_cache_estimate` as contracted.

## Amendment A8: repo notes mirror (operator, 2026-10-03 ~01:20 PDT)

- **Who maintains the record:** the top-level session keeps a reproducible campaign record on the
  campaign branch, under `notes/learned-routing/`:
  - `README.md` and `REPRODUCE.md`;
  - `campaign/`, a mirror of small curated files from CR, produced by `sync_from_campaign.sh`.

  It pushes the branch to `origin/rupei/learned-routing-public` (no PR) at milestones.
- **Size limit:** no bulky artifacts in git. The per-file cap is 1 MiB; traces, runs, caches and PDFs
  stay in CR and are identified by manifests and SHA-256s.
- **Stage agents:**
  - Write reproducibility-relevant results to `CR/facts/`, `CR/cells/`, `CR/audits/` and
    `CR/report/` as before.
  - Don't edit `notes/learned-routing/` and don't push. The top-level session syncs.
  - Commit in the WT only your own paths: `git commit -s -- <paths>`.

## Amendment A9: use the CPU-cluster lane eagerly (operator, 2026-10-03 ~01:40 PDT)

- **Operator request:** use CPU-cluster nodes as needed. Workflow `wf_e37792bc-dd5` is provisioning and
  validating the lane; it writes `facts/remote.json` and runner scripts under
  `WT/benchmarks/learned_routing/remote/`.
- **When to use it:** once `facts/remote.json` reports `"status": "ok"`, any stage (pilot, phase-2
  tuning, test eval, robustness) SHOULD run on the CPU cluster any batch or tuning job projected to take more
  than about 1 hour of local wall-clock. This replaces CONTRACT's earlier 8-hour gate threshold.
  - Run independent `lr-train` jobs one per node.
  - Shard large `lr-eval` batches.
  - Always ingest remote results into the cache and record the remote evidence.
- **Before it's ready:** if the lane is not ready when a stage starts, run locally. Re-check
  `facts/remote.json` before each long batch.
- **Allocations:** record every allocation in `facts/remote.json`, with exact cancel commands. Release
  idle holds. Never delete remote scratch; list it in `CLEANUP.md`.

### A9 addendum (operator, 2026-10-03 ~01:50 PDT)

- **Concurrency is fine.** Hold multiple CPU-cluster nodes at the same time, and run batches and tuning
  jobs in parallel across them.
- **Node type doesn't matter.** Offline replay runs in simulated time, so host speed never changes
  results, and mixed node types are acceptable.
- **The one requirement:** one bit-exact parity smoke against local per distinct node OS or glibc
  image, because remote runs use that host's libm. Record which images have been validated in
  `facts/remote.json`.
- **Unchanged:** release idle holds, and record every job with its cancel command.

## Amendment A10: CPU-cluster lane is ready (top-level, 2026-10-03 ~02:51 PDT)

- **Status:** `facts/remote.json` reports `"status": "ok"`, independently verified as bit-exact across
  4 node images. Runner scripts are in `WT/benchmarks/learned_routing/remote/` (commit <commit-18>). Use
  them per A9.
- **Hard deadline:** CPU cluster has a weekly maintenance cutoff. Every <qos> job must
  end by then, and longer `--time` is rejected.
  - Size remote jobs to finish about 30 min before the cutoff.
  - `lr-train` checkpoints sync back at least every 540 s; resume locally or resubmit after the maintenance window.
  - Node availability after the maintenance window is unmeasured, so plan for local fallback.
- **CMA-ES numerics are pinned** in `node_train.sh` (`OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=1
  NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`). Export the same variables for local
  `lr-train` runs that must stay bit-exact with remote runs; they match the workstation's defaults.
- **Co-location:** jobs share nodes, so use `--mem` above half the node's RAM (160G on 256 GB
  nodes, 400G on 768 GB nodes) to avoid stacking our own jobs. Slots default to physical cores.

## Amendment A11: phase-2 deltas on top of `facts/gate.json` full_plan (top-level, 2026-10-03 ~05:26 PDT)

The gate's `full_plan` governs phase 2, with these changes from verified literature lessons. They target
the pilot's headline risk (M1 vs the best heuristic was +0.030, below MDE 0.038, with M1 trailing on
AgentX).

1. **M1 multi-start (LR-07).** M1's 3 restarts use 3 different initializations, each with the same
   B = 400:
   - **s1:** θ0 = −e0, the default router.
   - **s2:** behavior-clone init. Fit θ by conditional-logit maximum likelihood (convex) to the
     decisions of the val-best default-arm heuristic (`llm-d-precise-prefix`), using candidate tables
     reconstructed from its per-request rows on TRAIN cells only.
   - **s3:** a cache-heavy init, overlap-dominant, which LR-07 recommends for AgentX. Alternatively,
     an ML fit to the second-best heuristic family if that's more informative; record the choice.

   **Not extra budget:** initializations use only train-split heuristic decisions, which is an
   initialization choice and not extra objective evaluations. Baselines start from their own defaults
   as the gate specifies. Document this in REPORT under "fairness". M2 warm-starts from the selected
   M1, per the gate.
2. **Concentration (LR-13).** No hard constraint up front. The stage-4 mid audit must test:
   - whether M1's worker-share cap violations (pilot: 5 Mooncake val cells, max share 0.646) are
     objective gaming (sacrifice or starvation) or legitimate cache affinity;
   - guard metrics.

   If it's gaming, re-run the affected M1 restarts with sign-constrained load coefficients (load must
   not attract), at the same B. REPORT includes the concentration and segregation flags.
3. **N = 6 is in both val and test.** Label test N = 6 results "selection-exposed" in REPORT (LR-10/11).
4. **Scale.** CPU cluster per A9, A10 and the addendum: as many nodes as are available. Respect the
   weekly maintenance cutoff. Breadth-first waves, per the gate.
5. **Lag patch (A5).** Implement it in a SEPARATE worktree with its own venv or bundle, so the tuning
   build's `build_id` and cache stay untouched. Prove lag = 0 gives identical results to the tuning
   build on sample cells. Use it only for the test-stage robustness pass.
