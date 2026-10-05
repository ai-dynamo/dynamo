# Audit: build checkpoint, lens "goodput-normalization", round 0 (dynamic)

- Auditor: independent dynamic auditor, 2026-10-02.
- Audited state: WT `rupei/learned-routing` at `<commit-07>` (clean), bindings build_id `4b4525a9…`.
- Evidence directory: `CR/runs/audits/build-goodput-normalization-r0/` (scripts in `scripts/`, outputs in `out/`,
  console logs in `logs/`).
- **Verdict: FAIL.** 0 blocker, 1 major, 8 minor.
  - The goodput math is correct. My independent recomputation matched the harness exactly.
  - The major finding is latent: `lr-report` pairs a policy with its reference by `(cell_id, repeat)` only. No
    output produced so far is affected.

## What I verified independently

### 1. Independent A2 goodput from raw `per_request`: exact match

- `scripts/replay_raw.py` re-ran 12 (cell, policy, k=1) evaluations directly through `dynamo.replay.run_trace_replay`.
  It does not import `learned_routing`, and each replay held a `CR/slots` flock.
  - It took only locators from the harness record (replicate trace by SHA, replay policy YAML, router mode).
  - Load arguments came from the cell, engine arguments from `engine.json`.
  - The raw rows are in `out/raw/*.jsonl.gz`.
- **Cells (5).** These cover every window regime:
  - Mooncake open-loop w0, N=8;
  - Mooncake open-loop w1, N=4. Its derived trace is rebased by 482,443 ms;
  - Mooncake closed-loop, concurrency 32;
  - synthetic sessions s0, open-loop, with causal turns;
  - AgentX A2, 4 lanes.
- **Policies.** default@defaults, lmetric@defaults, round_robin and sticky-hard.
- `scripts/goodput_indep.py` implements A2 from the CONTRACT text. It does not use the harness:
  - **Good:** completed, and mean ITL (e2e − ttft)/(OSL − 1) ≤ I unless OSL ≤ 1, and e2e ≤ S·E0(ISL, OSL).
  - **E0:** comes from my own isolated single-request replays (item 2), not the harness E0 table.
  - **Open-loop window:** derived from trace time, then mapped to replay time with the speedup. For Mooncake that
    is the transform window start + 240 s / + 720 s, minus the derived trace's `window_rebase_ms`. For sessions
    it is `measure_trace`. I did not use the cell's `measure` fields.
  - **Closed-loop and lanes:** I reproduced the harness's provisional rule, a completion basis over [first
    arrival, last arrival], and computed alternatives for sensitivity.
- **Result** (`out/goodput_compare.json`, `logs/goodput_indep.log`): all 12 records agree exactly. Absolute
  difference is 0 on:
  - `good`;
  - `window_good` / `window_requests`;
  - `good_frac_window` and `goodput_rps_window`;
  - `goodput_rps`;
  - window boundaries (`logs/window_boundaries.log`, max |Δ| 0.0 ms);
  - `rescore` at all six SLO scales.

  Example, Mooncake w0 N8 open, k=1: default 2047/2854 in-window good gives 3.434802 rps; lmetric 2414/2854 gives
  4.050617 rps.
- My raw rows equal the harness's cached `per_request` rows as a multiset of (arrival, ttft, e2e, ISL, OSL,
  worker) on 12/12 records.
- My ITL-only recount equals the native `goodput_request_throughput_rps` (relative error 0) on 12/12.
- None of the 963 cached records has `missing_rows`, `incomplete`, `window_fallback`, an error, or a nonzero native
  check error.

### 2. E0 against isolated single-request replays

- `scripts/e0_isolated.py` put every distinct (ISL, OSL) pair of the 12 replays (10,058 pairs) on ONE worker.
  Requests were 1,000 s apart with fresh hash ids. Each run asserts afterwards that the requests were isolated
  (arrival ≥ previous terminal), had no reuse, and completed.
- Against the harness table `CR/runs/cache/e0/01277503f73c1abe-ais-chunked-estimator-v1.json`, every pair was
  present:
  - maximum relative difference 9.7e-6, median 1.6e-6;
  - the harness value is lower on 10,035 of 10,058 pairs.

  The cause is finding F2.
- TTFT, which is prefill only, matches to 9e-9.
- Effect on goodput: none observed. Zero classification flips in 12 replays (41,956 requests) at any scale.

### 3. Edge cases

Evidence: `scripts/edge_replay.py`, a single smoke replay, then `scripts/edge_harness.py` driving
`learned_routing.goodput` (`out/edge_*.json`).

- **Rejected requests** (ISL ≥ 131072):
  - The row has `terminal_status: rejected`, an `arrival_time_ms`, `output_length` 0, and TTFT/e2e set to None.
  - The harness never counts them as good, and on an arrival basis counts them as in-window misses. This is
    correct under A2.3.
- **OSL ≤ 1:** the ITL check is skipped and E0 is prefill only. Mooncake cells have 20–22 such rows, all matched in
  item 1.
- **Router queueing is inside e2e:** aisimulate-core `report.rs:1997-1999` measures TTFT and e2e from
  `arrival_time_ms`. A hold-at-router policy therefore cannot hide queueing from the SLO.
- **Warm-up and drain:**
  - Open loop counts in-window arrivals that finish after the window (no defer-past-the-end incentive, LR-01).
  - The warm-up is excluded through `measure.warmup_ms`.
  - On w1, the integrator's `warmup_ms = 610,720.9` equals (480,000 + 240,000 − 482,443) / 0.388978, which is
    correct for the rebased trace.
- **End-of-window drain** (my hypothesis: Mooncake windows have no post-window arrivals, so the last arrivals see
  less contention):
  - Tested on 5 open Mooncake cells, all their policies, and up to 8 replicates (`logs/end_of_window.log`).
  - The last-10% minus middle-80% good fraction ranges from +0.022 (w0, default) to −0.090 (w5, N6). There is no
    consistent sign.
  - **Not supported; no finding.**

### 4. Normalization reference

- **lr-train.** I recomputed every candidate objective of both tiny lr-train runs (`train_m1_tiny`,
  `train_m2_tiny`; 18 candidates) from the full cached records (`logs/train_objective_recompute.log`).
  - Max |recomputed − history| = 0.
  - The reference is `38a79285…`, whose canonical YAML is `dynamo-default-cost-fn`, `parameters: {}`,
    `router_config: {}`. That is default@defaults.
  - Its seed is k+1 on every record.
  - Across all 139 records the candidate and its reference share the replicate trace SHA, `cell_sha`, build_id
    and protocol, with 0 mismatches. lr-train pairs inside one call and asserts that both sides cover the same
    (cell, k) set (CRN).
- **lr-report.** Its `norm_mean` in `runs/integrate/smoke/report/summary.json` equals my recomputation for all 7
  policies. Example: lmetric@defaults 1.0806832376250293. The pairing key is still unsafe in general (F1).

### 5. Cache-key coverage

Evidence: `scripts/key_coverage.py`, `out/key_coverage.json`.

- **Changes the cell SHA (covered):** `sla.itl_ms`, `sla.e2e_slowdown`, `measure.warmup_ms`, `load.value`,
  `num_workers`, `engine_overrides`, `replay_options` and `trace_block_size`.
- **Correctly ignored:** labels (`split`, `notes`, `expected_cost_s`, `measure_trace`).
- **Also covered:**
  - trace bytes (source SHA, checked against the declared `trace_sha256`);
  - `mock_engine_args` and model;
  - policy type, parameters and the `router_config` sidecar (`policy_sha`);
  - the `.so` SHA-256 plus the `aisimulate` version (the `aisimulate` distribution also ships `aisimulate_core`).
- **Gaps:** see F3.

## Findings

### F1 (MAJOR, latent): `lr-report` can pair a policy with a reference of different cell content, build or protocol

- **Defect.** `report.paired_ratios` keys the reference by `(cell_id, repeat)` only
  (`learned_routing/report.py:139-158`). `load_records` deduplicates only by `cache_key`.
- **Consequence.** If the input files hold the same `cell_id` under two contents, every ratio is silently computed
  against whichever reference record was loaded last. Two contents means a different `cell_sha` (load, SLA,
  `engine_overrides`), `build_id`, or protocol. Raw means and the bootstrap also pool the two contents under one
  cell.
- **Demonstration** (`scripts/report_pairing.py`, `out/report_pairing.json`):
  - A candidate at 1.1× its same-content reference is reported as **0.55**.
  - The reference's own ratios become [0.5, 1.0].
- **Why it is plausible in this campaign:**
  - calibration sweeps `load.value` or the SLA on a fixed `cell_id`;
  - Stage 5's LR-14 timing perturbation adds `engine_overrides` (`speedup_ratio` 0.8/1.2) to the same test cells;
  - any Rust fix rebuilds the bindings while result files span both builds.

  In every case the headline normalized goodput would be wrong, with no warning.
- **Current impact: none.** The smoke report has one content per `cell_id`, and lr-train pairs inside one call
  (item 4).
- **Fix:**
  - Pair on `(cell_id, cell_sha, repeat, replicate_protocol, build_id, harness_version)`.
  - Fail when one `cell_id` carries more than one `cell_sha` or `build_id` in the input, unless the user
    explicitly selects one.
  - lr-train's reduced `results.jsonl` lacks `cell_sha`, `build_id` and `trace_sha256`. Add them, or have
    lr-report resolve full records through `cache_key`.

### F2 (minor): E0 decode context is off by one versus the replay engine

- **Defect.** `e0.py:90-91` sums decode steps at KV context `ISL + j + 1`. The replay engine's batch-1 decode step
  j runs at `ISL + j + 2`.
- **Evidence** (`scripts/e0_convention.py`, `out/e0_convention.json`, 300 random pairs against isolated replays):

  | Context used | Max relative error | Pairs below / above replay |
  |---|---|---|
  | `ISL + j + 1` (harness) | 9.4e-6 | 299 / 1 |
  | `ISL + j + 2` | 9.9e-9 | 224 / 76 (float noise) |

  Over all 10,058 pairs, harness E0 is up to 9.7e-6 low (item 2). A2's "OSL decode steps" wording is loose. The
  harness correctly uses OSL − 1 steps after prefill, as the replay does.
- **Impact:** negligible. S·E0 is at most 1e-5 relative too strict, and there were zero flips in 41,956
  requests.
- **Fix:** use `+ 2`, **and** bump `METHOD` and `HARNESS_VERSION`. Otherwise the persisted E0 table and the result
  cache keep serving the old values (see F3b). Do it before calibration freezes SLAs, or leave it and record it.

### F3 (minor): Stale-cache risks: inputs outside the result cache key

a. **`seeded` is not part of `policy_sha`.**
   - `{"type": "dynamo-default-cost-fn", "seeded": false}` has the same `policy_sha`, and so the same cache key, as
     default@defaults.
   - Yet it replays with no seed, which means fresh entropy and nondeterministic ties. Both replay YAMLs are shown
     in `out/key_coverage.json`.
   - Whichever runs first fills the reference's cache entry. `seeded: true` on lmetric collides the same way.
   - Fix: put `seeded` into `identity()`, or reject `seeded: false` for `SEEDED_TYPES`.

b. **The E0 table has no build identity.**
   - It is keyed by the top-level `engine.json` `ais_perf_config` + chunk + method (`e0.py:46-48`). The replay
     itself uses `mock_engine_args.ais_perf_config`; the two are equal today.
   - The result key holds no E0 identity.
   - So a rebuild or aisimulate-core patch that changes AIS timing re-runs every replay but scores it against the
     stale E0 table. An E0 code fix without a `METHOD` bump is likewise masked (F2).
   - Fix: include `build_id` in the E0 file key, and include the E0 file identity in the result key.

c. **Code outside the `.so` is not hashed:**
   - the editable Python replay wrapper `WT/components/src/dynamo/replay/*.py`, which the contract allows patching
     for replay plumbing;
   - the `learned_routing` scoring and replicate code. It is covered only by the manual `HARNESS_VERSION = "lrh-1"`
     (`__init__.py` asks for a bump, but nothing enforces it);
   - the materialized replicate trace's SHA. The key holds only the source SHA + protocol id + k, so a change to
     the permutation code without a protocol bump reuses old results. The replicate SHA is already computed at
     plan time.

   Fix: add `rep.trace_sha256` and a hash of the scoring modules (`goodput.py`, `e0.py`, `worker.py`, `cells.py`,
   `replicates.py`) plus the replay wrapper to the key, or at least to `build_id`.

### F4 (minor): A missing `measure` silently scores the warm-up and the cold start

- Candidate cells carry `measure = {basis}` only. B2 put the trace-time hints in `measure_trace`.
- Without `warmup_ms` and `window_ms`, `goodput.measurement_window` scores from the first arrival to the last:
  - open loop includes the 4-minute warm-up;
  - closed loop and lanes include the cold-cache ramp.
- B2 and the integrator flagged this as calibration's job, but the harness raises no error.
- Fix: in `Cell.measure` or `plan()`, reject an open-loop cell whose `measure_trace.warmup_ms > 0` when
  `measure.warmup_ms` is unset. Require an explicit rule for closed loop and lanes once calibration records one.
- Nit: `window_ms: 0` is treated as unset (`goodput.py:173`, truthiness).

### F5 (minor): The objective's `eps` is an absolute 1e-3 rps

- `pair_score` uses `(m + 1e-3)/(m_ref + 1e-3)`. This shrinks every ratio toward 1 by about `eps/m_ref`, which
  depends on the cell's scale:
  - about **1.6%** on AgentX lanes (default `goodput_rps_window` 0.0624 on A2, 0.0671 on A3);
  - **< 0.08%** on Mooncake (1.28–5.69 rps).

  AgentX improvements are therefore slightly down-weighted in the pooled objective.
- `lr-report` instead drops pairs whose reference is 0 (`dropped_pairs`), so training and reporting treat
  degenerate cells differently.
- Fix: use a relative eps (`eps·m_ref`), or work in good-fraction units, and record the choice.

### F6 (minor): Closed-loop and lanes goodput mean something different from open-loop goodput

- **Open loop:** `goodput_rps_window` = offered in-window rate × good fraction. The window is
  policy-independent, so the ratio is an attainment ratio.
- **Closed loop and lanes:** good completions per second over [first arrival, last arrival]. That window is
  policy-dependent, so the ratio mixes throughput with attainment.
  - **Mooncake closed:** robust to the window choice. lmetric/default is 1.1986 under the harness rule, 1.1960 over
    the whole makespan, and 1.2059 with 10% trimmed from each end (`logs/closed_window_sensitivity.log`).
  - **AgentX lanes:** the window is dominated by the think-time span of the single slowest lane. On A2 at k=1,
    lmetric/default `good_frac_window` is **1.0021** while `goodput_rps_window` is **0.9959**: opposite signs,
    because lmetric's span is 0.6% longer.
  - **Sessions, open loop:** later turns are released causally, so the in-window denominator also depends on the
    policy (2,686 / 2,688 / 2,689 for default / sticky-hard / lmetric). This is small.
- Recommendation for calibration: fix and record the steady-state rule per mode (A2.3), choose the lanes metric
  deliberately, and report and test per load mode (LR-09, LR-11). Averaging open and closed cells in one mean
  combines two different quantities.

### F7 (minor, calibration note): Under the no-reuse E0, the E2E check barely binds on high-reuse families

Default replays at the provisional loads (`logs/slowdown_dist.log`):

| Family (cell) | e2e/E0 p50 | p90 | p99 | Token reuse |
|---|---|---|---|---|
| AgentX (A2) | 0.522 | 0.923 | 1.003 | 0.950 |
| Synthetic sessions (s0) | 1.054 | 1.168 | 1.301 | 0.742 |
| Mooncake (w0) | 1.62 | 4.10 | 12.47 | 0.355 |

- The fraction of requests with e2e < E0 is 93% on AgentX A2, 29–37% on sessions s0 and 3–49% on Mooncake (`logs/window_effects.log`).
- So for any S ≥ ~1.3, E2E never binds on AgentX or sessions. This explains the integrator's good_frac ≈ 1.0.
- To meet A2's band rule, AgentX likely needs S < 1, a regime where an idle worker cannot pass without prefix reuse,
  or much heavier lane loads. The 0.5× rescore then goes further into that regime. This follows from the A2 design
  (the operator chose a no-reuse reference). It is not a harness bug. Calibration should state which regime it
  picks.

### F8 (minor): Small scoring inconsistencies

None of these affects any current cell.

- Under the completion basis, `isl_buckets[*].good_frac_window` is not windowed. It equals the bucket's
  all-request `good_frac` (`goodput.py:293-299`).
- Requests missing from `per_request` are counted as in-window misses even when they would fall in the warm-up
  (`goodput.py:197`). This was never observed: 0 of 963 records.
- A truncated request (ISL + OSL > `max_model_len`) is scored on its truncated OSL and can be judged good. In the
  demonstration, 131,000/500 completed with 72 tokens and `a2_good` returned True. No candidate cell can
  truncate: the derived-trace `max_isl_plus_osl` / `max_context` is ≤ 131,072 in all 104 cells.

### F9 (minor): The contract wording for `goodput_rps` is stale

- The CONTRACT's lr-eval section defines `goodput_rps` as the TTFT/ITL recomputation, and lr-train as normalizing
  `goodput_rps`.
- Under A2 the harness's `goodput_rps` is A2-good over the makespan. It includes the drain and serves only as an
  integrity number.
- The objective is `goodput_rps_window`, which is the default in both lr-train (`identity.json` `metric`) and
  lr-report.
- A later stage that follows the contract text literally could pick the wrong field. Record the mapping in facts
  or in a contract note.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-01: checked windowed attainment, late finishers counted, warm-up excluded, and the closed-loop metric (F4,
    F6);
  - LR-03: CRN pairing on the same k and seed k+1, verified;
  - LR-08: the rescoring sweep, verified exactly. A2 supersedes its TTFT part;
  - LR-09: stratify by load mode (F6);
  - LR-11: normalization and pairing integrity (F1).
- **Out of lens:** LR-13 guard metrics (not re-verified beyond the scorer's code path), LR-02, LR-05 to LR-07,
  LR-10, LR-12, LR-14 and LR-15.
- **Rejected:** LR-04 (A2.5).

## Process notes

- **Runs:** 18 replays in total, each holding one `CR/slots` slot:
  - 12 raw re-runs;
  - 1 E0 probe of 200 pairs, then 4 E0 batches covering all 10,058 pairs;
  - 1 edge smoke.
- **Writes:** nothing in WT or the main checkout. The campaign E0 table and result cache were only read; harness E0
  objects were pointed at a scratch dir and never persisted.
- **No fixes applied.** None of the defects is a safe one-liner: F2 needs a METHOD and HARNESS_VERSION bump, and F1
  touches the report's grouping.
