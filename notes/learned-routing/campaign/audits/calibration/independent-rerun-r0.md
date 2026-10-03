# Audit: calibration / independent-rerun, round 0 (dynamic)

- Auditor: independent DYNAMIC auditor, lens "independent-rerun", round 0. Date: 2026-10-02.
- Audited state: WT `<commit-16>`, HARNESS_VERSION lrh-4, bindings build `6955b0ee…`, E0 method
  `ais-chunked-estimator-v2`, cells lr-cells-v4: `cells/train.jsonl` `6572232d…`, `cells/val.jsonl` `60635df6…`,
  `cells/test.jsonl` `b8648ac5…`.
- Evidence directory (EV): `runs/audit-calibration/independent-rerun-r0/`. It holds scripts, raw outputs and a private
  lr-eval root `EV/root`. That root has its own empty cache, replicates, policies and E0 table; `traces/` and `config/`
  are symlinked to CR.
- Replays I ran: 171, all through the shared 20-slot pool, 0 errors.
  - 128 lr-eval replays: runA 64, runB 32, longwarm 32.
  - 16 direct replays that bypass the harness replay path.
  - 27 isolated-E0 single-worker replays covering 141,012 (ISL, OSL) pairs.
- No test cell or test trace was replayed. The only reads of test artifacts were SHA-256 checks.

**Verdict: FAIL.** 0 blocker, 1 major, 5 minor, 2 info.

The harness, the scoring, the loads-as-declared, the freeze and the split hygiene all hold up under independent
re-execution. The failing point is the measurement window of the sessions **open-loop** cells. Their 180 s warm-up ends
before the session population has reached steady state. As a result, the levels, the sessions SLA anchor and the policy
gaps measured on those cells depend on the horizon (F1).

## What I checked and what held

### 1. Independent re-runs in fresh processes, bypassing the cache

**Sample.** I drew a seeded sample with `random.Random("lr-audit-calibration-independent-rerun-r0")`: one cell per
stratum (Mooncake, sessions and AgentX × open, closed and lanes), plus 3 more cells. Script: `EV/scripts/sample.py`.
The 8 cells (5 train, 3 val) are:

- `agentx-A2-think0.5-n4-lanes-L2`
- `agentx-A3-osl1.5-n4-lanes-L2`
- `mooncake-w4-base-n4-closed-L2`
- `mooncake-w4-islp1.5-n8-open-L2`
- `mooncake-w0-osl0.5-n8-open-L2`
- `mooncake-w1-base-n8-closed-L2` (val)
- `sessions-s5-base-n4-closed-L1` (val)
- `sessions-s5-base-n8-open-L3` (val)

The sample covers four transforms, and L1, L2 and L3.

**Run A (`EV/runA`).** I ran lr-eval in the private root, with an empty cache, freshly materialized replicates and a
fresh E0 table. Policies were default@defaults and round_robin, at k ∈ {0, 1} and at the new indices k ∈ {8, 9}: 64
replays, 0 errors.

- k = 0, 1 against calibration's cache entries under the same 32 cache keys: every metric field is identical (only the
  slot label differs), and the per_request bytes are identical 32/32 (`EV/scripts/compare_cache.py`).
- The fresh E0 table equals CR's on all 26,210 shared pairs.
- The re-materialized replicates have identical SHA-256s.

**Run B (`EV/runB`).** I re-ran k = 8, 9 with `--refresh`, so each new replicate was run twice. Run B equals run A
32/32, in every field and in per_request bytes.

**Direct replays (`EV/direct`, `EV/scripts/direct_replay.py`).** These replays do not use learned_routing's replay code.
I derived the load kwargs from the contract myself, wrote my own seeded default YAML, and used one fresh process per
replay. For k = 0 on all 8 cells × 2 policies, the per-request tuples equal the harness rows 16/16. The tuples have 10
fields: session, turn, arrival, ttft, e2e, ISL, OSL, worker, reused tokens and status.

So `load.value` is passed as the stated speedup, concurrency or lanes, N is passed correctly, and the policy is what
the harness says it is.

### 2. Goodput recomputed independently from raw per_request

**E0.** I did not reuse the harness estimator. I replayed every (ISL, OSL) pair alone on an idle single worker:
N = 1, round robin, requests 10⁷ ms apart, fresh hash ids, so there is no reuse. Its e2e is E0 by the A2 definition.
This covered 141,012 pairs (`EV/scripts/e0_isolated.py`, `EV/e0/out*.json`).

The isolated E0 agrees with CR's v2 table to between −2.4e-7 and +3.0e-7 relative. That is inside
`E2E_REL_TOL = 1e-6`; see F4.

**Scorer.** My scorer (`EV/scripts/myscore.py`) does not import learned_routing. It implements A2 good, the
arrival-basis window, identity warm-up and the full-occupancy completion window from the contract text.

Results:

- **Re-run records:** it equals the harness on 64/64 records, for window_good, window_requests, good_frac_window,
  goodput_rps_window and window start/end, at all 6 SLO scales (`EV/scripts/check_sample.py`).
- **All 1,008 step-4 reference records:** the default@defaults and round_robin normalization reference for 48 train/val
  cells plus 15 noise cells, k = 0..7. It is exact on 1,008/1,008 records at all 6 scales and on window bounds
  (`EV/scripts/check_step4.py`, `EV/step4_myscore.json`).
- **Pooled sd:** the pooled paired-ratio sd per family × N × mode in `facts/calibration.json` reproduces to 1.7e-16.

### 3. Does the SLA bind?

Over 8 replicates, default@defaults' good_frac_window ranges from 0.548 to 0.984 across the 48 train/val cells, and
round_robin's from 0.181 to 0.925. No cell is near 0. Only L1 cells come near 1, which is by design (I1).

Failure decomposition at k = 0, 1 (`EV/scripts/fail_decomp.py`) for default:

| Family | ITL-only | E2E-only | Both |
|---|---|---|---|
| Mooncake | 1.8–2.2% | 12.5–13.8% | 2–3% |
| Sessions | 0.05% | 7.7–11.6% | 0.5% |
| AgentX | 0.3% | 10.1% | 3.7% |

Both thresholds bind, and E2E slowdown dominates, as the rule intends. The AgentX e2e = E0 atom is 15.4% of default's
rows and 19.8% of round_robin's.

### 4. Default vs round_robin beyond noise (`EV/noise_binding.json`)

- **Per cell:** the RR/default paired goodput ratio is 0.267–0.959, and |mean − 1| > 3 × the replicate sd on 48/48
  train/val cells and on all 15 noise cells.
- **Out-of-sample replicates:** the new k = 8, 9 ratios lie within 3 sd of the k = 0..7 mean on 8/8 sampled cells.
- **Per segment:**
  - train: 11 segments, mean Δ −0.403, SE 0.045 (8.9 SE);
  - val: 6 segments, mean Δ −0.432, SE 0.049 (8.8 SE);
  - per load mode, at least 3.7 SE in every mode on both splits.

### 5. Loads as declared and N-scaling (`EV/per_worker.json`, `EV/scripts/levels_check.py`)

**Declared values.** All 108 train, val and test cells match the decision tables:

- per_worker comes from `levels` or `transform_levels`;
- value = per_worker × N, rounded for closed loop and lanes, or divided by the base-window token rate for Mooncake open
  loop;
- the SLA is the family's, with `ttft_ms` null.

**Realized offered load.**

| Cell type | Declared (per worker) | Realized |
|---|---|---|
| Base Mooncake open loop, N2–N32 | 5688 / 8798 / 8508 tok/s | 5700 / 8816 / 8526 tok/s |
| Sessions open loop | sessions/s | equal within Poisson noise |
| Closed loop and lanes | `value` | occupancy equal to `value` exactly |

**Arrival timing.** Open-loop arrivals equal the replicate trace stamps divided by the speedup, with a maximum error of
0 ms on 3 cells.

**N-scaling is knee-matched, not per-worker-matched.** Utilization therefore varies across N (F2):

- **Mooncake open L2:** per-worker in-flight requests are 6.1–7.6 across N = 2..32, and prefill throughput is
  5.0–5.8 k tok/s/worker.
- **Mooncake closed L2:** per-worker concurrency falls from 10.56 to 5.60 between N2 and N32. Prefill throughput falls
  from 6,838 to 4,676 tok/s/worker.

### 6. Open-loop and closed-loop semantics (`EV/semantics.json`)

**Open loop:**

- First-turn arrival times are identical between default and round_robin on every open cell at k = 0, so arrivals are
  exogenous.
- Sessions release turns causally: 0 of 235,610 later turns, summed over both policies, arrive before their
  predecessor finished.
- Session think time is invariant to load. The replay gap divided by the source-trace delay is exactly `think_mult`
  (p01–p99 within 1.3e-13) at speedups 0.86 to 10.8.

**Closed loop and lanes (31 cells × 2 policies):**

- `cap` units start at t = 0;
- peak occupancy equals `cap`;
- every later unit starts at the exact instant another ends (handoff gap 0);
- `window_below_cap_frac` is 0.

### 7. Test freeze (`EV/scripts/verify_freeze.py`)

89 file hashes match:

- `cells/test.jsonl`;
- 39 test traces and their 39 meta files;
- 7 sources;
- the engine;
- SPLIT_MANIFEST;
- the traces MANIFEST.

The 60 cell content SHAs also match, and every test cell's trace is listed. The step-4 inputs
`runs/calibrate/final_r4/{train,val,test}.jsonl` are byte-identical to `cells/`.

### 8. No test influence on calibration (`EV/leakage.json`)

I scanned 17 calibration `results.jsonl` files (14,780 records) and 44 cell files. There are 0 hits on any of these:

- test cell_ids;
- test segments;
- test trace SHAs;
- `split == test`.

Other checks:

- Every sweep, including the per-transform `sweep2/tx` (4,496 records), used only train segments:
  - Mooncake w0, w2, w4;
  - sessions s0–s3;
  - AgentX A1–A3.
- All Mooncake windows of the derived traces lie inside train or val slices.
- The 236 lowered AgentX gen files reference only train/val plays (54 plays). 0 test plays appear.
- 0 replicate files anywhere in `runs/replicates` derive from a test trace.
- Val segments (w1, s4, s5, V1–V3) appear only in the step-4 reference runs, not in any sweep feeding a level or SLA
  decision.

## Findings

### F1 (MAJOR): sessions open-loop windows open during the session-population ramp-up, so levels, SLA anchor and policy gaps depend on the horizon

**Rule.** Sessions open-loop cells use a 180 s warm-up and a 540 s window, with sessions starting in [0, 780 s).

**Why that is too short.** In replay time, session lifetimes have p90 170–222 s and p99 330–430 s, and they grow with
`think_mult` (`EV/scripts/sessions_rampup.py`). At the window start (t = 180 s) on the train/val cells:

- the active-session population is 0.64–0.93 of its 540–720 s level (0.72 for `sessions-s0-base-n4-open-L2`, 0.64 for
  `sessions-s5-think1.5-n6-open-L2`);
- the request arrival rate in the window's first minute is 0.84–0.95 of its late-window rate at N = 4..8.

Per-minute profiles (`EV/scripts/sessions_ramp.py`) show default good fractions of about 0.97–0.98 in the first two
window minutes, then 0.38–0.60 once the population saturates. Example: `sessions-s4-base-n4-open-L2`, val, k0.

**Horizon dependence, existing data only** (`EV/sessions_window_sens.json`). Over the same replays, scoring the late
window [360, 720] s instead of the official [180, 720] s gives:

- default good fraction 0.004–0.128 lower on 12/12 sessions-open cells;
- RR/default goodput ratio 7–43% lower on 12/12 cells (`s4-n4-L2`: 0.441 to 0.253; `s3-islu1.5-n8-L3`: 0.341 to
  0.220).

**Horizon dependence, new replays** (`EV/longwarm/`). I built four cells with the campaign's own transform: same
per-worker session rate, same 540 s window, 900 s warm-up. The official derived trace was reproduced byte-exactly first
as a control. Results over 4 replicates, 0 errors:

| Cell | RR/default, official | RR/default, warm900 | Default gf, official | Default gf, warm900 |
|---|---|---|---|---|
| `s0-n4-L2` | 0.456 | 0.283 | | |
| `s1-n4-L3` | 0.448 | 0.233 | 0.814 | 0.435 |
| `s4-n4-L2` | 0.441 | 0.157 | | |
| `s5-think1.5-n6-L2` | 0.469 | 0.299 | | |

**Why this is material:**

- **Level labels.** The sessions-open L1/L2/L3 levels were solved on a mix of light transient and saturated regime. At
  steady state, the "L3" train cell `s1-n4-L3` sits at default good fraction ≈ 0.44, not 0.65.
- **SLA anchor.** The sessions SLA anchor uses the ss-open N = 8 knee: drift between the last and first third of the
  window crossing 1.15. The light first third inflates that drift, so S = 1.90627 is contaminated too, and S applies to
  the closed-loop sessions cells as well.
- **Effect sizes.** Measured policy gaps on these cells are diluted by a third of each window in which every policy is
  good.
- **Transform axis.** The transient fraction grows with `think_mult`, so the think-time axis confounds think time with
  window transient. The test cells `sessions-s6-think4.0-n4-open-L2` and `s7-think0.25` sit at opposite ends.

The calibration rules list the sessions warm-up as a rejection of LR-01's 10-minute warm-up. The deferred LR-01
action 5 (Δ stable when the warm-up is doubled) fails here.

**Scope.**

- Affected cells:
  - 6 train cells (`sessions-s{0,1,1,2,3,3}-…-open-*`);
  - 3 val cells;
  - 3 noise cells;
  - 9 frozen test cells (`sessions-s6/s7/s8/s9-*-open-*`).
- Not affected: Mooncake open (single-turn rows), closed loop (population fixed at C from t = 0) and lanes.

**Hypothesis, not verified.** The test-only FAST25 conversation cells, which are multi-turn and use a 4-minute trace
warm-up, may share the issue. I did not replay test data.

**Fix.**

1. Set the sessions-open warm-up from the session-lifetime distribution at the cell's `think_mult`, for example
   ≥ p99 session duration × `think_mult`, or until active sessions are within 5% of their plateau. Regenerate the
   derived traces from the 48,000 s seeds with the window [0, (W + 540 + 60) s × speedup).
2. Re-run the sessions-open sweep and knee, then the sessions SLA fixed point and both sessions level tables.
3. Re-freeze `cells/test.jsonl`, which is allowed because no test cell has been evaluated, and record the change in
   DEVIATIONS.
4. Afterwards, re-check the reference cache for the changed train/val cells.

### F2 (minor): N-scaling is knee-matched only; per-worker load is not matched across N, and LR-12's "report both" arm is missing from the frozen test set

Per-worker levels relative to N = 8:

| Family, mode, level | N2 | N32 | max/min across N |
|---|---|---|---|
| Mooncake closed L1 | 1.45 | 0.74 | 1.95 |
| Mooncake closed L2 | 1.38 | 0.73 | 1.88 |
| Sessions closed L1 | | | 1.41 |
| Mooncake open L3 | | | 1.29 |

The realized per-worker prefill throughput at Mooncake closed L2 falls 32% from N2 to N32.

DEVIATIONS records knee-matching. But LR-12 action 1 asks to report N-extrapolation at both per-worker-matched and
knee-matched loads, all 26 worker-count test cells are knee-matched only, and `facts/calibration.json` lists LR-12 as
"applied".

This does not bias the headline, but REPORT must not describe the N axis as matched per-worker load.

**Fix.** Either pre-register a secondary per-worker-matched N-extrapolation set before the test stage, or state the
limitation in the report and mark LR-12 action 1 as partial.

### F3 (minor): realized band levels deviate from the stated 0.95/0.85/0.65 because of grid coarseness and segment heterogeneity

**Coarse grid.** The sessions closed sweep grid steps 8 sessions/worker across the shoulder: at N8, 24 gives 0.963 and
32 gives 0.658. Log-linear interpolation places L2 = 26.7, where default realizes about 0.91, not 0.85, on 5/5 cells:

| Cell | Default gf |
|---|---|
| s0 N8 | 0.918 |
| s4 N8 (val) | 0.907 |
| noise s0 N2 | 0.909 |
| noise s0 N16 | 0.915 |
| noise s0 N32 | 0.907 |

**Segment heterogeneity.** Sessions-open segments differ widely at the same load. At N2, load 0.40, s1 gives 0.992 and
s2 gives 0.729. Hence L2 cells realize anywhere from 0.658 (`s4-n4`, val) to 0.966 (noise `s0-n2`).

The levels are correct interpolations of the train-segment mean curve, which I reproduced exactly, so this is accuracy,
not bias. But the summary's "L1/L2/L3 are where the default's good fraction is 0.95/0.85/0.65" is not true per cell.

**Fix.** Use a finer grid around 0.85 for sessions closed, and report the per-cell realized band next to the level
label.

### F4 (minor): isolated-replay E0 differs from the harness E0 table by up to 3.0e-7 relative, which is 3.4× below the tolerance

Across 141,011 pairs, the extremes are −2.40e-7 and +2.96e-7. 6,635 pairs exceed 2e-8, mostly with OSL < 100: 18.5% of
those pairs exceed 2e-8. The earlier audits measured within 2e-8 on smaller samples.

No goodput changed, as the exact 1,072/1,072 matches above show, because `E2E_REL_TOL = 1e-6`.

**Fix.** Record the measured bound in `facts/calibration.json`, and keep the tolerance at 1e-6 or above.

### F5 (minor): replicate reuse trusts the sidecar's SHA-256

`materialize_replicate` (`replicates.py:421-438`) reuses a file when the `.rep.json` identity and byte size match. It
does not re-hash the file, so a same-size corruption would replay silently under the recorded `trace_sha256`. Of the
368 step-4 replicate files, 40 were sampled and 40/40 hash-verified.

**Fix.** Re-hash on first reuse per process, or compare the file's mtime against the sidecar's.

### F6 (minor): one OSL = 0 request per AgentX lanes run is never scored

Step-4 AgentX rows at k = 0..1 hold 64 rows per policy (128 in total) with output_length 0. They are all ISL 11008,
reported `terminal_status: completed`, and have ttft and e2e set to None. The harness's `completed()` therefore rejects
them, and the completion window drops them for every policy. This is the
already-known F2 of the build audits. It is equal across policies and has no effect on the reference, but it is still
open.

### I1 (info): L1 cells sit near the ceiling by design

- Default good fractions: `sessions-s1-base-n8-open-L1` 0.984, `sessions-s5-base-n4-closed-L1` 0.970,
  `mooncake-w4-base-n8-open-L1` 0.965.
- Open-loop headroom on these cells is at most 1/gf (about +1.6% to +3.6%). They work as no-regression cells, not as
  gain cells.

### I2 (info): round_robin is non-stationary on several lanes and Mooncake cells; default is not

- Round_robin's mean-slowdown drift (last third over first third) exceeds 1.15 on 6/8 AgentX train cells (1.20–2.72)
  and on 13/23 Mooncake train/val cells (up to 1.82).
- On train/val cells, default stays within 0.93–1.11 on AgentX and within 0.74–1.21 on Mooncake, mostly following the
  trace's own rate (`EV/stationarity.json`).
- On AgentX, default's first-third versus last-third good fraction moves by at most 0.07, so the full-occupancy AgentX
  windows are adequately stationary for the reference policy.

## Lessons

- **Applied:**
  - LR-01, action 5: the warm-up sensitivity check, run now rather than at the pilot. It produced F1.
  - LR-02: noise from CRN replicates, plus out-of-sample replicates k = 8, 9.
  - LR-09: causal release and think-time invariance verified, and stratified by load mode.
  - LR-11: the segment-level SE for the beyond-noise check.
  - LR-12: per-N utilization; F2.
- **Rejected:**
  - LR-04 (A2.5);
  - LR-08's TTFT part (A2: no TTFT SLO).

## Scratch

`runs/audit-calibration/independent-rerun-r0` holds 563M in total: the private root with replicates and cache, the
longwarm derived traces, the E0 outputs and `runA_cache_snapshot_results`. It is listed in CLEANUP.md.
