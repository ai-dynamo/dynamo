# Fix: calibration, round 0

- Fixer: calibration fixer, round 0. Date: 2026-10-02/03.
- Finding addressed: `audits/calibration/independent-rerun-r0.md` F1 (major). It is the only finding assigned to this
  round. The auditor's minors F2-F6 are not addressed, except that the finer sessions grids also cover part of F3.
- **Verdict on F1: VALID, FIXED at the root.** Nothing was rebutted, except the auditor's own unverified FAST25
  hypothesis (section 6).
- Worktree: no code change. The fix lives in the calibration scripts, as calibration r0's did. They are in
  `runs/calibrate-fix-r0/scripts/`, which holds copies of `runs/calibrate/scripts/` plus the new scripts. The original
  hashes are in `runs/calibrate-fix-r0/out/scripts_copied_from_calibrate.sha256`. This fixer made no WT change and
  has nothing to commit. WT head moved to `<commit-17>` during this round through a concurrent docs-only commit by
  another stage (`notes/learned-routing/campaign/`); `benchmarks/`, `lib/` and `components/` are identical to
  `<commit-16>`. That mirror still holds the lr-cells-v4 cells and facts and needs a refresh by its owner.
- New frozen state: lr-cells-v5.
  - `cells/test.jsonl` `7b998b8e…` (was `b8648ac5…`)
  - `cells/train.jsonl` `55b54ec6…`
  - `cells/val.jsonl` `e61ac577…`
  - `facts/test_freeze.json`
  - r0's files are kept at `cells/superseded/calibration-r0/` and `facts/superseded/calibration-r0/`.
- Integrity: every number below comes from a file under `runs/calibrate-fix-r0/` or `facts/`.

## 1. The finding reproduces

I checked it with my own measurements, independently of the auditor's scripts.

**Lifetimes are long relative to 180 s.** Session lifetimes measured in r0's step-4 replays (`out/lifetimes.json`,
sessions that start in the first 300 s, think 0.5 to 1.5) have p99 257-462 s for default and 336-515 s for round
robin.

**Near steady state they are longer still.** This was measured in the auditor's 900 s warm-up replays
(`out/late_lifetimes.json`):

| Policy | Lifetime p99 | Contended service stretch Σe2e/ΣE0 (mean) |
|---|---|---|
| default | 339-464 s | 1.68-2.00 |
| round robin | 423-547 s | 2.27-2.76 |

**Policy-free envelope.** I defined each session's lifetime envelope as m·Σthink + 3·ΣE0, over 8,033 train-segment
sessions with each transform applied (`out/warmup_table.json`). At 180 s, the M/G/∞ population fill
E[min(D, 180)]/E[D] is:

| think_mult | Fill at 180 s |
|---|---|
| 0.25 | 0.86 |
| 1 | 0.78 |
| 1.5 | 0.73 |
| 4.0 | 0.54 |

The think 4.0 test cell is the worst case, as the auditor said.

**Same sessions, different history.** For every open-loop session train, val and noise cell I replayed the same
in-window sessions with a history of 180 s and with a long history. The r0 180 s history over-states default's good
fraction by +0.061 on average (max +0.353, 9/12 cells |z| > 3), and RR/default by +0.217 (12/12 cells |z| > 3)
(`warm_check_r2/report.json`, arm `w180`). At r0's levels the bias was larger: +0.179 and +0.311 (`warm_check/report.json`).

**The knee was inflated.** With the population settled, r0's own within-window drift test moves the N = 8 sessions
knee from 0.4525 to 0.506-0.510 sessions/s/worker (`out/ssopen_old_vs_new.json` for the 600 s warm-up; `facts/calibration.json`
curves `knee_within_window_drift` for the final 1200 s warm-up). The light first third had been inflating the drift.

## 2. The fix

Sessions open loop only. Mooncake, FAST25 and AgentX are untouched.

### 2.1 Warm-up

The warm-up is two envelope lifetimes:

```
W = 2 × max(180 s, ceil60(p99 over train-segment sessions of m·Σthink_delay + 3·ΣE0(ISL, OSL)))
```

- m is the cell's `think_mult`.
- The rule is computed on train segments s0-s3, with sessions starting in [0, 2000) s of the long seeds, and the
  cell's transform applied.
- κ = 3 bounds the measured steady-state service stretch: mean ≤ 2.0 for default and ≤ 2.76 for round robin.

Resulting warm-ups:

| Transform | W |
|---|---|
| think 0.25 or 0.5 | 960 s |
| think 1 (including islu1.5 and root2) | 1200 s |
| think 1.5 | 1320 s |
| think 4.0 | 2160 s |

The derived trace keeps sessions that start in [0, (W + 540 + 60) s × speedup) of the seed. The window stays 540 s. The
code is in `scripts/calib.py`: `session_warmup_s` and `session_open_cell`.

**Why two lifetimes and not one.**

- **Round 1 failed the check.** It used one lifetime (W = 600 s at think 1) and failed the doubled-warm-up check
  (`warm_check/report.json`):

  | Metric | Mean Δ | SE | Max | Cells \|z\| > 3 |
  |---|---|---|---|---|
  | default good fraction | +0.040 | 0.014 | 0.152 | 6/12 |
  | RR/default | +0.078 | | | 9/12 |

- **Two lifetimes converge below the knee.** A convergence run (`conv_check/`) replayed default on the same sessions
  with 600, 1200, 1800 and 2400 s of history:

  | Load at N = 8 | 600 s | 1200 s | 1800 s | 2400 s |
  |---|---|---|---|---|
  | 0.45 | 0.727 | 0.659 | 0.662 | 0.641 |
  | 0.425 | 0.854 | 0.815 | 0.826 | 0.824 |

  At N = 4 and N = 32 the 1200 s history is within about 1-4% of 2400 s up to the same loads.

- **Near the true knee nothing converges.** At 0.475 (N = 8) the four histories give 0.517, 0.397, 0.366 and 0.339.

### 2.2 Knee

Once the population is settled, r0's within-window drift test (last third over first third, 1.15) cannot see
relaxation slower than one window. The sessions open-loop knee is now the load at which doubling the warm-up history
(W → 2W) lowers default's good fraction by more than 5% relative.

- **Measurement:** the same in-window sessions; train segments s0-s3 × 2 replicates; N ∈ {2, 4, 6, 8, 16, 32};
  672 replays (`hist_sweep2/`). Code: `scripts/hist_sweep.py` (`gf_knee`).
- **Fixed point:** the knee depends on (I, S), so it is solved jointly with r0's fixed point:
  knee → L3 anchor = 0.9 × knee → (I, S) → knee. It converged to 1e-4 in 7 outer iterations
  (`decisions.json` `outer_fixed_point_history`).
- **Shift at N = 8** (`facts/calibration.json` curves):

  | Load | 0.35 | 0.375 | 0.40 | 0.425 | 0.45 | 0.475 | 0.50 |
  |---|---|---|---|---|---|---|---|
  | Relative shift | 0.3% | −0.3% | 0.7% | −3.3% | 2.5% | 15.4% | 16.6% |

  The knee is 0.4548. The within-window drift knee on the same data is 0.5098.
- **Other N:** the knee is 0.476 at N = 4, 0.475 at N = 6, 0.449 at N = 16 and 0.449 at N = 32. N = 2 never crosses
  5% up to 0.5. Every sessions level lies below its N's knee (no `beyond_knee` flags).
- **The other families keep r0's knees.** A 1e-4-relative limit cycle of the outer loop is noted in `decide.py`.

### 2.3 Re-calibration

- **Re-swept** ss-open (1,440 replays, `sweep4/ss-open`) and the six session open-loop transforms (`sweep4/tx`) with
  the new warm-up.
- **Added finer grid points** where the corrected levels fall. This covers part of audit F3.

  | Group | Points added |
  |---|---|
  | ss-open | 0.375, 0.425, 0.475, 0.525 |
  | ss-closed | 28, 36, 44, 52 |
  | session open-loop transforms | 0.425, 0.45, 0.475 |
  | session closed-loop transforms | 36, 44, 52, 56 |

  Existing closed-loop points are byte-identical cells and therefore result-cache hits (168/168 and 104/104 checked).
- **Re-solved** the sessions SLA, both sessions level tables (open and closed, every N) and the session transform
  levels (`decisions.json`, `curves.json`). Everything else in the decisions is identical to r0: SLAs, levels, and the
  19 non-session transform levels.

| Sessions quantity | r0 (180 s) | Fixer r0 |
|---|---|---|
| I (ms) | 40.4165 | 46.0478 |
| S | 1.90627 | 2.11784 |
| N = 8 knee / L3 anchor | 0.4525 / 0.4073 | 0.4548 / 0.4093 |
| Open L1 / L2 / L3 at N = 8 (sessions/s/worker) | 0.3238 / 0.3645 / 0.4073 | 0.3497 / 0.3800 / 0.4093 |
| Closed L1 / L2 / L3 at N = 8 (sessions/worker) | 24.30 / 26.70 / 32.14 | 28.99 / 33.12 / 37.46 |
| think4.0, N = 4, open (test transform) | 0.322 / 0.370 / 0.429 | 0.300 / 0.315 / 0.347 |

ITL alone fails 0.6% of default's requests at the N = 8 L2 grid point.

### 2.4 Rebuild, reference and re-freeze

**Rebuild.** `final_r6/` was built with `build_final.py`.

- Only the 34 session cells changed, in every split: train 8, val 5, test 15 and noise 6.
- Every other cell is byte-identical to lr-cells-v4: train 26/34, val 9/14, test 45/60, noise 9/15. The 45 unchanged
  test cells also keep their content SHAs.
- Session cells carry `"calibration": "calibration-fix-r0"`. The rule is recorded in `measure_trace.warmup_rule`.

**Step 4** (`step4_r6/`, `step4_report_fix_r0.json`).

- Run: 304 new replays plus 704 cache hits, 0 errors. The offline rescoring matches the harness on 1,008/1,008
  records.
- RR differs from default beyond noise:

  | Split | Mean segment Δ | SE |
  |---|---|---|
  | train | −0.462 | 0.062 |
  | val | −0.474 | 0.065 |

- Default's lowest good fraction on a train or val cell is 0.423 (`sessions-s1-base-n4-open-L3`, a heavy segment).
  That is above LR-01's 0.3 floor.

**Re-freeze** (`freeze.py`). This was allowed because no test cell had been evaluated: there are 0 test cell ids, test
splits or test trace SHAs in 35,735 result records under `runs/`. The 39 test traces, 7 sources, engine, traces
MANIFEST and SPLIT_MANIFEST hashes and the 60 content SHAs were re-verified after writing.

**Facts updated.**

- `facts/calibration.json`: `sla`, `levels_per_worker`, `transform_levels_per_worker`, `curves`, `rules`,
  `step4_reference`, `noise`, `cache_pressure_open_loop`, `cost_projection`, `lessons` and `summary`, plus the new
  `session_open_warmup` and `fix_r0`.
- `facts/noise_calibration.json`.
- `facts/test_freeze.json`.
- `cells/SPLIT_MANIFEST.json` (`calibration.cells_version` lr-cells-v5).
- `facts/DEVIATIONS.md`.

## 3. Verification: the doubled-warm-up check passes

The check covers every open-loop session train, val and noise cell: 12 cells, default and RR, k = 0..3, 288 replays
(`warm_check_r2/report.json`). Each cell is replayed with the same in-window sessions under three histories:

- W, the rule's warm-up;
- 2W;
- 180 s, r0's rule.

| Comparison | Default good fraction Δ (mean, SE, max \|Δ\|, cells \|z\| > 3) | RR/default Δ (mean, SE, max \|Δ\|, cells \|z\| > 3) |
|---|---|---|
| W vs 2W (fixer rule) | −0.0006, 0.0019, 0.014, 0/12 | +0.0074, 0.0038, 0.038, 0/12 |
| 180 s vs 2W (r0 rule) | +0.061, 0.027, 0.353, 9/12 | +0.217, 0.016, 0.277, 12/12 |

So the levels, the sessions S and the policy gaps no longer depend on the warm-up horizon. The check has power: the
same check rejects r0's 180 s rule. The per-cell values are in the report.

This is LR-01 action 5 (Δ stable when the warm-up is doubled), run at calibration instead of the pilot.

## 4. What the auditor's fix text asked for, item by item

| Requested | Done |
|---|---|
| Warm-up from session lifetimes at the cell's think_mult (e.g. ≥ p99 lifetime × think_mult, or until active sessions are within 5% of plateau) | Yes. W = 2 × the p99 contended-lifetime envelope at the cell's think_mult (think time scales by m, service by κ = 3). The population criterion is checked by the doubled-history test, which is stricter than an active-count criterion. |
| Regenerate derived traces with window [0, (W + 540 + 60) s × speedup) | Yes, for every session open cell in every split and for all sweeps (`traces/derived`, content-addressed) |
| Redo the session open-loop sweep and knee | Yes. The knee definition was changed for sessions (section 2.2) because r0's test, freed of the ramp, no longer detects relaxation. |
| Redo the sessions SLA fixed point and both sessions level tables | Yes, plus the session transform levels |
| Refresh the reference cache | Yes (step 4, 304 replays) |
| Re-freeze `cells/test.jsonl` and record it in DEVIATIONS | Yes |

## 5. Residual limits (not papered over)

- **Near-critical loads stay horizon-dependent.** Above the sessions knee (for example 0.475 and 0.5 at N = 8, where
  the knee is 0.455), default's good fraction still falls by 15-17% between 1200 s and 2400 s of history. No warm-up removes that, because the system is
  near its stability limit. The levels sit at or below 0.92 × knee. Only N = 2 has no knee within the grid; its L3 is
  0.487, below the largest tested load of 0.5.
- **A policy that is unstable at a cell's load has no steady state.** Its windowed goodput keeps falling with the
  horizon. At the final levels round robin passes the check (RR/default Δ ≤ 0.038), but a much worse candidate during
  training might not. Such a candidate is penalized more with a longer horizon, which leaves rankings among stable
  policies unaffected.
- **Segment heterogeneity remains**, as in audit F3. At L3, s1 at N = 4 realizes 0.423 against the 0.65 target. Noise
  N = 2 open L2 realizes 0.746 against 0.85.
- **Calibration cells sit on val segments.** The doubled-warm-up check used val segments s4 and s5 as calibration-only
  cells. Like r0's step-4 reference, they verify; they do not feed any level or SLA decision.

## 6. Rebuttal: the FAST25 conversation hypothesis does not apply

The auditor marked it unverified. Every row of the derived traces of all 3 FAST25 conversation test cells carries a
`timestamp`, and none carries `delay` or `session_id`. Arrivals are therefore exogenous, and there is no session
population to ramp up. Their warm-up is the Mooncake 4-minute trace warm-up (LR-01 rejection already recorded). I did
not replay test data.

## 7. Cost and process

- **Replays:** 9,792 fixer replays, 112,561 replay-s, 0 errors (`facts/calibration.json` `costs.fixer_r0`). All ran
  locally through the harness `Evaluator` and the `CR/slots` pool.
- **Stopped process:** one chain was stopped by exact PID: 820838, and its child 820840. A spec-form mismatch made its
  build re-derive traces serially. It was restarted with the fix before any replay started.

## Lessons

**Applied:**

- **LR-01:** windowed goodput, action 5 (doubled warm-up as the acceptance test and as the sessions knee), floor ≥ 0.3.
- **LR-02:** CRN replicates and pooled paired-ratio sd.
- **LR-09:** causal session release kept; checks stratified by load mode; closed loop untouched.
- **LR-11:** segment-level beyond-noise verdicts.
- **LR-12:** levels and knees per N.

**Rejected:**

- LR-04 (A2.5).
- LR-08's TTFT part (A2).
- LR-01's 10-minute warm-up stays rejected for Mooncake windows (unchanged). The sessions warm-up is now 16-36
  minutes, which exceeds it.
