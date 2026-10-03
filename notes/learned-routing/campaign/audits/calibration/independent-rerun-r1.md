# Audit: calibration / independent-rerun, round 1 (dynamic)

- Auditor: independent DYNAMIC auditor, lens "independent-rerun", round 1. Date: 2026-10-03.
- Audited state: the state after calibration fixer r0. WT `<commit-18>`; the `learned_routing` package is byte-identical to
  `<commit-16>`, and only `benchmarks/learned_routing/remote/` has been added since. HARNESS_VERSION lrh-4, bindings build
  `6955b0ee…`, E0 `ais-chunked-estimator-v2`. Cells are **lr-cells-v5**:
  - `cells/train.jsonl` `55b54ec6…`
  - `cells/val.jsonl` `e61ac577…`
  - `cells/test.jsonl` `7b998b8e…`
- Note on the task context. The calibration summary handed to this audit describes lr-cells-v4, which fixer r0 has
  since superseded: test `b8648ac5…`, sessions SLA 40.4165 ms / 1.90627, and the r0 sessions levels. I audited the
  current files (F5).
- Evidence directory: `EV = runs/audit-calibration/independent-rerun-r1/`. It holds:
  - `scripts/`, with every check;
  - `root/`, a private lr-eval root with an empty cache, fresh replicates and policies, and `traces`/`config` symlinked
    to CR;
  - JSON outputs named below.
- Replays:
  - 414 lr-eval replays through the shared `CR/slots` pool, 0 errors, 2,753 replay-s:
    - runA 88, runB 44 (sampled cells);
    - warm2x 80, warm2x_tx 96, mc2x 64 (doubled-warm-up checks);
    - txlevel 42 (frozen test-transform levels on train data).
  - 231 isolated single-worker E0 replays, each holding a slot, 1,003 replay-s.
- No test cell and no test trace was replayed or opened. The only reads of test artifacts were SHA-256 checks and
  test.jsonl metadata: per-worker loads, warm-ups and SLAs.

**Verdict: PASS.** 0 blocker, 0 major, 8 minor, 4 info.

Fixer r0's change holds up under independent re-execution. The new sessions warm-up rule gives a horizon-stable
default reference, including at the two test think-time transforms it had never been checked on. Fresh-process
re-runs reproduce the cache bit-exactly. My own scorer and E0 reproduce every scored record. The levels and the
sessions SLA re-derive from raw rows. The freeze verifies, and no test data touched calibration.

The minors are:

- round-robin's collapse and horizon dependence on session cells (F1);
- two frozen test-transform levels interpolated across a coarse grid (F2);
- the open r0 minors (F3, F4, F7, F8);
- a stale downstream summary (F5);
- an over-broad statement in fix-r0 (F6).

## What I checked and what held

### 1. Fresh-process re-runs bypassing the cache (`EV/compare.json`)

**Sample.** `scripts/sample.py` draws a stratified sample seeded with
`random.Random("lr-audit-calibration-independent-rerun-r1")`. It takes 3 sessions-open and 2 sessions-closed cells,
because fixer r0 re-calibrated sessions, plus 2 each of Mooncake open, Mooncake closed and AgentX lanes. That gives 11
cells, 7 train and 4 val, covering L1, L2 and L3, N 4, 6 and 8, and the transforms islp1.5, islu1.25 and osl1.5:

- `agentx-A3-osl1.5-n4-lanes-L2`
- `agentx-A1_A2-base-n8-lanes-L3`
- `mooncake-w2-base-n8-closed-L1`
- `mooncake-w1-islu1.25-n6-closed-L2` (val)
- `mooncake-w4-islp1.5-n8-open-L2`
- `mooncake-w2-base-n8-open-L3`
- `sessions-s0-base-n8-closed-L2`
- `sessions-s5-base-n4-closed-L1` (val)
- `sessions-s1-base-n8-open-L1`
- `sessions-s5-base-n8-open-L3` (val)
- `sessions-s4-base-n4-open-L2` (val)

**Run A.** A private root with an empty cache, freshly materialized replicates and a fresh E0 table. Policies were
default@defaults and round_robin, at k = 0, 1 and at the new indices k = 10, 11, for 88 replays.

- k = 0, 1 against calibration's cache under the same 44 keys: every metric field is identical, the only differences
  being the labels `_emit` adds to results files. The `per_request` bytes are identical 44/44.
- The re-materialized replicates have the same SHA-256 as CR's, 20/20.

**Run B.** k = 10, 11 again with `--refresh` equals run A 44/44, all fields plus `per_request_canonical_sha256`.

**Cache reachability (`EV/levels_check.json`).** Under the current lr-cells-v5 cell content, every train/val reference
entry is reachable: default@defaults and round_robin, k = 0..7, 768/768 hits. lr-train will find its normalization
reference.

### 2. Goodput recomputed from raw per_request

**E0 (`EV/e0_summary.json`).**

- **Method.** I wrote my own isolated replay, without learned_routing's E0: each (ISL, OSL) pair alone on N = 1
  round-robin, with fresh hash ids at trace block 64. Isolation is verified per row: zero reused tokens, and each
  request ends before the next arrives.
- **First variant (rejected).** It used a fixed 5e7 ms spacing, and its ~1e11 ms absolute times alone cost up to
  6.5e-7 relative in double precision.
- **Final variant.** Tight variable spacing; the harness table is used only to space the requests. All 880,139 pairs
  needed by my checks agree with the harness v2 table within [−8.5e-9, +1.11e-8], with 0 isolation failures. See I1.

**Scorer (`scripts/myscore.py`).** It is written from CONTRACT A2/A3 and does not import learned_routing:

- the ITL and E2E clauses with the 1e-6 E2E tolerance;
- the arrival-basis window;
- the identity warm-up recomputed from the replicate trace;
- the full-occupancy end from my own occupancy profile.

**Agreement with the harness.** Window start and end, window_good and window_requests are exact, good_frac_window agrees
to 1e-12, goodput_rps_window to 1e-9 relative, and every SLO scale {0.5, 0.75, 1, 1.5, 2, 3} agrees:

| Records | Agreement |
|---|---|
| My 132 re-run records (`score_runAB.json`) | 132/132 |
| All 1,008 fixer-r0 step-4 reference records (`score_step4.json`) | 1,008/1,008 |
| warm2x, warm2x_tx, mc2x and txlevel records | 282/282 counts |

**Power (`EV/power.json`).** Planted mutations each produce mismatches against the harness on the 88 run-A records:

| Mutation | Mismatching records |
|---|---|
| S × 1.001 | 71/88 |
| I × 0.999 | 3/88 |
| window start + 1 ms | 56/88 |
| ITL clause dropped | 83/88 |

### 3. The SLA binds (`EV/binding.json`, `EV/faildecomp.json`)

**Good fraction range.** Over the 48 train/val cells × 8 replicates:

| Policy | good_frac_window |
|---|---|
| default@defaults | 0.423–0.965 (lowest `sessions-s1-base-n4-open-L3`, highest `mooncake-w4-base-n8-open-L1`) |
| round_robin | 0.072–0.925 |

No default cell is near 0. Only L1 cells come near 1, by design.

**Failure decomposition** (k = 0, 1, in-window, default@defaults):

| Family | E2E-only | Both | ITL-only |
|---|---|---|---|
| Mooncake | 13.5% | 2.8% | 1.8% |
| AgentX | 12.1% | 3.8% | 0.4% |
| Sessions | 16.8% | 0.9% | 0.08% |

E2E slowdown dominates, as the A2 rule intends. On sessions the ITL clause almost never binds alone (I3).

**Sessions SLA re-derived (`EV/itl_rule.json`, `EV/levels_recompute.json`).**

- **I.** 1.2 × p95 per-request ITL of default at N = 8 L2 (0.38001), from the 16 raw sweep4 records bracketing it
  (0.375, 0.4) and interpolated in log load, gives **46.0475 ms**. The decided value is 46.0478.
- **S.** Default's good fraction at the N = 8 L3 anchor (0.9 × knee = 0.40934), from my scores of sweep4 at
  (46.0478, 2.11784), is **0.65000**.

### 4. RR vs default beyond noise (`EV/binding.json`)

**Per cell.** |mean RR/default paired goodput ratio − 1| > 3 sd on 63/63 cells (48 train/val plus 15 noise). The
ratios are 0.111–0.959.

**Out-of-sample replicates.** My new k = 10, 11 ratios lie within 3 sd of the k = 0..7 mean on 22/22 (cell, k).

**Segment level** (mean segment Δ = ratio − 1):

| Split | Mean Δ | SE | t |
|---|---|---|---|
| train (11 segments) | −0.462 | 0.062 | −7.4 |
| val (6 segments) | −0.474 | 0.065 | −7.2 |

Per load mode, |t| ≥ 3.5 in every mode on both splits. The loads sit where routing matters. On session cells they sit
well past round-robin's own capacity (F1).

### 5. Loads as declared; N-scaling (`EV/levels_check.json`, `EV/levels_recompute.json`, `EV/per_worker.json`)

**Declared loads.** All 108 cells match the decision tables:

- per-worker level × N, with the integer rounding for closed loop and lanes and the base-window TR for Mooncake-format
  open loop;
- the family SLA, with TTFT null;
- closed-session identity warm-up = 2C × 1000 ms.

**Session levels re-derived.** All 18 sessions open-loop levels (L1/L2/L3 × N ∈ {2, 4, 6, 8, 16, 32}) re-derive from
the raw sweep4 rows, my scores and my E0 to within < 1e-5 relative. The rule is the mean over (segment, k), first
crossing, linear in good fraction and logarithmic in load.

**N-scaling is knee-matched, not per-worker matched** (F3). This is documented, but it is still not what the lens asks
for. Per-worker utilization in the scoring window, default@defaults, k = 0, 1:

| Group | Load per worker across N = 2 → 32 | Spread |
|---|---|---|
| Mooncake closed L2 | concurrency 10.5 → 5.6; prefill 6,817 → 4,677 tok/s | 1.88× / 1.46× |
| Sessions closed L2 | concurrency 40.5 → 30.7 | 1.32× |
| Sessions open L2 | 0.446 → 0.362 sessions/s | 1.23× |
| Mooncake open L2 | offered 7.5k–9.0k input tok/s | 1.20× |
| AgentX lanes L2 | 3.3–3.56 lanes, but prefill 1.0k → 2.3k tok/s | 2.4× prefill |

AgentX prefill rises because default's prefix reuse falls from about 0.75 at N ≤ 8 to about 0.48 at N = 16/32.

### 6. Open-loop and closed-loop semantics (`EV/semantics.json`, `EV/profile.json`, `EV/halves.json`)

**Open loop (k = 0, 10):**

- First-turn arrivals are identical between the two policies on every open cell.
- Session turn k + 1 arrives exactly `delay_k / speedup` after turn k's terminal, at relative error ≤ 2.7e-10. There are
  0 early releases among 56,293 later turns per policy and replicate. Release is causal and think-time-invariant.
- Session cells keep arriving 347–473 s past the window end. On Mooncake cells the trace's last arrival comes within
  1.3 s after the window end.

**Byte-level reproduction.** My own slicers rebuild the official derived traces byte-exactly:

- 7/7 sessions open-loop traces: seed window [0, (W + 600) s × speedup), delays × think_mult × speedup;
- 4/4 Mooncake window traces: slice, rebase, hash ids relabeled by first appearance.

**Closed loop and lanes.** All cap units start at t₀, the peak equals the cap, and every later unit starts exactly when
another ends (handoff gap 0).

**Within-window stationarity.** First-half vs second-half good fraction over all 63 reference cells × 8 replicates:
default's mean Δ by family × mode is −0.011 to +0.021 with mixed signs, and the pattern is reproduced across
replicates sharing content. This is workload content, not relaxation (I4).

### 7. Warm-up horizon: fix-r0 F1 confirmed, plus checks it did not cover

Every comparison below holds the in-window sessions or rows fixed and changes only the history.

**Sessions open loop, history W vs 2W (`EV/warm2x/report.json`).** Five train/val cells, my own trace construction,
4 replicates:

- default's good fraction moves by −0.022 to +0.014, with |z| ≤ 1.6 except s2-think0.5, where z = −3.5 in the direction
  opposite to ramp contamination;
- all |Δ| ≤ 4.3% relative.

The F1 contamination (+0.061 mean, max +0.353 under the 180 s rule) is gone.

**Test transforms think 4.0 and 0.25, never checked by fix-r0 (`EV/warm2x_tx/report.json`).** I built these on train
seeds s0–s3 at the test cells' frozen per-worker loads and warm-ups (2160 s and 960 s), 3 replicates each:

| Cell | Default good fraction Δ (W − 2W) |
|---|---|
| think4.0 N4 L2 | ≤ 0.006 (\|z\| ≤ 0.6) |
| think0.25 N8 L2 | ≤ 0.010 (\|z\| ≤ 2.7) |

**Mooncake 4-minute trace warm-up vs 8 minutes (`EV/mc2x/report.json`).** This is LR-01 action 5, which DEVIATIONS had
deferred to the pilot. The preceding 4 minutes are prepended: w2 (train) takes w1's tail and w1 (val) takes w0's. No
w3/w4/w5 data is used. Results over 8 replicates on `w2-n8-L3`, `w1-n8-L3`, `w2-n4-L2` and `w1-n6-L2`:

- default's good fraction Δ is ≤ 0.009 (|z| ≤ 1.55);
- RR/default Δ is ≤ 0.051 (|z| ≤ 1.43).

The 4-minute warm-up is adequate.

### 8. Test freeze (`EV/freeze_check.json`)

89 file hashes match `facts/test_freeze.json`:

- `cells/test.jsonl` `7b998b8e…`;
- the 39 traces and 39 metas;
- 7 sources;
- the engine;
- SPLIT_MANIFEST;
- the traces MANIFEST.

Also matching:

- the 60 cell content SHAs, recomputed with the harness definition;
- each cell's declared `trace_sha256`, against its file.

Every test cell's trace is listed in the freeze. Non-session cells in train, val and test are byte-identical to
lr-cells-v4 (26, 9 and 45 respectively); only session cells changed.

### 9. No test influence on calibration (`EV/leakage_calib.json`, `EV/leakage.json`)

**Calibration scope.** There are 0 hits across:

- 30,060 records in 29 `runs/calibrate*/**/results.jsonl` files (r0 and fixer r0);
- 23,428 cache records written since 2026-10-02 18:00 (records written by audits are skipped).

A hit would be any of:

- a test cell id or `split == test`;
- a Mooncake window overlapping w3/w5;
- session seeds 6–9;
- FAST25 sources;
- a test AgentX play.

**Which segments fed which decisions.** Every trace was classified from its derived or lowered meta:

| Runs | Segments |
|---|---|
| Sweeps, transform sweeps, hist_sweep(2), conv_check (all decision inputs) | train only: s0–s3, w0/w2/w4, A1–A3 plays |
| step4 reference, warm_check(_r2) | val segments s4/s5, w1, V1–V3 appear here only (verification) |

No calibration script reads build-stage run directories. Test cell metadata is read only by `build_final.py` (to set
load values) and the cost estimate.

**Build-stage replays (F6).** These pre-calibration build-stage replays exist outside that scope:
`build-fix-r1/candidates_check` and `build-fix-r0` postbuild and bundle.

## Findings

### F1 (minor): round robin is in collapse on session open-loop cells, and its windowed goodput there depends on the horizon

**Where round robin collapses.** Round robin's good fraction is 0.072–0.409 on all 9 train/val sessions-open cells,
including the L1 cell `sessions-s1-base-n8-open-L1` (default 0.940, RR 0.342). The RR/default ratio is 0.111–0.425; 5
train/val cells are below 1/3.

**Horizon dependence.** Holding the in-window sessions fixed, doubling the history:

- halves RR/default on `sessions-s1-base-n4-open-L3`: 0.118 → 0.055 (z 3.4, `EV/warm2x`);
- moves it 17–30% at think 0.25 N8 on train seeds s1/s2: 0.149 → 0.128 and 0.114 → 0.088 (z 10 and 64,
  `EV/warm2x_tx`).

**What this contradicts.** fix-r0 §5 says round robin passes the check at the final levels (RR/default Δ ≤ 0.038).
That holds on fix-r0's session set, but not on mine. The default reference is unaffected (§7).

**Consequences.** A policy past its stability limit has no steady state, so ratios in this range are not stable
quantities. Rankings against stable policies are unaffected, because collapsed policies sit far below them under any
horizon. The pre-registered headline statistic, a log-ratio clipped at ±log 3 in `facts/HEADLINE_TEST.json`, absorbs
everything below 1/3. lr-train, however, defaults to `objective: ratio`.

**Fix.** Use `clipped_log_ratio` in lr-train, which LR-01 already recommends. In REPORT, do not interpret ratios below
about 1/3 quantitatively. State in the pilot notes that sessions loads put non-affinity routers into collapse even at
L1.

### F2 (minor): two frozen test-transform levels were interpolated across one coarse, concave grid interval

**Where the grid is too coarse** (from the `facts/calibration.json` transform curves):

| Transform | Levels in one interval | Default good fraction across the interval |
|---|---|---|
| `mooncake\|osl4.0\|open_speedup\|4` | L2 = 5175 and L3 = 5640 tok/s/worker, both in [5000, 6000] | 0.930 → 0.506 |
| `synthetic_sessions\|think4.0\|open_speedup\|4` | L1 = 0.3005, L2 = 0.3152, L3 = 0.3469, all in [0.30, 0.35] | 0.953 → 0.631 |

Fix-r0's added transform grid points (0.425–0.475) are far from these levels.

**Realized level.** Default's good fraction at the frozen L2 load on train segments, using the official cell
construction (`EV/txlevel/report.json`, 3 replicates):

- osl4.0 (w0/w2/w4): **0.909** (0.882, 0.912, 0.933);
- think4.0 (s0–s3): **0.878** (0.97, 0.84, 0.75, 0.95).

So test cell `mooncake-w3-osl4.0-n4-open-L2` sits about 0.06 lighter than its L2 label on train data. The L3 levels in
the same intervals are likely further off; I did not measure them.

**Impact.** This has no direction between policies. It shrinks those cells' headroom and mislabels their band.

**Fix.** Refine the two grids around the frozen levels and re-level. Test may be re-frozen, because it has not been
evaluated. Alternatively, record the realized train-segment band for these cells in REPORT.

### F3 (minor, r0 F2 still open): N-scaling is knee-matched only

Per-worker load spreads 1.2–1.9× across N (§5). The AgentX per-worker prefill load doubles at N = 16/32.

LR-12 action 1's per-worker-matched arm is still absent from the frozen test set, yet `facts/calibration.json` lists
LR-12 as "applied". REPORT must not describe the N axis as per-worker matched.

### F4 (minor, r0 F3 still open): realized bands are wide per cell

Default's realized good fraction on sessions-open cells:

- L2: 0.715–0.961;
- L3: 0.423–0.804 (`EV/binding.json`).

The sessions curve is steep: at N = 8, default goes from 0.95 at 0.35 to 0.05 at 0.50 sessions/s/worker. The levels are
exact interpolations of the train mean (§5), so this is segment heterogeneity, not bias. Report the per-cell realized
band next to the level label.

### F5 (minor): the calibration summary handed to downstream stages is stale

The summary this audit received still cites the lr-cells-v4 values:

| Quantity | Stale (lr-cells-v4) | Current (lr-cells-v5) |
|---|---|---|
| test.jsonl SHA-256 | `b8648ac5…` | `7b998b8e…` |
| sessions SLA (I / S) | 40.4165 ms / 1.90627 | 46.0478 ms / 2.11784 |
| sessions N = 8 open levels | 0.324 / 0.365 / 0.407 | 0.350 / 0.380 / 0.409 |
| sessions N = 8 closed levels | 24.3 / 26.7 / 32.1 | 29.0 / 33.1 / 37.5 |

`facts/*`, `cells/*` and STATE are correct.

**Consequences.** Verifying against the stale hash fails loudly. Quoting the stale SLA or levels in REPORT would
mislead.

**Fix.** The orchestrator should hand later stages fixer r0's summary, or the facts, not r0's.

### F6 (minor): fix-r0's "0 test … trace SHAs in 35,735 result records under runs/" is not true of runs/ as a whole

304 records touch test segments: 235 cache entries and 69 results lines. All are from pre-calibration build stages:

| Directory | Lines | Replays |
|---|---|---|
| `build-fix-r1/candidates_check` (16:58) | 25 | default@defaults on test traces at provisional loads: w3/w5, s7–s9, FAST25, test AgentX plays |
| `build-fix-r0/postbuild` and `bundle/remote` | 28 + 16 | build-audit `aud-*` cells on a pre-B2 window spanning w4–w5 |

Scoped to calibration, the claim holds (§9). No calibration decision read these directories.

**Fix.** Reword the claim to "0 in calibration results". Optionally list these build-stage test-data replays in
DEVIATIONS for completeness.

### F7 (minor, r0 F6 still open): AgentX OSL = 0 rows are unscored

`agentx-V1-base-n4-lanes-L2` k = 0 has 64 rows that are completed with ISL 11008 and requested and actual OSL 0, but
have no TTFT or e2e. They are dropped for every policy. The effect is equal across policies.

### F8 (minor, r0 F5 still open): replicate reuse trusts the `.rep.json` sidecar without re-hashing

Today's data is intact: the 368/368 replicate files behind the step-4 reference hash to their recorded SHA
(`EV/rep_hash.json`).

### I1 (info): r0's F4 is a measurement artifact and is withdrawn

The 3.0e-7 isolated-vs-harness E0 gap came from absolute replay times of about 1e10 ms in that isolation trace. Double
precision there costs more than 1e-7 relative on short requests; my own fixed-spacing variant at about 1e11 ms showed
6.5e-7. With small absolute times, isolated replay and the harness v2 table agree within 1.11e-8 on 880,139 pairs, so
E2E_REL_TOL = 1e-6 has a margin of at least 90×.

### I2 (info): a stale label in decisions.json

`runs/calibrate-fix-r0/decisions.json` `session_open_warmup.rule` says `session-lifetime-p99-kappa3-v1`. The cells,
`facts/calibration.json` and the applied warm-ups use `…-x2-v1` (two lifetimes, W = 1200 s at think 1). Only the label
is wrong. I left it, because it is another stage's input artifact.

### I3 (info): the sessions ITL clause almost never binds alone

ITL-only failures are 0.08% of default's in-window requests, and 0.9% fail both clauses. The sessions SLA is
effectively E2E-only at these loads.

### I4 (info): within-window good-fraction movement follows trace content

Large first-half vs second-half moves are trace content. The clearest examples are the w1-based val Mooncake cells,
which all fall by 0.04–0.10, and `noise-sessions-s0-n2-open` (+0.21). These moves are reproduced across replicates and
have mixed signs across cells (§6). The doubled-history checks (§7) are the relaxation test, and they pass.

## Lessons

- **Applied:**
  - **LR-01:** windowed goodput re-derived. Action 5's doubled warm-up was extended to the untested test transforms and
    to Mooncake's 4-minute warm-up. The clipped log-ratio is recommended for lr-train (F1).
  - **LR-02:** CRN replicates, out-of-sample k = 10, 11, and pooled paired-ratio sd.
  - **LR-09:** causal release and think-invariance verified; all checks stratified by load mode.
  - **LR-11:** segment-level beyond-noise. The pre-registered clip neutralizes F1.
  - **LR-12:** per-N utilization (F3); realized bands of the test-transform levels (F2).
- **Rejected:**
  - LR-04 (A2.5);
  - LR-08's TTFT part (A2).

## Scratch

`EV` is about 5.1 GB:

- `root/` 3.0 GB: private cache with per_request gz, replicates and policies;
- `e0/` 1.0 GB and `e0_gap5e7/` 0.3 GB: isolation traces and outputs;
- `warm2x*/` 0.6 GB;
- `txlevel/` and `mc2x/` 52 MB.

It is listed in CLEANUP.md. No file was deleted. No WT, cell or fact file was modified.
