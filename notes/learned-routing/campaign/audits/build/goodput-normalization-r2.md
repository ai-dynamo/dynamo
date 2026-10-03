# Audit: build checkpoint, lens "goodput-normalization", round 2 (dynamic)

- Auditor: independent dynamic auditor, 2026-10-02 (about 17:03–17:35 PDT).
- Audited state: WT `rupei/learned-routing-public` at `<commit-12>` (build fixer r1 head), bindings build_id
  `6955b0ee…`, `HARNESS_VERSION` `lrh-3`, cells `lr-cells-v3` (SPLIT_MANIFEST `c42dacea…`).
- Evidence directory: `CR/runs/audits/build-goodput-normalization-r2/` (`scripts/`, `out/`, `logs/`,
  `cells/`). Key outputs: `out/goodput_compare.json`, `out/knife_edge_fixes.json`,
  `out/train_recompute.json`, `out/key_probe.json`, `out/edge_results.jsonl`, `out/audit_summary.json`.
- **Verdict: FAIL.** 0 blocker, 1 major, 7 minor.
  - The window, count, rate and normalization arithmetic is correct under lrh-3. My independent
    recomputation matches the harness exactly on every record once the same E0 values are used.
  - The major finding is in the E2E clause. When S × scale = 1, every request that ran uncontended
    with no prefix reuse is scored bad, because the harness E0 is biased low by up to 1.6e-5 and the
    comparison has no tolerance. This flips 11–12% of default's AgentX rows and 39–42% of
    round-robin's, and moves RR/default from 0.88 to 0.57. AgentX's e2e/E0 distribution has an atom
    at exactly 1 that contains default's p95, so calibration can easily land on this edge.

## What I verified independently

### 1. Independent A2 goodput from raw `per_request` (60 records, 6 cells)

- **Audit cells.** `scripts/make_cells.py` writes 6 PROVISIONAL `gn2-` copies of train/val candidates
  (no test cell), different from the r1 and fixer cells. Loads are B2's `load_provisional` unless
  noted. SLAs are chosen so that both clauses bind.

  | Cell | Mode | Load | SLA (I ms, S) | Measure |
  |---|---|---|---|---|
  | mooncake-w0-base-n8-open-L2 | open | speedup 0.805 | 35, 2.5 | trace warm-up/window ÷ speedup |
  | mooncake-w0-osl0.5-n8-open-L2 | open (15–16 rows OSL ≤ 1) | speedup 1.0 | 30, 2.0 | same |
  | sessions-s5-think1.5-n6-open-L2 (val) | open, causal turns, N=6 | speedup 0.663 | 20, 1.3 | same |
  | mooncake-w1-base-n8-closed-L2 (val) | closed C=32 | 32 | 35, 2.5 | identity warm-up, `end: full_occupancy` |
  | sessions-s4-base-n8-closed-L2 (val) | closed C=32, multi-turn | 32 | 20, 1.3 | identity warm-up, `full_occupancy` |
  | agentx-A1_A2-base-n8-lanes-L3 | lanes, 22 plays | 8 lanes | 30, 1.0 | `warmup_ms` 300 s, `full_occupancy` |

- **Harness run.** `lr-eval` over 5 policies (default@defaults, round_robin, lmetric@defaults,
  learned-choice θ0 one_draw, sticky-session hard) × replicates k = 3, 4: 60 records, 0 errors
  (`out/harness_results.jsonl`).
- **Direct replays.** `scripts/replay_direct.py` re-ran all 60 through `dynamo.replay.run_trace_replay`
  without importing `learned_routing`. From each record it took only locators: the replicate trace,
  found by SHA-256 and re-hashed; the replay YAML, with seed asserted to be k + 1 for seeded types
  and absent for lmetric; and the router mode. Every other argument came from the cell JSON and
  `engine.json`. Each replay held one `CR/slots` slot.
- **E0.** `scripts/e0_isolated.py` replayed all 17,233 distinct completed (ISL, OSL) pairs, each alone
  on one worker, 3,000 s apart, with fresh hash ids at block size 64 (r1 used 512). Isolation, zero
  reuse and completion are asserted (`out/e0_isolated.json`).
- **Scoring.** `scripts/goodput_indep.py` implements A2 from the contract text. Its windows come from
  slice facts, not from `measure`:
  - open loop: source window `[W0 + 240 s, W1)` (Mooncake) or `[180 s, 720 s)` (sessions), mapped
    through an affine map fitted on first-turn rows (max |Δ| 0 ms);
  - closed loop: warm-up sessions by source timestamp through the replicate's `request_<line>` IDs;
    the end is the first session departure strictly after the last session admission;
  - lanes: the end is the minimum over the driver's own `agentic.lane_id` of each lane's last
    terminal.
- **Result** (`out/goodput_compare.json`, `logs/goodput_indep.log`):
  - **50/50 exact** on every non-AgentX record. These all agree: `good`, `window_good`,
    `window_requests`, `good_frac_window`, `goodput_rps_window`, `goodput_rps`, window start and end,
    and `rescore` at all 6 scales.
  - Raw rows equal the harness's cached rows as a multiset on 60/60. Native ITL-only relative error is
    0 on 60/60, for the harness and for me.
  - On the 10 AgentX records the windows, starts, ends and counts agree exactly, but `good` differs
    (742 vs 694 for default k3). Substituting the harness's E0 table into my scorer gives **10/10
    exact** (`logs/goodput_indep_harnessE0_agentx.log`). The only difference is E0 at S = 1.0: see F1.
- **Other checks:**
  - ISL buckets: `good_frac_window` 260/260 exact (`logs/isl_buckets_check.log`).
  - `window_below_cap_ms` for windows pushed 25% and 50% past full occupancy equals my own midpoint
    integration exactly on closed Mooncake, closed sessions and lanes (`logs/below_cap_check.log`).

### 2. The full-occupancy end rule (fix r1)

- Closed loop: harness end = first departure strictly after the last admission on 20/20 closed
  records. Exactly one departure coincides with the last admission each time: the one that triggers
  it, netted by the profile.
- Lanes: end = first lane exhausted, by the driver's `lane_id`, on 10/10. Peak = cap and time below
  the cap is 0 on 30/30 completion records.
- The scored count in closed-loop Mooncake is policy-independent: 2,121 = 2,153 measured − 32. On
  closed sessions it is 2,703–2,706, and on lanes 278–306.

### 3. Edge cases (`scripts/make_edge.py`, `out/edge_results.jsonl`)

Real lr-eval replays of a 53-row trace with 3 rejected requests (ISL 131,072), 8 rows with OSL 1 and
a 2-turn session:

| Cell | Result |
|---|---|
| Open loop, warm-up 10 s, window 40 s | 2 in-window rejections count as misses: 40/42. Correct under A2.3. |
| Closed C=4, identity + `full_occupancy` | 2 in-window rejections are invisible: 36/36 (F5). |
| Closed `end: fixed` 15 s | 25/25; window = start + 15 s. |
| Closed, no warm-up | Window starts at 0. |
| Completion basis without `end` | `plan_error` with no replay. The guard works. |
| Open loop with `measure_trace` hints but no `measure.warmup_ms` | Silently scores [0, last arrival], warm-up included (F3). |

- OSL ≤ 1 rows (15–20 per Mooncake cell): the ITL check is skipped and E0 is prefill only. All match
  in item 1.
- **Zero-output requests (F2).** AISim reports a request with `requested_output_length` 0 as
  `completed` with null `ttft_ms` and `e2e_latency_ms`. The harness treats it as not completed.

### 4. E0

- Harness table against my 17,233 isolated replays: relative difference from −1.64e-5 to +6.3e-9.
  The harness value is low on essentially every pair.
- Root cause confirmed (`logs/e0_offbyone.log`, 400 random pairs). With decode context
  `ISL + j + 1` (the current code) the error ranges from −1.05e-5 to +1.4e-9. With `ISL + j + 2` it
  is within ±1.6e-8 of the isolated replay.

### 5. Normalization reference

- **lr-train.** A tiny run (`out/train_tiny`) used `clipped_log_ratio`, 3 train cells (open
  Mooncake, closed Mooncake, AgentX lanes), 1 closed-sessions validation cell, 2 generations of 4
  and 4 validation lines.
  - My recomputation from the cached rows matches all 12 objective values exactly, with the harness
    E0 (`out/train_recompute.json`).
  - The reference is `38a79285…` = `dynamo-default-cost-fn`, `parameters: {}`, `router_config: {}`,
    i.e. default@defaults. Its seed is k + 1 on 56/56 records.
  - On 56/56 pairs, candidate and reference share `cell_sha`, `build_id`, replicate `trace_sha256`,
    protocol and harness version.
- **lr-report.** `norm_mean` equals my paired ratios exactly (max |Δ| 0), overall and per load mode
  (`logs/report_recompute.log`).
- **Explicit defaults equal the reference.** A spec carrying the `default_cost_fn.yaml` init values
  explicitly is request-for-request identical to default@defaults on 8/8 (cell, k), across open,
  closed, sessions and lanes (`logs/init_parity.log`). So generation 0 of the default tuner is
  centred exactly on the reference.

### 6. Cache-key coverage (`scripts/key_probe.py`, `out/key_probe.json`)

- **The key changes with:**
  - cell load, SLA, `measure.end` and `warmup_trace_ms`, N, block size, `engine_overrides` and
    `replay_options`;
  - one changed trace byte;
  - `mock_engine_args`, including its `ais_perf_config`;
  - policy parameters and the `router_config` sidecar;
  - k, `HARNESS_VERSION`, and one appended `.so` byte.
- **Labels correctly do not change it:** split, notes, segment, `expected_cost_s`, `measure_trace`,
  holdout axis, policy name.
- **Gaps:** see F4.

## Findings

### F1 (MAJOR): the E2E clause is a knife-edge at S × scale = 1, and the low-biased E0 scores every uncontended no-reuse request as bad, in a policy-dependent way

**Defect.**
- A2 defines E0 as the latency of the request alone on an idle worker with no prefix reuse. A
  request that actually ran that way therefore has e2e = E0. Under A2 it must be good whenever S ≥ 1.
- `goodput.a2_good` tests `e2e_latency_ms > scale * slowdown * e0(...)` with no tolerance.
- `e0.py` sums decode steps at context `ISL + j + 1`, while replay uses `+ 2` (r0 F2 / r1 F4, rated
  minor with "0 flips"). This makes E0 low by 5.9e-8 to 1.64e-5 relative.

**Measured** on the gn2 AgentX lanes cell (`logs/knife_edge.log`, `out/knife_edge_fixes.json`):
- **The atom.** These requests have e2e within ±2.4e-9 of their isolated replay, and none has prefix
  reuse:

  | Policy | Requests in the atom (of 797) |
  |---|---|
  | default@defaults | 84–96 (11–12%) |
  | sticky-hard, lmetric@defaults, learned-choice θ0 | similar |
  | round_robin | 308–333 (39–42%) |

  Against the harness E0 every one of them is above E0 (+5.9e-8 to +1.64e-5), so all are bad at
  S × scale = 1.
- **Distribution.** The e2e/E0 distribution on AgentX has an atom at 1 (`P[1, 1 + 2e-5]` is 0.105–0.120
  for the default-like policies and 0.386–0.418 for RR). Default's p95 of e2e/E0 is 1.00000 on both
  replicates. Because E2E is otherwise non-binding on AgentX (r0 F7), an A2 band-rule S near 1 is
  natural, as is S = 1 ("no slower than idle"). So are S = 2 or S = 4/3, through the 0.5 and 0.75
  rescore scales.
- **Ratios against default at S = 1.0, I = 30 ms:**

  | k | Policy | Harness as is | Isolated E0, exact compare | Any tolerant compare (A2 intent) |
  |---|---|---|---|---|
  | 3 | round_robin | **0.5672** | 0.7501 | **0.8812** |
  | 3 | lmetric@defaults | 0.9860 | 0.9685 | 0.9576 |
  | 3 | sticky-hard | **1.0048** | 0.9925 | **0.9967** |
  | 3 | learned-choice θ0 | 0.9959 | 0.9924 | 0.9967 |
  | 4 | round_robin | 0.5615 | 0.7641 | 0.9420 |
  | 4 | lmetric@defaults | **1.0012** | 0.9965 | **0.9815** |
  | 4 | sticky-hard | 1.0103 | 0.9984 | 0.9874 |
  | 4 | learned-choice θ0 | 1.0078 | 1.0075 | 1.0036 |

  - The three tolerant variants agree exactly: harness E0 × (1 + 1e-4), E0 fixed to `+2` × (1 + 1e-6),
    and isolated E0 × (1 + 1e-6).
  - The exact compare against the true E0 is itself a coin flip at float noise: 41 of 89 atom rows
    lie above the isolated E0 by up to 2.4e-9.
  - The deviations from 1 change sign (sticky-hard at k3, lmetric at k4), at the scale of the A1
    noise floor (pooled sd 0.01–0.05).
- **Training.** Re-scoring the tiny lr-train run with the isolated E0 changes objectives by up to 0.084
  (clipped log-ratio) and swaps generation 0's 2nd and 3rd ranks (`out/train_recompute.json`). CMA-ES
  is rank-based, so its update changes.
- **Other families.** Mooncake has 0–1 atom rows per cell and sessions 7–16 of 3,740–3,872, so the
  effect there is small. It concentrates on AgentX: native lanes now, A3 lowered next.

**Why it matters now.** Calibration fixes I and S next and freezes them. It is a latent bug that
silently and policy-dependently changes the headline family's ratios, and above all the RR floor and
the session baselines that keep caches warm.

**Fix:**
1. In `e0.py`, use decode context `ISL + j + 2` (±1.6e-8 measured).
2. In `a2_good`, compare with a relative tolerance: `e2e <= scale·S·E0·(1 + 1e-6)`.
3. Bump `e0.METHOD`. `E0Table.__call__` serves stored values first, so without the bump the old table
   (63,065 entries) keeps being used.
4. Bump `HARNESS_VERSION`. The result key carries no E0 identity, so cached goodput scored with the old
   E0 would otherwise be reused (see F4).
5. Add a test: a single-request replay at S = 1 is good, and RR on an idle trace is all good at S = 1.

Calibration should record the atom fraction per family and avoid treating "p95 of e2e/E0" as a free
choice of S without the tolerance fix.

### F2 (minor): zero-output requests are never scored

- AISim reports `requested_output_length: 0` as `terminal_status: completed` with null
  `ttft_ms`/`e2e_latency_ms`, even though `terminal_time_ms` is set (660 ms after arrival).
- `goodput.completed()` requires TTFT and e2e, so such a request is never good:
  - completion basis: it is dropped from both numerator and denominator;
  - arrival basis: it is a forced miss for every policy.
- Scale: 1 of 3,214 base AgentX rows (play `0021`, val pool, cell agentx-V1). It is cached in 30
  records (29 under lrh-1, 1 under lrh-3). A3 recycling multiplies it, and A3 open-loop AgentX scores by
  arrival.
- The effect is policy-independent. Fix: score OSL 0 with e2e := terminal − arrival against S·E0(ISL, 0),
  or exclude such requests explicitly and record that in `facts/calibration.json`.

### F3 (minor, r1 F3 carried and extended): no guard against scoring the warm-up

- Open loop: a cell with `measure_trace.warmup_ms > 0` but no `measure.warmup_ms` still scores
  `[first, last arrival]`, warm-up included. The `gn2edge-open-nomeasure` cell scored [0, 47 s]
  instead of [10 s, 50 s].
- Lanes: all 22 lanes candidates carry only `{basis, end}`, so without `warmup_ms` the window starts at
  t = 0. That includes the synchronized first-play start and the cold caches. There is no guard
  either.
- Fix: plan-time errors for both cases. The completion-basis `end` guard shows the pattern works.

### F4 (minor, r1 F5 carried; now load-bearing for F1's fix): stale-cache risks outside the result key

Re-verified by `scripts/key_probe.py`:

a. **`seeded` is not in `policy_sha`.** `{"type": "dynamo-default-cost-fn", "seeded": false}` has
   **the same `policy_sha` as default@defaults**. An unseeded run can fill the reference's cache entry.
   Fix: add `seeded` to `identity()` only when it differs from the type default, which keeps
   existing SHAs.
b. **No E0 identity in the result key.**
   - Changing only the top-level `ais_perf_config` (the E0 source) leaves the result key unchanged
     while the E0 table path changes.
   - The E0 table is keyed by `ais_perf_config`, chunk and `METHOD` only, not by build ID,
     `aisimulate` version or `dynamo/_internal/ais.py`.
   - An aisimulate-core patch (the contract allows `[patch.crates-io]`) therefore re-runs replays but
     scores them against a stale E0.
c. **Code outside the key.** The scoring code (`goodput.py`, `e0.py`), the editable replay wrapper,
   and the replicate trace bytes (only the protocol ID and k are keyed) are covered only by manual
   `HARNESS_VERSION` and protocol bumps.
   - When calibration adds A3 `agentic_mooncake` replicates (GenSpec seed = k), the protocol ID or the
     replicate SHA must enter the key.
   - Fix: put `rep.trace_sha256`, the E0 `METHOD` and the E0 file SHA (or `build_id`) in the key. That
     is cheap now, while no calibrated result exists.
d. **Worker `_engine_cache`** memoizes `engine.json` by path, so an edit made during a run is not seen
   (unlikely).

### F5 (minor, r1 F6 carried)

Completion basis hides rejections: 2 in-window rejections were invisible (36/36). Rejection is still
policy-independent today (static admission limit).

### F6 (minor, r1 F7 carried)

`eps` is an absolute 1e-3 rps. On AgentX lanes, default's `goodput_rps_window` is 0.130–0.162 rps and
RR's 0.073–0.092. The (m + ε)/(m_ref + ε) form shrinks AgentX log-ratio deviations by about 0.6–1.4%,
against under 0.05% on Mooncake (3–5 rps). Use a relative ε or record the choice.

### F7 (minor, r1 F8 carried)

The A1 noise rule (`noise.differs_beyond_noise`) still has no caller in the harness (grep). The pilot's
glue must build its inputs from content-keyed records (`report.select_contents`).

### Closed since r1

r1 F1 (completion windows scored the drain) and F2 (sessions drain) are fixed. On 30/30 completion
records, time below the cap is 0, and the end rules agree with my independent derivations (item 2).

## What closed-loop and open-loop goodput mean (measured on the gn2 cells)

The normalized rate ratio decomposes as attainment ratio × completions-per-second ratio.

| Cell | Policy, k | Rate ratio | Attainment | Completions/s |
|---|---|---|---|---|
| Mooncake w0 open | RR k3 | 1.0845 | 1.0845 | 1.0000 |
| Mooncake w1 closed | RR k3 | **0.9042** | 1.0505 | 0.8607 |
| Mooncake w1 closed | lmetric k3 | 1.1917 | 1.1408 | 1.0446 |
| Sessions s4 closed | RR k3 | 0.9826 | 1.0011 | 0.9815 |
| Sessions s5 open | RR k3 | 0.9274 | 0.9290 | 0.9982 |

- **Open loop (flat traces):** a pure attainment ratio. The in-window count is fixed.
- **Open loop (causal sessions):** the in-window count moves slightly with the policy (2,844–2,856).
  Because the training metric is the rate, pushing turns out of the window lowers it, so there is no
  exclusion incentive.
- **Closed loop and lanes:** attainment × throughput at full occupancy. RR beats default in open-loop
  Mooncake (1.08) but loses in closed loop (0.90) even with *higher* attainment. Stratify gates and
  tables by load mode (LR-09). Pooling modes mixes two different quantities.
- In closed loop the C requests in flight at the end are excluded. That is policy-independent in
  count (M − C), but by inspection-paradox reasoning they are biased toward long requests, about
  1.5% of the scored set. This is noted, not a finding.

## Lessons (LESSONS.md)

- **Applied:**
  - LR-01: windowed attainment, and the full-occupancy windows re-derived;
  - LR-03: CRN pairing (same replicate trace, seed k + 1) verified on 56/56;
  - LR-08: the rescore sweep verified at 6 scales, and its S × scale = 1 hazard is F1;
  - LR-09: per-mode meaning, with the RR open/closed sign flip;
  - LR-11: content-keyed pairing in lr-train and lr-report, and the noise caller (F7).
- **Rejected:** LR-04 (A2.5) and the TTFT part of LR-08 (A2.2).
- **Out of lens:** LR-02, LR-05–07, LR-10, LR-12–15.

## Process notes

- **Replays**, all holding `CR/slots` slots, at most 12 at once:
  - 60 through lr-eval, 60 direct;
  - 5 isolated-E0 batches (17,233 single-request replays);
  - 10 edge replays, 16 init-parity replays, about 67 fresh in the tiny lr-train.
- **Writes:** none to WT or the main checkout; nothing committed; no fix applied. F1 needs a version
  bump plus a tolerance, so it is not a one-liner.
- **Cache:** 145 audit-only entries (`gn2-`/`gn2edge-` cell IDs, about 29 MB) were added to
  `CR/runs/cache`. Scratch is listed in `CLEANUP.md`.
