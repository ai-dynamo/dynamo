# Audit: build checkpoint, lens "goodput-normalization", round 1 (dynamic)

- Auditor: independent dynamic auditor, 2026-10-02 (~16:20–16:45 PDT).
- Audited state: WT `rupei/learned-routing` at `<commit-10>` (the build fixer r0 head; untracked
  `benchmarks/learned_routing/tools/` belongs to the A3 sidecar and was not touched), bindings
  build_id `6955b0ee…`, `HARNESS_VERSION` `lrh-1`, cells `lr-cells-v2` (SPLIT_MANIFEST `0f8ea452…`).
- Evidence directory: `CR/runs/audits/build-goodput-normalization-r1/` (`scripts/`, `out/`, `logs/`,
  `cells/`).
- **Verdict: FAIL.** 0 blocker, 1 major, 7 minor.
  - All the goodput arithmetic is correct. My independent recomputation matches the harness exactly on
    48/48 fresh records and 32/32 of the fixer's closed-loop records. The normalization reference is
    correct in both lr-train and lr-report.
  - The major finding is about what the completion-basis window measures. On lanes cells it scores the
    end-of-run drain, which dilutes policy differences by 23–41% (with one sign flip). The harness
    offers no drain-free rule for AgentX formats, and the latest stage note tells calibration that it
    only needs to set the open-loop windows.

## What I verified independently

### 1. Independent A2 goodput from raw `per_request`: exact on 48/48

- **Audit cells.** `scripts/make_cells.py` writes 6 PROVISIONAL `gn1-` copies of train/val candidates
  (none from test) to `cells/cells.jsonl`.
  - Loads are the fixer's `load_provisional` values.
  - SLAs are chosen so that requests fail: Mooncake I=40 ms, S=3; sessions S=1.15; AgentX S=0.8.
  - Open-loop `measure` is `measure_trace / speedup`; closed loop keeps the cell's own identity rule.

  | Cell | Regime |
  |---|---|
  | Mooncake w2 open N4 | first arrival 3,055 ms after the slice start (rebase) |
  | Mooncake w0-osl0.5 open N8 | 16 rows with OSL ≤ 1 |
  | Mooncake w4 closed N4, C=16 | identity warm-up |
  | Sessions s4 open N4 | causal turns |
  | Sessions s2-think2.0 closed N4, C=16 | identity warm-up over multi-turn sessions |
  | AgentX A2 lanes N4 | 4 lanes over 11 plays |

- **Harness runs.** `lr-eval` with 4 policies (default@defaults, round_robin, lmetric@defaults,
  sticky-session hard) × 2 replicates gave 48 records and 0 errors (`out/harness_results.jsonl`).
- **Direct replays.** `scripts/replay_direct.py` re-ran all 48 directly through
  `dynamo.replay.run_trace_replay`.
  - It does not import `learned_routing`, and each replay held one `CR/slots` slot.
  - It took only locators from the record: the replicate trace (SHA-verified), the replay YAML (seed
    asserted to be k+1) and the router mode. Every other argument came from the cell JSON and
    `engine.json`.
- **E0.** `scripts/e0_isolated.py` replayed all 17,606 distinct (ISL, OSL) pairs of those rows, each
  alone on one worker, 2,000 s apart, with fresh hash ids. It asserted isolation, zero reuse and
  completion (`out/e0_isolated.json`).
- **Scoring.** `scripts/goodput_indep.py` implements A2 from the contract text with my E0.
  - **Windows** come from trace and slice facts, not from the cell's `measure` block:
    - Mooncake open loop: source time `[W0+240 s, W1)`, mapped through the replay's affine time map,
      which is exact (max |Δ| 0 ms);
    - sessions open loop: absolute `[180 s, 720 s]`;
    - closed loop: warm-up sessions are those whose first trace arrival falls in the first 240 s of the
      slice (180 s for sessions); the window runs from the first measured dispatch to the last
      dispatch;
    - lanes: from the first arrival to the last arrival.
- **Result** (`out/goodput_compare.json`, `logs/goodput_indep.log`): **0 mismatches on 48/48 records.**
  - Raw rows equal the harness's cached `per_request` rows as a multiset on 48/48.
  - These all agree exactly:
    - `good`, `window_good` and `window_requests`;
    - `good_frac_window` and `goodput_rps_window`;
    - `goodput_rps`;
    - window start and end (relative difference 0);
    - `rescore` at all 6 scales.
  - The native ITL-only check has relative error 0 on every record.
  - Example: Mooncake w2 open, k=0. Default is 1375/2380 good, 1.355093 rps; lmetric is 1549/2380,
    1.526574 rps.
- **The fixer's identity smoke.** `scripts/drain_ratio.py` recomputed all 32 closed-loop records of
  `runs/build-fix-r0/identity_smoke` from their cached rows, with my own warm-up-ID derivation. All
  32 match (`out/drain_ratio_fix_smoke.json`).

### 2. The closed-loop identity warm-up rule

- **Admission order.** aisimulate-core admits closed-loop sessions in file order of first appearance:
  `driver.rs` `next_pending_session`, with no sort.
- **Static check** (`scripts/closed_order_check.py`, `out/closed_order_check.json`). On all 27
  closed-loop candidate cells:
  - the identity warm-up set is exactly a prefix of the admission order;
  - first timestamps are monotone in file order.

  Replicates only permute exact-timestamp ties, so this holds for every k.
- **Session IDs.** Line numbering matches `trace.rs` `request_{line_idx+1}`: 1-based, with blank lines
  counted. The rule raises a ValueError on Weka and agentic-Mooncake traces, which have no
  `timestamp`, so it cannot silently mis-score them.
- **Measured set.** In closed-loop Mooncake the scored set is policy-independent: exactly measured rows
  minus C (2027 − 32 = 1995; 2496 − 32 = 2464). The window end equals the last admission, so there is
  no drain (`out/drain_fix_smoke.json`, `out/drain_gn1.json`).

### 3. E0

- `scripts/e0_compare.py` (`out/e0_compare.json`) compares the harness table
  `runs/cache/e0/01277503…-ais-chunked-estimator-v1.json` with my 17,606 isolated replays.
  - Every pair is present.
  - Maximum relative difference is 1.03e-5, median 1.6e-6.
  - The harness value is lower on 17,576 of 17,606 pairs (r0 F2, still open).
- Classification flips: **0 at every scale (0.5–3) over 156,136 rows.** One row lies within 1e-5 of
  its bound.

### 4. Edge cases

- **OSL ≤ 1.** 424 rows in my replays. The ITL check is skipped and E0 is prefill only. All of them
  match in item 1.
- **Truncation.** 0 of 156,136 completed rows have `output_length != requested_output_length`.
- **Rejected requests** (`scripts/edge_smoke.py`, `out/edge_smoke.json`; real replay, ISL 131072,
  scored by the harness):
  - Open loop: the in-window rejected request counts as an in-window miss (22/23 good). This is
    correct under A2.3.
  - Closed loop, completion basis: the in-window rejection appears in neither numerator nor
    denominator (18/18 good). See F6.
  - Rejection only comes from the engine's static admission limit (`agg.rs` `signal.rejected`; the log
    says "exceeds a worker admission limit"), which is policy-independent.
- **AgentX primers.** No agentic warm-up requests leak into the scored rows: 486 per-request rows
  equal the 486 nested trace requests and `agentic_graph.node_count`.
- **Open-loop warm-up and drain.** The windows I derived from trace facts equal the harness's windows
  exactly, including w2's 3,055 ms rebase. In-window arrivals that finish late still count.

### 5. Normalization reference

- **lr-train.** A tiny run, `out/train_tiny` (default-cost-fn space, 2 train cells mixing open and
  closed loop, 1 validation cell, 6 fevals, `--keep-per-request`). I recomputed all 10 objectives
  (6 generation and 4 validation lines) from the cached rows with my own goodput code and E0:
  **maximum absolute difference 0** (`scripts/train_recompute.py`, `out/train_recompute.json`).
  - `reference_sha` `38a79285…` is `dynamo-default-cost-fn` with `parameters: {}` and
    `router_config: {}`, i.e. default@defaults.
  - The seed is k+1 on 12/12 reference records.
  - On 32/32 pairs, candidate and reference share `cell_sha`, `build_id`, replicate `trace_sha256`,
    protocol and harness version.
- **lr-report.** `norm_mean` matches my independent paired ratios exactly, with maximum difference 0,
  for 12 family groups and 4 overall rows (`logs/report_recompute.log`). The r0 F1 content-keyed
  pairing fix holds.

### 6. Cache-key coverage

`scripts/key_probe.py`, `out/key_probe.json`.

- **Covered:**
  - policy type, parameters and the `router_config` sidecar (`policy_sha`);
  - cell content, i.e. load, SLA, `measure` including `warmup_trace_ms`, `engine_overrides`,
    `replay_options` and block size;
  - trace bytes;
  - `mock_engine_args` and model;
  - replicate protocol and k;
  - the `.so` SHA plus the `aisimulate` version.

  AIS data comes from the pinned `aisimulate_core/systems` package, which that version covers.
- **The fix r0 change to `goodput.py`.** It added 109 lines without a `HARNESS_VERSION` bump. It is
  harmless: the new behavior applies only under `warmup_trace_ms`, which changes cell content, and the
  rebuild changed `build_id` anyway.
- **Gaps:** see F5.

## Findings

### F1 (MAJOR): completion-basis windows end at the last dispatch, so lanes cells score the end-of-run drain; no drain-free rule exists for AgentX formats

**Defect.** For completion basis, `goodput.measurement_window` ends at the last arrival of any row
(`end = last`, `goodput.py:248`) unless `measure.window_ms` is set.
- With one-pass Weka lanes, plays are dealt to lanes once, so lanes empty at very different times.
  Play completions in A2 range from 347 s to 7,545 s (`agentic_play_outcomes`).
- The window therefore runs on long after most lanes have gone idle.

**Measurements** on `gn1-agentx-A2-base-n4-lanes-L2` (`logs/lanes_window.log`, `logs/lanes_ratio.log`):
- **Window time below full occupancy:** 62.6–67.0% of the scored window (7,493–7,920 s) has fewer than
  4 active plays.
- **Scored completions below full occupancy:** 140–244 of 485 (28.9–50.3%). The share depends on the
  replicate: about 50% at k=0 and 29% at k=1, because replicates permute the deal of plays to lanes.
- **Dilution of policy differences.** Policy/default ratios under the harness window, against a window
  restricted to full occupancy:

  | Policy, k | Harness rate ratio | Full-occupancy rate ratio |
  |---|---|---|
  | round_robin k0 | 0.8471 | 0.8289 |
  | round_robin k1 | 0.8483 | 0.8033 |
  | lmetric k0 | **1.0003** | **0.9858** |
  | lmetric k1 | 0.9412 | 0.9009 |

  The harness window shrinks deviations from 1 by 23–41%, and lmetric k0 flips sign. Good fraction
  under full occupancy is 0.788–0.802 for default, against 0.816 over the whole window: the drain
  scores easy, under-loaded requests. The cells' L1/L2/L3 labels therefore do not describe most of
  what is scored.
- **Scale.** This affects 8/34 train, 4/14 validation and 10/56 test cells: every AgentX cell, all on
  the native lanes path.

**Why calibration will not fix it on its own:**
- The identity rule cannot express a lanes window. `warmup_ids_from_trace` raises on Weka and
  agentic-Mooncake traces (verified on the A2 derived trace and the A3 sidecar's
  `helper_lowered.jsonl`).
- A fixed `window_ms` is the only remaining tool, and the lane-exhaustion time varies by replicate:
  the last play starts at 4,247–5,451 s.
- `facts/build_fix_r0.json` `next_stage_notes` says calibration "sets only load.value, the SLA and
  open-loop warmup_ms/window_ms". That contradicts the integration note "Calibration must fix the
  closed-loop and lanes window rule".
- The harness accepts a completion-basis cell with no end rule, without warning.

**A3 does not remove the issue.** Lowered closed-loop AgentX also runs `agentic_lanes` with completion
basis. Recycling lowers the drain fraction but does not remove it, because the window still ends at
the last dispatch of the longest final play.

**Fix:**
- In the harness:
  - add a completion-basis end rule, such as `measure.end: full_occupancy`, that ends the window at
    the last time all `load.value` lanes or sessions were occupied. This is computable from
    per-play/session spans in `per_request`; for work-conserving drivers it equals the last
    admission;
  - reject completion-basis cells that declare no end rule.
- Or, in calibration: set an explicit `window_ms` per lanes cell that ends before the earliest lane
  exhaustion of default across all replicates, with margin, and record it in
  `facts/calibration.json`.
- Either way, correct the fixer's next-stage note.

LR-01 lists "a fixed window in which all lanes are active" as a hypothesis. These measurements support
it.

### F2 (minor): the same drain on closed-loop multi-turn sessions is small

- `scripts/drain_check.py` and `scripts/drain_ratio.py`. Sessions hold their slot across all turns, so
  the window from the last admission to the last dispatch is below the cap:

  | Cell | C | Window below cap | Completions after the last admission |
  |---|---|---|---|
  | s0 | 32 | 22.9–23.3% (302–310 s) | 175–180 of about 2,880 |
  | s2-think2.0 | 16 | 9.3–9.4% | 73–74 of about 2,753 |
  | s5 | 8 | 3.3–3.8% | — |

- Ending at the last admission instead moves ratios by at most 0.36 percentage points (lmetric s0 k0:
  1.00324 against 1.00688), with no consistent sign.
- Closed-loop Mooncake has none (0.0%).
- Fix: the same end rule as F1.

### F3 (minor, r0 F4 still open): open-loop candidates still carry only `measure = {basis: arrival}`

- Without `warmup_ms`/`window_ms`, the harness scores from the first arrival to the last, which
  includes the 4-minute warm-up. There is no guard.
- The conversion `measure_trace / speedup` is verified exact, including the w2 rebase (item 1).
- Fix: reject an open-loop cell whose `measure_trace.warmup_ms > 0` when `measure.warmup_ms` is unset.
  Also, `window_ms: 0` is still read as unset.

### F4 (minor, r0 F2 still open): E0 decode context is off by one

- The harness E0 is below the isolated replay on 17,576 of 17,606 pairs, by at most 1.03e-5 relative.
- Flips: 0 at any scale over 156,136 rows.
- Fix `e0.py` `ISL + j + 1` → `+ 2` with a `METHOD`/`HARNESS_VERSION` bump before SLAs are frozen, or
  record it as accepted.

### F5 (minor, r0 F3 still open, extended): stale-cache risks outside the result key

a. **`seeded`** is not part of `policy_sha`. `{"type": "dynamo-default-cost-fn", "seeded": false}` has
   the same SHA as default@defaults but replays with no seed, i.e. a nondeterministic tie RNG. Its
   result can fill the reference's cache entry.
b. **The E0 table has no build identity.** It is keyed only by the top-level `ais_perf_config`, the
   chunk and `METHOD`; the result key has no E0 identity at all.
   - E0 is computed through `WT/lib/bindings/python/src/dynamo/_internal/ais.py`, an editable Python
     file that is not hashed.
   - An `aisimulate` upgrade or an aisimulate-core patch therefore re-runs every replay but scores it
     against stale E0 values.
   - The top-level `ais_perf_config` is not part of the cell content. It equals
     `mock_engine_args.ais_perf_config` today, but an edit to one alone would not invalidate anything.
c. **Code outside the key:** the replicate trace SHA, the scoring code and the editable
   `components/src/dynamo/replay/*.py`. They are covered only by the manual `HARNESS_VERSION` (fix r0
   changed `goodput.py` without a bump; harmless this time).
d. **Worker memo:** `worker._engine_cache` memoizes `engine.json` by path for a worker's lifetime. An
   edit made while an lr-* process runs would replay the old engine under the new key. This is
   unlikely.

Fix: add `seeded` to `identity()`; add `build_id` to the E0 file key and the E0 file's SHA to the
result key; hash the scoring modules and the replay wrapper into `build_id` or the key.

### F6 (minor): completion basis makes rejected and incomplete requests invisible

- In closed loop and lanes, a rejected request appears in neither numerator nor denominator, and it
  frees its slot at once (`out/edge_smoke.json`: 18/18 good despite an in-window rejection).
- Today this is policy-independent: rejection comes only from the static admission limit, and no
  candidate cell exceeds 131,072 tokens.
- It would reward a future policy or knob that sheds load. Document it, or count closed-loop
  rejections as misses.

### F7 (minor): `eps` is still an absolute 1e-3 rps (r0 F5)

- Default's `goodput_rps_window` on the AgentX audit cell is 0.0512–0.0528 rps. `(m+ε)/(m_ref+ε)`
  therefore shrinks AgentX deviations by about 2%, against less than 0.1% on Mooncake.
- Use a relative ε, or record the choice.

### F8 (minor): the A1 noise rule has no content-keyed caller

- `noise.paired_ratios` takes dicts keyed by `(cell_id, k)` only, and no harness CLI applies the
  "differs beyond noise" gate. The pilot's glue code must therefore build its inputs from records
  restricted to one content (`report.select_contents` / `CONTENT_KEYS`).
- Otherwise the r0 F1 mixed-content hazard comes back in the gate.

## What closed-loop and open-loop goodput mean (for calibration and the report)

- **Open loop (arrival basis).** The window is policy-independent for flat traces, so
  `goodput_rps_window` = in-window arrivals / window × `good_frac_window`, and the ratio is an
  attainment ratio.
  - For causal sessions the in-window count moves slightly with the policy (2,885–2,888 on s4).
- **Closed loop, Mooncake (identity rule).** The scored count is fixed at M − C, so the ratio is
  attainment × throughput. For example, on w4 k0 lmetric/default is 1.121 = 1.072 (good fraction) ×
  1.046 (window).
- **Closed loop, sessions.** Think time dominates, and ratios stay within ±0.7%. This is the LR-09
  compression.
- **Lanes.** See F1.
- Pooling the modes in one objective mixes these quantities. Stratify gates and tables by load mode
  (LR-09).

## Lessons (LESSONS.md)

- **Applied:**
  - LR-01: windowed attainment and drain. F1 supplies evidence for the "window in which all lanes are
    active" hypothesis.
  - LR-03: CRN pairing (same replicate trace, seed k+1), verified.
  - LR-08: the rescoring sweep, verified exactly at all 6 scales.
  - LR-09: per-mode meaning and closed-loop compression.
  - LR-11: pairing integrity (content keys, F8).
- **Rejected:** LR-04 (A2.5) and the TTFT part of LR-08 (A2.2).
- **Out of lens:** LR-02, LR-05–07, LR-10, LR-12, LR-14 and LR-15, and LR-13's guard metrics (not
  re-verified).

## Process notes

- **Replays,** all holding a `CR/slots` slot:
  - 48 through lr-eval;
  - 48 direct;
  - 6 isolated E0 batches (17,606 single-request replays);
  - 2 edge smokes;
  - about 34 fresh replays in the tiny lr-train run.

  I used at most 8 slots, shared with the A3 sidecar.
- **Writes:** none to WT or the main checkout.
- **Cache entries:** about 80 audit-only entries for `gn1-` cells were added to `CR/runs/cache`
  (about 19 MB). They are listed in CLEANUP.md.
- **Fixes:** none applied. No finding has a safe one-line fix: F1 needs a new window rule and F4 needs
  a version bump.
