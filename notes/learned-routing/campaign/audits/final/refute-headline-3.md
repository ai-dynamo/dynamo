# Final audit, lens "refute-headline-3": robustness and simulator dependence

- **Auditor:** independent adversarial auditor, checkpoint final, 2026-10-04 23:15 to 2026-10-05 01:05 PDT.
- **Verdict: PASS.** 0 blockers, 0 majors, 4 minors.
- **Bottom line.** I could not refute the headline (M1-v2 beats ramjet in AIS-timed simulation). Its sign
  held under every condition I replayed or rescored:
  - router-state lag, up to 1,000 ms;
  - timing perturbations, including a fully perturbation-consistent SLO;
  - five warm-up and window variants;
  - four counterfactual simulator models that each remove or stress a known AISim simplification.

  Two things narrow the claim:
  - The win is significant only at the calibrated SLO scale and looser ones. At 0.75× the sessions
    family reverses (F1).
  - Select+Test left three robustness checks open: the mixed-prefill-pass exploit check owed to
    M1-v2 (F2), lag-0 identity for the headline pair (F3), and the LR-14 engine-state comparison (F4).
    This audit closes all three, and none changes the result.
- **Scope and integrity.** I never replayed the test split. Test evidence comes from rescoring the
  existing Select+Test records from their per-request rows. Every rescore re-derives the good flags with
  my own code. It asserts `window_good` and `goodput_rps_window` equal to the record on 3,600 test
  records (0 mismatches). My SLO-scale recompute matches the records' own `rescore` on 360 more. All
  my replays are on **val k3-10** (14 cells × 8 CRN replicates), in private roots under
  `runs/audits/final-refute-headline-3/root_*`. That is 3,248 replays with 0 errors, and the campaign
  cache was never written. Val k3-10 is the split that selected both the headline arm and
  the val-best baseline, so its absolute lead (+0.029) is optimistic. My evidence is the paired change
  between simulator variants on identical replicates. Nothing here feeds selection.

## What I ran

Scripts are in `runs/audits/final-refute-headline-3/scripts/`. Outputs are in `out/` (rolled up in
`out/summary.json`), raw replay records in `results/`, and logs in `logs/`.

| What | Records | Evidence |
|---|---|---|
| Independent recompute of the headline, SLO scales, lags 10/50/200 and timing (nominal E0), from raw records | 180 per policy per condition | `scripts/recompute_records.py`, `out/recompute_records.json` |
| Own SLO-scale recompute (0.5, 0.75, 1.5) from rows, compared with the records' `rescore` | 360 | `out/rows_scales.jsonl`: all match |
| Warm-up and window variants on test, rescored from rows (M1-v2, ramjet, llmdpp, default) | 720 | `scripts/rescore_rows.py`, `out/rows_nominal.jsonl`, `out/warmup_variants.json` |
| Common fixed window on completion-basis test cells | 348 | `scripts/rescore_common_window.py`, `out/common_window_completion_cells.json` |
| Timing records rescored with nominal E0, with E0', and with E0' plus I/(s·d) | 2,880 | `out/rows_timing.jsonl`, `out/timing_consistent.json` |
| Sub-E0 signature, mixing proxy and engine-state proxy on test | 720 each | `out/sub_e0_test_nominal.json`, `out/mixing_test_summary.json`, `out/engine_state_test_summary.json` |
| Lag build `a6c6b04e`, val k3-10, lag 0 and 1,000 ms | 896 | `results/val_k3_10_lag.jsonl` |
| Audit simulator build (mixed-prefill pricing `mean`/`attn`/`sum`), val k3-10 | 3 × 448 | `results/val_k3_10_prefill_{mean,attn,sum}.jsonl` |
| Audit build: decode overhead of 0.1 ms per running sequence per step, val k3-10 | 448 | `results/val_k3_10_dec0.1.jsonl`, `out/val_decode_overhead_consistent.json` |
| Audit build: a decode step inside a prefill pass charged at 0.25×, val k3-10 | 448 | `results/val_k3_10_mixdec0.25.jsonl` |
| Identity of every audit build (default env) and of the lag build at lag 0, against the campaign cache | 448 + 448 + 56 + 56 + 4 | `out/identity_*.json`: all identical |

**The audit simulator build.**
- **Code.** A detached worktree, `<other-worktree>`
  at the tuning HEAD <commit-44> (same Rust sources as the tuning build), with its own `.venv`. Its
  bindings Cargo.toml adds a `[patch.crates-io]` to a vendored copy,
  `CR/vendor/aisimulate-core-prefill-audit`, of aisimulate-core `=0.13.0-dev.202609300000000061`.
- **Knobs.** Three env knobs were added to `engine/scheduler/vllm/core.rs`:
  - `AISIM_AUDIT_PREFILL_PRICING`: `mean` (upstream), `attn` (attention-equivalent mean prefix: same
    Σn and same Σn(p + n/2)), or `sum` (sum of per-request batch-1 predictions). Single-request and
    uniform passes always take the upstream path.
  - `AISIM_AUDIT_DECODE_PER_SEQ_MS`;
  - `AISIM_AUDIT_MIXED_DECODE_FRAC`.
- **Identity.** With the env unset, every one of the three successive builds reproduces the tuning
  build byte for byte: per-request canonical SHA-256, `goodput_rps_window` and `window_good` match on
  448/448, 56/56 and 56/56 val records. So every difference below comes from the knob alone. The
  tuning build's `_core.abi3.so` is untouched (sha256 1d67c131…).

## Results

Headline statistic throughout: M1-v2 minus ramjet, segment mean of the clipped log-ratio, with the
one-sided exact Wilcoxon over segments (HEADLINE_TEST pipeline, own implementation in
`scripts/astats.py`).

### 1. The test-stage numbers reproduce

My recompute from raw records matches `facts/test_results.json` and `facts/robustness.json`:

| Condition | Δ | Segments ahead | p |
|---|---|---|---|
| Nominal | +0.0316 | 10/12 | 0.0017 |
| Lag 10 ms | +0.0312 | 9 | 0.0081 |
| Lag 50 ms | +0.0288 | 10 | 0.0024 |
| Lag 200 ms | +0.0304 | 9 | 0.0061 |
| s0.8 on E0' | +0.0814 | 9 | 0.0105 |
| s1.2 on E0' | +0.0219 | 10 | 0.0046 |
| d0.8 on E0' | +0.0391 | 10 | 0.0024 |
| d1.2 on E0' | +0.0304 | 9 | 0.0105 |

The E0' scoring is my own implementation of prefill/s + decode/(s·d) from the E0 table.

### 2. Timing, with the ITL bound made consistent as well

E0' re-anchors only the E2E bound. The ITL bound I stays absolute, so a slower engine still faces a
tighter ITL SLO. Rescaling it to I/(s·d) changes nothing material:

| Condition | Nominal E0 | E0' | E0' and I/(s·d) |
|---|---|---|---|
| s0.8 | +0.0790 (p 0.117) | +0.0814 (p 0.0105) | +0.0801 (p 0.0081) |
| s1.2 | +0.0151 (p 0.0081) | +0.0219 | +0.0225 (p 0.0046) |
| d0.8 | +0.0473 (p 0.026) | +0.0391 | +0.0378 (p 0.0017) |
| d1.2 | +0.0224 (p 0.013) | +0.0304 | +0.0310 (p 0.0081) |

### 3. Warm-up length and window boundaries (test, rescored)

For open loop the replay does not depend on the scoring window. Starting the window later is therefore
exactly a longer warm-up with a shorter window. For closed loop and lanes it lengthens the history in
the same way.

| Variant | Cells / segments | Δ | Ahead | p | Nominal on the same cells |
|---|---|---|---|---|---|
| Window starts 25% later | 60 / 12 | +0.0345 | 10 | 0.0024 | +0.0316 |
| Window starts 50% later | 60 / 12 | +0.0297 | 12 | 0.0002 | +0.0316 |
| Warm-up halved, arrival basis | 31 / 8 | +0.0309 | 6 | 0.074 | +0.0281 (p 0.074) |
| Warm-up doubled, where it fits | 22 / 4 | +0.0652 | 4 | 0.0625 | +0.0565 |
| Common fixed window, 0.75 × the shortest of 4 policies, completion basis | 29 / 10 | +0.0361 | 9 | 0.0029 | +0.0362 |

- The doubled-warm-up row fits only Mooncake/FAST25 open loop.
- The common-window row removes the policy-dependent full-occupancy end: window lengths of the same
  (cell, k) differ by up to 54% across the four policies.

The lead does not live in the warm-up transient, and the full-occupancy window end does not drive it.

### 4. Router-state lag beyond the registered range (val k3-10, lag build)

| Lag | M1-v2 − ramjet | Mooncake | AgentX | Sessions | M1-v2 − llmdpp |
|---|---|---|---|---|---|
| 0 ms (identical to the tuning build, 448/448) | +0.0290 | +0.079 | +0.032 | −0.0004 | +0.0203 |
| 1,000 ms | +0.0333 | +0.100 | +0.033 | −0.0001 | +0.0320 |

Staleness 5× beyond the largest registered lag widens the lead: the baselines lose more than M1-v2
does (vs default@defaults: M1-v2 0.178 → 0.159, ramjet 0.141 → 0.114, llmdpp 0.139 → 0.097).

### 5. Simulator-model counterfactuals (val k3-10, paired replicates)

| Simulator variant | M1-v2 − ramjet | Ahead | Mooncake | AgentX | Sessions | M1-v2 − llmdpp |
|---|---|---|---|---|---|---|
| Upstream AISim | +0.0290 | 5/6 | +0.079 | +0.032 | −0.0004 | +0.0203 |
| Mixed prefill pass priced attention-equivalent (`attn`) | +0.0274 | 4/6 | +0.080 | +0.028 | −0.0006 | +0.0209 |
| Mixed prefill pass priced as a per-request sum (`sum`) | +0.0291 | 5/6 | +0.082 | +0.031 | −0.0006 | +0.0220 |
| +0.1 ms decode per running sequence per step | +0.0268 | 4/6 | +0.084 | +0.027 | −0.0019 | +0.0180 |
| Same, with a consistent SLO (E0 + 0.1·(OSL−1), I + 0.1) | +0.0269 | 4/6 | | | | +0.0180 |
| Decode inside a prefill pass charged at 0.25× (fused-forward proxy) | +0.0269 | 5/6 | +0.067 | +0.032 | −0.0002 | +0.0202 |

The mixed-prefill-pass discount (phase2-mid F4, `UPSTREAM_FOLLOWUPS` #17) does real work in AISim:
under `attn`, all 448 records change. It accounts for at most 0.0016 of the lead (0.0035 on AgentX).

The other two variants are hypotheses about reality, not measurements: a batch-dependent decode cost,
which penalizes M1-v2's concentration, and a fused prefill+decode pass, which the upstream additive
pricing ignores. They bracket plausible fidelity errors that act on M1-v2's specific mechanism. The
lead moves by at most 0.0022 across the four simulator models (and the consistent-SLO rescoring). The 1,000 ms lag (§4) moves it by
+0.0043.

### 6. Exploit signatures and engine state (test, from rows)

- **Sub-E0 rows.** These are completed zero-reuse requests that finish below their uncontended E0, the
  F4 discount signature. M1-v2 has fewer than either baseline:

  | Family | M1-v2 | ramjet | llmdpp | default@defaults |
  |---|---|---|---|---|
  | AgentX | 43 | 90 | 214 | 79 |
  | FAST25 synthetic | 28 | 70 | 226 | 17 |
  | Mooncake and sessions | 0 | 2 | 3 | 0 |

- **Mixing proxy.** This is the share of short admissions (fewer than 2K uncached tokens) that land on
  a worker during another request's ≥ 16K-token prefill:

  | Family | M1-v2 | ramjet | llmdpp |
  |---|---|---|---|
  | AgentX | 0.067 | 0.049 | 0.069 |
  | Mooncake | 0.016 | 0.036 | 0.077 |

  M1-v2 does not systematically pair short prompts with long chunks.
- **Engine state.** This is the LR-14 action 4 proxy (in-flight requests per worker, time-weighted over
  the window).
  - On Mooncake, M1-v2's hottest worker carries 1.85× the per-worker mean (ramjet 1.04, default 1.18).
    On FAST25 conversation and synthetic the ratios are 2.21× and 1.67×. The peak is 59 in flight on
    Mooncake and 65 on FAST25 conversation.
  - That stays inside AIS's measured vLLM 0.24 H100 decode-attention grid (batch 1–2048, context up to
    131,071). It is also far below `max_num_seqs` 1024.
  - Concentration is real (it matches the LR-13 cap flags), but it does not push the engine into an
    AIS extrapolation regime.

## Findings

### F1 (minor): the headline is SLO-scale-dependent, and sessions reverse at tighter scales

- **Claim.** The win is significant at scales 1 to 3 only.
  - At 0.75× it is +0.0353 with 8/12 segments ahead and p = 0.117.
  - At 0.5× it is +0.0716 with p = 0.065.

  My own recompute from rows matches the records' `rescore` on 360/360 records.
- **What test_results.json does not show.** At 0.75× all four sessions segments turn against M1-v2:
  s6 −0.014, s7 −0.039, s8 −0.042, s9 −0.035. The same reversal appears at speedup 0.8 under nominal
  E0 (s6 −0.013 to s8 −0.048, overall p = 0.117).
  - On the consistent E0' at speedup 0.8, sessions return to about 0.
  - With 0.1 ms/seq decode overhead on val, sessions go to −0.0013 and −0.0022.
  - So "sessions at ceiling" is true only where the SLO does not bind. Once it binds, ramjet is better
    on sessions.
- **Low end.** At 0.5×, AgentX T3/T4 are −0.285/−0.174. This is the degenerate low end that
  phase2-mid F5 described: scale × S < 1, so only cache hits can be good.
- **Pre-registration.** HEADLINE_TEST lists the SLO sweep as "reported, not tested", so this is no
  test failure. LR-08's "must hold at one tighter scale" is not met for significance. The Select+Test
  summary already discloses the loss of significance.
- **Fix.** REPORT words the headline as holding "at the calibrated SLO and looser (scales 1–3)". Where
  it says sessions are at ceiling, it adds: "at 0.75× (and at speedup 0.8 on nominal E0) M1-v2 trails
  ramjet on all four sessions segments". It also notes that the AgentX low end is degenerate.

### F2 (minor): Select+Test did not run the M1-v2 mixed-pass exploit check; this audit closes it

- **Claim.** `UPSTREAM_FOLLOWUPS` #17, from phase2-mid F4, says: "Before trusting arms with
  prefill-attention … features (M1-v2 `prefill_attn` …), check they do not learn to pair long chunks
  with short prompts". The headline arm is M1-v2 (θ22 = −0.297 on `prefill_attn`). Neither
  `facts/test_results.json` nor `facts/robustness.json` contains such a check.
- **Evidence.** §5 and §6: two corrected pricings move the lead by −0.0016 and +0.0001. M1-v2 has the
  fewest sub-E0 rows of the three routers. The mixing proxy shows no pairing preference.
- **Fix.** REPORT cites this audit for the check. The AISim follow-up (#17) stays, because the discount
  is real and affects every record.

### F3 (minor): lag-0 identity was not proven for the headline pair; this audit closes it

- **Claim.** `robustness.json` `lag0_identity_lag_build_vs_tuning_build` covers only default@defaults
  and m1. The lag rows of M1-v2, ramjet and 13 other policies rely on the lag build's
  `learned-choice` v2 and ramjet code matching the tuning build's.
- **Evidence.** Lag build `a6c6b04e` at lag 0 equals the tuning build per request on val k3-10 for
  m1v2, ramjet, llmdpp and default: 448/448 (`out/identity_lag0_vs_tuning.json`). The sources agree
  as well: `learned_choice/` was last changed in <commit-08>, an ancestor of the lag base <commit-21>.
- **Fix.** Cite this file next to the lag table in REPORT.

### F4 (minor): the LR-14 engine-state comparison is missing from the test-stage facts

- **Claim.** LR-14 action 4 asks REPORT to compare engine-state distributions (batch, context, KV) for
  the learned and default policies. It also asks to flag shifts into weakly validated AIS regions.
  Neither test-stage fact file has it.
- **Evidence.** §6: concentration of up to 2.2× on the hot worker, within the AIS grid. §5: a
  batch-dependent decode-cost stress does not remove the lead.
- **Fix.** REPORT includes `out/engine_state_test_summary.json` (or an equivalent) and the 0.1 ms/seq
  stress, labeled as a hypothesis-level counterfactual.

## Not findings (checked and confirmed)

- **Magnitude.** The effect is below the MDE (0.032 against 0.038), and the LR-14 timing spread
  (0.060) exceeds it. Both are already disclosed. My additional conditions (lag 1,000 ms, four
  simulator models, window variants) spread far less (val 0.0268–0.0333, test +0.0297 to +0.0345),
  so the timing perturbations dominate the spread. Only the sign is a claim.
- **Sessions near zero.** The four sessions segments, at about 0, do not carry the Wilcoxon p. The
  eight non-sessions segments are all positive at nominal (minimum +0.005, fast25_synthetic:x1).
- **Coverage.** The FAST25 family exists only in test, so none of my simulator counterfactuals covers
  it. FAST25 synthetic x0 is a large contributor (+0.056) and the A19 outlier (−0.29). That residual
  simulator dependence is untested; the A13.2 live runs are the place to test it.
- **Simulated only.** Everything here is AIS-timed simulation. Agreement across simulator variants is
  not live evidence (LR-14).

## Housekeeping

- Nothing was deleted, nothing was committed, and the test split was not replayed.
- **Scratch, listed in `CLEANUP.md`:**
  - the audit worktree (5.8G, including a 3.4G target copy and its own `.venv`), with an uncommitted
    Cargo.toml/Cargo.lock patch only;
  - `CR/vendor/aisimulate-core-prefill-audit` (7.8M);
  - `runs/audits/final-refute-headline-3` (3.2G, mostly private-root caches and 8 copies of the
    50 MB E0 table).
- **Slots.** Replays used the shared `CR/slots` pool, alongside the concurrent refute-headline-2
  auditor.
- **Processes.** All my background PIDs (run scripts 1182012, 1186559, 1202158, 1220357; maturin
  1180572, 1200461, 1217091) have exited.
- **Lessons applied.**
  - LR-01, action 5: the warm-up length check.
  - LR-08, action 2: the SLO-scale sweep. Its "must hold at one tighter scale" is reported in F1.
  - LR-13: the concentration and engine-state context.
  - LR-14: lag beyond range, timing with a consistent SLO, the engine-state comparison, and
    simulator-model counterfactuals.
