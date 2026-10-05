# Audit phase2-mid, lens "overfit-selection" (tier A complete)

**Verdict: FAIL.** 0 blockers, 1 major, 4 minor.

The selection, budget and test-hygiene results hold up under independent recomputation and my own
replays. The failure is the LR-15 drift check:

- Its numbers reproduce exactly.
- Its recorded outcome, `within_mde: false` in `facts/tierA.json`, is not evidence of coefficient
  drift. The single above-MDE loss comes from an undertrained, underpowered specialist, and the
  pooled M1 beats every specialist on that specialist's own regime.

The fix is a correction to the record. No tier-A rerun is needed.

- **Auditor:** independent adversarial auditor, checkpoint phase2-mid, lens overfit-selection.
- **Date:** 2026-10-03, 19:53 to about 21:00 PDT.
- **Evidence directory:** `CR/runs/audits/phase2-mid-overfit-selection/` (`scripts/`, `out/`, `logs/`).
  - `out/SUMMARY.json` collects every number quoted below.
  - Each number also traces to the per-check file named next to it.
- **Lessons applied:**
  - LR-10: selection over restarts × checkpoints, the best-ever-sample bias, and the "extend if
    still improving" rule.
  - LR-11: segment-level view, counting arms, val-to-test.
  - LR-15: the drift check.
  - LR-03: CRN pairing.
  - LR-01: the clipped log-ratio, recomputed with my own code.

## What I ran (own evidence)

| # | Check | Evidence |
|---|---|---|
| 1 | Static audit of all 37 tier-A histories: 33 train runs and 4 drift runs. I took each run's directory from the authoritative `state.json` attempt and checked budgets, the k schedule, cell sets, val ks, the selection recompute and restart spreads. Stdlib only. | `scripts/hist_audit.py` → `out/hist_audit.json` |
| 2 | Own replays of all 12 selected configs and of each policy's own CMA start x0 (13 specs) on **train**: 34 cells × k0-3, 3,536 replays. | CPU cluster job <job> on <node>, COMPLETED in 24:39 with 0 errors. 3,234 records ingested and 302 already cached. Results in `out/own_train.jsonl`. |
| 3 | The same 25 specs on **val**: 14 cells × k0-2. | Local slot pool, lr-eval PID 1737402: 1,092 tasks, of which 546 were cached (default and the selected configs) and 546 fresh (the x0 specs). 0 errors. `out/own_val.jsonl` |
| 4 | **Fresh private root** with empty cache, replicates and E0: re-replayed the selected M1, ramjet, llmdpp and default on 3 val cells × k0,1. | `scripts/run_private.sh`, PID 1747464. Compared in `out/private_vs_cache.json`: **24/24 identical** with the campaign cache on all 69 record fields, including `per_request_canonical_sha256` and `goodput_rps_window`. |
| 5 | Post-reboot image check of the CPU cluster node I used. | Inside my own job: same libc6 (2.39-0ubuntu8.9) and the same libm.so.6 md5 (85d102c2…) as the workstation (`out/node_image_<job>.json`). Parity against independent local records: 166 matched, 136 identical. The other 30 differ only in the `policy_name` label. 1,116,108 per-request rows are byte-identical (`out/parity_<node>.json`). |
| 6 | Train-vs-val gap analysis, with a segment cluster bootstrap. | `scripts/gap_analysis.py` → `out/gap_analysis.json` |
| 7 | Val replicate noise, and M1's lead by segment, with leave-one-segment-out. | `scripts/val_noise.py` → `out/val_noise.json`, `out/m1_lead_by_segment.json`, `out/ramjet_vs_llmdpp.json` |
| 8 | Test freeze: every hash in `facts/test_freeze.json`, cell content SHAs, trace-level and segment-level split disjointness. | `scripts/verify_freeze.py` → `out/verify_freeze.json` |
| 9 | Test-touch scan of all 1,102,687 cached records, matching on test cell id, test cell content SHA and test trace SHA-256. Also scanned the materialized replicates, the phase-2 outputs and the bundles. | `scripts/scan_cache.py` → `out/scan_cache.json`, `out/scan_replicates.json` |
| 10 | Inventory of all val records with k ≥ 3, the "fresh" pool for Select+Test. | `scripts/scan_val_fresh.py` → `out/scan_val_k3_10.json` |
| 11 | LR-15 drift check: recomputed the matrix from raw records, computed per-cell SEs and the distance each theta moved from θ0, and compared pooled M1 with the regime specialists. | `scripts/drift_audit.py` → `out/drift_audit.json`, `out/drift_vs_pooled_k0_2.json` |

My objective code is independent stdlib: clip(ln((m + 1e-3)/(m_ref + 1e-3)), ±ln 3) on
`goodput_rps_window`, paired with default@defaults on the same cell and k. It reproduces the
stage's lr-train val objective exactly (|Δ| < 1e-12) for all 12 selected rows (`own_val_equals_stage`).
It also reproduces a best-so-far sample's in-run train values: ramjet-s3's generation-24 sample
scores 0.14347 on k0,1 and 0.13868 on its re-evaluation on k2,3, and my k0-3 value is 0.14107, their mean.

## Findings

### F1 (major): the LR-15 drift result is recorded as drift beyond the MDE, but the check cannot show drift

- **Claim.** `facts/tierA.json` → `drift` records `within_mde: false` under the rule "LR-15: if every
  cross-play loss is within the MDE, deprioritize M2". It carries no qualifier; the "undertrained"
  hypothesis appears only in relay i1's note, and nobody tested it. Read as LR-15 intends, the record
  says coefficients drift with load level, which triggers LR-15 action 1 ("add the drifting statistic
  to z_S"). `runs/phase2/spaces/m2_rank2.yaml` has not been written yet. The evidence does not
  support a drift reading.
- **The computation is correct.** My recompute from `runs/phase2/drift/val_k3_10.jsonl` matches the
  stage's matrix to 0.0. It covers 560 records with 0 errors and complete k3-10 coverage for all 4 thetas.
- **The single above-MDE loss is not significant, and one cell drives it.**
  - "l1-tuned on l3" = +0.0865 over **3** L3 val cells, with SE 0.0645 across cells, i.e. 1.3 SE.
  - The per-cell losses are +0.2148 (mooncake-w1-n8-open-L3), +0.0342 and +0.0105.
  - The other three losses are −0.0067 (2 cells), −0.0043 (5 cells) and +0.0021 (5 cells).
  - The MDE of 0.038 was measured for the full 6-segment val set. These columns have 2 to 5 cells.
- **The specialists are undertrained, and confounded with budget.** Each drift run had B/4 = 96
  evaluations. In feature-scale units ‖θ_j / s_j‖, the distance each moved from θ0 was:

  | Theta | Distance moved |
  |---|---|
  | drift-l1 (selected at generation 3, i.e. after 48 evaluations) | 0.08 |
  | drift-l3 | 0.25 |
  | drift-n8 | 0.26 |
  | drift-n4 | 0.28 |
  | m1-s1 after 400 evaluations (same θ0 and σ0) | 0.83 |
  | m1-s1's own CMA mean after 80 evaluations | 0.21 |

  So the l1→l3 "loss" mostly measures how far the least-trained theta got.
- **One pooled θ beats every specialist on the specialist's own regime.** This is what LR-15 asks: does
  a single θ suffice across regimes? On val k0-2, on exactly each specialist's own val cells, the
  pooled tier-A M1 (m1-s2 g15) scores:

  | Regime | Pooled M1 minus specialist |
  |---|---|
  | L1 | +0.0125 |
  | L3 | +0.0216 |
  | N4 | +0.0070 |
  | N8 | +0.0137 |

  The A12 row (m1-s1 g25) scores +0.0066, +0.0215, −0.0010 and +0.0151. A single θ is at least as
  good everywhere, which is the "no drift" case. The pooled runs had more budget and more cells; that
  is the same confound that makes the stage's matrix uninterpretable as drift.
- **Impact.**
  - Compute decisions are unaffected: A17.3 makes M2 rank 2 never-cut, so the "deprioritize M2" branch is moot.
  - The risks are in M2's context-source design and in the REPORT narrative. "The LR-15 drift check
    found a cross-play loss of 0.087 > MDE at L1→L3" would mislead.
- **Fix (documentation only; no reruns).**
  - Amend `facts/tierA.json` `drift`: record it as inconclusive, with no evidence of drift. Give the
    reasons (1.3 SE over 3 cells, undertrained specialists) and add this audit's pooled-vs-specialist
    table (`out/drift_vs_pooled_k0_2.json`).
  - Do not choose M2 sources on the strength of "load-level drift".
  - REPORT should describe the LR-15 check as underpowered at B/4.
  - If the operator wants a real drift test, rerun the specialists at full B on their subsets. That
    is optional and not required for the headline.

### F2 (minor): selection margins are near ties, M1's val lead rests on one segment, and the val-best baseline flips with weighting

These are pre-registered rules applied correctly, so this is not a stage error. They are risks the
Select+Test stage should pre-register around before it touches test.

- **Within-policy selection is among near ties.**
  - The selected candidate's margin over the runner-up is 0.0001 to 0.0020 for every policy.
  - The paired replicate SE of a val k0-2 objective is 0.0029 to 0.0056.
  - So the selected val k0-2 values are optimistic by at most about one SE. The table is below.
- **M1's val lead over the strongest baselines comes almost entirely from mooncake:w1.** That is one
  workload and one segment, with 5 of the 14 val cells. See the M1-versus-baselines table below
  (`out/val_noise.json`).
  - Against llmdpp: cell-mean +0.0222, but segment-mean +0.0085 (SE 0.0114), M1 ahead on 3 of 6
    segments. It trails on all three AgentX segments (−0.009, −0.012, −0.005). Without w1 the lead is −0.0006.
  - Against twotier and lmetric, the lead without w1 is also below 0 (−0.0030 and −0.0008).
  - Train confirms the pattern is family-wide and not w1-specific. M1's lead over llmdpp is +0.076,
    +0.058 and +0.039 on mooncake w0, w2 and w4, and −0.013 to +0.014 on the AgentX segments
    (`out/m1_lead_by_segment.json`).
  - So this is not overfitting to w1. It does mean the headline rests on mooncake-like segments.
- **The "val-best baseline" is a coin flip, and its winner depends on the weighting.** ramjet minus llmdpp on val k0-2:
  - cell-mean: +0.0006 (paired replicate SE 0.0016);
  - segment-mean: −0.0084, because llmdpp is better on all three AgentX segments (`out/ramjet_vs_llmdpp.json`);
  - on train k0-3, ramjet leads +0.0079 (SE 0.0014).

  The pre-registered val-best rule uses the cell mean on k3-10, where w1 weighs 5/14. The headline
  statistic is a Wilcoxon over 12 equally weighted test segments, 4 of them AgentX. M1 against ramjet
  (5 of 6 val segments ahead) and M1 against llmdpp (3 of 6) are materially different comparisons.
- **Recommendation for Select+Test, registered before the test pass:**
  - report the headline test against the val runner-up baseline as well, in particular both ramjet
    and llmdpp, as a secondary row;
  - give the segment-level val table next to the cell-mean selection.

  HEADLINE_TEST.json's "per-cell winner map and gap to the per-cell best tuned baseline" already
  covers part of this.

### F3 (minor): the UH re-evaluation diagnostic is not "identical for every run"

- `runs/phase2/MANIFEST.json` `common.budget` says the top-2 re-evaluation is "identical for every run".
- The histories show re-evaluations in only **13 to 23 of the 25 generations** per train run (for
  example chwbl-s2 13, m1-s2 23; `out/hist_audit.json` `reeval_generations`), because lr-train skips
  it when a chunk deadline falls inside it.
- It does not feed CMA-ES or selection. `train.py` 474-495: `tell()` and `best` use the
  first-evaluation values, and `rank_change` is written only to history. Budgets are unaffected.
- **Fix:** correct the wording in REPORT, and do not use `rank_change` as a per-policy noise
  statistic without normalizing by the generations covered.

### F4 (minor): evaluations outside B must be counted per method (LR-10 action 3, LR-11)

B = 400 per run is exactly equal (see below). Several design-time inputs differ by policy, and REPORT
must count them:

- **M1 and M0:** pilot CMA-ES runs of 4 × 480 evaluations on the 12-cell train subset shaped their
  spaces (σ0, bounds, maxstd).
- **M1-s2:** its init is a behavior clone fitted to 34 llm-d-precise-prefix@defaults train-k0
  records. The BC target was chosen as the val-best default-arm heuristic at the pilot (val k3-10).
- **The other baselines:** one-step flip calibration on reconstructed tables, plus 70 corner smoke replays.
- **The A12 row** (m1-s1 from −e0: val 0.16148, train 0.1700) shows M1's lead over every baseline
  does not depend on the BC init. Disclose this in REPORT's fairness section along with the counts.

### F5 (minor): val k3-10 is fresh for the selected configs, but not pristine, and it shares cells with k0-2

- **No selected config has val k ≥ 3 records.** None of the 11 selected configs or the A12 row has
  any val record with k ≥ 3 in the cache (`out/scan_val_k3_10.json`; 1,845 such records in total). The
  Select+Test re-check is therefore fresh for every candidate it compares.
- **Earlier phases did use val k3-10:**
  - the pilot gate: default, round_robin, ramjet, lmetric and llm-d-precise-prefix at defaults, and
    pilot m0-s1, m0-s2, m1-s1 and m1-s2;
  - the 4 drift thetas;
  - design decisions: choosing the BC target, and A11's AgentX-targeted s3 init.

  REPORT should not call val k3-10 untouched.
- **k3-10 differs from k0-2 only in CRN arrival-order permutations and policy seeds**, on the same
  14 cells and 6 segments (A1). It removes replicate-level selection optimism (≈ 1 SE here), but not
  cell- or segment-level optimism. Only the single test pass does that.

## Checks that passed

### Equal budgets, counted from the histories

- **Train runs.** All 33 have the same signature: 25 contiguous generation lines (0..24, no
  duplicates, including the runs moved across 2 or 3 hosts) × popsize 16 = **400 candidate
  evaluations**, plus **10 val candidates** (5 checkpoints × best_so_far and mean).
  - Every generation was evaluated on exactly the 34 train cells, with ks (2g, 2g+1) mod 8.
  - No failed candidates.
  - `identity.json` cell lists match the split files.
  - Violations: 0.
- **Drift runs.** All 4 have 96 evaluations (6 × 16) on their filtered train subsets (6, 8, 18 and
  16 cells), with 2 val checkpoints on their own val subsets (2, 3, 5 and 5 cells).

### Selection used only val k0-2

- All 330 tier-A val lines have `val_ks = [0,1,2]` and exactly the 14 val cells. Each `val_objective`
  equals the mean of its `val_per_cell`.
- Each run's `best.json` `selected_by_val` is the argmax of its own val lines.
- My per-policy argmax over 30 candidates (3 restarts × 10) equals `runs/phase2/selection/tierA.json`
  and `facts/tierA.json` for all 11 policies. The A12 row (m1-s1 g25 mean, 0.16148) also matches.
- My own replays reproduce every selected val objective exactly, and the fresh private root
  reproduces the underlying records byte for byte (24/24).

### No test cell was touched

- **Freeze.** `cells/test.jsonl` hashes to 7b998b8e… as frozen.
  - All 89 freeze hashes match: traces and their metadata, sources, engine, split manifest and
    traces manifest.
  - All 60 cell content SHAs match.
  - No train or val cell references a test trace file or SHA.
  - Test segments are disjoint from train and val segments.
  - `train.jsonl` (55b54ec6…) and `val.jsonl` (e61ac577…) are as recorded in `SPLIT_MANIFEST.json`
    (`out/verify_freeze.json`).
- **Cache.** 0 of 1,102,687 cached records match a test cell id, a test cell content SHA or a test
  trace SHA-256 (`out/scan_cache.json`, scanned 19:58 PDT). The newest record was 19:50:26, so the
  scan covered all of tier A.
- **Phase-2 outputs and bundles.**
  - 0 test cell ids in `runs/phase2`, `runs/phase2-ais`, `runs/remote/returned/p2a` and my own outputs.
  - The p2a train bundle holds 34 train and 14 val cells, and 0 test cells.
- **Replicates.** Only two sets of materialized replicate files derive from test traces. Neither
  involved a replay or an evaluation (`out/scan_replicates.json`):
  - 5 files from 2026-10-02 16:57, before the freeze (build stage);
  - the live loadgen parity audit's documented conv-w3 and fast25-x0 input files (10-03 10:47),
    marked "not replayed, to avoid test exposure" in `facts/live_loadgen_parity.json`.

### Train versus val (k0-2): no overfitting signature

The table compares each selected config with its own CMA start x0 on fresh common replicates. Here:

- **Tuning gain** = selected minus x0 on the same split. Comparing gains cancels the difference in
  cell composition between train and val. Val is "easier": most policies score higher on val.
- **Gap** = train gain minus val gain. Its 95% CI is a segment cluster bootstrap.

| Policy | Train k0-3 | Val k0-2 | x0 | x0 train | x0 val | Gain train | Gain val | Gap (95% CI) |
|---|---|---|---|---|---|---|---|---|
| m1 (m1-s2 g15) | 0.1662 | 0.1636 | s2 BC init | 0.1401 | 0.1360 | +0.0261 | +0.0277 | −0.0015 [−0.028, +0.026] |
| m1_a12 (m1-s1 g25) | 0.1700 | 0.1615 | −e0 | 0.0039 | −0.0050 | +0.1660 | +0.1665 | −0.0004 [−0.074, +0.109] |
| ramjet | 0.1411 | 0.1420 | defaults | 0.1337 | 0.1348 | +0.0073 | +0.0072 | +0.0001 [−0.005, +0.008] |
| llmdpp | 0.1332 | 0.1414 | defaults | 0.1253 | 0.1407 | +0.0079 | +0.0007 | +0.0072 [−0.002, +0.016] |
| stickybounded | 0.1278 | 0.1376 | defaults | 0.0708 | 0.0818 | +0.0570 | +0.0558 | +0.0012 [−0.065, +0.071] |
| twotier | 0.1263 | 0.1363 | defaults | 0.1199 | 0.1313 | +0.0064 | +0.0049 | +0.0014 [−0.007, +0.010] |
| m0 | 0.1266 | 0.1334 | defaults | −0.0021 | 0.0001 | +0.1287 | +0.1333 | −0.0046 [−0.067, +0.087] |
| lmetric | 0.1219 | 0.1334 | defaults | 0.1227 | 0.1317 | −0.0008 | +0.0017 | −0.0025 [−0.007, +0.003] |
| stickyhard | 0.1128 | 0.1160 | defaults | 0.0562 | 0.0563 | +0.0567 | +0.0597 | −0.0030 [−0.074, +0.070] |
| llmdob | 0.0855 | 0.1047 | defaults | 0.0859 | 0.1008 | −0.0004 | +0.0038 | −0.0042 [−0.026, +0.009] |
| dualmap | −0.0062 | 0.0402 | defaults | −0.1958 | −0.0825 | +0.1896 | +0.1227 | +0.0670 [−0.055, +0.169] |
| chwbl | −0.0374 | 0.0290 | defaults | −0.1571 | −0.1517 | +0.1196 | +0.1806 | −0.0610 [−0.211, +0.070] |

**M1 and A12.** Neither shows overfitting: its tuning gain carries over from train to val one for
one (−0.0015 and −0.0004).

**Every baseline CI includes 0.**
- llmdpp's +0.0072 is the largest gap among the strong policies.
- dualmap and chwbl are noisy, far behind, and irrelevant to the headline.
- lmetric and llmdob gained nothing from tuning on either split (|gain| ≤ 0.004). Their shipped
  parameters barely move decisions, which matches the flip calibration (σ0 capped at 0.3).

**M1's lead over each baseline is consistent by family between train and val.** For example,
against ramjet: mooncake +0.045 on train vs +0.049 on val; AgentX +0.008 vs +0.015; sessions
−0.002 vs −0.000 (`out/gap_analysis.json` `m1_vs_baselines`).

**Trajectories.** M1's val keeps rising with train for s1 and s3. s2 peaked at generation 15
(0.1636) and drifted to 0.1604 by generation 25 while its train best rose. That move is about
1 replicate SE.

chwbl-s1's val peaked at generation 5 while train kept improving. The LR-10 val selection correctly
kept the early checkpoint.

### Restart gaps and selection margins (val k0-2)

| Policy | Selected | Val k0-2 | Margin to runner-up | Restart spread | Replicate SE |
|---|---|---|---|---|---|
| m1 | m1-s2 g15 mean | 0.16363 | 0.00103 | 0.0058 | 0.0032 |
| ramjet | ramjet-s3 g25 best_so_far | 0.14202 | 0.00013 | 0.0002 | 0.0035 |
| llmdpp | llmdpp-s2 g25 mean | 0.14145 | 0.00203 | 0.0030 | 0.0035 |
| stickybounded | stickybounded-s2 g25 mean | 0.13758 | 0.00123 | 0.0034 | 0.0037 |
| twotier | twotier-s1 g15 mean | 0.13628 | 0.00081 | 0.0017 | 0.0032 |
| m0 | m0-s3 g20 mean | 0.13343 | 0.00045 | 0.0016 | 0.0037 |
| lmetric | lmetric-s2 g10 mean | 0.13341 | 0.00044 | 0.0005 | 0.0034 |
| stickyhard | stickyhard-s1 g25 mean | 0.11604 | 0.00202 | 0.0022 | 0.0033 |
| llmdob | llmdob-s1 g5 mean | 0.10466 | 0.00111 | 0.0018 | 0.0038 |
| dualmap | dualmap-s3 g10 mean | 0.04017 | 0.00181 | 0.0082 | 0.0056 |
| chwbl | chwbl-s1 g5 mean | 0.02896 | 0.00097 | 0.0060 | 0.0046 |

**M1's restart spread includes an init effect.** It is 0.0058, about 1.8 replicate SE, and the three
restarts used three different inits (A11.1).

**The selected restart and A12 are a noise-level choice between near-equals:**
- val: s2 (BC init) beats s1 by +0.0021 (paired replicate SE 0.0016; segment-mean +0.0072,
  SE 0.0050);
- train k0-3: s1 is ahead by 0.0038.

**M1's val lead over the best selected baseline is large next to all of these.** It is +0.0216 over
ramjet (paired replicate SE 0.0014), against restart spreads and selection margins of ≤ 0.008 and
≤ 0.002. F2 covers the segment-level caveat.

### M1 against each selected baseline, val k0-2, paired

| Baseline | Cell-mean lead | Replicate SE | Segment-mean lead | Segment SE | Segments M1 ahead | Leave-one-segment-out minimum |
|---|---|---|---|---|---|---|
| ramjet | +0.0216 | 0.0014 | +0.0169 | 0.0086 | 5/6 | +0.0065 |
| llmdpp | +0.0222 | 0.0020 | +0.0085 | 0.0114 | 3/6 | −0.0006 |
| stickybounded | +0.0261 | 0.0023 | +0.0183 | 0.0105 | 4/6 | +0.0062 |
| twotier | +0.0273 | 0.0015 | +0.0122 | 0.0144 | 3/6 | −0.0030 |
| m0 | +0.0302 | 0.0021 | +0.0184 | 0.0107 | 6/6 | +0.0074 |
| lmetric | +0.0302 | 0.0015 | +0.0167 | 0.0150 | 3/6 | −0.0008 |

The remaining baselines (stickyhard, llmdob, dualmap, chwbl) trail M1 by 0.048 to 0.141, with
leave-one-segment-out minimums of at least +0.036.

## Side effects of this audit

**CPU cluster jobs.** Both are recorded in `facts/remote.json` with their cancel commands.
- **<job>:** PENDING on Resources, because both epyc9654p nodes were taken by non-campaign 579G jobs
  under the same account. I cancelled it by its exact ID before it started, and recorded
  `cancel_issued` and `cancelled_by`.
- **<job>:** ran on <node> and COMPLETED. No allocation is held.

**Campaign cache.** I added deterministic records for specs nobody else needs (13 x0 controls on val
k0-2, and 25 specs on train k0-3). They were 546 local and 3,234 ingested. The fresh-root and node
parity checks show they are bit-identical to what any stage would produce.

**Scratch.** Listed in `CLEANUP.md`:
- the local bundle (890M);
- the fetched shard (2.3G);
- remote scratch: bundle 896M and returned shard 2.3G;
- the node-local `/tmp/lr-<user>-<job>-audit-p2mid-ofs-train`;
- the audit directory (132M, which includes the private root).

**Nothing else.** No deletions, no WT changes, no commits.
