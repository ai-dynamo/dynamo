# Final audit: refute-headline-2 (fairness)

- **Checkpoint:** final (after Select+Test).
- **Lens:** refute the headline through fairness. Were the baselines tuned as well as the learned
  model (equal B, sensible spaces, sigma0, restarts)? Did the learned model get any information
  advantage (features, inits from test data, leakage)?
- **Auditor run:** 2026-10-04 23:16 to 2026-10-05 ~01:05 PDT.
- **Evidence:** `CR/runs/audits/final-refute-headline-2/` (`scripts/`, `specs/`, `out/`, `logs/`).
  Every number below is in `out/SUMMARY.json` or in the `out/score_*.json` file named next to it.
- **Test split:** not touched. My outputs contain 0 records of any test cell. Everything below is
  train or val.
- **Lessons applied:** LR-04 (paper-faithful baselines), LR-07 (heuristics inside M1's class),
  LR-10 (restarts, selection, still improving at cut-off), LR-11 (counting arms), LR-05 (gauge
  pins), LR-01 (clipped log-ratio, with my own scorer).

## Verdict: FAIL. 0 blockers, 1 major, 3 minor.

**The tuning mechanics are fair.** I rebuilt them from the raw histories and they hold. My own
ceiling probes show the baselines are not under-tuned. I found no leakage and no runtime
information advantage.

**The headline is unfair in what the baseline pool contains.** The learned arm was given LMetric's
faithful decision rule, which includes the queued-prefill term, as features and as its s3
initialization. LR-04 flags the `lmetric` port in the baseline pool as missing exactly that term, and
CONTRACT A2.5 kept the faithful version out of the pool. That rule beats every tuned baseline
without any tuning on val k0-2 and on fresh val k3-10, and beats ramjet on train k0-1:

- **Untuned M1-v2 s3 start point** (default plus faithful LMetric): it reproduces 54% (cell) to 66%
  (segment) of M1-v2's fresh-val margin over ramjet.
- **The same family tuned with 8 train evaluations:** it reproduces 53% to 62%.
- **What is left for learning:** M1-v2 beats the in-class heuristic by a segment mean of +0.010 to
  +0.011 (SE about 0.007, 4 to 5 of 6 segments). That is about a quarter of the MDE.

The pre-registered primary test (M1-v2 vs ramjet, p = 0.0017) remains a valid statement about the
heuristics as ported. Read as "learning beats the best available heuristic", it misleads.

## What I ran (own code, own replays)

| # | Check | Evidence |
|---|---|---|
| 1 | **Equal-footing reconstruction from raw `history.jsonl` and `identity.json`** for all 48 runs of the headline learned pool (m1v2, m1, m1noaff, m2r2, ablam1) and the tuned-baseline pool (11 policies × 3 restarts, incl. ablabase). It covers generations, popsize, fevals, cells and ks per generation, val lines, val ks and val cells, objective, metric, eps, reference, space dimensions, sigma0, maxstd and init. | `scripts/budget_verify.py` → `out/budget_verify.json` |
| 2 | **Controls:** M1-v2, ramjet and default on val k0-2 and k3-10, all cache hits. My scorer reproduces the campaign's selection values exactly: M1-v2 0.1767 / 0.1783, ramjet 0.1420 / 0.1411. | `out/score_P0_PC.json`, `out/score_PC_k3_10.json` |
| 3 | **Baseline ceiling probes on val k0-2, deliberately favoring the baselines** (a val-oracle maximum over many configs): (a) ramjet, 38 configs: all three `basis` values (tier A pinned `relative`) and wider log ranges, 12 scrambled-Sobol points per basis plus the tuned point under the two other bases; (b) M0, 20 configs: the decode request weight up to 1e6 (tier A capped it at 4096, and m0/stickyhard selections sat near that cap), prefill scale up to 1e4, credit up to 16; (c) llm-d-precise-prefix, 12 configs, with weights on log [1e-3, 1e3]. | `specs/PA_trim.jsonl`, `PB_trim.jsonl`, `PD_trim.jsonl` → `out/score_PA_trim.json`, `score_PB_trim.json`, `score_PD_trim.json` |
| 4 | **Heuristics inside M1-v2's hypothesis class but absent from the baseline pool,** run as `learned-choice` v2 θ points: faithful LMetric θ = −(e_log_ptok + e_log_bs); the M1-v2 s3 start θ = −e0 − 20.7 (e_log_ptok + e_log_bs); the lmetric port score without its hot-spot filter; and three others. Run on val k0-2, then on **fresh val k3-10** (8 CRN replicates). | `specs/PC.jsonl`, `PC_k3_10.jsonl` → `out/score_P0_PC.json`, `out/score_PC_k3_10.json`, `*.paired.json` |
| 5 | **A fairly tuned version of that family.** I swept c in θ = −e0 − c (e_log_ptok + e_log_bs) on **train** (34 cells, k0-1; 8 evaluations, against 1,200 per baseline policy). The train argmax c = 10 ties c = 20.7 (0.15663 vs 0.15663). I picked c = 10 on val k0-2 (0.1609 vs 0.1596) and then scored it on fresh val k3-10. M1-v2 and ramjet ran on the same train ks for context. | `specs/PF_train.jsonl`, `PF_val.jsonl` → `out/score_PF_train.json`, `score_train_k0_1_all.json`, `score_PF_val_k0_2.json`, `score_PF_val_k3_10.json` |
| 6 | **Information attribution of the selected M1-v2** on val k0-2: θ6 (session affinity), θ11 (hash_home), θ21 (new_prefill × requests) and θ1 each zeroed. | `specs/PE.jsonl` → `out/score_PE.json`, `out/score_PE.paired.json` |
| 7 | **Leakage:** train/val/test trace files and SHA-256 overlap; M1-v2 prep inputs (`runs/phase2/m1v2/default_train_k0.jsonl`); the source split of the SLA and load calibration (`facts/calibration.json` rules). | inline below |

**Execution:**

- **Build:** tuning build 6955b0ee, through the shared CR slot pool, campaign root and cache.
- **Volume:** 4,530 fresh replays and 1,236 cache hits, with 0 errors, 0 timeouts and 0 test-cell
  records.
- **Process stop:** I stopped one oversized first probe pass (PA at 74 configs, runner PID 1179100,
  lr-eval PID 1182423) by exact PID and re-ran it trimmed to 38 configs. Its completed records were
  kept.

## Refutation attempts that failed (the tuning was fair)

**1. Equal budget holds exactly.** All 48 runs share one budget signature:

- 25 contiguous generations × popsize 16 = 400 fevals;
- all 34 train cells, K = 2 per generation;
- 10 val lines on the 14 val cells at k0-2;
- objective `clipped_log_ratio` on `goodput_rps_window`, eps 1e-3, same reference sha 38a79285.

Every policy has 3 restarts. The tasks requested, including UH re-evaluations, range from 32,056
to 34,096 per run on both sides (`out/budget_verify.json`, `equal_budget: true`).

**2. Baselines are not under-tuned.** This is a val-oracle bound, so it is biased toward the
baselines.

| Policy | Tuned config, val k0-2 | Best audit probe on val k0-2 |
|---|---|---|
| ramjet | 0.1420 | 0.1421 (38 configs). The other `basis` values at the tuned point give the same score; the cap rarely binds. |
| M0 | 0.1334 | 0.1299 (20 configs, widened bounds). Pushing the decode request weight past the old 4,096 cap lowers the score: 0.1299 at 4,096, 0.1160 at 16,384. |
| llmdpp | 0.1414 | 0.1202 (12 widened configs) |

Every probe stays far below M1-v2's 0.1767. In tier A the three ramjet restarts converged to
0.1419, 0.1418 and 0.1420, with CMA sigma shrinking from 0.27 to 0.06. The learned arm, not the
baselines, was the one still moving at the cut-off: m1v2-s1's sigma grew from 0.0027 to 0.0073.

**3. sigma0 and maxstd asymmetry is harmless.** Five baselines start at the 0.3 maxstd cap because
the 15–20% flip target was unreachable; the learned arms use the flip rule with no cap. Given the
ceilings above, the cap did not limit any baseline.

**4. No leakage:**

- train, val and test share 0 trace files and 0 trace SHA-256s;
- M1-v2's feature scales and inits come from 34 train cells only (0 val, 0 test);
- SLA and load levels were calibrated on train segments only (`facts/calibration.json` rules; FAST25
  inherits Mooncake's).

**5. No runtime information advantage.** M1-v2 reads only router-observable inputs, the same CACHE
and LOAD signals the baselines read. It does not rely on session IDs or prefix-hash homes. On val
k0-2, removing them makes it slightly better:

| Change | Score (vs 0.1767) | Paired cell delta, M1-v2 minus variant |
|---|---|---|
| θ6 = 0 | 0.1773 | −0.0005, SE 0.0016 |
| θ11 = 0 | 0.1778 | −0.0010, SE 0.0017 |
| θ21 = 0 (new_prefill × active_requests) | 0.1589 | +0.0178, SE 0.0058, 13/14 cells |

Its increment beyond the LMetric heuristic comes from θ21, the new_prefill × active_requests
interaction, which is a router-observable product.

## Findings

### F1 (major): the learned class contains a published heuristic that the baseline pool lacks, and that heuristic, untuned, beats every tuned baseline

**Claim.** The comparison is "one tuned config per policy with equal B" (A2.1), but it is not equal
in what each side may express:

- **Learned side:** M1-v2's feature set v2 adds `log_ptok` and `log_bs` so that faithful LMetric's
  queued-prefill product is representable (FEATURES.md, LR-07). Its s3 restart starts exactly there.
- **Baseline side:** the pool's `lmetric` port omits the queued-prefill term (LR-04, VERIFIED,
  high/high). CONTRACT A2.5 forbids paper-faithful variants.
  `facts/build_rust.json` records this choice: "lmetric-faithful/smetric baselines rejected per A2.5;
  LMetric's queued-prefill product is nevertheless representable inside learned-choice v2".

The omitted heuristic is the strongest simple router the campaign had. Most of M1-v2's margin is
that heuristic, not learning.

**Evidence** (objective = mean clipped log-ratio vs default@defaults; paired deltas vs ramjet,
the val-best tuned baseline):

| Policy (tuning) | train k0-1 | val k0-2 | **val k3-10 (fresh)** | vs ramjet on k3-10: cell / segment |
|---|---|---|---|---|
| M1-v2 (3 × 400 evals) | 0.1851 | 0.1767 | **0.1783** | +0.0372 (SE 0.0116) / +0.0290 (SE 0.0125, 5/6) |
| default + c·faithful-LMetric, c = 10 (8 train evals) | 0.1566 | 0.1609 | **0.1609** | +0.0197 (SE 0.0056) / +0.0179 (SE 0.0067) |
| M1-v2 s3 start point, c = 20.7 (untuned) | 0.1566 | 0.1596 | **0.1612** | +0.0200 (SE 0.0052, 13/14) / +0.0190 (SE 0.0067, 6/6) |
| faithful LMetric alone (untuned) | 0.1466 | 0.1493 | **0.1490** | +0.0079 (SE 0.0031, 11/14) / +0.0110 (SE 0.0042, 6/6) |
| ramjet (val-best of 11 tuned baselines, 3 × 400) | 0.1435 | 0.1420 | **0.1411** | — |

Sources: `out/score_train_k0_1_all.json`, `out/score_PF_val_k0_2.json`,
`out/score_PF_val_k3_10.json`, and the `*.paired.json` files.

- **Faithful LMetric alone beats every tuned baseline.** With no tuning and no evaluations, it beats
  all 11 tuned baselines on both val splits. On val k3-10 their ranking tops out at ramjet 0.1411
  (`facts/finalists.json`); on val k0-2 their selected configs top out at ablabase 0.1423
  (`facts/tierA.json`, `facts/tierBC.json`). On TRAIN k0-1, where ramjet was tuned, it is also ahead
  of ramjet: 0.1466 vs 0.1435. Ramjet is the only baseline I ran on train.
- **Share of M1-v2's fresh-val margin that is the in-class heuristic:** 54% / 66% (cell /
  segment) for the untuned s3 start, and 53% / 62% for the train-tuned c = 10. On AgentX, which
  drives the headline's test gain (+0.037), M1-v2's lead over the s3 start is only +0.002 to +0.009
  per segment. The learned residual is concentrated on Mooncake (+0.044).
- **Learned increment over the train-tuned heuristic** (val k3-10): cell +0.0174 (SE 0.0069,
  11/14), segment +0.0110 (SE 0.0065, 5/6). That is about 0.29 × the pre-registered MDE of 0.038.
  On val k0-2 it is +0.0089 (SE 0.0063).

**Consequence.**

- **What stays valid:** the pre-registered primary test (M1-v2 vs ramjet on test, +0.0316,
  p = 0.0017) is valid as a statement about the branch's ported heuristics, tuned. A2.5 binds, so
  this is not a blocker.
- **What misleads:** the PLAN story is "learn a routing function that beats every heuristic router
  we have", and REPORT plans a "fairness" section that discusses only budget and inits (A11.1).
  Both mislead. A simple, published rule gets more than half the margin, and the campaign's own
  lessons said to include that rule as a baseline.
- **Test-level split (HYPOTHESIS, untested):** the test split is single-pass, so this cannot be
  checked there. If the val split carried over, M1-v2's test margin over a default + faithful
  LMetric heuristic would be roughly +0.01 per segment. Its significance is unknown.

**Fix (no test re-evaluation; the test split stays single-pass):**

1. **Scope the claim.** REPORT must word the headline as "beats the branch's ported heuristics,
   each tuned with equal budget". It must not say "beats every heuristic" or attribute the full
   margin to learning.
2. **Add a val-only fairness table.** REPORT must carry the table above, from val, labeled "auditor
   diagnostic, not pre-registered, never used for selection". It must state that:
   - faithful LMetric was excluded by A2.5;
   - it is representable in M1-v2's class and was M1-v2's s3 initialization;
   - untuned, it beats every tuned baseline;
   - default + faithful LMetric reproduces 53–66% of M1-v2's fresh-val margin.
3. **Escalate to the operator** the option of a pre-registered follow-up that puts faithful LMetric
   (and the A2.5-excluded SMetric) into the pool. It would run on fresh data, either new segments or
   the live A13 runs, never the frozen test cells. Until then, REPORT should call the learned
   increment over the strongest simple heuristic unmeasured on test and about 0.01 on val.

### F2 (minor): the learned side was redesigned on val evidence, the baseline side was not, and the original M1 does not pass on test

**What happened on each side:**

- **Learned side.** The learned arms went through several rounds driven by val comparisons
  against the baselines:
  - A11 multi-start;
  - A11.2 sign rules from the val audits;
  - A17.1's M1-v2, written to target "the pilot's headline risk: M1 against the best heuristic
    (+0.030 < MDE), with M1 trailing on AgentX";
  - M2, M1-noaff and the queue-threshold arm.
- **Baseline side:** one extra arm, ablabase, and no added heuristic types (see F1).
- **Validity:** selection among the 5 learned arms was on val, with the same rule as the baselines,
  and the test pass was single. So the p-value is valid for the selected arm.
- **But on test**, the originally planned rungs do not pass against ramjet:
  - M1: +0.0186 per segment, p = 0.055;
  - M2 rank 2: +0.0211, p = 0.055.

  Source: `facts/test_results.json` `secondary.m1_vs_ramjet` and `m2r2_vs_ramjet`.

**Fix.** REPORT's fairness section should:

- disclose this asymmetry in iteration;
- show every learned arm's test row next to the headline;
- say that the passing arm is the post-pilot redesign;
- cite the LR-11 count (314 learned variants).

### F3 (minor): informed initializations went only to the learned side

**What happened:**

- M1 and M1-v2 got informed starts: a behavior clone of llm-d-precise-prefix (train) and the LMetric
  or cache-heavy points. Baselines started only from their shipped defaults (gate rule).
- The headline M1-v2 came from s1 (θ0 = −e0), so these inits did not produce the selected
  policy.
- The s3 init is the F1 heuristic. Scored as a policy, it beats every tuned baseline (0.1612 on val
  k3-10).

**Fix.** REPORT's A11.1 fairness note should say that the inits carry heuristic knowledge the
baselines never got. It should quote the s3 point's own score rather than calling the inits "not
extra budget" only.

### F4 (minor): auditor variants in the campaign cache

This audit added 4,530 val/train records to `CR/runs/cache` (3.93 GB with per-request rows; keys in
`out/new_cache_keys.txt`). They include 12 learned-choice θ variants evaluated on val and 5 more evaluated on train only.

- **Status:** they are audit diagnostics, not campaign variants, and they are not counted in the
  LR-11 total of 314.
- **Use:** REPORT may cite them as such.
- **Cleanup:** listed in CLEANUP.md.

## Not checked, or out of lens

- Re-derivation of the test statistics (refute-headline-1's lens), the AIS league's fairness, and
  the live runs.
- A full B = 400 × 3 CMA-ES tuning of faithful LMetric or SMetric. The 8-evaluation train sweep
  above is a lower bound on what tuning would give, and is not a substitute.

## Scratch

- `CR/runs/audits/final-refute-headline-2/` (31 MB: scripts, specs, outputs, logs).
- 4,530 new records in `CR/runs/cache/results` (3.93 GB, keys listed in `out/new_cache_keys.txt`).
- Both are recorded in `CR/CLEANUP.md`. Nothing deleted; nothing committed. No WT change, no CPU cluster
  job.
