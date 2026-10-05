# Final fix

- **Stage:** fixer at checkpoint final, 2026-10-05 about 01:05 to 01:35 PDT.
- **Status:** COMPLETE. Both majors are VALID and fixed. Nothing is rebutted. No replay was run,
  the test split was not re-evaluated, and no tuning was re-run (reasons below).
- **Inputs:** `audits/final/completeness.md` F1 and `audits/final/refute-headline-2.md` F1. The minor
  findings of all four final audits are not in this stage's scope. Their REPORT additions stay in
  the audit files.
- **Evidence:** `runs/final-fix/` (`scripts/`, `out/`, `logs/`).
- **Facts written:**
  - `facts/test_results.json`: only the keys `secondary_metrics` and `report_must_state` were added.
    Every other key is identical to the Select+Test version, which is preserved as
    `runs/final-fix/out/test_results.pre-fix.json` (sha256 ff43fff4…).
  - `facts/secondary_metrics_test.json` (new): full detail.
  - `facts/fairness_in_class_heuristic.json` (new): the val-only table.
- **WT:** unchanged; nothing committed (no WT paths touched).

## Verdicts

| Lens | ID | Severity | Verdict | Action |
|---|---|---|---|---|
| completeness | F1 | major | VALID | Secondary metrics computed from the cached test records and per-request rows for all 33 test policies. The long-prompt TTFT deficit vs ramjet is added to `report_must_state`. |
| refute-headline-2 | F1 | major | VALID | Headline scope wording, the val-only fairness table (re-derived from the cache) and the learned-increment statement are added to `report_must_state`. The faithful-LMetric/SMetric follow-up is escalated to the operator. |

## completeness F1: A2.2 TTFT reporting and PLAN secondary metrics

**Verification.**
- `facts/test_results.json` (Select+Test) had no TTFT, ITL-percentile, throughput, prefix-reuse or
  trajectory result. `isl_strata` held only `mean_good_frac_window`. CONTRACT A2.2 requires
  "TTFT is still reported (p50/p90, and by ISL bucket)", and PLAN's Objective lists the other
  metrics.
- My pipeline reproduces all 34 of the auditor's M1-v2 numbers
  (`runs/audits/final-completeness/out/secondary_metrics.json`, against ramjet and against
  default): segment means agree within 1e-5 after rounding to 6 significant digits, and the
  higher-segment and segment counts are identical (0 mismatches).

**Fix (root: the metrics now exist as facts).**

1. `runs/final-fix/scripts/secondary_per_record.py` reads each of the 6,120 test-pass records
   (`runs/select_test/test/{tuning,ais}.jsonl`) and its cached `per_request` rows. It writes
   `runs/final-fix/out/secondary_per_record.jsonl`. No replay is run.
   - **Guards** (all 6,120 pass): the cache record's metric fields and per-request digest equal
     the stage's record, and the in-window good count, recomputed with the harness's own window
     and warm-up rules (`learned_routing.window_guards.in_window_flags`, the same helpers as
     `rescore_records.py`), equals `window_good`.
   - **`rec.*`:** the harness's own record fields, over every completed request of the replay:
     - TTFT p50/p90/p99;
     - per-request mean ITL p50/p90;
     - native per-token ITL p99;
     - output and total throughput;
     - prefix reuse;
     - per ISL bucket: TTFT p50/p90, `good_frac_window` and n.
   - **`win.*`:** the same latencies over exactly the scored in-window requests: TTFT
     p50/p90/p99, mean-ITL p90, e2e p50/p90, and TTFT by ISL bucket.
   - **`traj.*` (AgentX):** per-play trajectory latency, from the first arrival to the last
     completion, over all plays and over plays inside the window. Every policy runs the same
     play set on every (cell, k): 42/42 checked by play-id digest, and every play completes.
2. `runs/final-fix/scripts/secondary_aggregate.py` writes the outputs.
   - **Aggregation:** the headline's cell → segment aggregation (mean over k, then over a
     segment's cells, then over the 12 segments).
     - Latency, throughput and reuse use ln(A/B); good fractions use A − B. Positive means A is
       higher.
     - Every policy is compared against ramjet (the val-best baseline) and against
       default@defaults.
     - The headline pair also gets per-family means, per-segment values and a two-sided exact
       Wilcoxon, which is descriptive.
   - **Outputs:**
     - full detail in `facts/secondary_metrics_test.json` (611 KB);
     - a compact `secondary_metrics` section in `facts/test_results.json`: the headline pairs,
       all-policy matrices for 34 core metrics, and absolute medians;
     - `report_must_state`, generated from the computed values, so no number in it is hand-typed.
   - **Checks:** 5,940 records (33 policies × 180), and the AIS build's default@defaults equals the
     tuning build's on every secondary metric (180/180).
3. **A sensitivity view chosen after seeing the data, labeled as such.**
   - The ISL-bucket percentiles of synthetic-sessions records rest on at most 13 prompts of
     32K-64K tokens per record. One sessions segment, in window, rests on 3 prompts and gives a
     log-ratio of −1.08.
   - So the bucket TTFT metrics also come in a `|n>=20` variant, which pairs a bucket only if
     both arms' records hold at least 20 such requests.
   - Both variants are reported, and `definitions` records the choice.

**What the data shows** (M1-v2 vs ramjet on test, segment mean log-ratio; from
`facts/secondary_metrics_test.json`):

| Metric | M1-v2 vs ramjet | Segments where M1-v2 is higher | M1-v2 vs default |
|---|---|---|---|
| TTFT p50, all requests | +0.006 | 9 of 12 | −0.573 |
| TTFT p90, all requests | −0.002 | 6 of 12 | −0.452 |
| TTFT p99, all requests | −0.082 | 4 of 12 | −0.116 |
| **TTFT p90, 32K–64K prompts** | **+0.135** (×1.145; descriptive p 0.012) | **11 of 12** | −0.107 |
| TTFT p90, 32K–64K, n ≥ 20 | +0.169 (p 0.0078) | 8 of 8 | −0.011 |
| TTFT p90, 32K–64K, in window, n ≥ 20 | +0.143 (p 0.0078) | 8 of 8 | +0.001 |
| TTFT p90, ≥ 64K, n ≥ 20 | +0.021 (p 1.0) | 3 of 7 | −0.237 |
| E2E p90, in window | −0.048 (p 0.0093) | 3 of 12 | −0.164 |
| Mean-ITL p90 | +0.006 | 3 of 12 | −0.271 |
| Per-token ITL p99 | −0.077 | 4 of 12 | −0.464 |
| Output throughput | +0.006 | 7 of 12 | +0.031 |
| Prefix reuse | −0.024 (p 0.001) | 1 of 12 | +0.238 |
| AgentX trajectory, mean (all plays) | −0.012 | 0 of 4 | −0.042 |
| AgentX trajectory, p90 (all plays) | −0.004 | 1 of 4 | −0.019 |

**Good fraction in window by ISL bucket** (M1-v2 minus ramjet; ramjet minus default in
parentheses):

| Bucket | M1-v2 − ramjet | ramjet − default |
|---|---|---|
| < 2K | +4.3 pp | +19.4 pp |
| 2K–8K | +4.3 pp | +16.6 pp |
| 8K–32K | −1.7 pp | +7.2 pp |
| 32K–64K | −3.8 pp | +4.5 pp |
| ≥ 64K | −3.7 pp | +1.1 pp |

**Where the deficit sits.** The 32K-64K deficit (in window, n ≥ 20, mean over cells) is
concentrated on Mooncake (+0.367) and the FAST25 cells. FAST25-conversation reaches +0.970, about
2.6×, on the same cells that carry the largest goodput gain. FAST25-synthetic is +0.160 and AgentX
+0.013.

**Ranking context.** M1-v2's long-prompt TTFT is on par with default@defaults (−0.011, n ≥ 20),
while ramjet's is −0.180.
- Ranked over the 32 non-default policies by the n ≥ 20 log-ratio vs default (lowest first):
  ramjet is 2nd and M1-v2 25th.
- The other router-observable learned arms rank between 16th and 23rd.

So the auditor's reading holds. M1-v2 trades long-prompt latency for short-request goodput against
ramjet. Overall TTFT, E2E p90 and AgentX trajectory latency do not get worse.

**Fix-side caveats (recorded in `definitions`).**
- `rec.*` percentiles include warm-up and drain, as the harness defines them.
- Open-loop throughput is bound by the offered load.
- These statistics are secondary and descriptive, chosen after the single test pass. They never
  select a policy or decide the headline.
- Sessions trajectories are not computed. PLAN's "agentic trajectory latency" maps to AgentX
  plays.

## refute-headline-2 F1: the baseline pool lacks the in-class heuristic

**Verification.** I re-derived the auditor's table without their scorer.
`runs/final-fix/scripts/verify_fairness.py` takes only the cache keys from the auditor's lr-eval
outputs. It reads every value from the campaign cache record and scores with the headline pipeline
(`st_common.pair_score`).

| Policy (tuning) | train k0-1 | val k0-2 | Fresh val k3-10 | vs ramjet on k3-10: cell / segment |
|---|---|---|---|---|
| M1-v2 (3 × 400) | 0.1851 | 0.1767 | **0.1783** | +0.0372 (SE 0.0116) / +0.0290 (SE 0.0125, 5/6) |
| default + 10 × faithful LMetric (8 train evaluations) | 0.1566 | 0.1609 | **0.1609** | +0.0197 (SE 0.0056) / +0.0179 (SE 0.0067, 4/6) |
| M1-v2 s3 start, c = 20.7 (untuned) | 0.1566 | 0.1596 | **0.1612** | +0.0200 (SE 0.0052, 13/14) / +0.0190 (SE 0.0067, 6/6) |
| faithful LMetric alone (untuned) | 0.1466 | 0.1493 | **0.1490** | +0.0079 (SE 0.0031, 11/14) / +0.0110 (SE 0.0042, 6/6) |
| ramjet (3 × 400) | 0.1435 | 0.1420 | **0.1411** | — |

**What the re-derivation confirms.**
- Every value equals the auditor's `out/SUMMARY.json`.
- My ramjet and M1-v2 val k3-10 values equal `facts/finalists.json` to 1e-12.
- **Faithful LMetric beats every tuned baseline:**
  - on val k0-2: 0.1493, against the 11 tuned baselines' selected configs, which top out at
    ablabase 0.1423 (`facts/tierA.json`, `facts/tierBC.json`);
  - on val k3-10: 0.1490, against ramjet's 0.1411.
- **Share of M1-v2's fresh-val margin over ramjet:**
  - default + c × faithful LMetric reproduces 53% / 62% (cell / segment) at c = 10;
  - the untuned s3 start reproduces 54% / 66%.
- **Learned increment over default + 10 × faithful LMetric:** cell +0.0174 (SE 0.0069, 11/14),
  segment +0.0110 (SE 0.0065, 5/6). That is 0.29 × the MDE of 0.0382.
  - On val k0-2 it is +0.0089 (SE 0.0063).
  - Against the s3 start on k3-10 the residual sits on Mooncake w1 (+0.044). AgentX V1-V3 gets
    only +0.002 to +0.009, and the sessions segments about 0.
- The class claim holds as well:
  - `facts/tierBC.json` records the s3 init as −e0 − 20.7 (e_log_ptok + e_log_bs);
  - FEATURES.md states that θ = −(e_log_ptok + e_log_bs) takes the faithful LMetric argmin;
  - `facts/build_rust.json` records that faithful LMetric and SMetric were rejected under A2.5.

**Fix.**
1. **Scope.** `facts/test_results.json` `report_must_state` item `MS-RH2-F1-scope`. The headline is
   "M1-v2 beats the branch's ported heuristics, each tuned with an equal budget" (pre-registered,
   p 0.0017). REPORT must not say "beats every heuristic" or "learning beats the best available
   heuristic", and it states why: A2.5 and LR-04.
2. **Table.** `facts/fairness_in_class_heuristic.json` (labeled "AUDITOR DIAGNOSTIC, not
   pre-registered, never used for selection; validation/train only, the test split was not
   evaluated") and item `MS-RH2-F1-table`.
3. **Increment.** Item `MS-RH2-F1-increment`: about +0.011 per segment on val, not measured on
   test.
4. **Escalation to the operator** (below).

**Not done, and why.**
- **No test re-evaluation.** The test split is single-pass, by contract and pre-registration.
- **No equal-budget tuning of a faithful-LMetric or SMetric baseline.**
  - Adding a paper-faithful baseline reverses the operator's binding A2.5 decision. That is a
    design decision the auditor itself routed to the operator.
  - Its result could only reach val, which is the selection split, never the frozen test.
  - The 8-evaluation train sweep already bounds what such a baseline gets on val (0.1609), and
    its share of the margin is not in dispute.

## Escalation to the operator (design decision)

A pre-registered follow-up.
- **Arms:** `lmetric-faithful`, i.e. default + c × faithful LMetric as a tuned baseline, and an
  `smetric`-style baseline (LR-04 action 4, "without global tier").
- **Budget:** each tuned with the equal budget (3 × 400 evaluations on pooled train, selected on
  val k0-2).
- **Evaluation:** on fresh data only, either new workload segments built under A4 split purity or
  the A13 live runs. Never the frozen test cells.
- **Primary test:** M1-v2 against the best of {ramjet, lmetric-faithful, smetric}, with the same
  HEADLINE_TEST pipeline.
- **Until it runs:** REPORT calls the learned increment over the strongest simple heuristic
  "unmeasured on test, about +0.01 per segment on val".

## Regeneration order

1. `runs/select_test/scripts/analyze_test.py` rewrites `facts/test_results.json` from scratch and
   drops the two keys added here. After any re-run of it, re-run
   `runs/final-fix/scripts/secondary_aggregate.py`.
2. `secondary_per_record.py` is resumable and keyed by cache key. Run
   `verify_fairness.py` before `secondary_aggregate.py`, because the latter reads
   `facts/fairness_in_class_heuristic.json`.
3. Use the tier-A WT venv, `PYTHONDONTWRITEBYTECODE=1`.

## Noted, not changed

- `facts/test_results.json` `inputs` lists a `runs/select_test/test/tuning.jsonl` sha256 (0f1e36d6…)
  that differs from the file's current sha256 (0c615def…). This is refute-headline-1's minor
  "input hashes stale". My section records current hashes in `secondary_metrics.inputs`. Record
  parity was established by refute-headline-1 (6,120/6,120) and completeness (31,508/31,508).

## Lessons applied

- **LR-04:** the faithful LMetric gap, scoped and escalated.
- **LR-07:** the heuristic inside the class.
- **LR-08:** good fraction and TTFT by ISL bucket.
- **LR-11:** segment unit, descriptive labeling, and the arm count stays as registered.
- **LR-13:** the long-request trade, now with its TTFT dimension.
- **LR-01:** windowed (`win.*`) variants next to the harness's whole-run fields.
- Rejected as out of scope: LR-14 engine-state distributions (completeness F9, minor).

## Scratch

`runs/final-fix/` (12 MB) holds the scripts, `out/secondary_per_record.jsonl` (11.9 MB, regenerable
in about 80 s), the pre-fix copy of `test_results.json` and the logs. It is listed in `CLEANUP.md`.
Nothing deleted, no CPU cluster job, no push.
