# Final audit: refute-headline-1

- **Checkpoint:** final (after Select+Test)
- **Lens:** refute the headline in `facts/test_results.json` by re-deriving it from the raw test records
  with independent code, and checking the pairing, the CIs and the multiple-comparisons handling.
  If it can't be reproduced, it counts as refuted.
- **Auditor run:** 2026-10-04, 23:15 to 23:40 PDT.
- **Evidence:** every number below comes from a file in `audits/final/refute-headline-1/`, which holds
  my scripts and their JSON outputs.

## Verdict: PASS. The headline is reproduced exactly and survives every refutation attempt.

The headline claim is that M1-v2 (`m1v2@p2-m1v2-s1-g20-best_so_far`, policy_sha 07f95e14) beats the
val-best baseline ramjet (`ramjet@p2-ramjet-s3-g25-best_so_far`, policy_sha 8c99a422) on the frozen
test set. My recomputation from the per-request rows gives:

- segment mean clipped log-ratio +0.031555, SE 0.009332;
- ahead on 10 of 12 segments;
- one-sided exact Wilcoxon p = 0.001709 (zero handling wilcox and Pratt);
- P(>) = 0.833, with 95% CI [0.583, 1.0] from 20,000 segment resamples on my own RNG;
- segment-mean CI [0.0151, 0.0500];
- cell mean +0.05058.

These match `test_results.json` to every printed digit. I found no blocker and no major finding.
The seven minor findings below concern how REPORT should word the secondary claims. None of them
affects the pre-registered primary test.

## What I did (own code; nothing reused from the stage's analysis scripts)

1. **Independent rescoring** (`recompute.py`).
   - **Re-implemented from CONTRACT A2/A2.3 and the goodput docstring:**
     - the A2 good rule: ITL ≤ I and E2E ≤ S·E0;
     - both window bases: arrival, and completion with identity warm-up and the full-occupancy
       end built from my own occupancy profile;
     - the missing-row rule.
   - **Scope:** all 6,120 test-pass records (5,220 from the tuning build 6955b0ee and 900 from the
     AIS build 555ca082), read from the cache's per-request rows.
   - **Result:** 6,120 of 6,120 records match the recorded `window_start_ms`, `window_end_ms`,
     `window_good`, `window_requests` and `goodput_rps_window`, bit for bit. All five off-nominal
     SLO-scale rescores match as well (`recompute_all.jsonl`).
   - The E2E tolerance of 1e-6 changes the good count in only 21 of the 6,120 records.
2. **Row integrity** (`rows_sha_check.json`). For all 6,120 records, the sha256 of the
   decompressed per-request rows equals the record's `per_request_canonical_sha256`, with 0
   mismatches. This closes the gap left by the ingest race in which one record's rows were
   restored by hand.
3. **E0 spot check** (`e0_spot.json`).
   - **Sample:** 400 seeded random (ISL, OSL) pairs from the 255,022 pairs used by the headline
     records.
   - **Method:** recomputed directly through the AIS session API, without `learned_routing.e0` and
     without the cache.
   - **Result:** the maximum relative difference from the cached table is 4.0e-14, which is float
     summation order only. The 207,315 pairs precomputed locally are sound.
4. **Pairing and identity** (`headline_stats.json` checks; cache keys recomputed).
   - **Coverage:** both arms cover the same 60 × 3 (cell, k) set.
   - **CRN pairing:** 0 mismatches in `replicate_seed`, `replicate_protocol`,
     `trace_content_sha256`, `cell_sha` and `build_id`.
   - **Policy seeds:** M1-v2 uses k + 1; ramjet is unseeded.
   - **Policy hashes:** one policy_sha per arm, equal to `finalists.json`.
     `runs/policies/<sha>.yaml` hashes to its own name and holds the frozen θ.
   - **Cache keys:** recomputed for 540 of 540 headline records, 0 mismatches.
   - **Cell hashes:** all 6,120 records' `cell_sha` equal `test_freeze.json`.
   - **Freeze:** I re-verified `cells/test.jsonl` (7b998b8e) and all 39 test trace files against
     `test_freeze.json`, with 0 mismatches.
   - **Spec files:** all 5 frozen stage spec files match the hashes in `finalists.json`.
5. **Test-once check** (`test_split_cache_files.txt`).
   - **Scan:** every cached record of any test cell_id, 30,060 records.
   - **Timing:** the earliest dates from 21:31:50, after the `finalists.json` freeze at 21:24:35.
     No record predates the freeze.
   - **Names:** each policy name maps to exactly one sha, and every evaluated name is in the frozen
     spec files. No alternative checkpoint of any arm was ever evaluated on test.
   - **Duplicates:** first-wave and r2 duplicates are identical for the 992 tasks that succeeded
     twice, including 33 M1-v2 and 34 ramjet tasks.
6. **Selection re-derivation.**
   - Using my own objective code on the val k3-10 records, all 21 val k3-10 objective values in
     `selection_final.json` reproduce exactly. M1-v2 at 0.17831 is the best headline-eligible arm,
     and ramjet at 0.14113 is the best baseline.
   - The reference default@defaults is identical across the three val files: 112 of 112 records.
7. **Independent replays** (`val_replay_parity.json`).
   - **Decision:** I replayed no test cell, to honor the evaluate-test-once rule.
   - **Setup:** a private cache root (`replay_root/`) that uses the shared 20-slot pool, the shared
     E0 table and the shared traces.
   - **What I replayed:** M1-v2 and ramjet on 3 val cells at k = 3, one each from Mooncake,
     sessions and AgentX.
   - **Result:** 6 of 6 replays reproduce the cached val k3-10 records byte for byte: goodput,
     per-request sha, cache_key and policy_sha.
8. **Statistics** (`headline_stats.py`, stdlib only).
   - **Wilcoxon:** an exact null built by enumerating all 2^n sign vectors with average ranks.
   - **Sign test, bootstrap and Holm:** my own implementations.
   - **Sensitivity variants:**
     - eps = 0;
     - no clipping;
     - ratio of replicate means;
     - each replicate alone;
     - leave-one-segment-out;
     - drop-a-family;
     - excluding N = 6, excluding the transform-extrapolation cells, and seen N only;
     - near-zero deltas treated as ties.

## Refutation attempts that failed (the headline holds)

| Attempt | Result (segment mean, one-sided exact Wilcoxon p) | Source |
|---|---|---|
| eps = 0, no clip, ratio of replicate means | +0.0316 / +0.0316 / +0.0315, all p 0.0017 | headline_stats.json |
| one replicate only (k0 / k1 / k2) | +0.028 p 0.0049 / +0.033 p 0.0012 / +0.034 p 0.0046 | extra_sens.json |
| leave any one segment out | max p 0.0034 | headline_stats.json |
| drop a whole family (agentx / fast25_synthetic / mooncake / sessions) | max p 0.027 (agentx dropped, n = 8) | headline_stats.json |
| exclude N = 6 (selection-exposed) | +0.0330, p 0.0012, 11/12 ahead | extra_sens.json |
| exclude transform-extrapolation cells | +0.0320, p 0.0005 | extra_sens.json |
| seen N only (4, 8) | +0.0346, p 0.0081 | extra_sens.json |
| exclude the 3 FAST25 conversation cells | +0.0260, p 0.0017 | (command output; see F3) |
| sign test instead of Wilcoxon | p 0.019 (10/12); 0.0039 with ties dropped | headline_stats.json |
| \|d\| < 1e-3 treated as ties | 8 non-zero, p 0.0039, P(>) CI [0.708, 0.958] | headline_stats.json |
| comparator = the best baseline ON TEST (oracle; llmdpp) | +0.0293, p 0.00024, 12/12 | secondary_stats.json |
| robustness: lag 10/50/200; timing on E0' | +0.031/+0.029/+0.030; +0.081/+0.022/+0.039/+0.030, all p ≤ 0.0105 | robust_check.json |

**Multiple comparisons.**
- **The primary test is a single pre-registered comparison.** The learned arm was chosen among 5
  arms, and the baseline among 11, both on val k3-10; I reproduced both choices. The test split
  was never touched before the freeze, so neither selection inflates the test p-value. LR-11's
  disclosure of 314 validated variants is about honesty in reporting, not a test correction.
- **The per-N Holm correction is correctly implemented:** the adjusted p values are 0.078, 0.094,
  0.219, 0.070, 0.094 and 0.078, reproduced.
- **The "beats all 11 baselines" conjunction:** the Wilcoxon-only intersection-union test is valid
  without correction, and its largest p is 0.0105, reproduced.

**Secondaries reproduced exactly.**
- AIS league: M2-ais vs M0-ais +0.0305 (p 0.0012), and M2-ais vs M1-v2 −0.0029.
- A19 headline: segment mean −0.00925, p 0.032, cell mean +0.0044.
- SLO sweep: at scale 0.5, +0.0716 (p 0.065); at 0.75, +0.0353 (p 0.117); at 1.5, 2 and 3, p 0.00024.
- Load-mode strata: open +0.0281 (p 0.074), closed +0.0354 (p 0.078), lanes +0.0374 (p 0.0625).
- Family-stratified P(>) CI: [0.667, 1.0].

## Findings (all minor)

### F1 (minor) "Beats every tuned baseline" fails the effect-size half of LR-11's own conjunction

**What fails.** REGISTRATION.json defines this intersection-union test as Wilcoxon-only, and on that
definition it holds. But LR-11, which the registration cites, pairs each Wilcoxon test with a
P(learned > b) leg: the CI lower bound must exceed 0.5. That leg fails for two baselines
(`iut_effect.json`):

| Baseline | P(>) | 95% CI | With \|d\| < 1e-3 as ties | Segments behind |
|---|---|---|---|---|
| lmetric | 0.667 | [0.417, 0.917] | [0.458, 0.917] | all 4 sessions segments, −0.0001 to −0.0050 |
| stickybounded | 0.75 | [0.50, 1.0] (not > 0.5) | [0.458, 0.917] | sessions s7, s8, s9, −0.0020 to −0.0030 |

**Fix.** REPORT should word this as "Wilcoxon-significant against each of the 11 tuned baselines".
It should add that M1-v2 trails lmetric and sticky-bounded slightly (≤ 0.5%) on the sessions
segments, where all policies are at ceiling, so the P(>) criterion is not met for those two.

### F2 (minor) Only the pooled 12-segment test is confirmatory; individual strata are not significant

**The pooled unseen-N result.** The stage reports unseen N (2, 16, 32) at +0.034, p 0.008, with no
correction.
- That stratum is not in HEADLINE_TEST's stratification, which specifies per-N tests with Holm.
- It is one of about a dozen strata computed: 3 load modes, the families, 6 values of N, transform
  extrapolation, unseen N and seen N.

**The other strata, reproduced.** None is individually significant:
- load modes: p 0.062–0.078;
- transform extrapolation: +0.025, p 0.109;
- per-N after Holm: smallest adjusted p 0.070.

**Fix.** REPORT should label every stratum descriptive. It should not state "significant at unseen
N" without the multiplicity caveat.

**Robustness note.** Speedup 0.8 under nominal E0 gives +0.079 with p 0.117, which is not
significant. The robustness summary quotes p only for E0', so REPORT should show both.

### F3 (minor) The magnitude is concentrated in request-count gains that token-weighted goodput does not share

**Where the gain sits.**
- The three FAST25 conversation cells contribute (0.481 + 0.494 + 0.077) / 60 ≈ 0.0175 of the
  0.0506 cell mean (about 35%), from 3 of 60 cells.
- Their token-weighted (A19) deltas are +0.017, −0.147 and +0.019.
- fast25_synthetic:x0 shows the same pattern: requests +0.084 and +0.028, tokens −0.389 and −0.193
  (`extra_sens.json` and command output).

**Without the conversation cells,** the segment mean is +0.0260 (p 0.0017, unchanged), against
+0.0316 with them.

**Disclosure so far.** The stage disclosed the A19 result as mixed and noted that the conversation
cells average +0.39. It did not decompose this.

**Fix.** REPORT should present the effect size as a request-count property. It should show the
request-vs-token contrast for these cells next to the headline. CONTRACT A19 keeps the request-count
headline itself.

### F4 (minor) The headline's "10 of 12 segments ahead" includes four sub-noise sessions segments

**Ceiling.** In the sessions cells, ramjet and M1-v2 both reach an in-window good fraction of at
least 0.987. Default ranges from 0.75 to 0.99. The largest |cell delta| in sessions is 0.0019.

**Signs.** sessions:s8 at +4.5e-5 and s9 at +5.9e-4 count as "ahead"; s6 at −1.5e-5 and s7 at
−9.7e-5 count as "behind".

**More accurate statement.** M1-v2 is ahead on all 8 informative segments, and the 4 sessions
segments are ties at ceiling. This version is equally significant: p 0.0039, P(>) CI [0.708, 0.958].

### F5 (minor) Provenance in `facts/test_results.json` is stale

**Stale hashes.** `inputs` records sha256 0f1e36d6… for `runs/select_test/test/tuning.jsonl` and
3643cb27… for `ais.jsonl`. The files on disk now hash to 0c615def… and 6a33ee32….
- `collect.sh` rewrote them from the cache at 23:04, after the 22:19 analysis.
- The new copies differ only in run-environment fields such as `run_id`, `wall_s` and `slot`.
- My recomputation from the current files reproduces every number in the file.

**Hand-added sections.** `summary` and `execution` were added after `analyze_test.py` ran: the
`written` field says 22:19:08, but the file's mtime is 23:11:53. The script does not write those
keys. Re-running it, as the hand-off notes instruct, would silently drop them.

**Fix:**
- re-hash the inputs, or hash a projection that excludes environment fields;
- move the hand-written sections to a file the script does not regenerate.

### F6 (minor) The registration timeline cannot be verified from the artifacts

**The record.** `selection_final.json`, written at 21:23:39, pins `REGISTRATION.json` as sha256
cfdfc10a…. The file now hashes to 702630e2… and was last modified at 21:24:07. The addendum was
written into the same file, and the pre-addendum version is not preserved.

**Inconsistent times.** The file's own timestamps ("~21:25", and "~21:35" for the addendum) do not
match the mtimes. The val k3-10 cache records of the newly evaluated arms date from 21:22:20 to
21:22:41.

**Impact.** None on the outcome: both registered choices are neutral.
- m1cont (0.1672) would not have beaten M1-v2 (0.1783).
- ablabase (0.1405) ranks below ramjet (0.1411).

**Fix.** In future, keep registrations append-only, as separate files.

### F7 (minor) This auditor's own incident: pid files written into the main checkout

**What happened.** A `cd X && nohup … &` construct backgrounded the `cd` along with the job. The
following `echo $! > *.pid` therefore ran in `<repo>` and wrote four pid files there:
- `e0_spot.pid`
- `recompute_all.pid`
- `recompute_headline.pid`
- `rg_scan.pid`

**Recovery.** Within minutes I moved them, not deleted them, into this audit directory.
`git status` in the main checkout is identical to its state at session start. I recorded the
incident in `facts/DEVIATIONS.md`.

## Not checked, or out of lens

- I replayed no test cell (test-once rule). Replay fidelity of test records rests on these checks:
  - row sha integrity (6,120 of 6,120);
  - duplicates identical across nodes (992 of 992);
  - image parity in `facts/remote.json`;
  - my val replay parity (6 of 6).
- The LR-13 concentration and segregation flags, live-validity, and feature leakage of feature set
  v2 belong to other lenses. I only confirmed that v2's features (FEATURES.md indices 8–22) are
  router-observable and use no AIS, OSL or `expected_output_tokens`.

## Scratch

`audits/final/refute-headline-1/` holds about 15 MB, including `replay_root/` with its 2.7 MB
private cache. It is listed in `CLEANUP.md`. I deleted nothing, committed nothing and pushed
nothing.
