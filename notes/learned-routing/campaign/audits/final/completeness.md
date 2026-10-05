# Final audit: completeness lens

- **Checkpoint:** final (after Select+Test, before REPORT)
- **Lens:** completeness critic: what's missing or unverified, checked against PLAN, CONTRACT A1-A19, `facts/gate.json` full_plan, `facts/HEADLINE_TEST.json` and LESSONS.
- **Date:** 2026-10-04, about 23:15-23:45 PDT
- **Verdict: FAIL.** 0 blockers, 1 major, 10 minor.

The headline result, the single test pass, the equal budgets and the selection hygiene all hold up
under my own checks. The failure is a reporting gap. The test stage never produced the TTFT
reporting that A2.2 makes mandatory, and the data it left out carries a consistent unfavorable
signal against the val-best baseline. A cheap fix needs no replays: add the missing tables from
the cached records. The remaining findings are disclosures that REPORT has to carry.

## Method and my own evidence

I ran no replays. The test split may be evaluated only once, and the lens questions are answered
by the records the stage already produced. Everything below comes from my own stdlib scripts, which
read the raw cache records directly (`runs/cache/results/<kk>/<key>.json`). None of them imports
`learned_routing` or reads the stage's analysis outputs. Scripts and outputs are in
`runs/audits/final-completeness/{scripts,out}/`.

| Check | Result | Evidence |
|---|---|---|
| **Test touched once.** Every cache record whose `cell_id` is a test cell, scanning 2,151,574 records. | 30,060 test-cell records = 5,220 tuning + 900 AIS + 10,800 lag + 900 lag-0 + 4 × 3,060 timing, exactly the expected set. 0 errors, every per-request file present. Earliest record 21:45:33; `facts/finalists.json` was frozen at 21:24:35, so 0 records predate the freeze. | `out/test_cache_rows.jsonl`, `scripts/scan_test_cache.py` |
| **Test-cell outcomes outside the cache.** Every other run root: audit private roots, returned shards, result jsonl files. | Only the stage's own `st-*` returns (all post-freeze), cell-file drafts, the static live-loadgen input checks (no replay outcomes), and 5 pre-freeze default-only AgentX lowering validations on test play pools (F7). | `out/other_hits.jsonl`, `scripts/scan_other_roots.py` |
| **Duplicate parity.** Every record returned by every `st-*` shard (first wave and r2) against the cache record with the same key, all fields except environment fields, including `per_request_canonical_sha256`. | 31,508 / 31,508 identical; 1,448 of them are duplicates of the cached source copy. This also closes a gap in the stage's own dup-parity check (F10). | `out/dup_parity_all.json`, `scripts/dup_parity_all.py` |
| **Headline recomputation** (HEADLINE_TEST primary). | M1-v2 vs ramjet: segment mean +0.031555, 10 of 12 segments ahead, one-sided exact Wilcoxon p = 0.001709, cell mean +0.050584. Equals `facts/test_results.json` to every printed digit. | `out/recompute_out.json`, `scripts/recompute.py` |
| **Intersection-union test over the 11 tuned baselines.** | Max p = 0.0105 (all pass). | same |
| **Per N, per mode, SLO scales, AIS league.** | All equal the stage's values: N6 +0.0296 with 3 of 6 segments ahead, p 0.219; unseen N p 0.0078; per-mode p 0.074, 0.078, 0.0625; SLO 0.5 / 0.75 p 0.065 / 0.117; M2-ais vs M0-ais +0.0305, p 0.0012. AIS default equals tuning default on 180 / 180 goodputs. | same |
| **"Beats every heuristic" at shipped defaults** (not computed by the stage). | M1-v2 beats all 9 heuristics at defaults, default@defaults and round_robin. Max p = 0.0105 (lmetric@defaults, 8 of 12 segments ahead); llm-d-precise-prefix@defaults +0.0316, 12 of 12, p 0.00024. | `out/m1v2_vs_defaults_heuristics.json` |
| **Sensitivity to the 3 FAST25-conversation cells** (+0.35 cell mean). | Excluding them, the headline is +0.0260, 10 of 12 segments ahead, p 0.0017. The headline does not depend on them. | ad hoc rerun of `recompute.py`'s `paired` |
| **Equal budget.** Every phase-2 `history.jsonl` for all finalists and AIS arms. | All 66 restarts: 25 generations, 400 candidate evaluations, 10 validation lines, all on val k0-2. (The 480-eval `m0-s1/s2` and `m1-s1/s2` histories are the pilot's runs of the same name.) | `scripts/budget_check.py` |
| **Val k3-10 freshness.** Cache scan of val cells, k ≥ 3, for the 21 selected policies. | Each has exactly 112 records (14 cells × k3-10). Learned and AIS arms were all created 21:17-21:22 by CPU cluster jobs <job>. The 10 tier-A baselines' records were created 02:49-03:07 the same day (tier-B/C i0); ablabase's at 21:22. No other k3-10 records exist for these policy SHAs. | `out/val_k3_rows.jsonl` |
| **Lag-0 identity on test.** | Lag build equals tuning build for default@defaults and m1: 360 / 360 (per-request digest, goodput, rescore). Lag-build image parity: 4 × 192 / 192 identical, train cells only. | `runs/select_test/robust/lag0.jsonl` cross-check; `runs/select_test/analysis/lagpar/` |
| **A14 constants from TRAIN only.** | `analysis/train_default_L2.jsonl`: 160 / 160 records are train cells. `sla_transfer_spec.json` (21:18) predates the first test record. | |
| **Missing secondary metrics** (F1). | Computed from cached test records: see F1. | `out/secondary_metrics.json`, `scripts/secondary_metrics.py` |

## Coverage matrix

| Requirement | Status |
|---|---|
| **HEADLINE_TEST primary**, P(>) with cluster bootstrap, IQM and geometric mean, per-cell winner map and gap to the virtual best | Done; verified |
| **Strata:** load mode, family, per N with Holm, transform-extrapolation, N = 2 parity, TOST, MDE | Done in `test_results.json`. The handoff omits the non-significant strata (F8). |
| **Robustness:** SLO-scale sweep, A5 lag {0, 50, 200} (+10), timing perturbation on E0 and E0', Kendall τ, LR-14 spread | Done for the 17 router-observable finalists. AIS league: none at all (F2). |
| **A2.2:** TTFT p50/p90 and TTFT by ISL bucket reported | **Missing (F1)** |
| **PLAN secondary metrics:** ITL percentiles, throughput, prefix reuse, agentic trajectory latency | **Missing (F1)** |
| **A12** M1-default-init | Done (the identical-policy row) |
| **A14** SLA transfer | Done for all 32 non-default policies × 10 variants |
| **A15** M1-noaff: retrain, post-hoc check, val k3-10, test row, lag | Done |
| **A16** AIS league: selection, test, secondary comparisons | Done; no robustness (F2) |
| **A17** M1-v2, symmetric selection, LR-11 disclosure | Done; 5 headline arms, 314 validated variants |
| **A19** token-weighted secondary | Done for every policy |
| **A13.2** live finalist runs | **Not done** (pending at top level; F11) |
| **Gate tiers A, B, C** | Tier A complete. B: M2 rank 2 and M1-continued. C: rank 3-4 gate failed as registered; ablation a run; ablation b skipped (F11); τ / hash_home run on M1 only (F3). |
| **PLAN baselines table** | All except thunderagent, which has no recorded decision (F6) |
| **LR-11:** performance profiles; val-to-test drop for every rung and baseline | Profiles missing; drop missing for the @defaults rows, round_robin and m1u (F9) |
| **LR-12:** knee- and per-worker-matched N extrapolation; gain vs cache pressure; N = 2 separately | N = 2 done; the other two missing (F9) |
| **LR-14:** engine-state distributions; "in AIS-timed simulation" label | Label done; distributions missing (F9) |
| **LR-15:** τ > 0 reported at N = 2/16/32 | Not per N (F3) |

## Findings

### F1 (major): A2.2's mandatory TTFT reporting and PLAN's secondary metrics are absent, and the omitted data is unfavorable to the headline arm

**The requirement.** CONTRACT A2.2 says "TTFT is still reported (p50/p90, and by ISL bucket) but is
not part of 'good'". PLAN's Objective lists TTFT and ITL percentiles, throughput, prefix-reuse ratio
and agentic trajectory latency as secondary metrics.

**What the stage produced.**
- `facts/test_results.json` contains no TTFT, ITL-percentile, throughput, prefix-reuse or
  trajectory-latency result. The only "ttft" string in it is the A14 variant name `ttft_len_itl`.
  Its `isl_strata` holds only `mean_good_frac_window` for 4 policies × 2 buckets.
- The REPORT handoff ("What REPORT must state") does not mention TTFT at all.

**What the data shows.** My own computation from the cached test records uses the headline's
cell → segment aggregation of the log-ratio, where a positive value means M1-v2 is higher
(`out/secondary_metrics.json`). Against ramjet, the val-best baseline:

| Metric | Log-ratio | Segments where M1-v2 is higher |
|---|---|---|
| TTFT p90, 32K–64K prompts | +0.135 (about +14%) | 11 of 12 |
| TTFT p90, ≥ 64K prompts | +0.033 | 4 of 8 |
| TTFT p50, all prompts | +0.006 | 9 of 12 |
| TTFT p99, all prompts | −0.082 | 4 of 12 |
| Prefix reuse | −0.024 | 1 of 12 |

Good fraction in window, M1-v2 minus ramjet:
- prompts under 8K: +4.3 percentage points;
- 32K–64K: −3.8 points;
- ≥ 64K: −3.7 points.

Against default, TTFT is much better: the p50 log-ratio is −0.57.

**Why it matters.** The learned arm trades long-prompt latency for short-request goodput relative to
the val-best baseline. LR-13 and A19 already point at this trade, but REPORT can't show its TTFT
dimension, which the contract requires, from the stage's facts.

**Fix (no replays).**
- Add a `secondary_metrics` section to `facts/test_results.json` from the cached records and their
  `per_request` rows. For every finalist and default@defaults, include:
  - TTFT p50/p90, overall and by ISL bucket;
  - ITL p90 and per-token p99;
  - throughput;
  - prefix reuse;
  - AgentX per-play trajectory latency;
  - paired segment log-ratios against the val-best baseline and default.
- Add "TTFT by ISL bucket vs ramjet (32–64K p90 +14%, 11/12 segments)" to the REPORT must-state list.

### F2 (minor): the AIS league has no robustness evidence at all, and the disclosure understates this

- The robustness set (`facts/finalists.json` robustness.set, `facts/robustness.json` set) holds only
  the 17 router-observable policies.
- Lag could not run, because the lag build predates feature_set v3. That part is disclosed.
- The four timing conditions could have run on the AIS build but were never registered or run. That
  part is undisclosed: REGISTRATION, robustness.json and the handoff say only "not lag-tested".
- So the secondary claim "M2-ais vs M0-ais +0.0305 (p 0.0012)" has no LR-14 assessment.

**Fix.** REPORT states that the AIS league has neither lag nor timing robustness, so under LR-14
even its direction is unverified under perturbation. Run no new test replays.

### F3 (minor): LR-14's herding guards were run on M1, not the headline arm, and LR-15's τ > 0 per-N reporting is missing

- The τ 0.747 / 2.75 and hash_home variants are M1 (m1c-s1 g20) only. That was registered before
  M1-v2 won, and it is disclosed in DEVIATIONS.
- LR-15 action 3 requires τ > 0 to be reported at N = 2/16/32. `robustness.json` m1_lr14_variants
  gives only pooled means.

**Fix.** REPORT states that the guards apply to M1, not M1-v2. Compute the τ-variant deltas per N
from the existing lag records.

### F4 (minor): lag-0 identity on the lag build is assumed, not shown, for the headline arm

- The lag-0 column of the 17-policy robustness ranking reuses tuning-build records.
- Identity of the lag build at lag 0 has been shown on test for default and M1 (v1) only (360/360).
- On val, the sim-exploitation audit additionally showed it for ramjet, llmdpp, stickybounded,
  twotier, m0 and lmetric (378/378).
- It is not shown for:
  - M1-v2 (feature_set v2);
  - m2r2 (named-source context);
  - ablam1 and ablabase (`router_queue_threshold`);
  - chwbl, dualmap, llmdob and stickyhard.
- Lag conditions are paired within the lag build, so the sign-under-lag claims don't depend on this.
  The "nominal vs lag" columns do.

**Fix.** Disclose it. Optionally, show M1-v2 at lag 0 equals the tuning build on train or val cells
(never test).

### F5 (minor): the pre-registration ordering of REGISTRATION.json is unverifiable, and its own timestamp contradicts it

The file says it was written "~21:25 PDT, BEFORE any val k3-10 record … exists". The recorded
timeline:

| Time | Event |
|---|---|
| 21:16:41 | val k3-10 bundles built |
| 21:17:05 / 21:17:34 | CPU cluster jobs submitted (`facts/remote.json` allocations[111-112]) |
| 21:22:22 / 21:22:44 | results fetched |
| 21:23:39 | `selection_final.json` written, recording registration sha `cfdfc10a`; no copy of that version is preserved |
| 21:24:07 | REGISTRATION.json final mtime, including the addendum (whose own text says "~21:35") |

Both registered decisions were outcome-neutral on the k3-10 values:
- excluding m1cont: 0.1672 < M1-v2's 0.1783;
- including ablabase: 0.1405 < ramjet's 0.1411.

Relatedly, `scripts/analyze_test.py` was last edited at 21:36:48, 13 s after the first test shard
landed locally (21:36:35). My independent recomputation matches the stage exactly, so this changes
nothing.

**Fix.** Correct the wording in REGISTRATION.json and DEVIATIONS to "registered at about 21:17-21:23;
ordering relative to the val k3-10 results is not evidenced; outcome-neutral". REPORT should not
claim strict pre-registration for these two choices.

### F6 (minor): the thunderagent baseline was dropped without a recorded decision

- PLAN's baselines table lists `thunderagent` (selection half), "reported as such or skipped".
- It is a linked catalog type (`facts/setup.json`), and its selection runs deterministically in replay
  (`audits/setup/ais-and-policy-r0.md` "Results / Repeats").
- No stage evaluated it, and no stage recorded the skip: it appears in no DEVIATIONS entry and nowhere
  in the gate, pilot or tierA records.

**Fix.** Add a DEVIATIONS entry. REPORT's "beats every heuristic" wording should exclude thunderagent
explicitly. My own check covers every evaluated heuristic, tuned and at defaults: max p 0.0105.

### F7 (minor): undisclosed pre-freeze default-only replays on AgentX test play pools

The A3 sidecar validated the lowering with `default-seed1` replays on test-pool plays:
- `runs/agentx_lowered/fix_r0/validate/n{4,8}-closed-lanes{20,40}-test-*-def.json`, 4 runs at
  10-02 17:20-17:21, labeled with today's test cell IDs `agentx-T1-think4.0-n4-lanes-L2` and
  `agentx-T2-osl2.5-n8-lanes-L2`;
- `runs/agentx_lowered/validate/n16-open-r0.002-test-def.json`.

These ran before calibration froze test.jsonl (10-02 21:19, re-frozen 10-03 01:53). They used
default only and pre-calibration loads, and compared no policies, so they cannot have favored any
policy. But they contradict the freeze rule "Nobody evaluates test cells until the test stage".

**Fix.** Disclose this in DEVIATIONS and REPORT.

### F8 (minor): the handoff omits the non-significant strata

The test-stage summary highlights unseen N (p 0.008) but omits:
- **Transform extrapolation** (split axis 2, 10 cells): +0.025, 4 of 7 segments ahead, p 0.109,
  P(>) 0.57, CI [0.14, 0.86]. Generalization to held-out transform values is not established.
- **N = 6:** 3 of 6 segments ahead, p 0.219, besides being selection-exposed.
- **Per load mode:** p 0.074 / 0.078 / 0.0625, each non-significant (descriptive by design).

**Fix.** Add these to REPORT's must-state list.

### F9 (minor): LESSONS reporting actions not produced

| Lesson | Missing item |
|---|---|
| LR-11 action 2 | Performance profiles |
| LR-11 action 2 | Val-to-test drop for the 9 @defaults heuristics, round_robin and m1u; `val_k3_10` is null for them in the leaderboard |
| LR-12 action 1 | N extrapolation at per-worker-matched loads. Calibration made only knee-matched cells, so state that as a limitation. |
| LR-12 action 2 | Learned-vs-default gain against cache pressure. Each cell carries `cache_pressure_ref`. |
| LR-14 action 4 | Engine-state distributions (batch, context, KV use) for learned vs default, and the flag for weakly validated AIS regions above 32K context |

**Fix.** Compute these in REPORT from the existing records, or state each as not done.

### F10 (minor): execution bookkeeping

- The ingest-race incident (one record ingested without its per-request rows, then restored) is
  recorded only in `facts/robustness.json` execution, not in DEVIATIONS as the handoff says.
- Neither harness defect is in `facts/UPSTREAM_FOLLOWUPS.md`: E0 table contention, and partial ingest
  during the sync window. Entry #13 sets the precedent for campaign-harness entries.
- The stage's `analysis/dup_parity/st-rob-s1.2-r2.json` compared shard 0 only. The 109 duplicates in
  shard 1 were unchecked. My all-shard check finds them identical.

**Fix.** Add the DEVIATIONS entry and the two UPSTREAM_FOLLOWUPS rows.

### F11 (minor): modalities not evaluated, to disclose

- **AgentX open loop.** AgentX was evaluated only in closed-loop lanes in every split. A3's open-loop
  Poisson play arrivals were built but never used in any cell.
- **Ablation b.** It was skipped by the cut order, but relay i8 recorded free capacity. The stated
  reason was implementation cost, not a compute shortfall.
- **Live runs.** A13.2 live finalist runs are not done yet, so every result is simulation-only.

**Fix.** REPORT discloses each item. The live runs stay validation-only, never selection.

## Additions to the stage's "What REPORT must state"

- TTFT p50/p90 overall and by ISL bucket, plus the PLAN secondary metrics (F1), including M1-v2's
  long-prompt TTFT and good-fraction deficit against ramjet.
- The AIS league has no lag and no timing robustness (F2).
- The τ and hash_home guards cover M1 only (F3).
- Non-significant strata: transform extrapolation, N = 6, per mode (F8).
- thunderagent was not evaluated (F6), and AgentX was evaluated in lanes only (F11).
- Pre-freeze default-only AgentX test-pool validation runs (F7).
- The selection-registration timing caveat (F5).
- M1-v2 beats every heuristic at shipped defaults too: max p 0.0105, from this audit.

## Scratch

`runs/audits/final-completeness/out/` (about 38 MB) holds the scan outputs and is listed in
`CLEANUP.md`. Working copies also sit in the session scratchpad.
