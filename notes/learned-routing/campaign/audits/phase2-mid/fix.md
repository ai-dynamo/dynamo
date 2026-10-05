# Phase2-mid fix

- **Stage:** fixer at checkpoint phase2-mid, 2026-10-03 21:43 PDT to 2026-10-04 ~02:45 PDT.
- **Status:** COMPLETE. All five majors are fixed; nothing is rebutted. The constrained M1 removes
  the dump worker and the load attraction. It still trips the pre-set gaming rules through size
  segregation, as gaming F2 predicted, so it carries flags (see "Re-audit result").
- **Inputs:** `audits/phase2-mid/{overfit-selection,gaming-concentration,sim-exploitation}.md`.
- **Machine-readable record:** `facts/phase2_mid_fix.json`. Evidence: `runs/phase2-fix/`
  (`scripts/`, `checks/`, `rescore_timing/`, `window_guards/`, `gaming_reaudit/`, `drift_verify/`).
- **WT commit:** <commit-42> (harness: space `clamp`, `rescore_timing`, `window_guards`).

## Verdicts

| Lens | ID | Severity | Verdict | Action |
|---|---|---|---|---|
| overfit-selection | F1 | major | VALID | Record fixed: LR-15 drift marked INCONCLUSIVE, with no evidence of drift |
| gaming-concentration | F1 | major | VALID | A11.2 re-run of m1-s1/s2/s3 with theta3, theta4, theta5, theta7 <= 0 |
| gaming-concentration | F2 | major | VALID | Re-audit pre-registered and run on the constrained selection; REPORT flags pre-registered |
| sim-exploitation | F1 | major | VALID | Covered by the same re-run; ISL-strata and share flags pre-registered |
| sim-exploitation | F2 | major | VALID | `learned_routing.rescore_timing` scores perturbed records under both E0s; LR-14 spread pre-registered on E0' |

Nothing was rebutted. Every number I re-derived from the findings matched the auditors' exactly.

## overfit-selection F1: the LR-15 drift record

**Verification.** My own lr-eval of the 4 drift thetas, the pre-fix selected M1 (m1-s2 g15) and the
pre-fix A12 row (m1-s1 g25) on all 14 val cells at k0-2 (294 records, 171 from the cache, 0 errors;
`runs/phase2-fix/drift_verify/`, `scripts/drift_verify.py`) gives pooled-minus-specialist +0.0125,
+0.0216, +0.0070, +0.0137 (L1, L3, N4, N8) and A12-minus-specialist +0.0066, +0.0215, -0.0010,
+0.0151. All 8 equal the audit's `drift_vs_pooled_k0_2.json` to 1e-12. The l1-on-l3 loss 0.0865 has
SE 0.0645 over its 3 cells, so it is 1.34 SE (`checks/drift_verify.json`).

**Fix.**
- `facts/tierA.json` `drift` keeps the matrix and the raw `within_mde: false`, and adds
  `within_mde_is_raw_rule_output: true` and `interpretation`:
  - verdict INCONCLUSIVE, `evidence_of_drift: false`, `use_for_m2_context_sources: false`;
  - the four reasons with their numbers, read from the audit's output files;
  - REPORT wording ("underpowered at B/4"; do not cite 0.087 as drift).
- The writer `runs/phase2/scripts/tier_a_facts.py` emits this on every rewrite, so a later relay
  cannot drop it. `runs/phase2/ORCHESTRATOR.md`'s drift section points to it.
- No rerun. A full-budget drift rerun stays optional, as the auditor said.

## sim-exploitation F2: timing robustness scored against nominal E0

**Verification.** `e0.py` does ignore `engine_overrides`. A new replay test
(`tests/test_replay_e2e.py::test_timing_perturbed_idle_requests_match_the_consistent_e0`) runs six
idle no-reuse requests (ISL 1 to 20,000, up to 3 prefill chunks) on an engine with speedup 0.8 and
decode speedup 1.25:
- every request's e2e equals E0' = prefill/s + decode/(s d) to 1e-7;
- lr-eval's own record scores all six bad at S = 1;
- under E0' all six are good.

**Fix (root).**
- `learned_routing.rescore_timing` (WT <commit-42>) rescoring from per-request rows, no replays:
  - for each record, both the nominal-E0 metrics and the consistent-E0 metrics: windowed goodput,
    good fraction, slowdown atom and the SLO-scale `rescore`;
  - the nominal recompute is asserted equal to the record's `goodput_rps_window`;
  - records are matched to cells by id and content SHA.
- `E0Table.prefill_ms` exposes the prefill part of E0.
- Checked against the auditor's reference on all 9 of their timing conditions (378 records each).
  The consistent-E0 windowed goodput is identical in 3,402 of 3,402 records, and every nominal
  recompute matched (`checks/rescore_timing_compare.json`). The val objective M1 minus the best
  baseline reproduces the audit's table from the module's output:

  | Condition | Nominal E0 | Consistent E0 |
  |---|---|---|
  | s0.8 | +0.0431 | +0.0257 |
  | s0.9 | +0.0507 | +0.0397 |
  | s1.1 | +0.0089 | +0.0115 |
  | s1.2 | +0.0111 | +0.0142 |
  | live | +0.0192 | +0.0232 |

- **Pre-registered** (`facts/phase2_mid_fix.json` `preregistration.timing_robustness`, and a dated
  addendum in `facts/HEADLINE_TEST.json` that leaves the primary test unchanged):
  - every timing-perturbed test record is scored under both E0s, and both are reported;
  - the LR-14 spread and Kendall tau are defined on E0';
  - AgentX at speedup 0.8 is flagged as degenerate under nominal E0.

## gaming-concentration F1 and sim-exploitation F1: sign-constrained M1 (A11.2)

**Verification.** From the unconstrained restarts' histories (`checks/m1u_signs.json`):
- the selected checkpoints have theta7 = +0.0707 (s1 g25), +0.0878 (s2 g15) and +0.669 (s3 g25);
- the active-prefill coefficient theta3 + theta7 P/8192 turns positive above 98.8K, 67.6K and 90.2K
  prompt tokens;
- theta3 > 0 or theta4 > 0 at validated checkpoints s1 g5-g20, s2 g5-g20 and s3 g5.

My window-basis guard tool (below) reproduces the gaming audit's guard metrics exactly on its 882
records, including the dump workers and the ISL >= 60K good fractions. The finding stands as
written.

**Constraint.**
- theta3 (active_prefill_tokens_k), theta4 (kv_load_frac), theta5 (active_requests_s) and theta7
  (isl_x_prefill_load) are each <= 0, with theta0 pinned at -1.
- So no load signal can attract for any prompt length.
- It implies the sim auditor's theta3 + 16 theta7 <= 0 at max_model_len 131072.

**How (a clamp, not a box bound).** The auditor asked the re-run stage to check the bound transform
for starts on the bound. A box bound of 0 puts the s1 and s3 starts (those coefficients are 0) on
pycma's BoundTransform edge, where its slope is zero. With the real lr-train CMA options, the
generation-0 physical steps on the four coordinates are 8e-5 to 3.5e-4 s-units, against 0.0099 to
0.017 in the interior (`runs/phase2-fix/checks/bound_transform.json`).
- So `space.yaml` gained an optional `clamp` (WT <commit-42>): `x = min(max(x, low), high)` after
  the transform.
- The re-run spaces `runs/phase2/spaces/m1c_*.yaml` are the original spaces plus
  `clamp: [null, 0.0]` on theta3/4/5/7.
- The internal CMA-ES search (box, start, sigma, seed) is identical to the unconstrained runs.
  Generation 0 of m1c-s1 and m1c-s3 asked exactly m1-s1's and m1-s3's 16 internal points
  (`checks/gen0_identity.json`); only the decoded policies differ.
- Tests: `tests/test_space_train.py` (clamp decode and validation; clamped and plain runs ask the
  same first generation).

**Inits (A11.1, disclosed under A12).**
- s1: theta0 = -e0, unchanged.
- s3: cache-heavy, unchanged and already feasible.
- s2: the behavior-clone init projected onto the constraint, theta3 +1.277 -> 0 and theta4 +12.53 -> 0,
  everything else unchanged (`runs/phase2/inits/m1c_s2.json`). On the BC fit's train tables, its
  agreement with llm-d-precise-prefix is 0.8793 (original 0.8748, reproduced exactly) and it flips
  0.425 of default's decisions (original 0.464) (`checks/projected_init_diag.json`).

**Equal footing.** Same seeds 1/2/3, B = 400 (25 generations x popsize 16), K = 2, replicate pool 8,
reeval-frac 0.125, validation every 5 generations on val k0-2, clipped log-ratio, pinned CMA
numerics, all 34 train cells (MANIFEST args identical to the m1 runs').

**Orchestration.**
- `runs/phase2/MANIFEST.json`:
  - m1-s1/s2/s3 keep their run ids under policy key `m1u` (unconstrained, gaming-flagged);
  - m1c-s1/s2/s3 are appended under policy key `m1`;
  - `scripts/make_manifest.py` regenerates exactly this (checked);
  - pre-fix copies of the manifest, state, selection, facts and scripts are in
    `runs/phase2-fix/checks/`.
- `p2orch.py select --tier A` therefore picks the headline M1 from the constrained restarts only,
  and `selection/tierA.json` already shows `m1u` (m1-s2 g15) and `m1` (incomplete).
- `tier_a_facts.py` puts the A12 row on m1c-s1 and keeps the unconstrained rows as
  `a12_m1_default_init_unconstrained` and per_policy `m1u` with a `gaming_flag`.
- `facts/tierA.json` is back to `in_progress` until the re-runs finish.
- The bundle `p2a` got the new `space.py` (the only harness file that differed from WT HEAD; pre-fix
  copy kept). CPU cluster jobs:

  | Run | Job | Node |
  |---|---|---|
  | m1c-s1 | <job> | <node> |
  | m1c-s2 | <job> | <node> |
  | m1c-s3 | <job> | <node> |

  All three are EPYC 7702P-class, `--time 12:00:00`, recorded in `facts/remote.json`. The epyc9654p
  nodes were left alone: this account's 579G jobs from another workload hold them. p2orch loop PID
  2000173.

**Results.**
- **Runs.** All three jobs COMPLETED between 02:16 and 02:24 PDT (4 h 24 min to 4 h 31 min each)
  and were ingested with 0 rejected: 30,664, 30,264 and 30,199 records.
- **Equal budget.** `p2orch budget` shows `tier_A_equal_budget: true` over all 36 tier-A train
  runs, signature (400 evals, 5 validation checkpoints). UH re-evaluation ran in 21, 16 and 17 of 25
  generations (diagnostic only).
- **Best val k0-2 per restart** (val k0-2 is selection-exposed):

  | Restart | Constrained | Unconstrained |
  |---|---|---|
  | s1 | 0.16599 | 0.16148 |
  | s2 | 0.16238 | 0.16363 |
  | s3 | 0.13750 | 0.15782 |

  The restart spread is 0.0285.
- **LR-10 selection.**
  - `selection/tierA.json` `m1` is m1c-s1 g20 mean, val k0-2 0.16599, with
    theta = [-1, 16.54, -2.274, 0, 0, -39.23, 0.2062, 0].
  - theta3, theta4 and theta7 sit at the clamp; theta5 is negative.
  - It comes from the s1 restart (theta0 = -e0), so the A12 row M1-default-init is the same config.
    The BC init (s2) no longer supplies the headline M1.
  - `checks/selection_recheck_fix.json` recomputed all 13 rows (11 policies, m1u and A12) from
    cache records exactly.
  - `facts/tierA.json` status: complete.
- **Val k0-2 comparison** (cache records, `runs/phase2-fix/checks/val_compare.json`; MDE 0.038;
  selection-exposed, and fresh val k3-10 decides the headline arm in Select+Test):

  | Constrained M1 minus | Cell mean | Segment mean (SE) | Segments ahead | Mooncake | AgentX | Sessions |
  |---|---|---|---|---|---|---|
  | ramjet | +0.0240 | +0.0152 (0.0101) | 3/6 | +0.0620 | +0.0079 | -0.0012 |
  | llmdpp | +0.0245 | +0.0068 (0.0148) | 3/6 | +0.0764 | -0.0165 | +0.0056 |
  | m1u (m1-s2 g15) | +0.0024 | -0.0017 (0.0038) | 2/6 | +0.0133 | -0.0073 | -0.0008 |

  The lead still rests on Mooncake and still trails llmdpp on AgentX (overfit F2's caveat stands).

## gaming-concentration F2: re-audit of the constrained M1

**Pre-registered** before the constrained selection was known (`facts/phase2_mid_fix.json`
`preregistration.gaming_reaudit`, `report_flags`):
- the gaming audit's own scripts, unchanged, on val k0-2 for the constrained selected M1 and the
  constrained A12 row, with the PRESET.md thresholds;
- plus `learned_routing.window_guards`;
- outcome rule: A11.2 prescribes one remedy, which is applied, so the constrained M1 stays the M1
  arm. Any remaining flag is reported on every M1 row, and retuning under a different objective is
  an operator decision.

**Tool.** `learned_routing.window_guards` (WT <commit-42>) recomputes, over exactly the requests the
record's goodput window scores:
- worker share against the cap;
- per-worker good minimum;
- good fractions at ISL >= 32K, >= 64K and the top decile;
- NMI(worker; ISL quartile);
- dump workers.

It uses the harness's own window, good and warm-up rules, and refuses a record whose in-window good
count it cannot reproduce. On the gaming audit's 882 val records it equals the audit's
`guards.json` on every compared field for 882 of 882 records, and its `giant_60000.json` on 100 of
100 (policy, cell) rows (`checks/window_guards_vs_audit.json`).

### Re-audit result

**Setup.** `runs/phase2-fix/gaming_reaudit/`:
- own 42 val k0-2 replays of the constrained M1 in a private root, 0 errors;
- merged with the audit's 882 records;
- the audit's scripts unchanged, with the analyze.py scorer check at 924/924;
- summary in `checks/reaudit_summary.json`.

**Removed.**
- **Dump workers:** 0 on every Mooncake val cell. m1u had 1.0 per replicate at n8-open-L3 and at
  n8-closed-L2.
- **ISL >= 60K good:** 0.920 to 1.000; m1u had 0.425 to 0.966.
- **Giants routed to the worker with the most outstanding prefill:** 0.071 to 0.203; m1u had 0.370
  to 0.580, and chance is 0.125 to 0.25.
- **Minimum per-worker good:** 0.546 to 0.905; m1u had 0.000 to 0.900.

**Remaining flags.** They go on every M1 row.

**Concentration.** The share exceeds the cap on 4 of 5 Mooncake cells, and at N8 it is higher than
m1u's. The hot worker's prefill-token share stays near 1/N, so the concentration is in requests,
not in work.

| Cell | Share (m1c) | Share (m1u) | Cap |
|---|---|---|---|
| n8-open-L3 | 0.558 | 0.402 | 0.25 |
| n8-closed-L2 | 0.567 | 0.364 | 0.25 |
| n6-open-L2 | 0.399 | 0.369 | 0.333 |
| islu1.25-n6-closed-L2 | 0.450 | 0.454 | 0.333 |

**Segregation.** NMI(worker; ISL quartile) is 0.211 at n8-closed-L2, 0.228 at n8-open-L3 and 0.120
at islu1.25-n6. The contender baselines are at or below 0.095.

**Long requests at n8-open-L3:**

| ISL class | m1c | default | ramjet | llmdpp |
|---|---|---|---|---|
| >= 32K good | 0.826 | 0.995 | 0.963 | 0.922 |
| >= 64K good | 0.910 | 1.000 | 1.000 | 0.846 |
| Top decile good | 0.821 | 0.880 | 0.882 | 0.920 |

**Pre-set rules.** S1-S4 still declare gaming on 3 cells:
- n8-closed-L2: S4;
- n8-open-L3: S1 and S4;
- islu1.25-n6-closed-L2: S1 and S4.

65-69% of the good-to-bad transitions against default fall in the top ISL quartile, which holds
25% of requests.

**Decision.** This follows the pre-registered outcome rule. The constrained M1 stays the M1 arm,
because A11.2's single remedy has been applied. The flags above are reported. Whether to retune
under a size-aware objective or constraint is a design decision for the operator, and the stage
escalates it.

## Pre-registered rules for later stages (summary of `facts/phase2_mid_fix.json`)

- **M1 identity.** Policy key `m1` = constrained (headline, warm starts, A12); `m1u` = disclosed,
  gaming-flagged rows only.
- **Derived arms** use the same sign rule:
  - m1cont and M2's theta block: same clamps, warm start from the constrained selection. M2's
    context term is free but re-audited with the same tools.
  - M1-noaff: the m1c spaces with theta6 pinned at 0.
  - M1-v2: clamp <= 0 on v2 features 3, 4, 5, 7, 8, 10 and 12-21 (checked against FEATURES.md).
  - Recommended for the AIS league (not started): the v3 load-increasing features.
- **REPORT flags** for every M1 row and finalist, in the test pass and every robustness condition:
  - concentration: share > cap;
  - segregation: NMI > 0.10;
  - long-request sacrifice: >= 32K / >= 64K / top-decile good more than 0.05 below default@defaults;
  - dump workers;
  - the ISL strata.
- **Timing robustness:** scored under both E0s, spread on E0'.
- **LR-15:** inconclusive; not used for M2 sources.
- **LR-11:** count both the m1u (30) and m1c (30) validated candidates.

## Minor findings

Out of scope here; left open for the Select+Test and report stages:
- gaming F3-F6;
- overfit F2-F5;
- sim F3, F5-F7.

Partly addressed:
- gaming F4: `window_guards` adds the missing guard views post hoc, without changing record content
  or the cache key;
- sim F4: recorded as `facts/UPSTREAM_FOLLOWUPS.md` #17.

## Files and side effects

**New harness code.** WT commit <commit-42> (signed off, not pushed). It holds the space `clamp`,
`rescore_timing.py`, `window_guards.py`, `E0Table.prefill_ms`, and tests in `test_space_train.py`,
`test_replay_e2e.py` and `test_window_guards.py`. The harness suite passes: 165 tests, including
the replay tests.

**Campaign files changed.**
- **Run setup:**
  - `runs/phase2/MANIFEST.json`, `runs/phase2/scripts/{make_manifest,tier_a_facts}.py`;
  - `runs/phase2/spaces/m1c_*.yaml`, `runs/phase2/inits/m1c_s2.json`;
  - `runs/phase2/ORCHESTRATOR.md`;
  - `runs/remote/bundles/p2a/site/learned_routing/space.py`.
- **Facts:**
  - `facts/{tierA,phase2_mid_fix}.json`;
  - `facts/HEADLINE_TEST.json` (addendum only);
  - `facts/{DEVIATIONS,UPSTREAM_FOLLOWUPS}.md`.

Pre-fix copies are in `runs/phase2-fix/checks/`.

**CPU cluster jobs and processes.**
- Jobs <job>, <job> and <job>, all COMPLETED and recorded in `facts/remote.json`; none held,
  none cancelled.
- p2orch loop PID 2000173 exited by itself at 02:25 PDT.
- 123 deterministic drift-verify replays went into the campaign cache. No test cell was touched.

**Scratch.** Listed in `CLEANUP.md`.
