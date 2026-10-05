# Audit: checkpoint "live", lens "claims"

- **Auditor:** independent adversarial auditor (phase 3, A20). Written 2026-10-05 ~07:15 PDT.
- **Scope:** `report/LIVE.md`, `facts/live_results.json` and `report/REPORT.md` (§15, plus the 10 lines
  the live-analysis stage changed in §1–§14), checked against `facts/live_plan.json` (the
  pre-registration), the raw per-run scores under `runs/live/finalists/scores/`, the frozen test pass
  `runs/select_test/test/tuning.jsonl`, and the frozen facts.
- **Verdict: FAIL.** There is 1 major finding (F1) and 7 minor ones. All registered numbers reproduce
  exactly, every pre-registered statistic is reported, the shortfalls that matter are disclosed, and
  the frozen headline is untouched. The failure is a framing problem in the narrative: the text names
  only the sim/live departures that widen the learned arms' lead and leaves out two that shrink
  relative gaps. One edit pass fixes it, with no new runs and no change to any number.

## What I re-derived myself (all match)

Scripts and outputs are in the session scratchpad (`audit/recompute.py`, `audit/a19.py`,
`audit/sens.py`; listed in CLEANUP).

| Item | My value | Stated | Source of my value |
|---|---|---|---|
| Valid runs | 42 cell runs (2 a1 attempts invalid, `payload_exit_1`) | 42/42 | `scores/*/summary.json` `valid` |
| Sim counterparts | 108/108 equal to the frozen test pass | "nothing re-simulated" | `tuning.jsonl` by policy_sha and repeat |
| Per-cell τ_b (k0) | 0.867, 0.867, 0.867, 0.867, 1.000, 0.867 | same | raw scores + registered D rule |
| Pooled τ_b, perm p, points, mean per-cell | 1.000; 1/720 = 0.00139; 0.738; 0.889 | same | |
| Sensitivity k0-2 / lag10 / lag50 | pooled 1.0/1.0/1.0; points 0.839/0.783/0.747; mean 0.889/0.950/0.911 | same | `sim_counterparts.json` derived |
| σ_live; 2√2σ | 0.009043; 0.02558 | 0.0090; 0.0256 | 6 default pairs |
| M1-v2 − ramjet, 5 informative cells | live +0.077/+0.190/+0.133/+0.663/+0.035, mean +0.2194; sim mean +0.1625; 5/5 positive and beyond 2√2σ | same | |
| A19 | pooled τ_b exactly 0.600 (12 concordant, 3 discordant); 3/5 positive, mean −0.0350; sim 2/5, mean −0.0439; signs agree 4/5 | same | |
| Mean Δ vs default, live (sim) | M1-v2 +0.273 (+0.219), M1 +0.256 (+0.218), ramjet +0.090 (+0.084), M0 +0.059 (+0.066), rr −0.295 (−0.332) | same | |
| Discordances | 5, all with \|sim diff\| ≤ 0.011; 2 beyond noise (M1-v2 over M1: Mooncake open L3 +0.032, conv +0.062) | same | |
| Close pairs within noise | M1-v2 − M1 4/6; M0 − ramjet 4/6 | same | |

Other checks:

- **Execution integrity.** Every pair's executed order matches the registered `run_order`. All 48
  pairs have `policy_evidence.ok = true`, and the policy YAML sha256 matches the plan. Each serve
  frontend's command line equals its check frontend's, apart from the dump path.
- **Registration and freezing.** `facts/live_plan.json` sha256 is `9c01e4a5…`, as recorded, and its
  mtime (02:03:24) precedes the first finalist job. `finalists.json` (e0bae29d…), `test_results.json`
  (mtime 01:17), `robustness.json` and `HEADLINE_TEST.json` were not modified by the live stages.
- **REPORT.** `assemble_report.py`, re-run in scratch, reproduces `report/REPORT.md` byte for byte.
  Against the pre-edit copy, the §1 headline block is unchanged, and no number token from §1–§14 was
  lost. The "lost" `11` and `16` tokens moved with the appendix, which now sits after §15.
- **Registered statistics.** Every item in `live_plan.json` `analysis` and `scoring.secondary` is
  reported:
  - paired deltas, with mean and range;
  - cross-node flags;
  - per-cell and pooled τ, the permutation p, the point and mean-per-cell variants, and the 3
    sensitivities;
  - sign agreement, raw and beyond noise;
  - per-cell noise with its flags, and calibration ratios;
  - A19, and live-idle E0 (pooled and own-node);
  - TTFT and ITL ratios, prefix hit rate (partial; the deviation is recorded), worker share, lateness,
    and AIPerf OSL errors.
- **Wording rules.** "Agrees" (τ 1.00 ≥ 0.6) and "M1-v2 > ramjet holds live" (5/5, mean > 0) are
  applied as registered. A19 "does not hold" is applied correctly.
- **A18.** `publish/scan.py tree` on copies of LIVE.md and `data/tables/live_validation.md` is CLEAN.
  LIVE.md contains no job IDs, node names or internal paths. (`facts/live_results.json` does contain
  job IDs; it is internal, and the paper must not copy from it verbatim.)

## Findings

### F1 (major): the departure narrative is selective; two relative-gap failures are not described

**Claim.** LIVE.md §15.2 (and REPORT §15.2) says "Two departures stand out". It then names FAST25
conversation, where the non-learned policies run at 0.72–0.81 of sim, "which widens the learned arms'
lead there", and sessions round_robin at 2.18×, and concludes "the relative comparisons below are the
intended use". Two departures in the relative deltas, the very quantity the validation is about, are
not described anywhere:

1. **Sessions, default@defaults.** Live default runs at **1.073×** sim (8.249 vs 7.691; default_a
   8.224, default_b 8.274), while every other policy is at 0.997–1.000. As a result, the simulated gain
   of every tuned policy over default on this cell (+0.084 to +0.086) is **+0.011 to +0.016 live**.
   That is about 0.07 lower, about 5.5 × sd(Δ) = √2·σ_live = 0.0128. The plan registered this cell
   to "check the ceiling and the default and round-robin gaps live" (`live_plan.json` `cell_choice`).
   The default-gap half of that check failed, and LIVE.md does not say so. REPORT §3.3 reports a
   simulated sessions-family gain over default of +0.178 for M1-v2 and for ramjet, and the one live
   sessions cell does not reproduce a gain of that kind beyond noise.
2. **FAST25 conversation, round_robin vs default.** The sim gap is −0.132, the live gap −0.010:
   round_robin's deficit vanishes live, an inconsistency of about 9 sd. The conv paragraph frames the
   cell only as widening the learned lead.

In 15.3 these deltas appear only in a neutral list of "the rest" that do not exceed 2 sd. The stage's
returned key facts attribute the four sessions deltas to "(ceiling)". In sim they are not at ceiling
relative to default: sim predicts +0.086. The attribution is wrong, and it could reach the paper
through the stage hand-off.

**Evidence.**
- `facts/live_results.json` `primary.per_cell["sessions-s6-base-n4-open-L2"].live_over_sim`:
  default 1.0726, m0 0.9998, m1 0.9971, m1v2 0.9973, ramjet 0.9967.
- `delta_vs_default_live`: ramjet +0.0161, m1v2 +0.0163, m1 +0.0159, m0 +0.0109. The sim values are
  +0.0864, +0.0860, +0.0858, +0.0842.
- Conv `delta_vs_default_live.round_robin` is −0.0101, against sim −0.1316.
- My recomputation from the raw `score.json` files gives the same values.

**Why major.** The text summarizes where sim and live disagree and names only the disagreement that
favors the learned arms. Gaps over default and the round_robin gap are relative results, the stated
intended use. A paper filled from LIVE.md (`next_stage_notes`) would inherit a one-sided account of
live fidelity.

**Fix (text only, no new runs).**
- In §15.2/15.3, replace "Two departures stand out" with a complete account:
  - (a) conv non-learned policies at 0.72–0.81 of sim, which widens the learned lead and erases
    round_robin's deficit vs default (−0.132 sim, −0.010 live);
  - (b) sessions default at 1.07× sim, which shrinks every tuned policy's gain over default from
    about +0.085 to +0.011–0.016 (within noise);
  - (c) sessions round_robin at 2.18×.
- In §15.7 "cannot show", add: the size of the gains over default (on sessions the simulated gain is
  not reproduced beyond noise live).
- Optional descriptive context, already in `descriptive.per_cell_policy`: vLLM preemptions on conv,
  20–23 for default, ramjet and M0 vs 1–6 for M1 and M1-v2.
- Regenerate through `live_results.py build` and `assemble_report.py`.

### F2 (minor): §1 limit 5 rewrite overstates and drops part of a final-audit limit

The new text says the live runs "reproduce the simulated policy ranking and the sign … on hardware".
The registered wording is "agrees". Per cell, 5 of 6 cells have a discordant pair (τ_b 0.87), so the
ranking is reproduced exactly only for the pooled means. The rewrite also drops the old sentence "AIS
fidelity at the policy level is not live-validated" without carrying its substance forward: absolute
live/sim goodput ranges 0.72–2.18, and the departure is policy-dependent (F1). Nor does it say that
the 6 cells are not a random sample. §1 limits are those "each of which the final audit requires".

**Fix.** Use "agree with" (pooled τ_b 1.00, per cell 0.87–1.00), and add one clause: absolute goodput
is not validated (live/sim 0.72–2.18, policy-dependent), and the cells were not drawn at random.

### F3 (minor): stale or self-contradictory REPORT lines outside §15

- Header line 7 says "Contract and amendments A1–A19"; §15 rests on A20.
- §11 item 1 ("Simulation only. Every result is AIS-timed offline replay") now contradicts §15.
- §13 lists the live runs under "Escalated to the operator, not done:" while saying they have run.
- §9 (line ~897) says the faithful-LMetric follow-up runs on "fresh segments or the live runs, never
  the frozen test cells", but §15's live runs were on frozen test cells. It should read "a new live
  matrix on non-test cells", or otherwise state that the live lane's cells are test cells.

**Fix.** Edit these lines in the template; no number changes.

### F4 (minor): the "cells are not random" context compares unlike statistics and omits selection timing

LIVE.md §15.7 compares the live cells' simulated M1-v2 − ramjet at k0 (a 6-cell mean, +0.135) with
the N = 4 stratum's **segment mean over k0–2** (+0.055). The like-for-like figure, the k0 cell mean
over all 13 N = 4 test cells, is **+0.071**; I recomputed it from `tuning.jsonl`. The live cells
include the 3 largest of the 13 N = 4 simulated gaps (conv +0.470, closed L3 +0.141, open L3 +0.122).
The cells were chosen after the frozen test pass, with per-cell sim results written into the
registration (`live_plan.json` `cells[].sim_k0`). LIVE.md says only "chosen at registration for family
and load-mode coverage". The selection rule itself looks defensible: all N = 4 base Mooncake, conv and
fsyn cells, minus AgentX (no load generator) and the long cells.

**Fix.**
- Report the like-for-like +0.071 next to +0.055.
- State that the cells were chosen after the test pass, with sim outcomes visible.
- Note that they include the three largest N = 4 gaps.

### F5 (minor): "beyond live noise" is not qualified for the uncertainty in σ_live

σ_live comes from 6 pairs, so it has 6 df. Its 95% interval is roughly 0.0058–0.020, which puts
2√2σ in about 0.016–0.056. At the upper end FAST25 synthetic (+0.035) is not beyond noise, so the
count would be 4/5, not 5/5. §15.7's "each time by more than the live noise" states this as shown.
Separately, the "sd ≈ 0.011 against default" (1.22σ) assumes D averages two runs. On conv and
sessions, D is a single same-job default, so the sd is √2σ = 0.0128. The 24/30 count does not change.

**Fix.**
- Add "(σ_live from 6 pairs; 95% CI about 0.006–0.020)".
- Soften "each time" to "in each cell, beyond the point estimate of live noise".
- Note the √2σ sd where D is one run.

### F6 (minor): pooled τ_b = 1.00 at k0 rests on a 0.0009 sim gap; the permutation p is reported for one metric only

The sim pooled means of M1-v2 and M1 are 0.21892 and 0.21800. Had they flipped, the primary would read
0.87. The k0-2 sensitivity (gap 0.0074) also gives 1.00, so the verdict is robust, but the text never
says the "1.00" is knife-edge at k0. The permutation p (0.0014) tests against random orderings of all
6 policies, a null that round_robin and default reject almost trivially. It is labelled descriptive,
correctly. Still, LIVE.md quotes it for the primary but not for A19 (p 0.068 in `live_results.json`)
or for live-idle E0 (0.0014).

**Fix.** One sentence on the M1-v2/M1 sim tie, and the A19 permutation p next to its τ.

### F7 (minor): items the relay hand-off asked to disclose are missing from LIVE.md

`facts/live.json` `finalists_run.handoff` says to "Disclose … the payload_hb.sh wrapper on later
packs, batch instead of interactive for P2/P3". LIVE.md discloses neither. The first is the AIPerf
heartbeat-watchdog threshold, raised from 3 to 60 for P3, P6, P7 and R1 (watchdog only, same argv).
The second can be worded without the partition names. LIVE.md also omits:

- `live_plan.json` notes 7 tie groups (14 requests) in FAST25 synthetic whose order live cannot
  enforce;
- §15.1's "replay's arrival schedule" wording also covers the closed-loop cell, which is
  concurrency-admitted;
- in the lag sensitivities round_robin keeps its k0 value (recorded only in DEVIATIONS).

**Fix.** One "Execution notes" bullet in §15.1.

### F8 (minor): the vLLM prefix-counter mechanism is stated as fact; the evidence is indirect

LIVE.md states as fact that vLLM 0.24.0 re-records a prefix-cache query each time it retries a
waiting request. I could not check the vLLM source locally. The data fit the mechanism: runs with zero
preemptions are still inflated, for example Mooncake open L2 M1-v2, counter 0.108 vs sim 0.377 with 0
preemptions. But preemption-recompute is a second re-query path: live preemptions reach 27 per run,
and LIVE.md does not mention them. The comparable-run subset (14/42, low-queueing cells) is not a
random subset either, so "within 0.009 of replay" is conditional on low queueing.

**Fix.**
- Cite the source location (file and function) or label the mechanism as the stage's reading of the
  source.
- Mention preemption as a second cause.
- Say that the comparable runs are the low-queueing ones.

## Checked and not a finding

- The 2 invalid first attempts were excluded by the registered rule and their reruns used. Using them
  instead changes nothing material (conv D 0.8998 → 0.8930 mean; Mooncake L2 round_robin 3.085 vs
  3.087).
- The headline (§1 quote), `finalists.json` and all frozen test numbers are unchanged. §15 is labelled
  validation-only throughout.
- The A19 "does not hold live", the "does not hold in sim by the same rule" addition (labelled a
  descriptive addition in DEVIATIONS) and the live-idle E0 conclusions match the data.
- The LR-13 cap lists live and sim, the TTFT and ITL ratio ranges, the lateness numbers (35 open-loop
  runs, p99 ≤ 2.6 ms, max 15.2 ms) and the AIPerf OSL errors (18 requests in 16 runs, spread across
  policies) match `live_results.json`.

## Scratch

`<session-scratch>`
(72K) and `.../scratchpad/rep` (1.2M, the scratch REPORT re-assembly). Both are listed in `CLEANUP.md`
and disposable. Nothing was deleted, committed or pushed, and no GPU or CPU job was run.
