# Fix: checkpoint "live"

- **Fixer:** phase 3 (A20), 2026-10-05.
- **Input:** 1 finding passed to the fixer, claims-lens F1 (major). The fidelity lens passed with no blocker or
  major.
- **Result:** F1 is **VALID and FIXED** at its root, the generator `runs/live/finalists/scripts/live_results.py`.
  Nothing is rebutted.
- **What did not change:** no run, no replay, no GPU or CPU job. `facts/finalists.json`, `facts/test_results.json`,
  `facts/robustness.json`, `facts/HEADLINE_TEST.json` and `facts/live_plan.json` have the same sha256 before and
  after (`runs/live-fix/frozen_sha_before.txt`). Every value that `facts/live_results.json` held before is unchanged
  (checked key by key); the fix only adds keys. The REPORT §1 headline block is byte-identical, and no number
  token in REPORT is lost.

## F1 (major, claims): the departure narrative was selective — VALID, FIXED

**Verdict.** Valid. The old §15.2 named two departures: the FAST25-conversation one, which widens the learned
lead, and sessions round_robin. It left out the sessions default@defaults departure, which removes every tuned
policy's gain over default beyond noise, and round_robin's vanished FAST25-conversation deficit. Both are
departures in the relative deltas, the quantity the validation is about.

The auditor's numbers reproduce exactly from `facts/live_results.json`:

- Sessions live/sim: default 1.0726; tuned policies 0.9967–0.9998.
- Sessions Δ vs default: live +0.0109 to +0.0163, against +0.0842 to +0.0864 in sim.
- Conv round_robin Δ vs default: −0.0101 live, against −0.1316 in sim.

**The narrative also missed a broader point.** Departures in the relative deltas are not limited to those two
cells. Under a uniform rule they are common, in both directions. So the fix replaces the hand-picked list with a
rule-based, complete account.

### What changed

All of the following is generated, so any edit goes through `live_results.py build`, then `assemble_report.py`.

1. **New keys in `facts/live_results.json`, `primary.per_cell[c]`:**
   - `default_ref_runs`: the n_D of the default reference D(c, p), as `analyze_live` builds it. `build` asserts
     that recomputing D reproduces `delta_vs_default_live` to 1e-12.
   - `sd_delta_vs_default_live` = σ_live·√(1 + 1/n_D), and `sd_delta_vs_ramjet_live` = σ_live·√2 (for default vs
     ramjet, σ_live·√(1 + 1/n_D)).
   - `departure_vs_{default,ramjet}_live_minus_sim` and `departure_vs_{default,ramjet}_beyond_2sd`.
   - `good_frac_window_live` and `good_frac_window_sim_k0`.
   - `preemptions_live`, per run.

   There is also a new `primary.departures` block with the rule text, the counts and the full entry list.
2. **The departure rule is descriptive and was not registered. It is labelled as such everywhere it appears.**
   - **Rule:** an entry departs when |live Δ − sim Δ| > 2 × the sd of the live Δ.
   - **Why only live noise:** the sim counterpart replays the same requests, so live noise is the only sampling
     term.
   - **Caveat stated in the text:** σ_live itself rests on 6 default pairs.
3. **LIVE.md / REPORT §15.2.** "Two departures stand out" is replaced:
   - **Count.** 18 of 30 Δ vs default depart (9 higher live, 9 lower), and 13 of 30 Δ vs ramjet.
   - **FAST25 conversation.** Every gain over default is larger live, which widens the learned lead over ramjet
     (+0.663 vs +0.470). Round_robin's deficit vs default is all but erased (−0.010 live vs −0.132), within live
     noise, because default@defaults loses the most live. Preemptions per run are 18–24 for default, ramjet and
     M0, 12 for round_robin, and 1–6 for M1 and M1-v2. TTFT p50 live/sim is 1.43–1.76× for the non-learned
     policies, against 1.08–1.13×.
   - **Sessions.** Default runs at 1.07× sim, while the tuned policies are at 0.997–1.000×. Default's in-window good
     fraction is 97.8–98.4% live, against 91.6% in sim; the tuned policies are at 99.9–100% in both. Every tuned
     gain over default is +0.084 to +0.086 in sim but only +0.011 to +0.016 live, inside 2 sd (0.026): **"not
     reproduced beyond live noise"**. Round_robin is at 2.18×, with its deficit shrinking from −1.099 (the clip) to
     −0.770.
   - **Mooncake and FAST25 synthetic.** Round_robin's deficit is larger live on all 4 cells (by 0.033–0.086), and
     every other flagged entry is listed.
   - **Mechanism, labelled a hypothesis and not tested.** The cause is engine timing; fidelity minor F2 supplied
     the reading. Conversation runs closer to prefill saturation. On sessions, live mean-ITL p50 is 0.84–0.85 of sim
     for every policy, which loosens the AIS-E0 SLA for the policies that were not at ceiling.
   - **Ending.** "the relative comparisons below are the intended use" becomes: neither absolute goodput nor the
     size of a gap is validated; what the live runs test is the ranking and the sign of M1-v2 − ramjet.
4. **§15.3.**
   - **New table.** A generated "Live − sim" table lists all 60 entries (Δ vs default and Δ vs ramjet), with \* on
     the departures.
   - **sd wording.** The live-delta sd now distinguishes 0.011 (two same-job defaults) from 0.013 (one, on FAST25
     conv. and sessions). This also covers the second half of fidelity F3 and claims F5. The "24 of 30 exceed
     2 sd" count is unchanged under the per-entry bar (0.022 or 0.026), and so is its list.
5. **§15.7.**
   - **New "cannot show" bullet: the size of the gaps between policies.** It gives the counts, says the sessions
     gain over default (+0.084..+0.086 sim, +0.011..+0.016 live) is not supported live, which also covers the
     simulated sessions-family gains in §3.3, and gives the conv round_robin reversal.
   - **"Absolute goodput".** Now ends "only the ranking and the sign of M1-v2 − ramjet are validated".
   - **"It shows".** Now says "agrees with" (pooled τ_b 1.00 = the same order of cell-mean gains; per cell
     ≥ 0.87), with the registered noise bar named.
6. **"Ceiling" wording.**
   - **§15.5 and §15.7 "Scope".** "The sessions cell is at ceiling" now reads "the four tuned policies tie at
     ceiling on the sessions cell, so it cannot inform M1-v2 − ramjet". In sim, default@defaults is not at ceiling
     there (91.6% good, gain over default +0.086).
   - **The live-analysis stage's returned key facts.** They attributed the four sessions deltas to "(ceiling)".
     That output is past and cannot be edited, so the correction is carried in this file, in `facts/STATE.md` and
     in the hand-off: the shrinkage is a live departure, not a ceiling effect.
7. **REPORT template, outside §15.** These lines carried the same one-sided "relative result validated" framing:
   - **Header line 13:** "validation of the policy ranking and of the sign of M1-v2 − ramjet".
   - **§1 limit 5:** now reads "agree with" (pooled τ_b 1.00, per cell 0.87–1.00). It adds that the cells were
     chosen after the test pass and not at random, and that live runs validate neither absolute goodput (0.72–2.18,
     policy-dependent) nor the size of the gaps (sessions gain over default not reproduced beyond noise). This also
     closes claims F2.
   - **§11 item 2:** now reads "agree with the simulated ranking and the sign …; neither absolute goodput nor the
     size of the gaps between policies is validated … in both directions (§15.2)".

### Verification

- **`live_results.py build`.** The unmodified code was run first: it reproduced LIVE.md, the fragment and
  `facts/live_results.json` byte for byte, apart from `written_local`. After the fix, `build` still passes all 5
  reproduction checks: `live_scores.jsonl` and the 4 analysis files byte-equal, and the pooled τ recomputation
  equal. New assertions guard every sentence of the new §15.2 text: the departure directions, the within-noise
  claims, the preemption and TTFT orderings, and the good-fraction ordering.
- **Independent recompute** (`runs/live-fix/indep_departures.py`, output `indep_departures.out`). It reads raw
  `scores/*/score.json`, the frozen sim counterparts and the job of each run, and shares no code with
  `live_results.py`. It gives σ_live 0.009043, 18 departures vs default (9 higher, 9 lower), 13 vs ramjet, and
  sessions default good fractions of 0.9780 and 0.9841 live against 0.9162 in sim. All of these match.
- **`assemble_report.py`.** It writes REPORT.md (1,539 lines, 21 generated tables).
  - **§1 headline block:** byte-identical from `## 1.` up to limit 5.
  - **Number tokens:** none of the pre-fix tokens is lost.
  - **Diff:** limited to header line 13, §1 limit 5, §11 item 2 and §15.
- **A18.** `publish/scan.py tree` on copies of LIVE.md and `data/tables/live_validation.md` is CLEAN. The new
  text names no cluster, node, job or path.
- **Pre-fix copies.** `runs/live-fix/pre/` holds LIVE.md, REPORT.md, the fragment, `live_results.json`,
  `live_results.py` and `REPORT.template.md`.

## Minor findings not addressed here (left open for the paper and publication audit)

These were not passed to the fixer. Each needs only a text change, and none of them changes a number or a
registered verdict.

- **Claims lens:**
  - **F3:** stale lines. The header still says "A1–A19", §11 item 1 says "Simulation only", §13 still lists the
    live runs as "not done", and §9 says "never the frozen test cells".
  - **F4:** the like-for-like N = 4 k0 cell mean is +0.071, not +0.055. Selection timing is now stated in §1
    limit 5, but §15.7 still quotes the segment mean.
  - **F5, first half:** the CI of σ_live.
  - **F6:** the M1-v2/M1 sim near-tie at k0, and the A19 permutation p of 0.068.
  - **F7:** execution notes: the heartbeat wrapper, the scheduling class, the fsyn tie groups, and the closed-loop
    arrival wording.
  - **F8:** prefix-counter mechanism wording and preemption.
- **Fidelity lens:**
  - **F1:** the usage-based prefix hit rate for all 42 runs.
  - **F4:** no runtime evidence that the SessionContext reaches learned-choice.

The paper's live section should take these into account when it is filled from LIVE.md.

## Integrity

- **Writes.** This stage wrote:
  - `runs/live/finalists/scripts/live_results.py` (CR, not WT);
  - `facts/live_results.json`, `report/LIVE.md`, `report/data/tables/live_validation.md`;
  - `report/scripts/REPORT.template.md`, `report/REPORT.md`;
  - `runs/live-fix/`;
  - this file, and one row each in `facts/DEVIATIONS.md`, `CLEANUP.md` and `facts/STATE.md`.
- **Not done:** no WT change, no commit, no push, no cluster job, and nothing deleted. The main-checkout copy
  `notes/learned-routing/REPORT.md` (01:39, pre-live) was not touched. Refreshing it is a later report or
  publication stage's job.
