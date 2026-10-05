# Publication audit: paper and final report (phase 3, A20)

- **Stage:** publication audit, audit skill in publication-prep mode (local passes; not an
  independent reviewer).
- **Date:** 2026-10-05, about 07:55 to 08:15 PDT.
- **Targets:**
  - the paper in `WT/notes/learned-routing/paper/` at `<commit-47>` (113 pp);
  - `report/REPORT.md` (sha256 `03f0d0e1…`, unchanged by this stage).
- **Result:** paper fixed and committed as `<commit-48>` (`docs(learned-routing): tighten the
  paper's claims after the publication audit`, `git commit -s`, no co-author trailer, 11 paper paths).
  The rebuilt PDF has 112 pp; its copy is `report/paper/learned-routing-paper-pubaudit.pdf`
  (sha256 `7033428a4d7d1b2190c7984b4ba92910f2ba2adc02c929e3a1b0995f5553f923`).
- **Verdict:**
  - **Paper:** PASS after fixes, for publication as a **draft**. Two `\pending` items remain (the
    author list, which is the operator's call, and a REPRODUCE.md revision).
  - **REPORT.md:** content PASS. **Publication BLOCKER P-R1:** the fail-closed scan rejects 4 of its
    lines (false positives), so the notes mirror cannot be published until that is fixed. The item
    is queued in `publish/TODO.md` #6.
- **Frozen facts:** nothing under `facts/` was edited; `finalists.json`, `test_results.json`,
  `robustness.json`, `HEADLINE_TEST.json` and `live_plan.json` are untouched.

## Coverage

**Read in full:**
- abstract, introduction, results, gaming, live validation, discussion, related work and conclusion;
- the reproducibility appendix;
- the evaluation section through the strata, and the opening of the baselines section;
- background: the catalog table and the caveat table;
- REPORT §1, §7.1, §9 and §11, plus targeted greps over the rest of REPORT.

**Scanned for claim wording, not re-read line by line:** setup, simulator, model, training,
notation, the pilot appendix and the tables appendix. The paper-final stage reconciled every number
in these sections (`facts/paper_final.json` `number_reconciliation`), and none of them changed
after that.

**Visual:**
- every page of the rebuilt PDF at 40 dpi, as 4×2 contact sheets;
- pages 1, 22, 83 and 104 at 90 dpi.

## Findings and fixes (paper)

Severity: major = a statement the evidence contradicts or that reads stronger than it; minor =
imprecise or loosely scoped; cosmetic = layout only. Every finding below was checked against the
cited fact file.

| ID | Sev. | Where | Finding | Evidence | Fix |
|---|---|---|---|---|---|
| P1 | major | intro "What we found" bullet 3; results §9.2 limit 4 | "prompts of 32K tokens and more see higher TTFT / wait longer than under ramjet" | `test_results.json` `report_must_state` MS-F1-ttft: 32K–64K TTFT p90 +0.135 (×1.145), 11/12 segments; **≥64K filtered +0.021 (×1.021), higher on 3 of 7 segments, p 1.0**. MS-F1-goodfrac-by-isl: good fraction −3.8 pp at 32K–64K and −3.7 pp at ≥64K | Intro: "prompts of 32K to 64K tokens". Limit 4: TTFT at 32K–64K, plus "prompts of 32K tokens and more lose about 4 points of good fraction on average", with a src comment |
| P2 | major | abstract; results limit 2; discussion "What the results say" and "Ports, not papers"; conclusion bullet 2 | Margin attributed to faithful LMetric itself: "a faithful LMetric … reproduces most of the margin", "faithful LMetric alone carries most of the validation margin", "which a faithful LMetric captures by itself" | MS-RH2-F1-table: **default + 10× faithful LMetric** reproduces 53–66% of M1-v2's fresh-val margin over ramjet. Faithful LMetric alone (untuned) is +0.0079 cell / +0.0110 segment over ramjet against M1-v2's +0.0372 / +0.0290, that is 21–38% | Each place now names "the default cost plus (a) faithful LMetric (term)". The discussion keeps the true statement that untuned faithful LMetric beats every tuned baseline |
| P3 | major | discussion, "SLOs and loads are defined by the default router" | "re-scoring … under scaled SLOs and four other definitions … kept M1-v2 first under every definition" | `test_results.json` `a14_sla_transfer.variants`: rank_of_headline 1 under the four definitions and at scales 0.5–1.5, but **2 at scales 2.0 and 3.0**. AIS-league M2-ais is first there (vs-default cell mean 0.0643 vs 0.0641 and 0.0470 vs 0.0466). The paper's own robustness table shows "2 / 10 of 32" | Reworded: first under each alternative definition; first among router-observable policies at every scale; edged by M2-ais overall at scales 2 and 3. Src comment added |
| P4 | minor | conclusion, first paragraph | "beats … on held-out windows, plays, seeds, a held-out trace family and unseen worker counts" reads as one claim per hold-out axis | Only the pooled 12-segment test is a test. FAST25 conversation has 2 segments (p 0.25), transforms beyond train ranges are "not established" (p 0.109), and the unseen-N strata are descriptive (Holm) | "on a test set that holds out …", plus "the per-axis results … are descriptive" |
| P5 | minor | discussion "What the results say" | "beats every one of the branch's ported heuristics" departs from the mandated wording | MS-RH2-F1-scope. The IUT secondary holds, but LR-11's effect-size leg fails for lmetric and sticky-bounded | The exact scoped wording "beats the branch's ported heuristics, each tuned with an equal budget" |
| P6 | minor | abstract; conclusion; background caveat 1 | "the ranking survives lag / timing perturbations": the full ranking changes (τ_b 0.78–0.95) | `robustness.json` `lr14_summary`: headline_rank 1 and sign_holds_in_all across lag 10/50/200 ms and the timing conditions | "M1-v2 stays first, ahead of Ramjet, under …"; caveat 1 is worded the same way, and its live clause now says the live ranking agrees |
| P7 | minor | related work, "Position of this work" | "supports LMetric's argument that the two should multiply rather than add" overreaches | M1-v2 keeps the additive default anchor (θ0 = −1). The heuristic that reproduces the margin is default **plus** LMetric. SMetric measured pure LMetric 36% below a tuned linear score without a global tier | "supports LMetric's point that the two should interact, though here as a correction added to the default's additive cost rather than in place of it" |
| P8 | minor | related work, goodput paragraph | Schroeder et al. rule of thumb quoted as "sessions of ten or more requests" | `literature/txt/eval-methodology/schroeder2006-open-vs-closed.txt` l.976–978: "a high number of requests per session (more than 10) suggests a closed model" | "more than ten requests" |
| P9 | minor | live §11.4 | "(of 0.011 or more)" vs the registered informative rule | `live_plan.json`: informative if the simulated \|M1-v2 − ramjet\| > 0.01 (FAST25 synthetic's sim lead is 0.0108) | "a lead of more than 0.01" |
| P10 | minor | reproducibility appendix | "Off the original host, git and the public sources are enough", while the lag and AIS branches are not published | Only `origin/rupei/learned-routing-public` exists; A20 lets the publish stage push only that branch. The lag records come from the lag build (`robustness.json` `execution`) | Scoped to "the steps REPRODUCE.md covers"; added that only the first branch is published, so the lag part of the robustness pass and the AIS league can be checked against the mirrored facts but not yet rebuilt from public sources |
| P11 | minor | results §9.11 follow-up bullet | "was escalated to the operator" is internal coordination language | reader context | "has not been run and is not part of this paper" |
| P12 | cosmetic | title box (`\textsc` inside `\sffamily`) | LaTeX font warning: T1/lmss/m/sc unavailable | build log | Plain "PENDING" in the box. The build now has 0 LaTeX warnings |
| P13 | cosmetic | notation table, page 22 | A single orphaned row ("R restarts per policy") on the continuation page | page render | `\\*` on the two preceding rows, so three rows carry over |

**Not changed:**
- **"operator" in the text.** Most uses name whoever set the study's requirements, which is
  coherent within the paper.
- **A*n* / LR-*n* labels.** The introduction explains them.
- **BibTeX empty-year warnings.** There are seven, all on software or data entries; plainnat
  prints no year for them.
- **Small type in the appendix tables.** It is legible at 90 dpi.

## Argument and evidence

**Headline wording.** Every statement of the headline now carries the `report_must_state`
MS-RH2-F1-scope wording, "beats the branch's ported heuristics, each tuned with an equal budget".
The places are:
- the abstract;
- the introduction;
- the results box;
- the discussion (after P5);
- the conclusion.

No passage says "beats every heuristic" or "learning beats the best available heuristic" except to
deny it. The four mandated limits travel with every headline statement: sign only (below MDE
0.038 and the 0.060 spread); the in-class LMetric scope; a request-count property; simulated.

**Checked against `facts/`:**
- **The headline numbers.** Segment mean +0.0316 (SE 0.0093); 10/12 segments; one-sided exact
  Wilcoxon p 0.0017; P(>) 0.833 [0.583, 1.0]; geometric mean 1.032 [1.015, 1.051]; IQM 1.026; cell
  mean +0.0506 with 50/60 cells. The segment interval is quoted from REPORT's 10,000-draw bootstrap,
  [0.0150, 0.0501]; the facts file's own bootstrap gives [0.0150, 0.0498], as the paper's source
  comment says.
- **Strata.** Load-mode p 0.06–0.08; unseen N 7/7 (p 0.0078); N = 6 3/6 (p 0.219).
- **Robustness.** Lag deltas and p ≤ 0.0081. "+0.022 to +0.080, p ≤ 0.0081" is for E0′ with the
  ITL bound rescaled to I/(s·d) (`audits/final/refute-headline-3.md` §2), not the E0′ table rows. τ_b
  ranges are as stated.
- **A14 definitions.** Deltas, p values and τ_b 0.85–0.99.
- **Per-cell results.** M1-v2 is the single best policy on 31 of 60 cells; it is at or above the
  virtual best tuned baseline on 42 of 60, with mean gap +0.037 (`test_results.json`
  `per_cell_winner_map` and `per_cell_gap_to_best_tuned_baseline`).
- **Fairness table.** It matches `report_must_state` MS-RH2-F1-table.
- **The 32K–64K TTFT figures and the A19 secondary.**

**Live statements.** These match `facts/live_results.json` through `facts/paper_final.json`:
- the registered words "agrees" and "holds live";
- 5/5 informative cells; mean +0.219 vs sim +0.162;
- σ_live 0.0090 with 6 df; the bar is 0.049 at the one-sided upper bound, which FAST25 synthetic's
  +0.035 does not clear;
- 18/30 and 13/30 departures, labelled descriptive and not registered.

The live section's can-and-cannot-show list carries every limitation the live audits raised.

**Claim stages:**
- **Headline:** pre-registered, and independently reproduced by the final audit.
- **Live verdicts:** independently re-derived by the live claims and fidelity audits.
- **Fixes in this stage:** checked by me against the fact files. That is an unchecked local
  derivation, not an independent recomputation. The "about half" in conclusion bullet 2 is my
  arithmetic, recorded in its source comment: θ21 = 0 costs 0.0178 on val k0-2, against margins of
  0.0344 over the val k0-2 best (ablabase) and 0.0372 over ramjet on fresh validation, that is
  0.52 and 0.48.

## Related-work accuracy (`literature/`)

**Spot-checked against the extracted text or the PDF:**

| Source | Claim the paper makes | Check |
|---|---|---|
| LMetric | Dynamo's linear score "tuned for each workload" | l.1790–1793 "we also tune its hyperparameters for each workload" ✓ |
| LMetric | best weight 0.7 vs 0.55 | l.768 ✓ |
| LMetric | could not sweep for GPU cost | l.767–772 ✓ |
| SMetric | ideal reuse 50–70% vs 82% | l.824–826 ✓ |
| SMetric | no global tier: LMetric 920 vs 1,437 TPS (−36%) | l.830–831 ✓ |
| SMetric | reuse 45% vs 74% | ✓ |
| SMetric | quote "except for setups without any global KV$ store" | l.1683 ✓ |
| Lodestar | k = 2 consistent-hash filter above 80% cluster memory | l.829–843 ✓ |
| DualMap | best-of-all "effectively equivalent to using d = n choices" | pdf p.5 ✓ |
| DualMap | Min TTFT "may oscillate between cache-aware and load-aware decisions" | pdf p.6 ✓ |
| Calibrate-then-Route | widths 3/4/6: ties JSQ 0.816 vs 0.822, leads 0.864 vs 0.835, reconverges 0.858 vs 0.860 RR | ✓ |
| Calibrate-then-Route | 0.819 vs 0.864 | ✓ |
| Calibrate-then-Route | residual 0.068 and the winner not reproduced | ✓ |
| Calibrate-then-Route | starved pool 0.292 vs 0.680 | ✓ |
| GORGO | w_queue → 0 and 100% of requests to the closest replica under p95-TTFT tuning | ✓ |
| AgentServeSim | 0.5% and 2.8% | ✓ |
| Decima | 630 vs 610 s with 10× fewer executors | ✓ |
| Decima | 104.8 vs 91.2 s | ✓ |
| Pensieve | slow-start-restart disabled | ✓ |
| Webb et al. | set sizes 2–12 | ✓ |
| Tomlinson & Benson | d + 1 affinely independent sets | ✓ |
| Schroeder et al. | rule of thumb | corrected (P8) |

The remaining citations rely on `literature/LESSONS.md` lines marked VERIFIED with page locators,
which I did not reopen. Sources the bibliography marks as read at abstract or web depth are cited
only for what they say there.

**Precedence and novelty.** This check ran as a local pass and is **not independent**. Four bounded
web searches found nothing that anticipates or subsumes the contributions beyond the campaign
bibliography, whose closest works are Lodestar, Calibrate-then-Route, Wu et al. 2026 and GORGO. The
searches covered:
- learned or conditional-logit KV-aware routing tuned in a simulator;
- CMA-ES or ES-tuned routing costs;
- learned routers tested at unseen replica counts;
- self-precedence: Dynamo learned-choice, AISimulate.

Self-precedence turned up a public NVIDIA blog on the DynoSim simulator. It concerns the simulator,
not learned routing, and is not a priority conflict. The paper's novelty language is already
bounded: "we did not find together in the work above" and "among the learned routers we read". No
"first" claim was found.

## LaTeX and build

**Build.** `make check`: latexmk, pdflatex and BibTeX with the installed TinyTeX (no tlmgr).
- 112 pages;
- 0 LaTeX warnings (P12 removed the only one);
- 0 overfull or underfull boxes;
- 0 undefined or multiply defined references and citations;
- 7 BibTeX empty-year warnings, on software entries.

The `\pending` markers are the author list and the REPRODUCE.md revision.

**Visual inspection, all 112 pages.** There is no clipping, overflow or broken float. Floats stay in
their sections (placeins), and the longtables continue with repeated headers. Pages 61 and 62 carry
float-driven whitespace (the audit-rounds table and the headline box): acceptable.

## A18 sanitization

All scans use `publish/scan.py` with the rules in `publish/denylist.txt`, read in place and never
copied into the public tree. Every scan adds `--internal-ref` for rupei/learned-routing, -ais and
-lag, and `--deny-literals publish/work/known_ids.txt`.

| Target | Result |
|---|---|
| Paper tree copy without `build/` (64 files), before and after the fixes | CLEAN |
| `pdftotext` of the rebuilt PDF | CLEAN |
| `git` HEAD~1..HEAD of `<commit-48>` (paths, added lines, message, author) | CLEAN |
| Dry run of `publish/sanitize.py run` on the paper plus REPORT/LIVE (scratch output) | The paper changes only by `rupei/learned-routing` → `rupei/learned-routing-public` (2 lines in the appendix); the sanitized paper scans CLEAN |
| `report/REPORT.md`, raw and sanitized | **FAIL, 4 hits** (P-R1) |

An extra grep went beyond the denylist (slurm, sbatch, srun, partition, qos, node names, wf_ IDs,
`<trace-corpus>/`, `.claude`, worktree, /home, shared filesystem, scratch, internal, PDT, job, allocation and
cluster names). It found only generic words (routing "partition", CMA-ES "internal units",
"scratch buffers") and source comments with campaign-relative paths and dates. These raise no A18
issue.

**What the published text says about the execution:** "a shared file system", "a different queue
than planned", "a pool of shared CPU nodes", "the workstation", "an 8×H100 SXM node", and
"EPYC 7702P / 9654P". None of these names a cluster, node, partition, account, job or path.

## REPORT.md findings

Not edited: the instruction was to fix in the paper. REPORT is generated by
`report/scripts/assemble_report.py`, and the items below are queued in `publish/TODO.md`.

- **P-R1 (publication blocker for the notes mirror).** REPORT lines 149, 167, 176 and 589 print
  CMA-ES checkpoints as `<arm>-s<k> g10 mean`. The denylist rule `node-class-label` reads `g10` as a
  CPU-class label. `sanitize.py` rewrites only `epyc7702p` and `epyc9654p`, so the token reaches the fail-closed
  scan. Because `sync_from_campaign.sh` copies `report/*.md` into the mirror, the next
  `publish.sh sync` fails.
  - Fix at the generator: print `gen 10`, as the paper's `scripts/make_tables.py` already does. Or
    add a narrow allowlist entry.
  - Separately, a `epyc9654p` checkpoint label would be silently rewritten to `epyc9654p` by the
    sanitizer.
- **P-R2 (minor, the still-open live claims F3).**
  - The header says "amendments A1–A19"; A20 exists.
  - The §9 "Escalation" bullet says the follow-up may use "the live runs, never the frozen test
    cells". The live finalist runs used frozen test cells, so that route is closed. The paper's
    wording is already correct.
- **Unchanged and correct.** REPORT §1 states the headline in the exact mandated wording with all
  five limits. Its long-prompt limit specifies TTFT at 32K–64K. Its fairness bullets say "Default +
  faithful LMetric". Its A14 bullet claims rank 1 only for the four definitions. None of P1–P3
  applies to REPORT.

## Open items before the paper is final

These are not blockers for publishing a draft:
1. the author list (`\pending`, operator decision);
2. the REPRODUCE.md revision (`\pending`);
3. P-R1 and P-R2 in REPORT (`publish/TODO.md` #6–7);
4. `publish/TODO.md` #1–5 from 2026-10-03;
5. `WT/notes/learned-routing/README.md`, which carries uncommitted edits by an earlier stage. Its
   changes were left alone and are not in `<commit-48>`.

## Scratch and housekeeping

**Scratch.** All of it is in the session scratchpad and listed in `CLEANUP.md`:
- `pub/` holds the scan copies and the sanitizer dry run;
- `vis/` holds the page renders;
- `dualmap.txt` holds the PDF text for the DualMap check;
- `commitmsg.txt` held the commit message.

`commitmsg.txt` was written with `cat >`. If an earlier stage had left a file of that name in the
shared scratchpad, it was overwritten; no ledger entry names one. Recorded in `facts/DEVIATIONS.md`.

**What this stage did not do:** run a replay, submit a GPU or CPU job, push, or delete anything
outside its own scratch.
