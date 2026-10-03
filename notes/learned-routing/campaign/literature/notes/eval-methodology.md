# Evaluation methodology for the learned-routing campaign: literature notes

Scout angle: evaluation methodology for LLM serving and learned systems (goodput and SLO definitions,
SLO thresholds, open- versus closed-loop load, statistics across many cells, fair baseline tuning,
simulator fidelity). Written 2026-10-02 for the campaign in
`<campaign-root>/` (CONTRACT.md) and `<repo>/notes/learned-routing/PLAN.md`.

Every claim below carries a pointer of the form [key, section/figure]. PDFs are in
`/tmp/learned-routing-lit/pdfs/eval-methodology/`, text dumps in
`/tmp/learned-routing-lit/txt/eval-methodology/`. Where I checked campaign code, the file and line are
cited; those are facts about this checkout, not literature. Anything labeled **Hypothesis** has not been
verified.

## Sources

| Key | Paper | Venue | Local PDF | Read depth |
|---|---|---|---|---|
| DistServe | Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving" | OSDI 2024, arXiv 2401.09670 | zhong2024-distserve.pdf | §1, §6.1-6.4, Table 2 |
| OpenClosed | Schroeder, Wierman, Harchol-Balter, "Open Versus Closed: A Cautionary Tale" | NSDI 2006 | schroeder2006-open-vs-closed.pdf | §2, §5 (Principles i-vi), §6 (vii-viii), §7 |
| SmoothGoodput | Wang et al., "Revisiting Service Level Objectives and System Level Metrics in LLM Serving" | arXiv 2410.14257v2 (2025) | wang2024-revisiting-slo-goodput.pdf | §1, §3, §4 |
| Vidur | Agrawal et al., "Vidur: A Large-Scale Simulation Framework for LLM Inference" | MLSys 2024, arXiv 2405.05465 | agrawal2024-vidur.pdf | §3 (cascading errors), §7.2, App. A.1 |
| CalRoute | Tumkur et al., "Calibrate, Then Route: A Measured Study of Learned Request Routing for Disaggregated LLM Serving" | arXiv 2609.16206 (Sep 2026), preprint | tumkur2026-calibrate-then-route.pdf | full |
| Puffer | Yan et al., "Learning in situ: a randomized experiment in video streaming" | NSDI 2020, arXiv 1906.01113 | yan2020-puffer-learning-in-situ.pdf | §1, §3 (uncertainty), §5.2-5.3 |
| DRLMatters | Henderson et al., "Deep Reinforcement Learning that Matters" | AAAI 2018, arXiv 1709.06560 | henderson2018-deep-rl-matters.pdf | seeds, codebases, reporting, recommendations |
| Precipice | Agarwal et al., "Deep RL at the Edge of the Statistical Precipice" | NeurIPS 2021, arXiv 2108.13264 | agarwal2021-statistical-precipice.pdf | Table 1, §3, §4.1-4.3, App. A.5 |
| Demsar | Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets" | JMLR 7, 2006 | demsar2006-statistical-comparisons.pdf | §3.1-3.2 |
| Cawley | Cawley & Talbot, "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation" | JMLR 11, 2010 | cawley2010-overfitting-model-selection.pdf | abstract, §4.4, §6 |
| ShowWork | Dodge et al., "Show Your Work: Improved Reporting of Experimental Results" | EMNLP 2019, arXiv 1909.03004 | dodge2019-show-your-work.pdf | abstract, §1, §3 |
| Variance | Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks" | MLSys 2021, arXiv 2103.03098 | bouthillier2021-variance-ml-benchmarks.pdf | §1 recommendations, §4, §7 |
| Melis | Melis, Dyer, Blunsom, "On the State of the Art of Evaluation in Neural Language Models" | ICLR 2018, arXiv 1707.05589 | not downloaded | abstract only |
| Etalon | Agrawal et al., "Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems" | arXiv 2407.07000 | not downloaded | abstract only |
| Playbook | Krishnamachari, "How to Do Statistical Evaluations in ECE/CS Papers" | arXiv 2605.00428 (May 2026) | not downloaded | HTML §3, §8, §11, §14, §17, §20 (via fetch) |
| LoadTest | Abdelfattah et al., "Load Testing for Machine Learning Model Serving Systems at Scale" | arXiv 2606.22013 (Jun 2026) | not downloaded | abstract only |
| OldInfo | Mitzenmacher, "How Useful Is Old Information?" | IEEE TPDS 2000 (DEC SRC TN 1998-002) | not downloaded | summary only |

## 1. Goodput and SLO attainment: definitions and their failure modes

**Definitions in the literature.**
- DistServe defines per-GPU goodput as the *maximum request rate* at which an SLO-attainment target
  (for example 90% of requests meeting both TTFT and TPOT) is met. This is a capacity-at-SLA metric, not
  a count at fixed load [DistServe, §1 and §6.1 "Metrics"]. It reports SLO attainment at 90% and, in
  an appendix, at 99% [DistServe, §6.1].
- SmoothGoodput formalizes the two system-level metrics used since then. SLO attainment is the fraction
  of requests meeting every per-token deadline (eq. 6). Goodput is the sum over SLO-meeting requests of
  their tokens, divided by the serving interval T (eq. 7) [SmoothGoodput, §4.1].
- TTFT+TPOT SLOs constrain only the first and last token. TBT constrains every gap [SmoothGoodput, §3.1,
  eqs. 2-3]. TPOT "is too loose ... a long stall in the middle of the request can be averaged out"
  [SmoothGoodput, §1].

**What the campaign's metric actually computes (code-verified, aisimulate-core
0.13.0-dev.202609300000000061, the version pinned in the worktree's Cargo.lock).**
- `goodput_request_throughput_rps = goodput_requests / duration_s`, where `duration_ms` is the maximum
  over completed requests of `terminal_time_ms - report_start_ms` (`src/replay/report.rs` lines
  1807-1892). The denominator is the **policy-dependent makespan**, including the drain tail after the
  last arrival.
- "Good" is TTFT ≤ T and aiperf-style mean ITL `(e2e - ttft)/(output_len - 1)` ≤ I, with ITL skipped
  when output ≤ 1 token (`report.rs` lines 934-950). That is a TPOT-type constraint, so stalls are
  averaged out [SmoothGoodput, §1].
- The report also keeps a per-token ITL distribution (`itl_distribution`, `report.rs` lines 980 and
  1066-1076), so tail ITL is available as a guard metric.

**Failure modes documented in the literature.**
- *Abandonment and sacrifice.* Requests that miss the SLO contribute 0 to goodput and attainment, so
  the "goodput-optimal scheduling strategy should kill this request and prioritize the next request
  that can meet the SLOs" [SmoothGoodput, §4.2, eq. 8 and Counterintuitive Example 2]. A router can do
  the same without killing anything: it can concentrate requests that will miss anyway on one worker.
  **Hypothesis:** CMA-ES on binary goodput can discover this.
- *Output delay.* Delaying token delivery can improve tail-TBT attainment while making the user
  experience worse [SmoothGoodput, §3.2].
- *Greedy concentration.* A greedy argmin router "concentrates requests on whichever instance prices
  cheapest". Under extreme scarcity it collapsed to 0.292 goodput while round-robin held 0.680. This
  reproduced four times, including with calibrated constants [CalRoute, abstract and §V-E].
- *Proposed fix.* SmoothGoodput replaces the indicator with a benefit `n_r - α f(l_r)`, where `l_r` is
  the maximum user-idle lateness against a reading-speed deadline `d_i = V·i` (eqs. 5, 9-11)
  [SmoothGoodput, §3.3 and §4.3]. Etalon likewise argues that TTFT, TBT, TPOT and normalized latency
  "fail to fully capture the nuances of LLM inference" and proposes a fluidity index [Etalon, abstract].

## 2. Choosing SLO thresholds

- DistServe sets SLOs "empirically based on their service target because there exists no available
  SLO settings for these applications", per application (chatbot, code, summarization). It then sweeps
  a single **SLO Scale** multiplier on both TTFT and TPOT at a fixed rate to find the tightest SLO each
  system sustains (Fig. 8, second row) [DistServe, §6.1 and §6.2 "SLO Scale"].
- CalRoute derives its SLO targets (TTFT 175 ms and TPOT 39.7 ms tight, plus a 3× loose tier) "from the
  same calibration as the router constants", and evaluates both tiers [CalRoute, §IV-B].
- Sarathi-Serve uses strict and relaxed P99 TBT SLOs per model, for example Mistral-7B at 0.1 s strict
  and 0.5 s relaxed [arXiv 2403.02310, from the search snippet; not downloaded].
- Implication for the campaign: one (T, I) pair per family, taken from the default router's own runs,
  is a single point on DistServe's SLO-scale curve. Because `per_request` stores TTFT and ITL, the whole
  attainment-versus-SLO-scale curve can be recomputed from runs that already exist, at zero replay cost.

## 3. Open, closed and partly-open load

From [OpenClosed, §5-§7]:
- **Principle (i).** For a given load, mean response time is much lower in closed systems than in open
  ones.
- **Principle (ii).** Closed systems approach open ones as the multiprogramming level (MPL) grows, but
  slowly. Even at MPL 1000 they differ when job sizes are highly variable.
- **Principle (v).** Scheduling helps a closed system significantly only at moderate load and high MPL.
  At both high and low load, policies perform similarly.
- **Why.** In a closed system with zero think time, Little's law N = X·E[T] with N fixed means every
  work-conserving policy gives the same throughput and mean response time. In a closed system "the
  scheduling policy actually affects the throughput, and hence the load", which compresses differences
  further [OpenClosed, §5.2].
- **Principle (vii).** A partly-open system (open session arrivals, closed turns within a session)
  behaves like an open system when sessions have ≤ 5 requests, and like a closed one when they have
  ≥ 10.
- **Principle (viii).** In a partly-open system, think time has little effect on mean response time,
  because it does not change load.
- **Caveat for routing.** These results are for single-queue scheduling with fixed work per job. With
  prefix caching, routing changes the *amount* of work (cache hits remove prefill), so the
  Little's-law argument does not fully transfer. **Hypothesis:** throughput can differ across routers
  even in a closed loop, through hit rate.

Campaign-relevant facts (code-verified): aisimulate-core already models partly-open agentic sessions. A
follow-up turn is released only after its dependency completes plus a delay (`replay/loadgen/driver.rs`
around line 3320, test `agentic_mode_releases_turn_after_dependency_completion_plus_delay`; also
`replay/disagg_tests.rs` line 2845). The 82 AgentX plays average about 38 requests each (3,132/82,
PLAN.md), which puts them well past the ≥ 10 rule of thumb: closed-like behavior.

CalRoute adds an overload note: "the aggregate in-flight count under overload is set by arrivals minus
capacity regardless of policy, so queue depth carries no routing signal; the tail is where policies
differ" [CalRoute, §V-F]. It ran each workload where round-robin "lands mid-collapse ... because
routing policies only separate under contention" [CalRoute, §IV-B].

## 4. Warm-up, drain, and measurement windows

- Discard a warm-up window or analyze it separately, and document how its length was chosen (cache
  effects). Measure latency from the intended dispatch time to avoid coordinated omission. Report the
  median plus p95/p99 and the CDF for heavy-tailed latency [Playbook, §14].
- Proper warm-up handling improved capacity-estimate accuracy by 22.2% across 14 industrial cases
  [LoadTest, abstract].
- Code fact: aisimulate-core has a warm-up phase only for agentic lanes (`AGENTIC_WARMUP_REQUESTS_PER_LANE
  = 10`, `replay/loadgen/phase.rs` line 25), and it excludes that phase from metrics. Open-loop
  time-window slices of Mooncake and toolagent start with empty prefix caches and empty queues, and the
  native goodput denominator includes the drain tail (section 1).

## 5. Statistics across many workload cells

**Unit of analysis and independence.**
- When comparing over multiple datasets, "the sample size ... will refer to the number of data sets
  used", and the variance of interest comes from differences across *independent* datasets
  [Demsar, §3]. No standard test handles dependent repeated observations per cell [Demsar, §3.2.3].
- In the campaign, cells built from the same trace segment (the same window at L1/L2/L3, at several N)
  are not independent.

**Tests.**
- Demšar recommends the Wilcoxon signed-rank test for two methods over many datasets, and Friedman with
  post-hoc tests for many methods. Compared against a control, Holm's step-down procedure is more
  powerful than Bonferroni-Dunn and makes no extra assumptions [Demsar, abstract, §3.2 and §3.2.2].
- The paired t-test requires commensurable differences. Averaging over non-commensurable datasets makes
  "as little sense as computing the averages over data sets". Webb's fix is the geometric mean of
  ratios [Demsar, §3.1.1-3.1.2].

**Aggregates and intervals** [Precipice, Table 1 and §4].
- Report interval estimates using a **stratified bootstrap** that resamples runs within each task
  [Precipice, §4.1].
- The arithmetic mean across tasks is "often dominated by performance on outlier tasks". The median
  needs many runs. The **interquartile mean (IQM)** is robust and more efficient [Precipice, Table 1,
  §4.3].
- Use **performance profiles**, the score distribution P(X > τ) with bootstrap bands
  [Precipice, §4.2].
- Report the **average probability of improvement** P(X > Y), computed via Mann-Whitney U per task
  [Precipice, §4.3 and App. A, eq. A.2].
- Bootstrapping over tasks instead of runs answers "what if I used a different set of tasks" and gives
  much wider intervals [Precipice, App. A.5].

**Decision rule** [Variance].
- Randomize as many sources of variation as possible (data sampling, initialization, the whole
  hyperparameter search). Of these, variance from data sampling dominates [Variance, §1 recommendations
  and §7].
- Declare A better than B when P(A > B) is significantly above 0.5 and its CI upper bound exceeds
  γ = 0.75 ("meaningful") [Variance, §4]. Prefer several random splits to one fixed test set
  [Variance, §7].

**Seeds and power.**
- Two groups of 5 seeds of the same TRPO configuration gave statistically different curves
  (t = −9.09, p = 0.0016) [DRLMatters, Fig. 5 and "Random Seeds and Trials"].
- Bootstrap power analysis: apply a uniform lift (for example 1.25×) to the observed samples and check
  what fraction of bootstrap resamples come out significant [DRLMatters, "Power Analysis"].
- Rule of thumb: n ≈ 8/(Δ/s)² from a pilot of about 10 seeds. Tail metrics need more observations per
  run [Playbook, §17].

**Uncertainty outside the simulator.**
- Trace-based emulation "eliminat[es] the effect of the play of chance", but "it is difficult to
  characterize the systematic uncertainty that comes from selecting a set of traces that may omit the
  variability or heavy-tailed nature of a real deployment" [Puffer, §5.3].
- In the real-world RCT, the 95% CI on stall ratio after 1.75 stream-years per scheme was ±10% to ±17%
  of the mean, "comparable to the magnitude of the total benefit reported by some academic work"
  [Puffer, §3].

**Equivalence and multiplicity.**
- To claim equivalence, pre-declare a margin Δ and require the whole 95% CI to lie within ±Δ. Do not
  rely on a non-significant p-value [Playbook, §8].
- Pre-register metrics and cutoffs, "report every configuration evaluated, including the unfavorable
  ones", and use Benjamini-Hochberg for many tests [Playbook, §11].
- A conjunctive claim ("beats every baseline") is an intersection-union test: each component at level α
  gives an overall level of α, so the conjunction needs no correction. This is a standard result (Berger
  1982, *Technometrics*), stated from memory and not read in this session.

## 6. Fair baselines and tuning budgets

- Properly regularized standard LSTMs, tuned with large-scale black-box search, beat newer architectures
  that had been evaluated on "differing code bases and limited computational resources" [Melis,
  abstract].
- Implementation differences change results with everything else held fixed. Baseline implementations
  should match the original codebase's reported results [DRLMatters, "Codebases" and
  "Recommendations"].
- Ask "in what setting would this work"; there is "often no clear winner among all benchmark
  environments" [DRLMatters, Discussion].
- The best model is a function of the tuning budget. Report expected best validation score versus the
  number of hyperparameter trials. A simple model can win at small budgets and lose at large ones
  [ShowWork, abstract, §1 and §3].
- Optimizing a selection criterion over a finite sample over-fits it. The resulting bias is "of
  comparable magnitude to differences in performance between learning algorithms" and creates
  selection bias in reported performance. Remedies: regularization, early stopping, and averaging models
  or hyperparameters [Cawley, abstract, §4.4 and §6].
- CalRoute reports its boundaries as results: the learned router is beaten on homogeneous chat
  (heuristic 0.855 versus 0.760), ties at pool width 3, and inverts under scarcity [CalRoute, §V-C to
  §V-E].

## 7. Simulator fidelity

- **Cascading errors.** In inference, "small errors in individual batch predictions cascade over time
  and lead to aggregate errors" [Vidur, §3].
- **Errors near capacity.** Vidur evaluates at 85% of capacity because "as we approach capacity point,
  any small deltas in prediction can lead to significant blow up of the errors" [Vidur, §7.2]. The
  error reaches up to 12.65% at 95% of capacity for a CPU-overhead-bound 7B model [Vidur, App. A.1].
- **DistServe's simulator.** It reproduces real SLO attainment within 2% [DistServe, §6.4, Table 2].
  That is one validation regime, not a ranking guarantee.
- **"Simulation predicts structure, not magnitudes".** After re-fitting the simulator to measured
  physics, it "reproduces goodput to a mean absolute residual of 0.068, but it does not reproduce the
  measured winner on any workload". With batching modeled, "all four policies fall within 0.01 of each
  other on bursty where the hardware separated them by 0.03" [CalRoute, §VI].
- **Simulators hand policies information that deployments lack.** CalRoute's simulator fed the router
  its true backlog, while the deployed proxy had to estimate it. "A simulator can hand a policy
  information no deployment will have, and ... this advantage stays invisible until the policy meets
  hardware" [CalRoute, §VI].
- **Calibration.** Simulator-derived constants were 7-13× off and cost 4.5 goodput points plus about
  40% of the tail advantage [CalRoute, abstract, §III-B and §V-B]. *Quality caveat:* this is a
  single-node, 3B-model preprint from a small lab, unreviewed.
- **Stale load information.** Choosing the lightest-loaded server on stale data causes herd behavior.
  A little randomness, such as the lighter of 2 random choices, recovers most of the benefit
  [OldInfo, summary].
- **Code fact.** The offline replay router applies KV events synchronously:
  `SyncReplayIndexer::apply_event` in
  `lib/mocker/src/replay/offline/extensions/kv_router/mod.rs` lines 201-235, inside `on_kv_events` at
  line 688. In replay the router's prefix view therefore has **zero staleness**, while live KV events
  cross the event plane with nonzero lag. The magnitude of live lag is not measured here
  (**Hypothesis** that it matters).
- **Learned controllers in the wild.** ML-based ABR schemes that looked good in emulators did not beat
  simple buffer-based control in a blinded randomized trial (14.2 stream-years, 56,000 users, arXiv version). Emulation "do[es] not
  capture the vagaries of the real-world paths" [Puffer, abstract and §5.2-5.3].

## 8. Checks the campaign should run (mapped to stages)

Concrete, in priority order. "CR" is the campaign root and "WT" the worktree.

1. **Metric definition (Build/Calibrate).**
   - In `lr-eval`, emit `good_frac` over requests arriving inside a measurement window (excluding
     warm-up), `makespan_ms`, `arrival_span_ms`, and `goodput_rps_window` = window-good / window span,
     next to the native field.
   - For open-loop cells, make windowed attainment the primary metric, or show at the pilot that the
     ranking under native `goodput_rps` ratios matches the ranking under attainment ratios.
   - Rationale: the native denominator is the policy-dependent makespan (section 1).
2. **Noise floor that is not zero (Setup and the pilot gate).**
   - If replay is deterministic, gate condition 1 (Δ > 3 × repeat sd) is vacuous.
   - Build replicates from independent sources: distinct trace windows and plays, Poisson re-draws for
     synthetic sessions, arrival jitter seeds, and tie-break seeds [Variance; Puffer §5.3].
   - Gate on segment-level paired differences.
3. **Aggregation (Training and Reporting).**
   - Train on the mean of clipped log-ratios (a geometric mean), not the arithmetic mean of ratios.
   - Report IQM and geometric mean with stratified-by-family, segment-clustered bootstrap CIs,
     performance profiles, and P(learned > b) for each baseline b [Precipice; Demsar §3.1].
4. **Pre-registered headline test (Calibrate, before training).**
   - For each baseline b, run a one-sided Wilcoxon test on segment-level paired log-ratios against the
     *tuned* b, on the test split. All must pass.
   - Apply Holm (or BH) to secondary per-family and per-N claims.
   - Write this into `facts/` before training [Demsar; Playbook §11; IUT note].
5. **SLO-scale robustness (Reporting, no extra replays).**
   - Re-score every replay at SLO scales {0.5, 0.75, 1, 1.5, 2, 3}× (T, I) from `per_request` and plot
     attainment against scale for every policy [DistServe Fig. 8].
   - Also report one continuous metric: mean TTFT/T clipped at 10, or SmoothGoodput.
6. **Anti-gaming guard metrics (Stage 4 mid-run audit).**
   - p99 TTFT and the fraction of requests with TTFT > 3T over *all* requests.
   - Per-worker assigned-work share (max/mean).
   - p99 per-token ITL from `itl_distribution`.
   - Flag any policy that gains goodput while worsening these [SmoothGoodput §4.2; CalRoute §V-E].
7. **Tuning curves and seeds (Stage 4).**
   - Run ≥ 3 CMA-ES seeds for every tuned baseline and every rung. Report median-of-seeds, and plot best
     incumbent against evaluations [ShowWork; DRLMatters].
   - Select the CMA-ES mean (or re-evaluate the top-k on validation) rather than the best-ever sample
     [Cawley §4.4].
   - Extend the budget of any baseline still improving at the cut-off.
8. **Port fidelity (Build).**
   - Decision-parity tests of each ported baseline (LMetric, Ramjet, DualMap, CHWBL, llm-d) against its
     upstream reference on randomized candidate tables, or mark the port "unverified" in REPORT
     [DRLMatters, "Codebases"].
9. **Unseen-N regime control (Calibrate and Test).**
   - Also calibrate L1-L3 at each test N with the default router (no leakage).
   - Report N-extrapolation at both per-worker-matched and knee-matched loads. Treat N = 2 as a likely
     failure region [CalRoute §V-D and §V-E; OpenClosed Principle v].
10. **Load-mode stratification (Calibrate and Report).**
    - Never pool open- and closed-loop cells into one mean. Gate condition 3 must hold per mode.
    - Expect smaller deltas in closed-loop and AgentX cells [OpenClosed, Principles v and vii].
11. **Warm-up and drain (Calibrate).**
    - Prepend a warm-up segment to each open-loop window and score only the measurement window. Record
      the warm-up length.
    - At the pilot, check that Δ(learned − default) is stable when the warm-up is doubled
      [Playbook §14; LoadTest].
12. **Sim-to-real perturbations (Test).**
    - Beyond `speedup_ratio` and `decode_speedup_ratio` at 0.8 and 1.2, add KV-event and load-signal
      staleness of 10, 50 and 200 ms in the replay router (patch on WT and log an upstream follow-up),
      plus a temperature > 0 or power-of-two variant.
    - Report the rank stability (Kendall τ) of the policy ordering across perturbations, and claim only
      gaps larger than the perturbation spread [Vidur §7.2; CalRoute §VI; OldInfo].
13. **Equivalence claims (Integration and Test).**
    - Use TOST with a pre-declared margin for "learned-choice@θ0 ≡ default" and for any "no regression
      at unseen N" claim [Playbook §8].
14. **Validation-test gap (Report).**
    - Report validation versus test for every rung and baseline. Validation is optimistically biased by
      ladder, checkpoint and gate selection [Cawley §6].
15. **Where it wins (Report).**
    - Add a per-cell winner map and the learned policy's gap to the per-cell best baseline (a "virtual
      best" oracle). State the losing regimes explicitly [CalRoute Fig. 3; DRLMatters].
