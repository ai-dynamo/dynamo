# Literature lessons for the learned-routing campaign

> **Adversarial verification, 2026-10-02.** For every lesson, I re-read each cited local PDF
> page by page (pdftotext output indexed by physical page) and re-checked each [code] and [setup] fact against WT
> `<commit-01>`, the pinned aisimulate-core crate and `CR/runs/setup/`.
> `pdf p.N` means the physical page of the local PDF, which may differ from the printed page number.
>
> - **Lessons:** 15 checked, 0 deleted, 11 corrected (LR-01–07, 11, 13–15). All 15 survivors are
>   **VERIFIED**: each one's core claim is backed by a full-text read. None rests on an abstract alone.
> - **Evidence items:** 105 bullets in total.
>   - 98 checked: 82 PDF citations (including the 2 removed below) and 16 code, setup or
>     arithmetic facts. All 16 code and setup facts held.
>   - 2 removed because the source does not support the lesson: Salimans ES §2.1 in LR-03, and
>     Seshadri & Ugander 2020 in LR-11.
>   - 17 corrected because they were misread, mis-sectioned or overstated. Each is flagged
>     "corrected" inline. A few more bullets are now labeled as our inference rather than the
>     paper's claim.
>   - 6 remain UNVERIFIED because they were read on the web or as an abstract only, or are cited
>     from memory: Krishnamachari ×2, the llm-d blog ×2, Melis 2018 and Berger 1982.
>   - 1 is an explicit [hypothesis] and was left as is.
>   - Each lesson now has a **Status** line, and each evidence bullet ends with VERIFIED (pdf p.N)
>     or UNVERIFIED.
> - **Corrections that change actions:**
>   - LR-07 and LR-04: without a global KV tier (replay has none), SMetric's own data show that
>     LMetric's product lost 36% to a tuned linear score on agentic traffic, and SMetric's
>     session-centric gain does not hold.
>   - LR-05: the proposed context form is a column-restricted LCL, not DLCL.
>   - LR-06: the leave-one-out rank fraction fails the proposed duplication test.
>   - LR-11: the P(A>B) rule needs both CI bounds, and Wilcoxon needs at least 5 segments per test.
>   - LR-03: UH-CMA-ES does not support "add seeds before raising λ".
>   - LR-13: GORGO optimized p95 TTFT, not goodput.

Written 2026-10-02 by the literature sidecar. The inputs were:

- four scout notes under `/tmp/learned-routing-lit/notes/` (`dcm-context.md`, `llm-routing.md`,
  `learned-systems-policies.md`, `eval-methodology.md`);
- about 70 sources, 53 of them papers downloaded under `/tmp/learned-routing-lit/pdfs/<area>/`;
- the campaign's own files: PLAN.md, CONTRACT.md, `CR/facts/` and `CR/runs/setup/`.

Code facts cited here were re-checked on 2026-10-02 in WT (head `<commit-01>`) and in the pinned
aisimulate-core crate `0.13.0-dev.202609300000000061`. The scouts' quoted claims that carry the most
weight (LMetric's P-token, SMetric's SLO formula, GORGO's concentration failure, and
Calibrate-then-Route's "winner not reproduced") were spot-checked against the PDFs.

**Labels.**

- **[paper]** A claim made by the cited source.
- **[code]** Verified in source code.
- **[setup]** Measured by this campaign's setup stage. Setup cells are uncalibrated smoke cells.
- **[hypothesis]** Not verified. It needs a campaign measurement.

**How stages use this file.** Each stage from calibration onward cites lessons by ID. It records
`LR-xx applied` or `LR-xx rejected: <reason>` in its summary and in its STATE.md line.

**Campaign state when written.**

- Setup is done (`facts/STATE.md`, 13:26).
- `cells/` is empty.
- WT has no `learned_choice` module yet.

So feature set v1 is **not frozen**, and the v1-level changes below are still cheap.

## Stage index

| Stage | Lessons to apply |
|---|---|
| Build: `learned-choice` and policies | LR-03 (RNG), LR-05, LR-06, LR-07, LR-04 (ports), LR-14 (information-parity audit lens) |
| Build: harness (`lr-eval`, `lr-train`, `lr-report`) | LR-01, LR-03, LR-10, LR-11, LR-13 (guard metrics) |
| Calibration and freeze | LR-01, LR-02, LR-08, LR-09, LR-11 (pre-registration), LR-12 |
| Pilot and gate | LR-02, LR-15 (drift check) |
| Stage 4 training and audit | LR-03, LR-05, LR-10, LR-13, LR-14, LR-15 |
| Stage 5 test | LR-11, LR-12, LR-14 |
| Stage 6 report | LR-08, LR-11, LR-15 |

## Ranked lessons

The ranking is by expected reward to this campaign times confidence. Ties go to the lesson whose
decision freezes sooner. Section paths in the evidence lines are relative to
`/tmp/learned-routing-lit/pdfs/`.

### LR-01. The objective as written measures drain time and chases collapsed cells: train on windowed attainment with a clipped mean log-ratio

- **Status.** VERIFIED (code report.rs 1807–1895; setup determinism.json; DistServe pdf p.9–10;
  Decima pdf p.7; Agarwal pdf p.2–3; SMetric pdf p.10). Corrected: the Demšar citation is a
  caution, not support.
- **Lesson.** Two problems make the contract objective unreliable:
  - The native `goodput_request_throughput_rps` divides good requests by the policy-dependent
    makespan, which includes the drain tail, so it mixes SLO attainment with drain speed.
  - The mean of per-cell ratios to default is dominated by heavy cells where default's goodput is
    near zero, so a few cells drive CMA-ES.
- **Action.**
  1. In `lr-eval`, emit these fields:
     - `good_frac_window`: SLO attainment over requests that arrive inside the measurement window,
       after warm-up;
     - `goodput_rps_window`: window-good requests divided by the window length. At fixed open-loop
       load this equals the offered rate × `good_frac_window`;
     - `makespan_ms`, `arrival_span_ms` and `warmup_ms`.

     Keep `goodput_rps`, `goodput_rps_report` and the 1e-6 cross-check, but only as an integrity
     check.
  2. Choose the metric per load mode:
     - Open-loop cells: optimize `goodput_rps_window`.
     - Closed-loop and lane cells: keep makespan goodput, because throughput is legitimately
       policy-dependent there. A fixed window in which all lanes are active is a possible
       refinement [hypothesis].
  3. Set the training objective to the mean over cells of `clip(log((m + ε)/(m_def + ε)), −log 3,
     +log 3)`. Here `m_def` is the default run under the **same tie seeds** (LR-03). Also report the
     arithmetic mean of ratios and the IQM.
  4. During calibration, choose L2 and L3 so that default's `good_frac_window` is at least about 0.3
     in every train and validation cell. Re-level or reject any cell below 0.1. Keep the L1 cells as
     no-regression checks.
  5. Warm up open-loop cells:
     - Prepend a warm-up segment to each open-loop Mooncake and toolagent window, for example the
       preceding 10 minutes of trace.
     - Score only the measurement window.
     - At the pilot, check that Δ(learned − default) is stable when the warm-up length is doubled.
- **Stage.** Build (`lr-eval`, `lr-train`), calibration, training.
- **Evidence.**
  - [code] aisimulate-core `src/replay/report.rs` lines 1807–1895:
    `duration_ms = max(report_end, max terminal_time) − report_start`, and
    `request_throughput_rps = goodput_requests / duration_s`. VERIFIED.
  - [code] Warm-up exists only for agentic lanes (`replay/loadgen/phase.rs:25`,
    `AGENTIC_WARMUP_REQUESTS_PER_LANE = 10`). VERIFIED; no other warm-up path was found in `src/replay`.
  - [setup] In the overloaded N=4 smoke cell (`runs/setup/determinism/determinism.json`), goodput was
    default 0.281, LMetric 0.479 and RR 0.637. Those are ratios of 1.70 and 2.26 to default, from a
    single collapsed cell. VERIFIED.
  - DistServe §6.1: SLO attainment is "the major evaluation metric", swept over rate and SLO scale —
    `eval-methodology/zhong2024-distserve.pdf`. VERIFIED (pdf p.9–10).
  - Decima §5.3: under a deterministic, fixed horizon the agent learned to defer large jobs until
    termination; the fix was memoryless, exponentially distributed termination —
    `learned-systems-policies/mao2019-decima.pdf`. VERIFIED (pdf p.7). Implication: the window
    metric must keep scoring in-window arrivals that finish after the window, or it recreates this
    defer-past-the-end incentive.
  - Agarwal et al. 2021 Table 1 and §4.3: the mean is "often dominated by performance on outlier
    tasks"; use IQM — `eval-methodology/agarwal2021-statistical-precipice.pdf`. VERIFIED (pdf p.2–3, p.7).
  - Demšar 2006 §3.1 (corrected): Demšar describes Webb's geometric mean of ratios but calls its
    utility "rather questionable", because it is just the mean of log-scores. He notes that arithmetic
    means of ratios are skewed, and recommends the Wilcoxon signed-ranks test —
    `eval-methodology/demsar2006-statistical-comparisons.pdf`. VERIFIED (pdf p.6). This is a
    caution, not support. The log-ratio is chosen here for scale invariance across cells and for
    clipping. It is not a statistical test, and LR-11 still tests with Wilcoxon.
  - SMetric p10: the gap to the best baseline is within 1% at 6× load, +4% at 8×, +13% at 9× and
    +26% past saturation (10×) — `llm-routing/wang2026-smetric.pdf`. VERIFIED (pdf p.10).
- **Plan effect.** CHANGES the PLAN Objective, the CONTRACT `lr-train` objective and the `lr-eval`
  record. The operator's "goodput at loadgen-defined load" semantics are kept. Only the denominator
  changes, from makespan to the arrival window.
- **Reward/confidence.** high/high

### LR-02. Replay is deterministic per seed but chaotic across tie seeds: build the noise floor from seeds and workload segments, not repeats

- **Status.** VERIFIED (setup determinism.json; Bouthillier pdf p.4, p.9–10; Puffer pdf p.10;
  UH-CMA-ES pdf p.7–8; AgentServeSim pdf p.8; CtR pdf p.4). Corrected: the effect-size comparators
  are not all learned routers.
- **Lesson.** Repeat sd is not the noise in this campaign:
  - Re-running one (policy, cell, seed) gives zero spread.
  - Changing only the tie-break seed moved default goodput by about 1.8% on the N=8 setup cell. That
    is as large as the published margins of searched or engineered policies over their best simple
    baseline: +0.5% and +2.8% mean JCT for LLM-searched retention and scheduling policies, and +1.7
    goodput points for a hand-weighted marginal-cost router.
  - So the contract's "3 × pooled repeat sd" rule is vacuous. The real noise is frozen noise plus
    workload-sampling variance.
- **Action.**
  1. Always pass a seed. Setup already added a seeded `dynamo-default-cost-fn` (WT `<commit-01>`).
     Never use an unseeded default as the normalization reference.
  2. Redefine replicates as K tie seeds (K ≥ 3) × independent workload segments:
     - distinct Mooncake and toolagent windows;
     - Poisson re-draw seeds for synthetic sessions;
     - disjoint AgentX play subsets.
  3. Redefine "differs beyond noise" as the mean paired per-segment Δ > 2 SE over segments, per load
     mode, instead of repeat sd.
  4. At the pilot, measure the tie-seed spread per family and N. Set the MDE to max(1–2%, 2 × that
     SE). Size segments and seeds with a bootstrap power check, using n ≈ 8/(Δ/s)².
  5. Record the change in `facts/DEVIATIONS.md`.
- **Stage.** Calibration, pilot gate.
- **Evidence.**
  - [setup] `runs/setup/determinism/seeded/determinism.json`: `n8_default_seed1` gave 2.8687 in both
    repeats and `n8_default_seed2` gave 2.8180 in both. In `runs/setup/determinism/determinism.json`,
    the unseeded catalog default gave 2.8934 vs 2.8290 across identical runs. VERIFIED. The seed
    spread is 1.80%.
  - Bouthillier et al. 2021 (corrected sections): §3.2 Fig 1, "bootstrapping data stands out as the
    most important source of variance"; §5, "randomize as many sources of variations as possible" —
    `eval-methodology/bouthillier2021-variance-ml-benchmarks.pdf`. VERIFIED (pdf p.4, p.9–10).
  - Puffer §5.3: trace-based emulators eliminate "the play of chance", but not "the systematic
    uncertainty that comes from selecting a set of traces" —
    `eval-methodology/yan2020-puffer-learning-in-situ.pdf`. VERIFIED (pdf p.10).
  - UH-CMA-ES §IV (corrected wording): "a small perturbation **can** be applied, before the
    reevaluation is done, to cover 'frozen noise'". Re-evaluating on fresh tie seeds (LR-03) is the
    resampling alternative — `learned-systems-policies/hansen2009-uh-cmaes.pdf`. VERIFIED (pdf p.7–8).
  - AgentServeSim §5 Table 2 (corrected scope): LLM-driven evolutionary search improved mean JCT over
    hand-written seeds by 0.5% for **KV retention** and 2.8% for **scheduling**. These are not routing
    policies, and the policies are deployment-specific by design —
    `llm-routing/rajib2026-agentservesim.pdf`. VERIFIED (pdf p.8).
  - Calibrate-then-Route Table II: 0.864 vs 0.847 for the best heuristic. Corrected scope: the
    "learned router" is a marginal-cost scorer with hand-set weights; only its hardware constants
    were fitted (§VI Scope) — `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf p.4, p.6).
  - Krishnamachari 2026 §17. UNVERIFIED (web only).
- **Plan effect.** CHANGES the CONTRACT "Noise, determinism, gates" section and pilot gate
  condition 3.
- **Reward/confidence.** high/high

### LR-03. Use common random numbers per CMA-ES generation, including tie randomness

- **Status.** VERIFIED (Decima pdf p.7, p.10; Mao ICLR'19 pdf p.3, p.5; PEGASUS pdf p.4–7;
  UH-CMA-ES pdf p.4, p.8–9; code selector.rs and picker.rs). Corrected: the Salimans citation is
  removed, and the noise-handling order in action 3 is fixed.
- **Lesson.** The largest variance-reduction lever in learned-systems policy search is to score every
  candidate on identical inputs and seeds and to normalize per input. The router's RNG design decides
  whether that pairing survives the first decision on which two candidates differ.
- **Action.**
  1. In `lr-train`, for each generation:
     - draw the cells, transform seeds and tie seeds once;
     - assert that every candidate's rows cover exactly that (cell, seed) set;
     - normalize each candidate against default under the same tie seeds;
     - compute the objective only over shared cells.
  2. In `learned-choice` and `sticky-session`, consume exactly one draw per decision, regardless of
     ties. Alternatively, derive each draw counter-style from hash(seed, request id), if the plugin
     API exposes a stable id (verify this first). Add a unit test that two policies differing in one
     early decision still share later draws. The size of the benefit is a [hypothesis]: diverging
     trajectories still decorrelate later state.
  3. Re-evaluate a fraction r_λ of each generation on fresh tie seeds and track rank changes
     (UH-CMA-ES; `cma.NoiseHandler`).
     - The paper's default is r_λ = max(0.1, 2/λ); it uses 30% only as an example.
     - When rank changes exceed the threshold, UH-CMA-ES's own treatments are to raise the
       evaluation effort (here: more tie seeds or cells per candidate) or to raise σ.
     - §II says raising λ alone is inferior to resampling, but raising µ and λ together (pycma's
       `popsize` does this) is preferable to resampling, provided step sizes adapt properly.
     - So the corrected ordering is: raise `popsize` or the per-candidate seeds and cells. Do not
       assume that "seeds first" is better.
- **Stage.** Build (policy RNG), training.
- **Evidence.**
  - Decima §5.3 and §7.4: the same arrival sequence is fixed across several episodes, with a separate
    baseline per sequence. Above 75% load, removing the arrival-sequence variance improved average JCT
    by 2× — `learned-systems-policies/mao2019-decima.pdf`. VERIFIED (pdf p.7, p.10).
  - Mao et al. ICLR'19 §3, Fig 2 and §5: on two-server load balancing, the input-dependent baseline
    gave 50× lower gradient variance and +33% test reward. It relies on "input-repeatability" —
    `learned-systems-policies/mao2019-input-driven-variance.pdf`. VERIFIED (pdf p.3, p.5). Scope:
    this is a policy-gradient baseline. For rank-based CMA-ES the analogue is paired CRN scoring.
  - PEGASUS §3, §4.3 and §5, Fig 1b: fixed scenarios, and how random numbers are consumed changes
    search quality — `learned-systems-policies/ng2000-pegasus.pdf`. VERIFIED (pdf p.3–7). Fig 1b: a
    "complex" deterministic simulative model g′ that scrambles the uniforms finds worse policies than
    the natural g.
  - Removed: Salimans et al. ES §2.1. Its "shared random seeds" let workers reconstruct each other's
    parameter perturbations to save bandwidth (pdf p.3). It is not CRN across candidate evaluations,
    so it does not support this lesson.
  - UH-CMA-ES (corrected): §II says increasing only λ is inferior to resampling, and increasing µ and
    λ is preferable to resampling when step sizes adapt properly. §IV gives the default
    r_λ = max(0.1, 2/λ) and uses r_λ = 0.3 as an example —
    `learned-systems-policies/hansen2009-uh-cmaes.pdf`. VERIFIED (pdf p.4, p.8–9).
  - [code] `WT/lib/router-plugins/builtin/src/default/selector.rs:25-35`: "Clones share its random
    stream". VERIFIED.
  - [code] `default/picker.rs:94-134`: tie handling draws once per tied candidate (reservoir), so the
    number of draws varies. VERIFIED.
- **Plan effect.** CHANGES the CONTRACT `lr-train` (CRN assertion, noise handling) and the
  `learned-choice` decision rule. CONFIRMS PLAN's "common random numbers".
- **Reward/confidence.** high/high

### LR-04. Make baselines paper-faithful and calibrate them to this deployment, or "beats every heuristic" is inflated

- **Status.** VERIFIED (LMetric pdf p.2, p.8–9; SMetric pdf p.2, p.6, p.8–10; CtR pdf p.3–5;
  DualMap pdf p.7; Henderson pdf p.5; code lmetric.rs, signals.rs, dualmap.rs,
  optimized_baseline.rs). Corrected: SMetric's gains are conditional on a global KV tier, and its
  first-turn rule is restated.
- **Lesson.** Three ports are weaker than the methods they represent:
  - The `lmetric` port omits the queued-prefill term that the paper credits for its win.
  - The `dualmap` and `llm-d-optimized-baseline` ports carry physical constants from other
    deployments.

  Mispriced constants cost a router 4.5 goodput points in Calibrate-then-Route. That study measured
  disaggregated serving of a 3B model on A40s; it is the nearest measured study, not a close match.
- **Action.**
  1. Add `lmetric-faithful`, scoring `max(active_prefill_tokens + uncached_tokens, 1) ×
     (active_requests + 1)`. Report both variants and log the gap in `facts/UPSTREAM_FOLLOWUPS.md`.
  2. In the "defaults" arm, derive the physical constants:
     - dualmap's `pending_prefill_token_budget` = the family TTFT SLO × AIS uncontended prefill
       throughput for Qwen3-32B TP2 H100;
     - llm-d's `peak_prefill_tokens_per_second` from the same AIS profile.

     This is calibration-time AIS use only, with no runtime coupling. Record the derivation in
     `facts/`. Keep the CMA-ES-tuned arm separate.
  3. Run decision-parity tests of each port against its upstream reference on randomized candidate
     tables. A port that cannot be checked is marked "unverified" in REPORT.
  4. Add an `smetric`-style baseline, following SMetric Fig 13 (pdf p.8; corrected):
     - First turn, or a follow-up whose session was evicted (best overlap < HIT_RATIO × expected
       hit): send to the least-loaded worker by `estimate_load = q_i + prefill_cost(req, c_i)`. This
       is queued prefill plus the request's uncached prefill, not the LMetric product.
     - Follow-up: send to the highest-overlap worker if `estimate_load / PREFILL_RATE ≤ SLACK ×
       TTFT_SLO(req)`. Otherwise send to the least-loaded worker. If no worker meets the deadline,
       stick to the highest-overlap worker.
     - PREFILL_RATE is a calibration-time constant from AIS, like action 2.

     Replay has no global KV tier, so migration loses KV. Label this baseline "SMetric without global
     tier". SMetric reports its advantage over the best baseline holds "except for setups without any
     global KV$ store" (pdf p.10), so it is a reasonable heuristic here but not a known winner.
- **Stage.** Build and baselines, before Stage 4 tuning.
- **Evidence.**
  - LMetric §1 p2 and §5.1 p9: P-token is "the number of new prefill tokens if the request is routed
    to an instance, considering KV$ hits". The paper says its edge over 1 − hit ratio (14.4% lower
    P50, 42.8% lower P95 TTFT) comes because "it additionally considers the queued prefill tokens in
    each instance" — `llm-routing/zhang2026-lmetric.pdf`. VERIFIED (pdf p.2, p.8–9).
  - SMetric §4.1 restates LMetric as f = (q_i + L − c_i) × l_i, with q_i the queued prefill work.
    §4.2 and Fig 13 give its policy. The +9–15% peak TPS over the best of LMetric, llm-d, Preble
    and Dynamo holds "under colocation with a provisioned global tier" (corrected qualifier). The
    advantage holds "except for setups without any global KV$ store" — `llm-routing/wang2026-smetric.pdf`.
    VERIFIED (pdf p.2, p.6, p.8, p.10).
  - [code] `WT/lib/router-plugins/builtin/src/lmetric.rs:151-157` scores
    `uncached_prompt_tokens(..).max(1) × (active_requests() + 1)`; `signals.rs:18-28` subtracts
    cached blocks from the prompt only. VERIFIED.
  - [code] `dualmap.rs:57` sets the budget to `65_536`; `llm_d/optimized_baseline.rs:79` sets
    `15_928.0`. VERIFIED.
  - Calibrate-then-Route §III-B, §V-B and §VI (corrected sections):
    - §III-B: the simulator-derived τ_dec was 7.4× too small;
    - §VI: the constants were 7–13× off overall;
    - §V-B: the same policy with simulator constants lost 4.5 goodput points (0.819 vs 0.864). Its
      router "balances request counts almost perfectly", which is a degeneration into queue
      counting.

    `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf p.3–5). The mechanism is a unit
    mismatch between token-priced work and observed seconds. It applies here to dualmap's token
    budget vs the SLO and to llm-d's tokens/s conversion.
  - DualMap §3.2: "Given a predefined TTFT SLO, we compute the maximum number of pending prefill
    tokens that a GPU can process within the SLO" — `llm-routing/yuan2026-dualmap.pdf`. VERIFIED
    (pdf p.7). With a length-scaled T(L) (LR-08), evaluate it at a stated reference length.
  - Henderson et al. 2018, "Codebases": baseline implementations differ materially —
    `eval-methodology/henderson2018-deep-rl-matters.pdf`. VERIFIED (pdf p.5).
  - Melis et al. 2018. UNVERIFIED (abstract only).
- **Plan effect.** CHANGES the PLAN Baselines table: new `lmetric-faithful` and `smetric` rows, and
  calibrated defaults.
- **Reward/confidence.** high/high

### LR-05. Remove exact flat directions and gauge freedoms from the CMA-ES space, and structure the context as named sources

- **Status.** VERIFIED (Tomlinson pdf p.4–5, p.13; Ko & Li pdf p.5; Seshadri pdf p.4, p.8; Rosenfeld
  pdf p.7–8; Hansen tutorial pdf p.24, p.29; ARS pdf p.7–8; ES pdf p.4). Corrected: the proposed
  form is a column-restricted LCL, not DLCL, and "vanish" was overstated.
- **Lesson.** Free search over (θ, p, q, τ) wastes budget along flat directions, where CMA-ES σ
  drifts upward. Three are exact:
  - at τ = 0, scaling (θ, p) together never changes a decision;
  - at τ > 0, scaling (θ, p, τ) together never changes one;
  - the bilinear p_k q_kᵀ term has a gauge (p → cp, q → q/c, and sign).
- **Action.**
  1. Pin the scale. Fix one coefficient whose sign is known and nonzero:
     - θ₀ = −1 for starts near default;
     - the `log_ptok` coefficient = −1 for starts near LMetric (LR-07).

     Alternatively, decode θ/‖θ‖ and add a small penalty on (‖θ_raw‖ − 1)².
  2. At τ = 0, do not search τ. Treat τ > 0 as a separate named variant with θ's scale fixed (LR-15).
  3. Replace the free q_k with fixed unit vectors over a named context vector z_S:
     u_i = θ·x_i + Σ_k z_{S,k} (p_k·x_i).
     - Corrected attribution: this is Tomlinson & Benson's **LCL**, u_i = (θ + A z_S)ᵀx_i, with A
       restricted to one free column p_k per named source. It is **not** their DLCL, which is a
       mixture of d logits with separate intercepts B_k and mixing weights π_k.
     - The term is linear in the free p_k, so it has no gauge.
     - Pre-register 2–3 sources (LR-06).
     - Add a source only if validation improves beyond the MDE.
  4. Condition the search:
     - set `CMA_stds` per coordinate ∝ 1/(the feature's within-set sd), computed offline from logged
       candidate tables of default-router train replays;
     - choose σ0 so that about 10–30% of decisions flip relative to θ₀ [hypothesis range];
     - watch the σ and condition-number traces in `history.jsonl`.
- **Stage.** Build (policy schema), training (`space.yaml`).
- **Evidence.**
  - Tomlinson & Benson 2021 (corrected):
    - §3.1 defines LCL as F(C) = A x̄_C, and says a "low constant-rank approximation of A" keeps
      the parameter count linear in d;
    - §3.2 defines DLCL as a mixture;
    - Tables 5–7 (sushi, expedia, car-alt) show that the largest full-model A entries change a lot
      when each is fitted alone: many shrink (0.24 → 0.04, −0.47 → −0.13), while some hold or grow
      (0.15 → 0.20, −0.20 → −0.26). Full-A magnitudes are not interpretable in isolation, but they
      do not uniformly "vanish".

    `dcm-context/tomlinson2021-lcl-dlcl.pdf`. VERIFIED (pdf p.4–5, p.13).
  - Ko & Li 2024 (eq. 3): without a Frobenius penalty the low-rank parameters are "only unique up to
    a multiplicative factor" — `dcm-context/ko2024-choice-self-attention.pdf`. VERIFIED (pdf p.5).
  - Seshadri et al. 2019 §2.2 and §4: on SFwork and SFshop, the unfactorized CDM "fails to
    outperform low-rank CDMs" out of sample — `dcm-context/seshadri2019-cdm.pdf`. VERIFIED (pdf p.4,
    p.8). Scope: an item-level model, not a feature-based one.
  - Rosenfeld et al. 2020 Fig 3: "roughly 90% of the gain in accuracy is achieved by ℓ = 4", where ℓ is
    the number of aggregated score functions, which is analogous to rank —
    `dcm-context/rosenfeld2020-set-dependent-aggregation.pdf`. VERIFIED (pdf p.7–8).
  - Hansen tutorial §5: under random selection, E[σ^(g+1)] > σ^(g). App. A: the initial ranges
    "should not disagree by several orders of magnitude. Otherwise a scaling of the variables should
    be applied" — `learned-systems-policies/hansen2016-cmaes-tutorial.pdf`. VERIFIED (pdf p.24, p.29).
  - ARS §3.2: "We were not able to train a linear policy for this task [Humanoid] without the
    normalization of the states" — `learned-systems-policies/mania2018-ars.pdf`. VERIFIED (pdf p.7–8).
  - ES §2.2: it is crucial that perturbations occasionally produce individuals with better return.
    Perturbed Atari policies that always took one action gave inadequate exploration —
    `learned-systems-policies/salimans2017-es.pdf`. VERIFIED (pdf p.4).
- **Plan effect.** CHANGES the CONTRACT `learned-choice` utility and parameter schema
  (`context: {sources, p}` instead of `{p, q}`) and the `space.yaml` conventions.
- **Reward/confidence.** high/high

### LR-06. Make the context and set-relative features N-stable: linear set-relative terms and z-scores do nothing or break at N = 2

- **Status.** VERIFIED (Tomlinson pdf p.4–6; Pfannschmidt pdf p.7, p.11, p.22–23; Rosenfeld pdf
  p.4; Webb pdf p.26–30; Lodestar pdf p.5). The no-op and z-score claims are exact algebra.
  Corrected: the Deep Sets and SDA claims were overstated, and the LOO rank fraction fails the
  proposed duplication test.
- **Lesson.** Four feature choices in the current plan fail across N:
  - Mean-pooling a one-hot feature leaks 1/N into the context: the mean of `session_affinity` is 1/N.
  - A set-constant request covariate cancels inside x_i, and it duplicates a hand-made interaction
    once it is placed in the context.
  - x_i − min_S or x_i − mean_S entering linearly is an exact no-op under argmax.
  - Z-scores are ±1 at N = 2, whatever the gap.
- **Action.**
  1. Build the context vector z_S from request-level, N-stable quantities:
     - mean_S `kv_load_frac`;
     - mean_S `active_prefill_tokens_k`;
     - mean_S `active_requests_s`;
     - max_S `overlap_frac`;
     - `isl_k`;
     - `is_first_turn`.

     Never mean-pool `session_affinity` or `hash_home`.
  2. Let ISL enter only through z_S. Drop v1 feature #7 `isl_x_prefill_load`, because it equals the
     (ISL source, `active_prefill`) entry of P. Alternatively, freeze θ₇ = 0 whenever ISL is a
     source.
  3. In the PLAN M2 row, delete "x_i − min_S, rank". In v2 only, use nonlinear, N-normalized
     set-relative features:
     - x_ik/(ε + mean_S x_k) for load features;
     - the strict-below fraction #{j: x_j < x_i}/N (corrected from #{j≠i: x_j < x_i}/(N−1); see
       action 4);
     - relu(x_ik − mean_S x_k).

     Use no z-scores, and no sums or counts over S.
  4. Add a unit test: duplicating the worker set (N → 2N identical copies) leaves z_S and the argmax
     class unchanged.
     - Corrected: the leave-one-out rank fraction fails this test. For a worker with k strictly
       smaller peers, it becomes 2k/(2N−1) instead of k/(N−1); for example, 1 becomes 2/3 at N=2.
     - #{j: x_j < x_i}/N is invariant (2k/2N), and so are the ratio and relu features.
  5. Document all of this in `FEATURES.md` before calibration.
- **Stage.** Build (features), before the v1 freeze.
- **Evidence.**
  - Tomlinson & Benson 2021:
    - §3: F(C) = (1/|C|) Σ f(x_j), so each item's effect is diluted by 1/|C|;
    - Lemma 4.3: β_{i,C} = (θ + A x̄_C)ᵀ(x_i − x̄_C), so set constants cancel;
    - Prop 4.5: A is identifiable only from at least d+1 choice sets with affinely independent
      mean features.

    `dcm-context/tomlinson2021-lcl-dlcl.pdf`. VERIFIED (pdf p.4–6).
  - Pfannschmidt et al.:
    - §4.1: FETA averages pairwise terms with 1/(|Q|−1). VERIFIED (pdf p.7).
    - Added, and stronger than the original evidence: §4.3 shows that with linear sub-utility and
      linear aggregation, FATE's set term "does not depend on x", so the singleton choice "is
      independent of the context Q". This is the linear set-relative no-op in their notation.
      VERIFIED (pdf p.11).
    - §6.4.1 (corrected): SDA had the worst accuracy on LETOR and Expedia. The authors only
      "suspect" that fixed-size training is the cause, and did not test it. VERIFIED (pdf p.22).
    - §6.4.3 Fig 8: FATE-Net and FETA-Net trained at size 10 generalize to sizes 3–21, but
      generalization "depends on the dataset". On Hypervolume, all models degrade as size grows,
      with FATE and FETA degrading more slowly. VERIFIED (pdf p.23).

    `dcm-context/pfannschmidt2019-fate-feta.pdf`.
  - Rosenfeld et al. 2020 eq 4: μ(F(x) − r(F(s))) with an asymmetric, kinked-tanh μ (App. D.4) —
    `dcm-context/rosenfeld2020-set-dependent-aggregation.pdf`. VERIFIED (pdf p.4, p.20–21). Note:
    that a reference point matters *only* through a nonlinearity is our inference (with μ = identity,
    r(s) cancels). The paper does not state it.
  - Webb et al. 2021 §4.2: the divisive normalization model "captures the sample choice probabilities
    for all set sizes" (2 to 12 alternatives) better than probit or range normalization —
    `dcm-context/webb2021-divisive-normalization.pdf`. VERIFIED (pdf p.26–30). Scope: this is human
    consumer choice, used here only as design inspiration for ratio features.
  - Deep Sets (corrected): Lemma 3 characterizes permutation-equivariant layers, with a max-pool
    variant in eq 4. In §4.1.2, a **sum**-pooled DeepSets trained on sets of size ≤ 10 degrades more
    slowly than LSTM or GRU on sets up to 100, on a digit-**sum** task —
    `learned-systems-policies/zaheer2017-deep-sets.pdf`. VERIFIED (pdf p.2–5). This is weak support
    for permutation-invariant pooling in general, not evidence that mean or max pooling is N-stable.
  - Lodestar §4.1: "the same parameters θ are shared across all instances, and instance identity is
    never an input", so the architecture is "instance-count independent" —
    `llm-routing/lim2026-lodestar.pdf`. VERIFIED (pdf p.5).
- **Plan effect.** CHANGES the PLAN M2 row, CONTRACT feature v1 (#7) and the v2 list (drops
  z-scores).
- **Reward/confidence.** high/high

### LR-07. Put the strongest heuristics inside M1's hypothesis class, and multi-start from them

- **Status.** VERIFIED (LMetric pdf p.6, p.11, p.13; SMetric pdf p.6, p.8, p.10; DualMap pdf p.5–6;
  Lodestar pdf p.6, p.12; Park pdf p.4). Corrected: which heuristic wins depends on the regime,
  so the case for multi-starting gets stronger.
- **Lesson.** A linear utility over v1's raw features cannot exactly represent the three
  strongest heuristic decision rules. No single one of them dominates in this campaign's setting:
  - LMetric's product score beat per-workload-tuned linear scores, including tuned Dynamo, on
    LMetric's own workloads. These include chatbot, API, coder and two agent traces; SMetric (pdf
    p.6) puts their ideal reuse at "typically around 50–70%".
    - Corrected: on an agentic workload with 82% ideal reuse and **no global KV tier**, which is
      this replay's setting, SMetric measured LMetric at 920 TPS within SLO. That is 36% below
      BAILIAN's tuned linear cache-plus-load score, because LMetric's reuse ratio fell to 45% vs 74%.
    - So the product must be representable, but it is not a safe default for AgentX.
  - Greedy best-of-N disperses shared prefixes, where restricting to two consistent-hash candidates
    does not. Lodestar applies its k = 2 filter only when cluster KV utilization exceeds 80%.
  - Session-centric gating (SMetric) is SLO-aware stickiness.
    - Corrected: its published advantage depends on a global KV tier that makes migration cheap.
      Without one it is not shown to win.
    - v1's `session_affinity` with a large weight already gives a soft version of it.
- **Action.**
  1. Add these features to v1, or to v2 if v1 is already frozen:
     - `log_ptok = ln(max(active_prefill_tokens + new_prefill_tokens, 1)/1024)`;
     - `log_bs = ln(1 + active_requests)`;
     - `hash_home ∈ {0, 1}`: 1 if the worker is among the top-2 rendezvous winners for the
       request's prefix root or session key. Reuse `signals::rendezvous`
       (`WT/lib/router-plugins/builtin/src/signals.rs:53`);
     - `is_first_turn`, as a context source (LR-06).
  2. Add a parity test: θ = −(e_log_ptok + e_log_bs) reproduces the `lmetric-faithful` argmin on
     random tables.
  3. Multi-start CMA-ES from θ_default and θ_LMetric, and optionally from a start weighted toward
     `hash_home`. Report which start wins, per family. Given the SMetric counter-evidence, also add
     a cache-heavy start (large `overlap_frac` and `session_affinity` weights) for AgentX
     [hypothesis].
  4. Lodestar gates its hash filter on cluster KV utilization. With `hash_home` in x_i and mean_S
     `kv_load_frac` in z_S (LR-06), the LCL context term can express that gate linearly as a
     load-dependent `hash_home` weight [hypothesis].
- **Stage.** Build (features), training.
- **Evidence.**
  - LMetric:
    - §4.4: the optimal linear KV weight is 0.7 on ChatBot and 0.55 on API;
    - §6: it outperforms all baselines, including Dynamo "tune[d] ... for each workload";
    - §6 production canary: mean TTFT −39% and mean TPOT −51% vs BAILIAN's prior scheduler. This
      was a one-day split with 1/3 vs 2/3 of the traffic.

    `llm-routing/zhang2026-lmetric.pdf`. VERIFIED (pdf p.6, p.11, p.13).
  - SMetric §4.1 (added counter-evidence): without a global tier, LMetric served 920 TPS within SLO,
    36% below BAILIAN's 1,437, even though its load balance was better —
    `llm-routing/wang2026-smetric.pdf`. VERIFIED (pdf p.6).
  - DualMap §2: a global best-of-all strategy "is effectively equivalent to using d = n choices", which
    severely degrades KV reuse; Min-TTFT "may oscillate between cache-aware and load-aware decisions" —
    `llm-routing/yuan2026-dualmap.pdf`. VERIFIED (pdf p.5–6).
  - Lodestar §4.1 and §5.6 Fig 14:
    - the k = 2 consistent-hash filter applies when cluster memory utilization exceeds 80%;
    - removing it raised average TTFT 1.72× and P99 TTFT 1.78× on ToolAgent;
    - corrected page: Fig 14 is on pdf p.12.

    `llm-routing/lim2026-lodestar.pdf`. VERIFIED (pdf p.6, p.12).
  - SMetric §4.2 Fig 13: first turn goes to the least-loaded worker; a follow-up goes to the
    highest-hit worker if it meets SLACK × TTFT SLO — `llm-routing/wang2026-smetric.pdf`. VERIFIED
    (pdf p.8). Its gain is conditional on a global tier (pdf p.10).
  - Park §3.1: "bootstrapping from existing policies can improve policy search efficiency" —
    `learned-systems-policies/mao2019-park.pdf`. VERIFIED (pdf p.4).
  - Exact representability is our algebra, not a paper claim: ln(P × BS) = ln P + ln BS, and argmin
    is invariant to monotone transforms.
- **Plan effect.** CHANGES the CONTRACT feature set and the PLAN M1 row (start points).
- **Reward/confidence.** high/medium-high. Confidence is high for the log features and medium for
  `hash_home` and `is_first_turn`.

### LR-08. One fixed TTFT threshold per family is wrong for 1K–128K prompts

- **Status.** VERIFIED (SMetric pdf p.3, p.8, p.10; DistServe pdf p.10; CtR pdf p.4; Wang et al.
  pdf p.6; setup engine.json and STATE.md). No correction. One supporting citation added.
- **Lesson.** With prompts up to 128K, a single family-level TTFT bound makes long requests
  unattainable even on an idle worker. Goodput then ignores them, or the optimizer learns to
  sacrifice them. A single (T, I) pair is also only one point on the SLO-scale curve.
- **Action.**
  1. During calibration, set T(L) = b + a·L per family:
     - a is a slack multiple of the AIS uncontended per-token prefill time for this deployment
       (calibration-time use only);
     - b comes from default's near-knee runs;
     - I is set as planned.

     Record, per family, the fraction of requests that are unattainable on an idle worker.
  2. Re-score every evaluation from `per_request` at SLO scales {0.5, 0.75, 1, 1.5, 2, 3} × (T, I).
     This needs no extra replays. The headline must hold at at least one tighter and one looser
     scale, and REPORT states where it flips.
  3. In `lr-report`, add `good_frac` by ISL bucket.
- **Stage.** Calibration, reporting.
- **Evidence.**
  - SMetric:
    - p10: TTFT budget b = 1 s plus a = 62.5 ms per 1K prompt tokens, TPOT ≤ 20 ms, and Fig 17
      shows SMetric is robust across SLO settings;
    - p8 (added): the TTFT deadline "is a function set by the provider, typically linear in the
      request length";
    - p3: "a single 128K-token request consumes about 32 GB of KV$ on the popular Qwen3-32B model".

    `llm-routing/wang2026-smetric.pdf`. VERIFIED (pdf p.3, p.8, p.10).
  - DistServe §6.1–6.2 and Fig 8: attainment is swept over rate and over SLO scale (1.5× down to
    0.75×) — `eval-methodology/zhong2024-distserve.pdf`. VERIFIED (pdf p.10).
  - Calibrate-then-Route §IV-B: "TTFT 175 ms / TPOT 39.7 ms tight; 3× loose" —
    `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf p.4).
  - Wang et al. 2024 §4.2 eq 8: a request that already missed its SLO contributes 0, so the
    "goodput-optimal scheduling strategy should kill this request" —
    `eval-methodology/wang2024-revisiting-slo-goodput.pdf`. VERIFIED (pdf p.6).
  - [setup] `CR/config/engine.json` gives 301,808 KV tokens per worker, so one 128K context fills
    about 43% of a worker. `facts/STATE.md`: single-request TTFT is 55 ms at 1K and 476 ms at 8K.
    VERIFIED (STATE.md gives 55.49 ms and 475.97 ms; 131,072/301,808 = 0.434).
- **Plan effect.** CHANGES the PLAN Objective "T and I: one pair per family" to a length-scaled
  T(L) per family plus an SLO-scale sweep.
- **Reward/confidence.** High reward. Confidence is high that the problem exists and medium for the
  exact (a, b) recipe.

### LR-09. Session workloads must release turns causally; stratify everything by load mode

- **Status.** VERIFIED (CausalSim pdf p.1–3; AgentServeSim pdf p.1–2, p.4; Schroeder pdf p.10–11;
  code driver.rs:3320; setup.json). Scope note added: Schroeder's principles concern scheduling
  order inside a server, so applying them to routing is an analogy.
- **Lesson.** Load mode distorts results in two ways:
  - Open-loop timestamp replay of multi-turn and agentic sessions treats arrival times as exogenous,
    although they embed the recording system's latency.
  - Closed and partly-open loops compress differences between policies.

  Pooling modes therefore both biases and dilutes the effect.
- **Action.**
  1. Make `agentic_lanes` the primary AgentX mode, and report trajectory latency.
  2. For AgentX `trace_timestamps` and for synthetic sessions, verify that replay releases turn k+1
     only after turn k completes.
     - Mooncake rows with a `session_id` already use completion-relative `delay`
       (`facts/setup.json`, `session_context`).
     - Where release is not causal, log per policy the fraction of turns that arrive before their
       predecessor finished, and label those cells open-loop stress cells, outside the headline.
  3. Stratify every gate, table and bootstrap by load mode (open, closed, lanes). Pilot gate
     condition 3 must hold per mode.
  4. Calibrate closed-loop and lane levels at moderate concurrency, where scheduling differences can
     appear, not at saturation.
- **Stage.** Calibration, reporting.
- **Evidence.**
  - CausalSim §1 and §3.1: the exogenous-trace assumption. The baseline simulator that ignored bias
    predicted that the tuned BOLA1 "should stall 1.34× the stall rate of BBA"; deployment measured
    0.7× — `learned-systems-policies/alomar2023-causalsim.pdf`. VERIFIED (pdf p.1–3).
  - AgentServeSim abstract and §1: "Replaying its turns at recorded timestamps fixes the release
    schedule and cache state that a different policy is supposed to change"; successors must be
    released causally — `llm-routing/rajib2026-agentservesim.pdf`. VERIFIED (pdf p.1–2, p.4).
  - Schroeder et al. 2006:
    - Principle (v): scheduling helps closed systems only at moderate load and high MPL;
    - Principle (vii): a partly-open system behaves as closed when requests per session are
      "≥ 10 as a rule-of-thumb".

    `eval-methodology/schroeder2006-open-vs-closed.pdf`. VERIFIED (pdf p.10–11). Scope: these
    principles are about scheduling order (e.g. SRPT vs FCFS), and applying them to routing is an
    analogy.
  - [code] aisimulate-core `replay/loadgen/driver.rs:3320`, test
    `agentic_mode_releases_turn_after_dependency_completion_plus_delay`. VERIFIED.
  - AgentX averages about 38 requests per play (3,132/82, PLAN). That is past Schroeder's threshold
    of 10 requests per session, beyond which a partly-open system behaves as closed. Arithmetic
    VERIFIED (38.2).
- **Plan effect.** CHANGES the PLAN Workloads load-mode table (a primary mode per family) and the
  gate definition. CONFIRMS that lanes exist.
- **Reward/confidence.** high/high

### LR-10. Select by validation over several CMA-ES restarts, never the best-ever train sample, and treat baselines the same way

- **Status.** VERIFIED (Cawley pdf p.1, p.16, p.25; Dodge pdf p.1–3; Henderson pdf p.5; ARS pdf
  p.3, p.13, p.17–18; Jain 2020 pdf p.4–5, p.8; PEGASUS pdf p.4; Decima pdf p.19; Hansen tutorial
  pdf p.32). No correction.
- **Lesson.** Three effects distort comparisons:
  - The best-ever sample of a noisy search is selection-biased by an amount comparable to the
    differences between methods.
  - A single optimizer seed misleads.
  - Which method wins depends on the tuning budget.
- **Action.**
  1. Run at least 3 independent CMA-ES restarts per rung and per tuned baseline. IPOP-style growing λ
     is optional.
  2. Select the final candidate as the argmax of validation goodput, under fresh tie seeds, over the
     last-k distribution means and the top-k samples. `best.json` records both the selected
     candidate and the best-ever.
  3. Plot best incumbent against evaluations for every method. Extend any baseline still improving at
     the cut-off. Report the total evaluations per method, including sweeps.
  4. Budget each rung by its parameter count: M1 (about 10 parameters) needs fewer cells per
     generation than M2. Use N = 4 and short windows early and full cells late, with CRN within each
     generation.
- **Stage.** Training, baselines.
- **Evidence.**
  - Cawley & Talbot 2010:
    - abstract: over-fitting the model-selection criterion has effects "of comparable magnitude to
      differences in performance between learning algorithms";
    - §4.4: remedies are regularization, early stopping and averaging;
    - §6: model selection must be repeated in every trial.

    `eval-methodology/cawley2010-overfitting-model-selection.pdf`. VERIFIED (pdf p.1, p.16, p.25).
    Scope: the paper concerns hyper-parameter criteria; the best-ever CMA-ES sample is the same
    mechanism.
  - Dodge et al. 2019 §2–3: expected validation performance as a function of budget; which model
    wins "often depends on the" budget — `eval-methodology/dodge2019-show-your-work.pdf`. VERIFIED
    (pdf p.1–3).
  - Henderson et al. 2018 Fig 5: two 5-seed groups of one configuration differ (t = −9.09,
    p = 0.0016) — `eval-methodology/henderson2018-deep-rl-matters.pdf`. VERIFIED (pdf p.5).
  - ARS:
    - §4.2: over 100 seeds, some seeds "lead ARS to discover locally optimal behaviors";
    - §5: "should we not count the number of rollouts used for every tested hyperparameters?".

    `learned-systems-policies/mania2018-ars.pdf`. VERIFIED (pdf p.13, p.17–18).
  - Jain et al. 2020 §4.3 and Alg 1 line 17 ("Select θ with best validation performance"), Fig 6 —
    `dcm-context/jain2020-generalization-new-actions.pdf`. VERIFIED (pdf p.4–5, p.8).
  - PEGASUS §4.1 Thm 1: the required number of scenarios grows with the VC dimension of the policy
    class — `learned-systems-policies/ng2000-pegasus.pdf`. VERIFIED (pdf p.4).
  - Decima App. I Table 3: an agent trained on a 10× smaller cluster had 3% higher average JCT
    (630 vs 610 s) — `learned-systems-policies/mao2019-decima.pdf`. VERIFIED (pdf p.19).
  - Hansen tutorial App. A: "Independent restarts with increasing population size ... are a useful
    policy" — `learned-systems-policies/hansen2016-cmaes-tutorial.pdf`. VERIFIED (pdf p.32).
- **Plan effect.** CHANGES the CONTRACT `lr-train` outputs and the PLAN Training section (restarts,
  selection rule). CONFIRMS the equal CMA-ES budget.
- **Reward/confidence.** high/high

### LR-11. Cells are not independent: pre-register a per-baseline headline test over trace segments

- **Status.** VERIFIED (Demšar pdf p.5, p.7, p.12–13; Agarwal pdf p.5–8, p.21–22; Bouthillier pdf
  p.8–10; Genet pdf p.3–4, p.12). Corrected: the P(A>B) criterion and the minimum segment count are
  fixed, and the Seshadri & Ugander citation is removed.
- **Lesson.** One trace window at L1–L3 and at several N is one workload, so confidence intervals
  over cells overstate significance. "Beats every heuristic" is a conjunction of paired tests and
  should be fixed before training.
- **Action.**
  1. Before training, write `facts/HEADLINE_TEST.json`:
     - for each tuned baseline b, a one-sided Wilcoxon signed-rank test on segment-level paired
       log-ratios on the test split, at α = 0.05. All must pass; as an intersection-union test, the
       conjunction needs no correction;
     - corrected: the exact one-sided Wilcoxon cannot reach p < 0.05 with fewer than 5 segments.
       At n = 5 its minimum p is 1/32 ≈ 0.031, and every segment must favor learned. Pre-register
       at least 6–8 independent test segments per pooled test, and more for per-family claims. If
       a family has fewer, its claim is descriptive only. Demšar notes sample sizes "as small as
       five"; Bouthillier warns that tests over few datasets have "very limited statistical power";
     - P(learned > b), following Bouthillier §4.1 (corrected): the CI lower bound must exceed 0.5
       ("significant") **and** the CI upper bound must exceed γ = 0.75 ("meaningful");
     - Holm correction for per-family and per-N claims;
     - TOST margins (for example ±0.5%) for learned-choice@θ₀ ≡ default parity and for any "no
       regression at unseen N" claim.
  2. In `lr-report`:
     - use a cluster bootstrap that resamples independent segments within each family, carrying all
       of a segment's cells;
     - report IQM and geometric mean with CIs, and performance profiles;
     - add a per-cell winner map and the gap to the per-cell best tuned baseline (virtual best);
     - report the validation-to-test drop for every rung and baseline.
  3. Count every M2 variant tried on validation, and report that number.
- **Stage.** Calibration (pre-registration), reporting.
- **Evidence.**
  - Demšar 2006:
    - §3: the "sample size" is the number of independent data sets, "as small as five";
    - §3.1.3: the Wilcoxon signed-ranks test;
    - §3.2: Holm's step-down procedure.

    `eval-methodology/demsar2006-statistical-comparisons.pdf`. VERIFIED (pdf p.5, p.7, p.12–13).
  - Agarwal et al. 2021:
    - §4.1: stratified bootstrap CIs, which resample runs within each task. For segments within a
      family, our cluster bootstrap is the analogue;
    - §4.2: performance profiles;
    - §4.3: IQM and average probability of improvement;
    - App. A.5: bootstrap details.

    `eval-methodology/agarwal2021-statistical-precipice.pdf`. VERIFIED (pdf p.5–8, p.21–22).
  - Bouthillier et al. 2021 §4 (corrected):
    - P(A > B) is "significant" when P − CI_min > 0.5 and "meaningful" when P + CI_max > γ;
    - γ = 0.75 was robust across their case studies;
    - §6 notes that Wilcoxon over 3–5 datasets has "very limited statistical power".

    `eval-methodology/bouthillier2021-variance-ml-benchmarks.pdf`. VERIFIED (pdf p.8–10).
  - Genet §2 Fig 2b: "Even if RL schemes perform better on average, they are worse than the baselines
    on a substantial fraction of test environments", including a load-balancing use case. The
    discussion suggests "an 'ensemble' of existing baselines (i.e., measuring the maximum gap to any
    baseline from a set)" — `learned-systems-policies/xia2022-genet.pdf`. VERIFIED (pdf p.3–4, p.12).
  - Removed: Seshadri & Ugander 2020. It bounds the sample complexity of *testing IIA* on choice data.
    The abstract and paper say what was claimed, but this campaign never fits or tests choice
    probabilities, so the citation does not support any action here.
  - Krishnamachari 2026 §8 and §11. UNVERIFIED (web only).
  - Berger 1982 (intersection-union). UNVERIFIED (cited from memory). The IUT property itself is
    standard: each component test is run at α with no multiplicity adjustment.
- **Plan effect.** CHANGES the PLAN Stage 5/6 statistics: the bootstrap moves from "over cells and
  repeats" to over segments. Adds a pre-registration artifact.
- **Reward/confidence.** high/high

### LR-12. Matched per-worker load does not hold the routing regime fixed across N: calibrate per N and span cache pressure

- **Status.** VERIFIED (CtR pdf p.4–5; setup determinism.json; van der Boor pdf p.4, p.7, p.11–12;
  CacheRoute pdf p.4; Tahir pdf p.9). No correction. A supporting detail from CtR §V-D is added.
- **Lesson.** The regime in which routing matters moves with N and with cache pressure:
  - Informed routing ties at small widths, leads at moderate widths, and reconverges when capacity
    pools.
  - It separates from simpler policies only when the active prefix working set approaches cluster KV
    capacity.

  So linearly scaled loads can put N = 16/32 at a ceiling and N = 2 in a scarcity regime unrelated to
  policy generalization.
- **Action.**
  1. During calibration, for each test N in {2, 6, 16, 32}, find L1–L3 with the default router only,
     so no learned policy leaks in. Report N-extrapolation at both per-worker-matched and knee-matched
     loads. Flag ceilings, where default's good_frac ≈ 1 and RR ≈ default.
  2. For each cell, compute the cache pressure: unique prefix tokens touched in a ~100 s window
     / (N × 301,808).
     - Make sure train, validation and test each span values below 0.5, near 1 and above 1.
     - Plot learned-vs-default gain against it.
  3. Pre-register in the frozen test plan that |Δ| within the MDE at N = 2 is parity, not a failure.
     Report N = 2 separately.
- **Stage.** Calibration, generalization.
- **Evidence.**
  - Calibrate-then-Route:
    - §V-D: the router ties JSQ at decode width 3 (0.816 vs 0.822), leads at 4 (0.864 vs 0.835)
      and reconverges at 6 (0.858 vs 0.860 RR). The paper attributes width 6 to an operating
      point that "leaves headroom", which is exactly the unmatched-load confound;
    - §V-E: on the starved 1P+2D pool, learned 0.292 vs RR 0.680, "reproduced ... four times".

    `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf p.4–5). Scope: this is
    disaggregated serving with a 3B model on A40s.
  - [setup] The overloaded N=4 smoke cell shows the same inversion: RR 0.637 vs default 0.281. VERIFIED.
  - van der Boor et al.:
    - §1: under JSQ, "the mean waiting time vanishes as N grows large for any fixed λ < 1";
    - §2 eq 2.1: in the Halfin–Whitt regime, "the relative capacity slack behaves as β/√N";
    - §2.4 Table 1: comparison of load-balancing algorithms.

    `learned-systems-policies/vanderboor2018-scalable-lb.pdf`. VERIFIED (pdf p.4, p.7, p.11–12).
  - CacheRoute Table 2: when topK128 fits the warm allocation, the three balanced cache-aware
    policies tie at 100 QPS; at topK256 they separate, because "the active set outgrows the warm
    allocation" — `llm-routing/cheng2026-cacheroute.pdf`. VERIFIED (pdf p.4).
  - Tahir et al. Fig 4: mean-field policy performance approaches its limit only as system size grows
    (N = M²), and is worst at small M — `learned-systems-policies/tahir2022-meanfield-lb.pdf`.
    VERIFIED (pdf p.9).
  - The llm-d blog: the large win came at 73% cache demand. UNVERIFIED (web only).
  - [hypothesis] LLM engines are not M/M/N servers, so verify this at calibration.
- **Plan effect.** CHANGES the PLAN Loads section ("scaled with N so per-worker load stays matched"):
  add knee-matched loads at each test N and stratify by cache pressure.
- **Reward/confidence.** high/medium

### LR-13. Black-box search exploits gaps in its objective (concentration, sacrifice, reservation): constrain and audit for them

- **Status.** VERIFIED (GORGO pdf p.7–8, p.11–12; Wang et al. pdf p.1, p.6; Park pdf p.5; CtR pdf
  p.4–5; Lehman pdf p.9–10; code report.rs 934–950). Corrected: the title said "binary goodput",
  but GORGO's ES optimized p95 TTFT. The 2/N flag is also fixed.
- **Lesson.** Three documented failure modes apply:
  - Binary goodput rewards sacrificing requests that will miss anyway. This is an argument about
    the metric (Wang et al.), not an observed search result.
  - ES on a routing cost whose objective was **p95 TTFT only** zeroed the queue weight and sent
    about 100% of traffic to one replica, while E2E and ITL tails collapsed. Goodput here also checks
    mean ITL, which bounds this failure, but mean ITL averages out stalls (Wang et al. §1), so the
    bound is weak.
  - Learned load balancers drift into reservation or segregation policies that break under shift.
- **Action.**
  1. In `space.yaml`, sign-constrain the load coefficients to ≤ −ε through a transform.
  2. Add guard metrics to `lr-eval`:
     - p99 TTFT over all requests;
     - the fraction of requests with TTFT > 3T;
     - p99 per-token ITL, from `itl_distribution`;
     - max/mean per-worker assigned prefill tokens;
     - max worker request share;
     - session replication ratio;
     - per-worker share by ISL bucket;
     - one continuous metric: mean min(TTFT/T, 10).
  3. In the Stage 4 audit, flag:
     - any candidate that gains goodput while a guard worsens beyond a pre-set tolerance;
     - any policy whose max worker share exceeds min(2/N, 1/N + 0.25). Corrected: plain 2/N is
       vacuous at N = 2;
     - persistent long-ISL segregation. Stress-test it on the ISL-stretch hold-out.
- **Stage.** Training, Stage 4 audit.
- **Evidence.**
  - GORGO:
    - §5 and App. D Table 8: a (1+1)-ES tuned on p95 TTFT "learns to set wqueue to 0", and
      "GORGO chose to send 100% of requests to the closest replica", winning every TTFT percentile
      while being worst on the E2E and ITL tails;
    - §7 Limitations: "because the online tuner optimizes p95 TTFT, it can exploit continuous
      batching by concentrating load", and a "lower bound to the queue term mitigates but does not
      eliminate this trade-off".

    `llm-routing/toniolo2026-gorgo.pdf`. VERIFIED (pdf p.7–8, p.11–12).
  - Wang et al. 2024 §4.2 eq 8 (sacrifice) and §1: TPOT is loose because "a long stall in the
    middle of the request can be averaged out" — `eval-methodology/wang2024-revisiting-slo-goodput.pdf`.
    VERIFIED (pdf p.1, p.6).
  - Park §3.3 Fig 2: an RL load balancer learned "a 'reservation' policy that keeps a server empty
    for small jobs" and was not robust to workload change — `learned-systems-policies/mao2019-park.pdf`.
    VERIFIED (pdf p.5).
  - Calibrate-then-Route §V-E: "greedy cost minimization concentrates load on whichever instance
    momentarily prices cheapest" — `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf
    p.4–5).
  - Lehman et al., "Unintended Debugging": "search will often learn how to exploit bugs in
    simulations" — `learned-systems-policies/lehman2018-digital-evolution-creativity.pdf`. VERIFIED
    (pdf p.9–10).
  - [code] Replay's `is_good` uses mean ITL, (e2e − ttft)/(osl − 1) (`report.rs` lines 934–950).
    VERIFIED.
- **Plan effect.** CHANGES the CONTRACT `lr-eval` record and the Stage 4 audit checklist. CONFIRMS the
  PLAN's degenerate-policy audit.
- **Reward/confidence.** high/medium

### LR-14. Replay gives the router perfectly fresh state: test ranking stability under staleness and timing perturbation

- **Status.** VERIFIED (code kv_router/mod.rs; Mitzenmacher pdf p.2; Tahir pdf p.1–2; CtR pdf
  p.5–6; ThunderAgent pdf p.4–5; Peng pdf p.7; Vidur pdf p.9, p.15; Puffer pdf p.9–10).
  Corrected: the Vidur error figure is model-specific, and the CtR simulator differs in kind from
  AIS replay.
- **Lesson.** Simulated gains smaller than the perturbation spread are not claims, for three reasons:
  - Replay applies KV events to the router's index synchronously, while live routers see lagged
    events.
  - A deterministic argmax over stale state herds.
  - A re-fitted simulator can match goodput yet fail to reproduce the measured winner.
- **Action.**
  1. Add a Stage 1 audit lens, "information parity". For each feature, show that replay computes it
     from the same router-side state, with the same update timing, as the live router. Cover KV-event
     lag, `accounting_cache_estimate` vs device events, and when active prefill tokens are
     decremented.
  2. In Stage 5:
     - patch a KV-event and load-signal delay of 10, 50 and 200 ms on WT, and log it in
       `UPSTREAM_FOLLOWUPS.md`;
     - re-run learned vs default vs the best baseline on a test subset;
     - also run learned-choice with τ > 0 and a variant restricted to `hash_home` candidates, as
       guards against herding.
  3. Randomize timing during training: draw per-cell prefill and decode speedups from [0.9, 1.1],
     fixed per cell for CRN. Keep 0.8 and 1.2 as out-of-range test perturbations.
  4. In the report:
     - give the Kendall τ of the policy ranking across all perturbations;
     - claim only gaps larger than the spread the perturbations induce;
     - label the headline "in AIS-timed simulation";
     - compare engine-state distributions (batch size, context length, KV use) for learned vs
       default, and flag shifts into weakly validated AIS regions, such as context above 32K
       [hypothesis].
- **Stage.** Build audit, Stage 4 audit, Stage 5 test.
- **Evidence.**
  - [code] `WT/lib/mocker/src/replay/offline/extensions/kv_router/mod.rs:229-235`
    (`SyncReplayIndexer::apply_event`) and `:688-694` (`on_kv_events` applies events immediately).
    The directory has no delay knob. VERIFIED.
  - Mitzenmacher 2000: "having tasks go to the least loaded server can significantly hurt
    performance" when load information is old, while two random choices stay robust —
    `dcm-context/mitzenmacher2000-old-information.pdf`. VERIFIED (pdf p.2; the abstract and §1
    restate the §3–§6 results).
  - Tahir et al. §1: JSQ "fails when Δt > 0 mainly due to ... 'herd behaviour'" —
    `learned-systems-policies/tahir2022-meanfield-lb.pdf`. VERIFIED (pdf p.1–2). Scope: their model
    has many concurrent dispatchers. A single router with lagged state herds by Mitzenmacher's
    mechanism.
  - Calibrate-then-Route §VI: after refitting, the simulator matched goodput to a mean absolute
    residual of 0.068 "but it does not reproduce the measured winner on any workload". Also: "a
    simulator can hand a policy information no deployment will have" (there, the true backlog) —
    `llm-routing/tumkur2026-calibrate-then-route.pdf`. VERIFIED (pdf p.5–6). Corrected scope: their
    simulator was a hand-built serial-service DES, with batching added later, and constants 7–13×
    off. How far this transfers to AIS-timed engine replay is a [hypothesis], not a measured fact.
  - ThunderAgent §3.2: SGLang's "router-side radix tree that approximates worker KV state" goes stale
    under eviction thrashing — `llm-routing/kang2026-thunderagent.pdf`. VERIFIED (pdf p.4–5).
  - Peng et al.:
    - Table II: a feedforward policy without randomization succeeded 0.0 ± 0.0 on the real robot;
    - Table III: fixing the action timestep or removing observation noise dropped success to 0.29
      and 0.25, against 0.89 with all randomization.

    `learned-systems-policies/peng2018-dynamics-randomization.pdf`. VERIFIED (pdf p.7).
  - Vidur (corrected):
    - §7.2: near the capacity point, small prediction deltas "lead to significant blow up of the
      errors";
    - App. A.1: Vidur "retains high fidelity even at 95% of maximum system capacity for larger
      models". The 12.65% maximum error is for LLaMA2-7B, where CPU overheads cause cascading errors.

    `eval-methodology/agrawal2024-vidur.pdf`. VERIFIED (pdf p.9, p.15).
  - Puffer §5.2: emulation-trained Fugu performed "horrible" in the real world, and emulation results
    "differ markedly from the real world" — `learned-systems-policies/yan2020-puffer.pdf`. VERIFIED
    (pdf p.9–10).
  - The llm-d blog: a precise vs approximate prefix index gave TTFT p90 of 0.54 s vs 31 s.
    UNVERIFIED (web only).
- **Plan effect.** CHANGES the PLAN Stage 1 audit lenses, the Stage 5 robustness axes, and training
  (randomized speedups).
- **Reward/confidence.** high/medium

### LR-15. Climb the ladder only on measured need: drift check before M2, warm starts, τ = 0, and M3 as nest-level features

- **Status.** VERIFIED (Tomlinson pdf p.10–11, p.13; Seshadri pdf p.6; Train pdf p.2–9; Aouad pdf
  p.8; Webb pdf p.14–16; DeepHalo pdf p.5–6; Decima pdf p.11; code picker.rs:22-46). Corrected: the
  Decima Table 2 reading and the Aouad section number. The nested-logit argmax corollary is
  labeled as inference.
- **Lesson.** Three facts limit the value of richer rungs:
  - Context terms pay off only where the optimal coefficients drift with set statistics.
  - A larger model that nests a smaller one can still lose to it under a fixed optimizer budget.
  - Nest correlations in a nested logit have no effect on a deterministic argmax.
- **Action.**
  1. Run a drift check in the pilot or early in Stage 4:
     - tune M1 at a small budget separately on L1-only vs L3-only train cells, and on N = 4 vs N = 8;
     - build the cross-play matrix;
     - if the cross-play losses are within the MDE, deprioritize M2;
     - otherwise, add the drifting statistic to z_S, log-transformed if the drift looks
       multiplicative.
  2. Start the M2 CMA-ES mean at (θ*_M1, p = 0), with context `CMA_stds` about 0.3× θ's. Compare M2
     against continuing M1 for the same number of added evaluations.
  3. Train and ship at τ = 0:
     - τ > 0 is a separate named variant, reported at N = 2/16/32. At fixed τ, the softmax mass on
       non-best workers grows with N.
     - The `dynamo-default-cost-fn` temperature is range-normalized (`default/picker.rs:22-46`), so
       the two temperatures are different parameters. Run parity tests only at τ = 0.
  4. Build M3 from nest-level features (node or DP-group aggregate load) in x_i, not as a GEV model.
     An optional M2b adds one more mean-pooled interaction layer (DeepHalo).
  5. Interpret coefficients by leave-one-out ablation (zero each source and briefly re-tune), not by
     raw magnitudes.
- **Stage.** Pilot, model form, reporting.
- **Evidence.**
  - Tomlinson & Benson 2021:
    - §6.3 Fig 2: binned-MNL coefficients drift linearly in the log-transformed feature on
      mathoverflow, nonlinearly on email-enron, and not at all on the synthetic-MNL control. "Absent"
      is therefore shown only on a control;
    - footnote 3: "LCL does not beat MNL within the 500 training epochs", although it nests MNL;
    - Tables 5–7: full-model vs single-entry A magnitudes differ (see LR-05), which supports
      ablation over raw magnitudes.

    `dcm-context/tomlinson2021-lcl-dlcl.pdf`. VERIFIED (pdf p.10–11, p.13).
  - Seshadri et al. 2019 §4: "The CDM parameter optimization is initialized with values
    corresponding to a Luce MLE" — `dcm-context/seshadri2019-cdm.pdf`. VERIFIED (pdf p.6).
  - Train 2009 §4.2.1–4.2.4: the nested logit is GEV with correlated unobserved utility within nests,
    and IIA holds within each nest — `dcm-context/train2009-gev-ch4.clean.pdf`. VERIFIED (pdf p.2–9).
    That nest parameters drop out of a deterministic argmax over V is our corollary: they shape
    only the error distribution. Train does not state it.
  - Aouad & Désir (corrected section §2.2, not §2.3): without added noise, log-likelihood "ha[s] a
    gradient equal to zero on a set of measure 1"; Gumbel noise turns argmax into softmax to make
    training possible — `dcm-context/aouad2023-rumnet.pdf`. VERIFIED (pdf p.8).
  - Webb et al. §3.2: range normalization vs divisive normalization —
    `dcm-context/webb2021-divisive-normalization.pdf`. VERIFIED (pdf p.14–16). Used only as a naming
    analogy for the range-normalized default temperature.
  - Zhang et al. 2026 §3.2 eqs 4–5: recursive, residual, mean-pooled interaction layers —
    `dcm-context/zhang2026-deephalo.pdf`. VERIFIED (pdf p.5–6).
  - Decima §7.4 Table 2 (corrected reading):
    - average JCT is 91.2 s for the tuned heuristic, 65.4 s trained on the test workload, 104.8 s
      on an anti-skewed load, 82.3 s on mixed loads, and 76.6 s on mixed loads plus an
      interarrival hint;
    - training on a mismatched load is worse than the heuristic. Among agents *not* trained on the
      test workload, mixed loads plus a hint is best.

    `learned-systems-policies/mao2019-decima.pdf`. VERIFIED (pdf p.11).
  - [code] `default/picker.rs:22-46` normalizes by the cost range: scale = −1/((max − min) × τ).
    VERIFIED.
- **Plan effect.** CHANGES the PLAN M3 row and adds a drift gate before M2. CONFIRMS "a rung counts
  only if it improves held-out results".
- **Reward/confidence.** medium/high

## Plan deltas

These are the specific edits recommended to PLAN.md and CONTRACT.md. The lesson IDs give the
rationale.

**Features (CONTRACT feature set v1, before the calibration freeze)**

- Drop #7 `isl_x_prefill_load` (LR-06).
- Add `log_ptok`, `log_bs` and `hash_home` (LR-07).
- Make `isl_k` and `is_first_turn` request-level context inputs, not columns of x_i (LR-06).
- Pre-register these v2 candidates in `FEATURES.md`:
  - the ratio to the set mean for load features;
  - the strict-below fraction #{j: x_j < x_i}/N. Corrected from the leave-one-out rank fraction,
    which fails the duplication test (LR-06);
  - relu(x − mean_S);
  - `affined_ctx_frac`: session-committed context on worker i within a ~120 s TTL, divided by KV
    capacity (SMetric p5 and p7; ThunderAgent p5);
  - `prefill_attn` = n(L − n/2)/8192² (SMetric p9 and p12);
  - `new_prefill_k × active_requests_s` (Jain24 p4 and p7–8; Preble p6);
  - a session-history EWMA of output length, as an ablation only.

  Remove z-scores and x_i − min_S from the v2 list.
- Keep the exclusions. Setup confirmed that `expected_output_tokens` equals the true OSL
  (`facts/setup.json`).

**Model form (CONTRACT `learned-choice`)**

- Change the context to u_i = θ·x_i + Σ_k z_{S,k}(p_k·x_i), with `context: {sources: [...], p: [[d], ...]}`
  (LR-05).
- Fix the scale by pinning one anchor coefficient (LR-05).
- Make the RNG one draw per decision, or keyed by request id (LR-03).
- Default to τ = 0, and document that the default policy's temperature is range-normalized (LR-15).
- Add an optional `clamp` parameter (per-feature training quantiles [q0.001, q0.999]). Report the
  fraction of clamped decisions at N = 16/32 and at 128K. Corrected citation: Lodestar §4.3.2 (pdf
  p.8) says predictions outside the training support "are extrapolations with no accuracy
  guarantee". Its Algorithm (pdf p.17) does not clamp: it returns OOD and falls back ("per-feature
  extrapolation guard"). Clamping is our variant; consider an OOD fallback to the default cost as
  the alternative.
- In PLAN, the M2 row loses "x_i − min_S, rank". M3 becomes nest-level features. Add an optional M2b
  (DeepHalo layer) (LR-06, LR-15).

**Training budget (PLAN Training, CONTRACT `lr-train`)**

- Run at least 3 restarts per rung and per tuned baseline (LR-10).
- Use CRN per generation with K ≥ 3 tie seeds, and noise-handle by re-evaluating a fraction r_λ on
  fresh seeds. The UH-CMA-ES default is r_λ = max(0.1, 2/λ); 0.3 is the paper's example (LR-02, LR-03).
- Set `CMA_stds` from feature spreads (LR-05).
- Warm-start M2 from M1 (LR-15).
- Multi-start from θ_default and θ_LMetric, plus a cache-heavy start for AgentX (LR-07).
- Sign-constrain the load coefficients (LR-13).
- Draw train-time speedups from U[0.9, 1.1], fixed per cell (LR-14).
- Size the budget from the measured tie-seed SE and an MDE of 1–2% (LR-02). Setup measured about
  1.5 s per 2,000-request replay, flat in N (`facts/STATE.md`), so K seeds × 3 restarts is
  affordable. Use CPU cluster only if the projection exceeds 8 h.
- Select on validation over the CMA means and the top-k samples (LR-10).

**Splits (PLAN Splits, `cells/`)**

- The unit of analysis is the independent trace segment. Set the number of validation and test
  segments per family with a bootstrap power check at the pilot (LR-02, LR-11).
- **N = 6 is in both validation and test** (CONTRACT "Worker counts"). Either drop 6 from validation
  and use a validation-only N = 12 probe, or label the N = 6 test result "selection-exposed" (LR-10,
  LR-11; Cawley §6).
- Train cells span L1–L3, low- and high-reuse regimes, and cache pressure below 0.5, near 1 and
  above 1 (LR-12; Decima Table 2; Lodestar §5.3).
- Test N uses knee-matched loads in addition to per-worker-matched loads (LR-12).
- Prepend a warm-up segment to every open-loop window (LR-01).

**Metrics (PLAN Objective, CONTRACT `lr-eval`)**

- `lr-eval` adds `good_frac_window`, `goodput_rps_window`, `makespan_ms`, `arrival_span_ms`,
  `warmup_ms`, the guard metrics (LR-13), and good_frac by ISL bucket (LR-08).
- The objective is the mean of clipped log-ratios, with default paired by tie seed (LR-01).
- TTFT becomes length-scaled T(L) per family, with an SLO-scale re-score at {0.5, …, 3}× (LR-08).
- "Differs beyond noise" becomes a paired per-segment test per load mode (LR-02, LR-09).
- Pre-register `facts/HEADLINE_TEST.json` before training (LR-11).

**Baselines (PLAN Baselines)**

- Add `lmetric-faithful` and an `smetric`-style baseline (LR-04). Label the latter "no global tier",
  since SMetric's published advantage requires one.
- Derive the defaults-arm constants for dualmap and llm-d from AIS and the SLO (LR-04).
- Add decision-parity tests against upstream implementations, or label ports "unverified" (LR-04).
- Report against the per-cell best tuned baseline (LR-11).
- Promote ablation (a), the joint `router_queue_threshold` tuning, to a reported arm on AgentX and
  128K cells. Holding at the router and pausing programs were levers in both papers:
  - Jain24's RL router has an explicit no-op "hold" action (pdf p.8); its better variants waited
    2–4 s at the router (pdf p.9), but the paper does not isolate the effect by ablation;
  - ThunderAgent §4 uses state-aware pausing.

**Audits**

- Stage 1 gains an "information parity" lens (LR-14).
- Stage 4 gains guard-metric, concentration and segregation flags, plus an engine-state shift check
  (LR-13, LR-14).
- Stage 5 adds staleness delays of 10, 50 and 200 ms, τ > 0 and `hash_home`-restricted variants, and
  the Kendall τ of the ranking (LR-14).

## Risks the literature flags

1. **The simulator's ranking may not transfer.** Calibrate-then-Route's re-fitted simulator matched
   goodput (MAE 0.068) but reproduced the measured winner on no workload (Calibrate-then-Route §VI).
   That simulator was a hand-built serial-service model, so the size of this risk for AIS-timed
   replay is a [hypothesis].
   Emulation-trained ABR controllers lost to simple control in a real RCT (Puffer §5.2). The headline
   must say "in AIS-timed simulation", and a small live A/B is the required next step.
2. **Replay's router state has zero staleness.** KV events are applied synchronously (LR-14). A
   τ = 0 argmax tuned on fresh state can herd live (Mitzenmacher 2000; Tahir et al. 2022), and router
   radix trees go stale under eviction (ThunderAgent §3.2).
3. **Effect sizes are close to frozen noise.** Corrected. Published margins of searched or
   engineered policies over the best simple baseline are small:
   - +0.5% and +2.8% mean JCT for LLM-searched retention and scheduling policies (AgentServeSim
     Table 2), which are not routers;
   - +1.7 goodput points for a hand-weighted cost router (Calibrate-then-Route Table II).

   Jain24 §6.1 reports a larger simulated RL-router gain: 11.4% over RR, while the classical
   heuristics gained only 0.5–2.6% over RR. Tie seeds alone moved default goodput by about 1.8% at
   N = 8 (setup). Without seed-paired CRN and segment-level statistics, the campaign can "find"
   gains that are noise.
4. **Objective gaming.** Concentration, sacrificing doomed requests, and horizon or drain effects
   (GORGO App. D, whose ES optimized p95 TTFT; Wang et al. 2024 §4.2; Decima §5.3). LR-01 and LR-13
   address these.
5. **Herding under scarcity.** Greedy argmin lost to round-robin under scarcity in Calibrate-then-Route
   (0.292 vs 0.680), and setup's overloaded N=4 smoke cell already shows default 0.281 vs RR 0.637. A
   heavy L3 or N = 2 cell may reward randomness over intelligence.
6. **Weak or mis-ported baselines inflate claims.** The `lmetric` port omits queued prefill, and the
   dualmap and llm-d constants are foreign (LR-04). Well-tuned standard baselines often beat new
   methods (Melis et al. 2018; Henderson et al. 2018).
7. **Selection bias.** Best-ever CMA samples, ladder gates, and the N = 6 overlap between validation
   and test all bias reported performance optimistically (Cawley & Talbot 2010).
8. **Invalid trace replay for sessions.** Open-loop timestamps for agentic sessions violate trace
   exogeneity (CausalSim §3.1; AgentServeSim).
9. **The regime moves with N.** Ceilings at N = 16/32 from pooling, and ties or inversions at small N
   (van der Boor et al.; Calibrate-then-Route §V-D/E), can be misread as generalization success or
   failure (LR-12).
10. **Narrow or mismatched training distributions.** A policy trained on one load or sharing regime
    is worse than a tuned heuristic after a shift (Decima Table 2; Lodestar §5.3, where a model
    frozen at 5% sharing failed at 50%). Wide ranges dilute gains and hide per-cell regressions
    (Genet §2).
11. **Simulator exploitation.** Black-box search exploits simulator artifacts (Lehman et al.). AIS
    and engine-model errors grow near capacity (Vidur §7.2). Vidur's worst case was 12.65% for
    LLaMA2-7B at 95% capacity, while larger models stayed accurate (App. A.1). That is the regime
    the near-knee cells target. Long-context accuracy is unverified [hypothesis].
12. **Coupling between routing and eviction.** The learned router will exploit whatever eviction
    policy the mocker implements (Wu et al. ICLR'26 §3–4) [hypothesis for this campaign].
13. **The learned router may lose on homogeneous traffic.** There, a queue count is close to a
    sufficient statistic (Calibrate-then-Route §V-C: 0.760 vs 0.855). REPORT should name the regimes
    where heuristics win.
14. **Replay-specific accounting in features.** `overlap_frac` uses `accounting_cache_estimate`
    because replay leaves device-tier counts at 0 (CONTRACT v1 #1). The live value may differ, which
    the information-parity lens must check (LR-14).
15. **Published router winners can depend on a global KV tier this replay lacks.** Added at
    verification. SMetric's advantage over the best baseline holds "except for setups without any
    global KV$ store" (pdf p.10). On an agentic workload without a global tier, LMetric fell 36%
    below a tuned linear score (pdf p.6). Do not import a baseline ranking from those papers into
    AgentX cells; measure it (LR-04, LR-07).

## Bibliography

Local paths are relative to `/tmp/learned-routing-lit/pdfs/`. "web" means read online and not
downloaded. "abstract" means only the abstract was read. Verification note: `llm-routing/yuan2026-dualmap.pdf`
contains form-feed bytes in its figure fonts, so whole-file `pdftotext` page counts are wrong (78
instead of 23). Its pages were checked one at a time with `pdftotext -f N -l N`.

**Discrete choice and context effects**

- Tomlinson & Benson 2021. Learning Interpretable Feature Context Effects in Discrete Choice. KDD.
  https://arxiv.org/abs/2009.03417 — `dcm-context/tomlinson2021-lcl-dlcl.pdf`
- Seshadri, Peysakhovich & Ugander 2019. Discovering Context Effects from Raw Choice Data. ICML.
  https://arxiv.org/abs/1902.03266 — `dcm-context/seshadri2019-cdm.pdf`
- Rosenfeld, Oshiba & Singer 2020. Predicting Choice with Set-Dependent Aggregation. ICML.
  https://arxiv.org/abs/1906.06365 — `dcm-context/rosenfeld2020-set-dependent-aggregation.pdf`
- Pfannschmidt, Gupta, Haddenhorst & Hüllermeier 2022. Learning Context-Dependent Choice Functions.
  IJAR. https://arxiv.org/abs/1901.10860 — `dcm-context/pfannschmidt2019-fate-feta.pdf`
- Aouad & Désir 2023. Representing Random Utility Choice Models with Neural Networks. Management
  Science. https://arxiv.org/abs/2207.12877 — `dcm-context/aouad2023-rumnet.pdf`
- Yousefi Maragheh, Chronopoulou & Davis 2018. A Customer Choice Model with HALO Effect. arXiv.
  https://arxiv.org/abs/1805.01603 — `dcm-context/maragheh2018-halo-mnl.pdf`
- Ko & Li 2024. Modeling Choice via Self-Attention. arXiv. https://arxiv.org/abs/2311.07607 —
  `dcm-context/ko2024-choice-self-attention.pdf`
- Zhang, Wang, Gao & Li 2026. DeepHalo: A Neural Choice Model with Controllable Context Effects.
  arXiv. https://arxiv.org/abs/2601.04616 — `dcm-context/zhang2026-deephalo.pdf`
- Train 2009. Discrete Choice Methods with Simulation, Ch. 4 (GEV). Cambridge UP.
  https://eml.berkeley.edu/books/choice2nd/Ch04_p76-96.pdf — `dcm-context/train2009-gev-ch4.clean.pdf`.
  The `.pdf` without `.clean` is MacBinary-wrapped.
- Seshadri & Ugander 2020. Fundamental Limits of Testing the Independence of Irrelevant Alternatives
  in Discrete Choice. EC 2019 (extended). https://arxiv.org/abs/2001.07042 —
  `dcm-context/seshadri2020-iia-testing-limits.pdf` (abstract). Removed from LR-11 at verification
  because the campaign never tests IIA.
- Webb, Glimcher & Louie 2021. The Normalization of Consumer Valuations. Management Science.
  https://www.neuroeconomicslab.org/s/Normalization-of-Consumer-Valuations.pdf —
  `dcm-context/webb2021-divisive-normalization.pdf`
- Jain, Szot & Lim 2020. Generalization to New Actions in Reinforcement Learning. ICML.
  https://arxiv.org/abs/2011.01928 — `dcm-context/jain2020-generalization-new-actions.pdf`
- Jain, Kosaka, Kim & Lim 2022. Know Your Action Set: Learning Action Relations for RL (AGILE). ICLR.
  https://openreview.net/forum?id=MljXVdp4A3N — `dcm-context/jain2022-agile-action-relations.pdf`.
  This file is the slide deck, not the paper.
- Mitzenmacher 2000. How Useful Is Old Information? IEEE TPDS.
  https://www.eecs.harvard.edu/~michaelm/abstracts/tpds2000.html —
  `dcm-context/mitzenmacher2000-old-information.pdf`

**LLM request routing**

- Tumkur et al. 2026. Calibrate, Then Route: A Measured Study of Learned Request Routing for
  Disaggregated LLM Serving. arXiv. https://arxiv.org/abs/2609.16206 —
  `llm-routing/tumkur2026-calibrate-then-route.pdf` (also under `eval-methodology/`)
- Lim et al. 2026. Lodestar: An Online-Learning LLM Inference Router. arXiv.
  https://arxiv.org/abs/2606.00946 — `llm-routing/lim2026-lodestar.pdf`
- Jain et al. 2024. Intelligent Router for LLM Workloads. arXiv. https://arxiv.org/abs/2408.13510 —
  `llm-routing/jain2024-intelligent-router.pdf`
- Zhang et al. 2026. Simple is Better: Multiplication May Be All You Need for LLM Request Scheduling
  (LMetric). OSDI. https://arxiv.org/abs/2603.15202 — `llm-routing/zhang2026-lmetric.pdf`
- Wang et al. 2026. SMetric: Rethink LLM Scheduling for Serving Agents with Balanced Session-centric
  Scheduling. arXiv. https://arxiv.org/abs/2607.08565 — `llm-routing/wang2026-smetric.pdf`
- Ricci Toniolo et al. 2026. GORGO: Online Tuning for Cross-Region Network-Aware LLM Serving. arXiv.
  https://arxiv.org/abs/2602.11688 — `llm-routing/toniolo2026-gorgo.pdf`
- Wu, Silwal & Zhang 2026. Randomization Boosts KV Caching, Learning Balances Query Load. ICLR.
  https://arxiv.org/abs/2601.18999 — `llm-routing/wu2026-randomized-eviction-learned-routing.pdf`
- Cheng 2026. CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM Serving. arXiv.
  https://arxiv.org/abs/2608.19677 — `llm-routing/cheng2026-cacheroute.pdf`
- Yuan et al. 2026. DualMap: Enabling Both Cache Affinity and Load Balancing for Distributed LLM
  Serving. ICLR. https://arxiv.org/abs/2602.06502 — `llm-routing/yuan2026-dualmap.pdf`
- Srivatsa et al. 2025. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. ICLR.
  https://arxiv.org/abs/2407.00023 — `llm-routing/srivatsa2024-preble.pdf`
- Kang et al. 2026. ThunderAgent: A Simple, Fast and Program-Aware Agentic Inference System. ICML.
  https://arxiv.org/abs/2602.13692 — `llm-routing/kang2026-thunderagent.pdf`
- Rajib, Zheng & Lou 2026. AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent
  Programs. arXiv. https://arxiv.org/abs/2606.09613 — `llm-routing/rajib2026-agentservesim.pdf`
- Luo et al. 2025. Autellix: An Efficient Serving Engine for LLM Agents as General Programs.
  https://arxiv.org/abs/2502.13965 (web, §4.3)
- Nixon et al. 2026. A Year in LLM Serving: Workload Evolution, Caching and Load-Balancing.
  https://arxiv.org/abs/2608.13573 (web, §7.2)
- llm-d project 2025. KV-cache wins you can see. https://llm-d.ai/blog/kvcache-wins-you-can-see (web)
- Jha et al. 2024. Learned Best-Effort LLM Serving. https://arxiv.org/abs/2401.07886 (abstract)
- Da & Kalyvianaki 2026. RouteBalance. https://arxiv.org/abs/2606.17949 (abstract)
- Cao et al. 2025. Locality-aware Fair Scheduling in LLM Serving. https://arxiv.org/abs/2501.14312
  (abstract)
- Li et al. 2025. Continuum: Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live.
  https://arxiv.org/abs/2511.02230 (abstract)
- Ramjet (Helix), not a paper. https://github.com/helixml/ramjet

**Learned systems policies, black-box search, sim-to-real**

- Mao et al. 2019. Learning Scheduling Algorithms for Data Processing Clusters (Decima). SIGCOMM.
  https://arxiv.org/abs/1810.01963 — `learned-systems-policies/mao2019-decima.pdf`
- Mao et al. 2019. Variance Reduction for RL in Input-Driven Environments. ICLR.
  https://arxiv.org/abs/1807.02264 — `learned-systems-policies/mao2019-input-driven-variance.pdf`
- Mao et al. 2019. Park: An Open Platform for Learning-Augmented Computer Systems. NeurIPS.
  https://proceedings.neurips.cc/paper/2019/hash/f69e505b08403ad2298b9f262659929a-Abstract.html —
  `learned-systems-policies/mao2019-park.pdf`
- Mao, Netravali & Alizadeh 2017. Neural Adaptive Video Streaming with Pensieve. SIGCOMM.
  https://web.mit.edu/pensieve/ — `learned-systems-policies/mao2017-pensieve.pdf`
- Yan et al. 2020. Learning in situ: a randomized experiment in video streaming (Puffer). NSDI.
  https://arxiv.org/abs/1906.01113 — `learned-systems-policies/yan2020-puffer.pdf` (same paper as
  `eval-methodology/yan2020-puffer-learning-in-situ.pdf`)
- Xia, Zhou, Yan & Jiang 2022. Genet: Automatic Curriculum Generation for Learning Adaptation in
  Networking. SIGCOMM. https://arxiv.org/abs/2202.05940 — `learned-systems-policies/xia2022-genet.pdf`
- Alomar et al. 2023. CausalSim: A Causal Framework for Unbiased Trace-Driven Simulation. NSDI.
  https://arxiv.org/abs/2201.01811 — `learned-systems-policies/alomar2023-causalsim.pdf`
- Peng et al. 2018. Sim-to-Real Transfer of Robotic Control with Dynamics Randomization. ICRA.
  https://arxiv.org/abs/1710.06537 — `learned-systems-policies/peng2018-dynamics-randomization.pdf`
- Mania, Guy & Recht 2018. Simple random search provides a competitive approach to RL (ARS). NeurIPS.
  https://arxiv.org/abs/1803.07055 — `learned-systems-policies/mania2018-ars.pdf`
- Salimans et al. 2017. Evolution Strategies as a Scalable Alternative to RL. arXiv.
  https://arxiv.org/abs/1703.03864 — `learned-systems-policies/salimans2017-es.pdf`. Its §2.1 is no
  longer cited for CRN (LR-03); §2.2 is still cited in LR-05.
- Hansen 2016/2023. The CMA Evolution Strategy: A Tutorial. arXiv. https://arxiv.org/abs/1604.00772 —
  `learned-systems-policies/hansen2016-cmaes-tutorial.pdf`
- Hansen, Niederberger, Guzzella & Koumoutsakos 2009. A Method for Handling Uncertainty in
  Evolutionary Optimization (UH-CMA-ES). IEEE TEC.
  http://www.cmap.polytechnique.fr/~nikolaus.hansen/TEC2009.pdf —
  `learned-systems-policies/hansen2009-uh-cmaes.pdf`
- Ng & Jordan 2000. PEGASUS: A policy search method for large MDPs and POMDPs. UAI.
  https://arxiv.org/abs/1301.3878 — `learned-systems-policies/ng2000-pegasus.pdf`
- Tahir, Cui & Koeppl 2022. Learning Mean-Field Control for Delayed Information Load Balancing. ICPP.
  https://arxiv.org/abs/2208.04777 — `learned-systems-policies/tahir2022-meanfield-lb.pdf`
- van der Boor, Borst, van Leeuwaarden & Mukherjee 2022. Scalable load balancing in networked systems.
  Statistical Science. https://arxiv.org/abs/1806.05444 —
  `learned-systems-policies/vanderboor2018-scalable-lb.pdf`
- Zaheer et al. 2017. Deep Sets. NeurIPS. https://arxiv.org/abs/1703.06114 —
  `learned-systems-policies/zaheer2017-deep-sets.pdf`
- Lehman, Clune, Misevic et al. 2020. The Surprising Creativity of Digital Evolution. Artificial Life.
  https://arxiv.org/abs/1803.03453 — `learned-systems-policies/lehman2018-digital-evolution-creativity.pdf`
- Heidrich-Meisner & Igel 2009. Racing for policy selection in CMA-ES (ICML). The download failed:
  `learned-systems-policies/heidrichmeisner2009-races-cmaes.FAILED-DOWNLOAD.html` is a bot-check
  page. Not read.

**Evaluation methodology**

- Zhong et al. 2024. DistServe. OSDI. https://arxiv.org/abs/2401.09670 —
  `eval-methodology/zhong2024-distserve.pdf`
- Schroeder, Wierman & Harchol-Balter 2006. Open Versus Closed: A Cautionary Tale. NSDI.
  https://www.cs.utoronto.ca/~bianca/papers/nsdi_camera.pdf —
  `eval-methodology/schroeder2006-open-vs-closed.pdf`
- Wang et al. 2024/2025. Revisiting SLOs and System Level Metrics in LLM Serving. arXiv.
  https://arxiv.org/abs/2410.14257 — `eval-methodology/wang2024-revisiting-slo-goodput.pdf`
- Agrawal et al. 2024. Vidur: A Large-Scale Simulation Framework for LLM Inference. MLSys.
  https://arxiv.org/abs/2405.05465 — `eval-methodology/agrawal2024-vidur.pdf`
- Henderson et al. 2018. Deep Reinforcement Learning that Matters. AAAI.
  https://arxiv.org/abs/1709.06560 — `eval-methodology/henderson2018-deep-rl-matters.pdf`
- Agarwal et al. 2021. Deep RL at the Edge of the Statistical Precipice. NeurIPS.
  https://arxiv.org/abs/2108.13264 — `eval-methodology/agarwal2021-statistical-precipice.pdf`
- Demšar 2006. Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7.
  https://www.jmlr.org/papers/v7/demsar06a.html — `eval-methodology/demsar2006-statistical-comparisons.pdf`
- Cawley & Talbot 2010. On Over-fitting in Model Selection and Subsequent Selection Bias in
  Performance Evaluation. JMLR 11. https://www.jmlr.org/papers/v11/cawley10a.html —
  `eval-methodology/cawley2010-overfitting-model-selection.pdf`
- Dodge et al. 2019. Show Your Work: Improved Reporting of Experimental Results. EMNLP.
  https://arxiv.org/abs/1909.03004 — `eval-methodology/dodge2019-show-your-work.pdf`
- Bouthillier et al. 2021. Accounting for Variance in Machine Learning Benchmarks. MLSys.
  https://arxiv.org/abs/2103.03098 — `eval-methodology/bouthillier2021-variance-ml-benchmarks.pdf`
- Melis, Dyer & Blunsom 2018. On the State of the Art of Evaluation in Neural Language Models. ICLR.
  https://arxiv.org/abs/1707.05589 (abstract)
- Agrawal et al. 2024. Etalon: Holistic Performance Evaluation Framework for LLM Inference Systems.
  https://arxiv.org/abs/2407.07000 (abstract)
- Krishnamachari 2026. How to Do Statistical Evaluations in ECE/CS Papers.
  https://arxiv.org/abs/2605.00428 (web, selected sections)
- Abdelfattah et al. 2026. Load Testing for Machine Learning Model Serving Systems at Scale.
  https://arxiv.org/abs/2606.22013 (abstract)
- Berger 1982. Multiparameter hypothesis testing and acceptance sampling. Technometrics. The
  intersection-union result is cited from memory and was not read.
