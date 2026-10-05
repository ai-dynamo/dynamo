# Learned routing in AISim: campaign plan (v2, 2026-10-02)

## Story

For one model on one deployment, use AISim offline replay to **learn a routing function that beats
every heuristic router we have** on held-out workloads and held-out worker counts. AISim is the
development environment only. **The learned router doesn't depend on AIS at runtime.** It reads only
signals the live router already has (prefix overlap, router-tracked load, worker KV capacity, session
metadata) and evaluates a small classical choice model in O(N·d) per request. It ships as a
builtin-catalog policy.

**Target:** Qwen3-32B, vLLM 0.24.0, H100 SXM, TP2, aggregated, with AIS timing (the AIS support matrix
passes this). Disaggregated serving and other models come later.

## Starting point (verified)

- **Your stack:** `rupei/router-policy-ports` (#15450) and `rupei/router-policy-aic-ttft` (#15453).
  They run builtin-catalog worker-selection policies in offline replay; on main, replay rejects any
  custom policy.
- **Worktree:** a new one off `origin/rupei/router-policy-aic-ttft`. The main checkout has uncommitted
  kv-router work and is not touched.
- **Decision hook:** a new builtin-catalog policy, `learned-choice`, with its coefficients passed as YAML
  `parameters`. Training writes a policy YAML and calls replay, with no Rust rebuild per iteration. The
  same artifact runs in the live router.
- **Gaps the campaign closes:**
  - Replay times by AIS only when built with `ais-forward-pass`.
  - Python replay has no seed control (the default picker's tie-break is unseeded).
  - Replay sets `affinity_target: None`, so sticky-session routing has to be emulated by a policy.
  - There are no replay knobs for stretching input or output lengths or multiplying prefixes, so traces
    are transformed before replay.
  - Per-request goodput has to be recomputed from `per_request`.

## Baselines

Each baseline runs at its published defaults **and** tuned on the train split, with the same CMA-ES
budget the learned model gets.

| Baseline | Notes |
|---|---|
| `round_robin` | floor |
| `dynamo-default-cost-fn` | defaults; tuned credit, load scale, decode request weight, temperature, credit decay |
| `dynamo-two-tier-cost-fn` | sgl-router `cache_aware_zmq` port |
| `lmetric`, `ramjet`, `dualmap`, `chwbl` | ports from your stack |
| `llm-d-precise-prefix` | port from your stack |
| `llm-d-optimized-baseline` | `ttft_source: throughput`. The `modeled` (AIS) variant is reported separately as an AIS-coupled reference, not a like-for-like rival. |
| `sticky-session-hard` / `sticky-session-soft` | new campaign policies mirroring live `SessionAffinityMode::{Hard,Soft}`. The first turn goes through the default cost; later turns pin (hard) or follow the last turn (soft). |
| `thunderagent` (selection half) | replay can't run its classifier, so it's reported as such or skipped |

## Workloads

| Family | Source | Load modes |
|---|---|---|
| Mooncake | `<traces>/mooncake_trace.jsonl` (23.6k rows); time windows act as distinct workloads; Mooncake FAST25 conversation and synthetic traces downloaded only if missing | open-loop timestamps (`arrival_speedup_ratio`); closed-loop `replay_concurrency` |
| Toolagent (flat) | `<traces>/toolagent_trace.jsonl` | same as Mooncake |
| AgentX (Weka) | HF `semianalysisai/cc-traces-weka-062126@23f152f6` (1.85 GB, 393 plays; SHA-256 `29b6a19e…`). **Simulate 128K context** (`max_model_len` 131072, YaRN): the 82 intact plays that fit (3,132 requests, 39 subagent groups), labeled as Opus-recorded agent traffic projected onto Qwen3-32B. | `trace_timestamps` with speedup; closed-loop `agentic_lanes` |
| Synthetic sessions | replay `synthetic-session` source | Poisson `request_rate` |

**Transforms** (deterministic, seeded, with a SHA-256 manifest): ISL stretch (unique part and prefix
part separately), OSL multiplier, prefix-root multiplier, think-time scale, and time-window slicing.

**Loads** are fixed loadgen parameters, either an open-loop rate or speedup, or a closed-loop
concurrency or lane count. They are scaled with N so per-worker load stays matched. Calibration picks,
per family, three levels around the regime where routing matters (light, near-knee, heavy) using the
default router, then freezes them as plain loadgen values.

**Splits.** The test manifest is frozen and SHA-256 recorded before any training, and evaluated once at
the end. Each test cell is held out along one axis:
1. a held-out Mooncake time window,
2. transform values outside the training range,
3. a held-out family or plays (e.g. disjoint AgentX plays),
4. **held-out worker counts**: train N ∈ {4, 8}, interpolate to 6, extrapolate to {2, 16, 32}.

A separate validation split handles model selection and the pilot gate.

## Objective

- **Goodput at the loadgen-defined load**: good requests per simulated second, from the report's
  `goodput_request_throughput_rps`, cross-checked against a per-request recomputation. "Good" means
  TTFT ≤ T and mean ITL ≤ I.
- **T and I:** one pair per family, set from the default router's seed runs at the near-knee load, then
  frozen.
- **Training objective:** mean over train cells of goodput / default-router goodput, so every cell
  weighs the same.
- **Secondary metrics:** TTFT and ITL percentiles, throughput, prefix-reuse ratio, agentic trajectory
  latency.

## Learned model ladder

Every rung uses one shared per-worker utility with softmax or argmax over the candidates, so it handles
any N. A rung counts only if it improves held-out results.

| Rung | Model |
|---|---|
| M0 | default cost, tuned |
| M1 | conditional logit over router-observable features. Per worker: new prefill tokens, overlap fraction, active prefill tokens, decode blocks / KV capacity, active requests, session-affinity flag. Per request: ISL, as interactions. Starts from the θ that reproduces the default cost, with a replay parity test. |
| M2 | **linear context logit**: `u_i = θᵀx_i + x_iᵀ A x̄_S`, with low-rank A and set-relative features (x_i − min_S, rank). This captures interactions between workers at any N. |
| M3 | nested logit over node/DP groups, only if DP > 1 matters |
| Ablations | (a) jointly tune `router_queue_threshold`, the dispatch-or-hold boundary, for the learned model and the best baseline; (b) add simulator-only signals (engine waiting counts, KV usage %) to measure what the router can't see. Neither ships. |

**Excluded features:** anything from AIS (including `modeled_prefill_backlog_ms`), `expected_output_tokens`
(replay may fill it with the true OSL), and the true output length.

## Training

- **Primary method: CMA-ES** over normalized train goodput, evaluated by parallel replays with common
  random numbers.
- **Every long job is a resumable, checkpointed CLI** with `--max-wall-seconds`, so any agent can
  continue it after an interruption.
- **Compute:** local 24 cores, spilling to CPU-cluster allocations if the measured cost projects past
  the local budget.

## Campaign control: pilot, auto-escalation, audits

| Step | Content | Audit at checkpoint |
|---|---|---|
| 0 Setup | worktree, venv, `maturin develop --release` with `ais-forward-pass`, uv installs (cma, scipy, lightgbm, matplotlib); smoke-replay a catalog policy under AIS timing; measure wall time per replay vs N; check `session_context`, seeding and long context; pick the replay entry point (Python API or `offline_replay_bench`) on determinism, AIS, formats and per-request output | **static**: adversarial check that AIS timing and the selected policy are actually in effect |
| 1 Build | parallel: (a) the `learned-choice` and `sticky-session` policies plus a parity test; (b) trace acquisition, transforms and the split tool; (c) the evaluation harness, CMA-ES driver, baseline tuner and report utilities. Then integrate. | **static**, three lenses: feature leakage, goodput math and normalization, harness and determinism correctness. A fixer then addresses confirmed findings. |
| 2 Calibrate | load levels per family × N, SLA thresholds, frozen manifests | **dynamic**: re-run sampled cells twice, recompute goodput independently, confirm loads sit where round-robin and default differ beyond noise |
| 3 Pilot | tune M0; train M1 on a train subset; score on validation | **gate** (below) |
| 4 Full | tune every baseline (equal budget); train M1 → M2 (→ M3); run the ablations | **dynamic**, midway: overfitting (train vs validation), degenerate policies (starving or pinning workers), sim-artifact exploitation |
| 5 Test | one pass on the frozen test set; robustness under timing perturbation (`speedup_ratio` / `decode_speedup_ratio` at 0.8 and 1.2); bootstrap CIs | **static**: completeness critic, plus three adversarial refuters on the headline claim |
| 6 Report | `REPORT.md`: tables, worker-count extrapolation plot, coefficient interpretation, porting notes, fidelity caveat (results are simulated) | — |

**Auto-escalation gate** after the pilot. The run escalates to the full campaign only if all of these
hold:
1. Fewer than 2% of replays fail.
2. Measured cost projects the full run within budget, after shrinking the cell grid if needed.
3. Some policy differs from the default beyond seed noise on validation.

If (3) fails, recalibrate the loads once and retry. Otherwise stop with a diagnostic report.

**Rules:**
- Agents delete nothing; they list scratch paths and sizes for later cleanup.
- Running agents are never messaged; late facts go into the next stage's prompt.
- The top-level session keeps a worklog in the worklog.

**Paths:** campaign root `<campaign-root>/`, worktree
`<worktree>` on branch `rupei/learned-routing`; the final report
is copied here.

## Decisions (operator, 2026-10-02)

- **Model:** Qwen3-32B, vLLM 0.24.0, H100 SXM, TP2, aggregated.
- **Objective:** goodput at loadgen-defined loads, either open-loop rate or closed-loop concurrency or
  lanes. No capacity-at-SLA search.
- **No AIS coupling at runtime:** AIS-derived signals are excluded from the learned model's features.
- **AgentX:** simulate 128K context. If AIS or replay can't handle 128K, token-scale the 82 plays to fit
  32K.
- **Hard stop:** none; run to completion, or stop at the gate.
- **Pilot gate failure:** after one failed recalibration, stop with a diagnostic report.
- **Dynamo/AISim bugs or missing plumbing:** patch on the local campaign branch, list each as an upstream
  follow-up, push nothing.
- **CPU-cluster spillover:** allowed.
- **Amendment A2 (2026-10-02 ~14:30)** — see CONTRACT.md for the full text:
  - The headline is the learned model against the best baseline, on the same footing: one config per
    policy, tuned on pooled train data, selected on val.
  - No TTFT SLO. A request is good iff mean ITL ≤ I AND E2E ≤ S × its uncontended no-reuse latency
    (an E2E slowdown). TTFT is reported only, and an SLO-scale sweep is reported.
  - Goodput is windowed (LR-01).
  - Baselines run as implemented, with only a brief sanity check; no paper-faithful variants.
