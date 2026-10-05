# Live audit, lens "fidelity": live-vs-sim fidelity and scoring

- **Auditor:** independent adversarial auditor, checkpoint live, 2026-10-05.
- **Verdict: PASS.** 0 blockers, 0 majors, 4 minors.
- **Bottom line.** The live finalist runs replayed what the simulator replayed, under the scorer the
  simulator used. That holds for the policies, cells, worker count, load and A2 scorer.
  - **Same requests and policies.** Every valid run used the frozen policy YAML, byte-identical to the
    test pass's k0 YAML, and the frozen cell, SLA, window and load, at N = 4 × TP2 on H100 80GB HBM3.
    Arrivals, ISL, forced OSL, prompts and session headers match the replay request for request.
  - **Scores reproduce.** My scorer shares no code with `score_live` or `learned_routing`. It
    re-derives every live `goodput_rps_window` and every A19 value from the raw AIPerf exports,
    exactly.
  - **Analysis and noise reproduce.** σ_live is computed by the registered formula, and the
    registered analysis (deltas, τ_b, sign test) reproduces exactly.
  - **No blocker or major.** I found nothing that invalidates or biases the registered live
    conclusions.

  The 4 minors concern omitted or under-stated context:
  - **F1:** a registered descriptive was dropped although it was computable.
  - **F2:** the reasons for the absolute live/sim departures are not stated.
  - **F3:** how firm the "beyond noise" wording is.
  - **F4:** one policy input with no runtime evidence.
- **Scope.** This audit is read-only: no GPU or CPU job, no replay, no write to the campaign cache,
  facts or report. My scripts and outputs are in `runs/audits/live-fidelity/` (about 100 KB).

## What I checked, with my own evidence

All paths below are relative to CR. All scripts are in `runs/audits/live-fidelity/`.

| Check | Result | Evidence |
|---|---|---|
| **Policy specs.** For each live slug, the plan spec equals `facts/finalists.json` (`learned_headline_pool.m1v2`, `.m1`; `tuned_baselines.ramjet`, `.m0`). The live `policy.yaml` sha256 equals the content-addressed YAML that the frozen test pass replayed at k0 (`runs/policies/replay/<sha>.yaml`). | 5/5 equal; YAML file hashes re-verified | inline check; the YAML names are in the `runs/live_sim/results.jsonl` k0 rows |
| **Serving code.** The live wheel's source commit has the same `lib/`, `components/` and `Cargo.lock` git trees as the tuning build's commit (`<commit-10>`, build 6955b0ee). | identical tree hashes | `git rev-parse <c>:lib` etc. |
| **Policy actually served, per pair.** For each pair: the serve-phase frontend command line points at the plan's slug and flags. Every `KvRouterConfig` kwarg in the dumped `frontend-config.json` equals the plan's kwargs (`plan.py` proved those equal to replay's defaults plus `router_config`, which is `{}` for all 5 policies). The check phase's `policy_evidence` sha256 equals the plan YAML. The cold reset is ok. There is no session-affinity TTL. | 44/44 pairs (42 used + 2 invalid first attempts), 0 issues | `deploy_check.py`, `deploy_check.out` |
| **Engine.** Every worker command line equals `engine_plan.json`'s vLLM args, which mirror `config/engine.json`. Every worker log shows `num_gpu_blocks_override=18864` and a KV cache of 301,824 tokens (replay: 18,863 usable blocks). The GPUs are H100 80GB HBM3 and the engine is vLLM v0.24.0. | 32/32 workers in 8 jobs | `deploy_check.out`; `nodes/*/gpus.csv`; worker logs |
| **Cells, worker count and load.** Each input manifest's `sla`, `load`, `num_workers`, `measure`, `split` and `family` equal the cell in frozen `cells/test.jsonl` (sha 7b998b8e…). Its `sla`, `measure`, `num_workers`, `num_requests`, replicate protocol and seed, and E0 method equal the sim counterpart record. The policy seed is 1. `verify_replay` is exact. | 44/44 inputs, 0 mismatches | inline check |
| **Sim counterparts.** All 108 `runs/live_sim/results.jsonl` records equal the frozen test-pass records in `runs/select_test/test/tuning.jsonl` on goodput, per-request sha, cache key, N, load and measure. | 108/108 | inline check |
| **Run order.** `random.Random(int(sha256('lr-live-order-v1\|cell')[:16],16)).sample(labels,7)` gives the registered order. Each job's `pairs.tsv` follows it; the sessions split keeps the order within each half. | 6/6 cells | `deploy_check.py`; `pairs.tsv` |
| **Request fidelity from raw exports.** I hashed every prompt token list as `<u4`, compared it with `requests.jsonl`, and checked `max_tokens = min_tokens = OSL` with `ignore_eos`. The session header is present exactly when the manifest requires it. | 42 used runs (40 + 2 reruns), 0 mismatches | `rawcheck.py`, `rawcheck_a2.py`, `*.out` |
| **Independent A2 scorer.** I wrote it from CONTRACT A2 and the cell's `measure` spec: the ITL and E2E-slowdown rule, the 1e-6 E2E tolerance, the arrival window, and the closed-loop identity warm-up with the full-occupancy end. First I validated it on sim: it reproduces all 36 sim k0 `goodput_rps_window` values exactly from their `per_request` rows. Then I applied it to the raw AIPerf `profile_export.jsonl`. | 36/36 sim and 44/44 live goodputs equal to the bit; 44/44 A19 `good_tokens_rps_window` equal; max abs diff 0 | `indep_score.py`, `check_sim.out`, `check_live.out`, `live_rescore.json` |
| **E0.** The AIS E0 precomputed in every input's `requests.jsonl` equals the simulator's cached table (`runs/cache/e0/01277503f73c1abe-ais-chunked-estimator-v2.json`) for every request. | 0 mismatches over 44 inputs | `check_live.out` |
| **Windows.** On open-loop single-turn cells the live in-window request sets equal the sim's (Mooncake L2 2,503; L3 2,494; FAST25 conv 1,271; synth 1,275). Closed loop: occupancy peaks at exactly C = 61 in all 7 runs, below the cap for 0.5–0.7% of the window. | as stated | `check_live.out`; `score.json` |
| **Live noise as registered.** r(c) = s(default_a, default_b) and σ_live = √(mean r²/2) = 0.009043, from 6 cells; conv and sessions are cross-node. D(c, p) is the mean of the same-job defaults: default_a only on conv and on the sessions P6 half. | reproduced exactly | `analysis.py`, `analysis.out` |
| **Registered analysis.** Per-cell τ_b 0.87/1.00/0.87/0.87/0.87/0.87 (mean 0.889). Pooled τ_b 1.00, permutation p 1/720. Points τ_b 0.738. M1-v2 − ramjet is positive in 5/5 informative cells, every one above 2√2·σ = 0.0256, mean +0.2194. Sensitivities: k0–2, lag 10 ms and lag 50 ms all give pooled 1.00. | all equal to `facts/live_results.json` and LIVE.md | `analysis.out`, `sens.out` |
| **Live-idle E0 secondary.** I refit the pooled E0 from the 4 idle runs (72 requests), then re-scored every run with my scorer. | fit equal (d0, d1, 9 points); 42/42 goodputs equal | `e0live.py`, `e0live.out` |
| **Completeness.** 48 pair directories: 42 used `ok`, 2 `payload_exit_1` with their reruns used, 4 idle E0 runs. There is no other hidden attempt. All 42 used runs have every request matched (missing 0), 4 workers seen, and 0 ISL, OSL, prompt, usage or header mismatches. | as stated | `status.json`; `score.json` `live` |

## Findings

### F1 (minor): the prefix-cache descriptive was computable for all 42 runs, not only 14

- **Claim in LIVE.md §15.6 and the analysis stage's issues.** The registered descriptive "prefix-cache
  hit rate" "cannot be computed comparably from vLLM counters for 28 of 42 runs", so the column
  shows only 14 runs (DEVIATIONS +1).
- **Evidence.** Every completed AIPerf record carries the server's per-request
  `usage.prompt_tokens_details.cached_tokens` (`usage_prompt_cache_read_tokens`). This is the same
  quantity as replay's per-request `reused_input_tokens`, and it does not depend on scheduler
  re-queries. Only the 18 AIPerf-error records lack it. Hit rate = Σ cached / Σ ISL:
  - On all 42 runs, live is within 0.015 of the sim counterpart's `prefix_reuse`, and within 0.010
    on 40 of them.
  - The largest gaps are on the at-ceiling sessions cell: round_robin 0.513 vs 0.498, and default_b
    0.706 vs 0.694.
  - FAST25 conv: default_a 0.059 vs 0.062, M1-v2 0.062 vs 0.068, ramjet 0.064 vs 0.066.
  - Source: `prefix.py`, `prefix_out.txt`.
- **Why it matters.**
  - The deviation was avoidable.
  - The full column is stronger fidelity evidence than the 14-run version: routing produced the
    replay's cache reuse on hardware for every policy and cell.
  - This rules out cache-hit fidelity as the cause of the absolute departures in F2.
- **Fix.** Report the usage-based hit rate for all 42 runs, keeping the vLLM-counter column as a
  footnote. Reword "cannot be computed comparably" in `facts/live_results.json`, `DEVIATIONS.md` and
  LIVE.md. No registered conclusion changes.

### F2 (minor): the absolute live/sim departures have an identifiable timing cause that is not stated

- **Claim in LIVE.md §15.2.** It reports, without a cause:
  - FAST25 conv at 0.72–0.81 of sim for default, ramjet, M0 and round_robin, but 0.91–0.98 for M1
    and M1-v2;
  - sessions round_robin at 2.18× sim.
- **Evidence on in-window completed requests** (`fails.py`, `fails.out`):
  - **FAST25 conv.** Prefix reuse matches sim (F1), but live TTFT p50 is much higher for the
    non-concentrating policies. Default: 2,827 vs 1,225 ms (2.3×). Ramjet: 2,420 vs 1,260 ms.
    Round_robin: 8,956 vs 4,669 ms. M1-v2: only 714 vs 630 ms. Almost every failure is an
    E2E-slowdown failure (default 847 live vs 694 sim).
    - The live engine's prefill under load is slower than AIS: the smoke measured 5–10% at idle,
      and this cell shows more under load. So the cell runs closer to prefill saturation on
      hardware.
    - Saturation amplifies the gap between policies that cut queueing and those that do not. This
      is the likely source of conv's larger live M1-v2 − ramjet (+0.663 vs +0.470). It is a
      hypothesis consistent with the TTFT data, not tested.
  - **Sessions.** Live mean ITL is about 14% below AIS: default p50 24.7 vs 28.8 ms, ramjet 20.0 vs
    23.7 ms. E2E/E0 p50 is 1.53 vs 1.79 for default. The cell's E2E-slowdown bound S = 2.118 sits
    near default's sim p90 (2.09).
    - So the AIS-E0 SLA is effectively looser on hardware. That compresses Δ vs default (ramjet
      +0.016 live vs +0.086 sim) and lifts round_robin (2,069 vs 949 good).
- **Why it matters.** LIVE.md correctly limits the claim to relative results. It also says live
  deltas above sim are not evidence of a larger hardware effect. But a reader could still take the
  conv "widening" as hardware favouring the learned arms. The mechanism is simpler: on these two
  cells the hardware runs at a different effective SLO scale or load point than AIS.
- **Fix.** Add one sentence to §15.2/§15.7, and to the paper's §11 when it is filled:
  - the departures come from engine timing (slower live prefill under load, faster live decode);
  - they do not come from routing or cache behaviour, which match (F1);
  - the two cells therefore run at a different effective load or SLO scale on hardware.

  No number changes.

### F3 (minor): the "beyond noise" wording leans on a 6-pair σ

- **The estimate is imprecise.** σ_live = 0.0090 is computed as registered, from 6 pairs (2
  cross-node). With 6 degrees of freedom, the one-sided 95% upper bound is about 1.92 × σ = 0.0173.
  That puts 2√2·σ at 0.049.
  - FAST25 synth M1-v2 − ramjet (+0.035) clears the registered threshold, 0.0256. It also clears a
    within-job-only σ (0.0105 → 0.030). It does not clear the upper bound.
  - The noise of the concentrating learned policies is unmeasured. LIVE.md says σ is assumed for
    the other policies.
- **The 2-sd bar is wrong for single-default cells.** "sd of a single live delta is about 0.011
  against default" assumes D is the mean of 2 defaults. On FAST25 conv and on both sessions halves,
  D is a single same-job default, so sd = √2·σ = 0.0128 and 2 sd = 0.026, not 0.022. The count
  "24 of 30" is unchanged: the same 6 deltas fall below either bar.
- **Fix.** In §15.7, soften "each time by more than the live noise" to "by more than the registered
  noise bar (2√2·σ_live, σ from 6 default repeats)". Note that FAST25 synth is the marginal case.
  Footnote the 0.026 bar for single-default cells.

### F4 (minor): learned-choice receiving the session context live has no runtime evidence

- **What is verified.** The header `X-Dynamo-Session-ID` is sent exactly when replay emits
  `SessionContext`: 0 header mismatches in 42 runs. Its code path to the policy exists:
  `lib/llm/src/protocols/agents.rs` → `agent_context` → `routing_host/kv.rs`
  `to_worker_selection_session_context`.
- **What is not verified.** No GPU or CPU check shows a learned-choice policy actually seeing a
  `SessionContext` (feature 6, `session_affinity`) on the live frontend. The check-phase smoke runs
  only single-turn requests with `expect_affinity: false`.
- **Impact is negligible here.** Only the multi-turn sessions cell can exercise this, and it is
  non-informative (at ceiling). On it, M1 and M1-v2 live reuse equals sim within 0.0013. The final
  audit's θ6 = 0 ablation on val moves M1-v2 by only +0.0005.
- **Fix.** Before any future live claim about session or affinity effects (A15), add a CPU
  mocker-frontend check that a learned-choice policy sees a non-empty session ID, for example a
  debug counter of affinity hits.

## Checked and not a finding

- **YaRN warning.** transformers warns that the explicit YaRN factor 4.0 differs from
  `max_position_embeddings / original` (1.25) and uses 4.0, as the recipe intends. It does not
  affect timing, and outputs are forced-length.
- **AIPerf empty-text errors.** 18 errored requests in 16 runs (OSL 1–3) count as not good. That is
  at most 2 per run, well below 0.001 of in-window goodput, and they are not policy-specific.
- **Client timing.** Open-loop lateness is p99 ≤ 2.6 ms (max 15.2 ms). The sessions think-time
  residual is p99 ≤ 6.6 ms. The HTTP latency base is about 2 ms of start lag. None of these matter
  at these SLAs.
- **The registered τ_b = 1.00 includes a sim near-tie.** M1-v2 vs M1 differ in sim by only 0.0009 in
  pooled mean (k0). The ranking holds under the k0–2, lag 10 ms and lag 50 ms sensitivities, and
  LIVE.md already lists M1-v2/M1 as an unresolved close pair.

## Integrity

- I did not modify `facts/finalists.json` (sha e0bae29d…, re-verified), `facts/test_results.json`,
  `cells/test.jsonl` (sha 7b998b8e…, re-verified), any live record, or any report file.
- No GPU or CPU cluster job was run. Nothing was deleted, committed or pushed.
- My only writes are this file, `runs/audits/live-fidelity/` (scripts and outputs), and one row
  each in `CLEANUP.md` and `facts/STATE.md`.
