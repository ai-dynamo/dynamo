<!-- # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Learned routing in AISim: campaign notes

This directory is the durable record of the learned-routing campaign on branch
`rupei/learned-routing`. Start here. To rebuild up to the current state, follow
[REPRODUCE.md](REPRODUCE.md). Every small file the reproduction needs is mirrored under
[`campaign/`](campaign/): the facts, the cells, all six AgentX base manifests, both sets of
calibration driver scripts and the engine-config script (REPRODUCE section 17). Off the original
host, git plus the public trace sources are enough; the bulky artifacts (traces, replay caches) are
regenerated. Every number on this page comes from a file under `campaign/`, and the file is named
next to it. There are two kinds of exception: newer state marked "live CR, not mirrored", and the
sizes of the mirror and of artifacts outside it, measured while writing this page.

**Public mirror.** This branch, `rupei/learned-routing-public`, is the sanitized public copy of the
internal campaign branch `rupei/learned-routing`, which is not published; the records name the
internal branch as it was. The internal history is regrouped into a few commits, so a commit named
in these records appears as `<commit-NN>`, its position in the internal history; where this page or
REPRODUCE says to check out a commit, use this branch's head, which holds the final code. Cluster,
node, account, job and path details are replaced by placeholders such as `<campaign-root>`,
`<worktree>`, `<scratch>`, `<node>` and `<job>`, the clusters are called "the CPU cluster" (a Slurm
CPU cluster) and "the GPU cluster" (8 x H100 SXM nodes), and the per-allocation records of both are
replaced by [`campaign/facts/compute_summary.md`](campaign/facts/compute_summary.md), which keeps
the CPU cluster's hardware classes, job counts, node-hours, throughput and parity evidence and the
GPU cluster's jobs and GPU-hours. Mirrored files that recorded host paths are path-redacted, so
their SHA-256s can differ from the values the records cite; REPRODUCE ("Path-redacted files") lists
the nine that its steps check and what that changes. The built paper is
[`paper/learned-routing-paper.pdf`](paper/learned-routing-paper.pdf).

**Mirror snapshot:** the maintainers' publish pipeline refreshes `campaign/` from the campaign root
with `sync_from_campaign.sh` and sanitizes it; the newest stage entry this snapshot holds is the
last line of [`campaign/facts/STATE.md`](campaign/facts/STATE.md). REPRODUCE step 1 pins the commit
its steps were written against. This page was last revised on 2026-10-05, after the live finalist
runs and the paper's publication audit (phase 3); its Status and Results sections describe that
state.

## Story

The campaign uses AISim offline replay (Dynamo's mocker, timed by AIS) to learn a worker-selection
function for one model on one deployment. The goal is a function that beats every heuristic router
we have on held-out workloads and held-out worker counts.

AISim is the development environment only. The learned policy, `learned-choice`:

- reads only signals the live router already has: prefix overlap, router-tracked load, worker KV
  capacity and session metadata;
- evaluates a small classical choice model in O(N·d) per request;
- ships as a builtin-catalog router policy whose coefficients are plain YAML parameters.

So training writes a policy YAML and calls replay, with no Rust rebuild per iteration
([`PLAN.md`](campaign/PLAN.md), Story).

## Target deployment

From [`config/engine.json`](campaign/config/engine.json) and
[`facts/setup.json`](campaign/facts/setup.json):

| Item | Value |
|---|---|
| Model and backend | Qwen/Qwen3-32B on vLLM 0.24.0, `h100_sxm`, TP2, aggregated |
| Engine timing | AIS, through `mock_engine_args.ais_perf_config`. Needs bindings built with `ais-forward-pass`. |
| Router-side AIS | off. The `ais_perf_config` argument of `run_trace_replay` stays `None` (`engine.json` `replay_call.top_level_ais_perf_config: null`), and `router_prefill_load_model` is `none` (`replay_call.router_prefill_load_model`). |
| KV cache | block size 16; `num_gpu_blocks` 18,863, pinned to the AIS estimate (301,808 tokens per worker) |
| Context | `max_model_len` 131,072 (YaRN). 128K works, so the 32K fallback wasn't needed. |
| Batching | `max_num_seqs` 1,024; `max_num_batched_tokens` 8,192; chunked prefill and prefix caching on |
| Worker counts | N ∈ {2, 4, 6, 8, 16, 32}. All workers are identical; heterogeneous sets are out of scope (CONTRACT A5.2). |

Setup confirmed AIS is in effect: the estimator and replay agree on TTFT, 55.49 ms at 1K tokens and
475.97 ms at 8K tokens. Disaggregated serving and other models come later (PLAN).

## Objective (Amendment A2)

**Good request.** A request is good iff both clauses hold:

- **ITL:** mean ITL ≤ I, where mean ITL = (e2e − ttft) / (OSL − 1). The check is skipped when
  OSL ≤ 1.
- **E2E slowdown:** e2e ≤ S × E0(ISL, OSL), with a 1e-6 relative float tolerance.

E0 is the request's AIS-timed latency alone on an idle worker with no prefix reuse (method
`ais-chunked-estimator-v2`). It's used for evaluation only; policies never see it.

There is no TTFT SLO. TTFT is reported only.

**Metric: windowed goodput** (`goodput_rps_window`, adopted from LR-01):

- **Open loop:** good requests that arrive inside the measurement window, divided by the window
  length. Requests that finish after the window still count; incomplete ones count as not good.
- **Closed loop and AgentX lanes:** good completions per second inside a half-open window that
  ends at full occupancy, the last instant all sessions or lanes were busy.

**SLA per family** ([`facts/calibration.json`](campaign/facts/calibration.json) `sla`). Each pair
is knee-anchored at N = 8:

| Family | I (ms) | S |
|---|---:|---:|
| Mooncake, plus FAST25 conversation and synthetic (inherited) | 252.117 | 3.46625 |
| Synthetic sessions (calibration fix r0; r0 had 40.4165 and 1.90627) | 46.0478 | 2.11784 |
| AgentX | 38.6205 | 1.16134 |

**Load levels.** L1, L2 and L3 are where the default router's good fraction is 0.95, 0.85 and
0.65. They are calibrated per N and per transform on train segments only. Every level follows from
the mirrored load curves by log-linear interpolation (REPRODUCE step 10).

**Training objective.** `lr-train` maximizes the mean over train cells of
goodput(policy) / goodput(default@defaults) (`--objective ratio`). The LR-01 clipped log-ratio is
also available (`clipped_log_ratio`); the training stage must choose one and record the choice.

**Headline.** The learned model against the validation-selected best baseline. Every policy is one
config tuned on pooled train data with an equal CMA-ES budget (A2.1). The test is pre-registered in
[`facts/HEADLINE_TEST.json`](campaign/facts/HEADLINE_TEST.json): a one-sided exact Wilcoxon
signed-rank test over 12 independent test segments, using a clipped log-ratio and 3 CRN replicates.

## Model ladder

From [`PLAN.md`](campaign/PLAN.md) and [`CONTRACT.md`](campaign/CONTRACT.md). Every rung uses one
shared per-worker utility, with softmax or argmax over the candidates, so it works at any N. A rung
counts only if it improves held-out results.

| Rung | Model |
|---|---|
| M0 | `dynamo-default-cost-fn`, tuned |
| M1 | conditional logit over feature set v1, starting from θ = −e₀, which reproduces default routing |
| M2 | linear context logit `u_i = θ·x_i + Σ_k (p_k·x_i)(q_k·x̄_S)`, low rank |
| M3 | nested logit over node/DP groups, only if DP > 1 matters |
| Ablations | (a) jointly tune `router_queue_threshold`; (b) add simulator-only signals. Neither ships. |

**Feature set v1**, in order:

0. `default_logit_scaled`
1. `overlap_frac`
2. `new_prefill_tokens_k`
3. `active_prefill_tokens_k`
4. `kv_load_frac`
5. `active_requests_s`
6. `session_affinity`
7. `isl_x_prefill_load`

An optional v2 has 23 features. Full definitions are in
`lib/router-plugins/builtin/src/learned_choice/FEATURES.md`.

**Excluded inputs:** anything from AIS, `expected_output_tokens` (replay fills it with the true
OSL) and the true OSL.

## Baselines

Each baseline is reported at its shipped defaults and also tuned on the train split with the same
CMA-ES budget as the learned model (PLAN). Per A2.5, baselines run as implemented on this branch:
there are no paper-faithful variants, only a brief sanity check.

| Baseline | Notes |
|---|---|
| `round_robin` | floor |
| `dynamo-default-cost-fn` | the reference (`default@defaults`). This branch adds an optional `seed` for deterministic tie-breaks. |
| `dynamo-two-tier-cost-fn` | sgl-router `cache_aware_zmq` port. Valid in replay: device-tier overlap is in the base (A7 correction). |
| `lmetric`, `ramjet`, `dualmap`, `chwbl`, `llm-d-precise-prefix` | ports from the operator's stack (#15450, #15453). The `lmetric` port was fixed after the campaign froze to count the worker's queued prefill in its P-token (ai-dynamo/dynamo#15450); the frozen results use the pre-fix port. |
| `llm-d-optimized-baseline` | `ttft_source: throughput`. The `modeled` (AIS) variant is an AIS-coupled reference only. |
| `sticky-session` (`mode: hard` or `bounded`) | new campaign policy that mirrors live `SessionAffinityMode`, because replay never sets `affinity_target` |
| `thunderagent` (selection half) | replay can't run its classifier, so it is reported as such or skipped |

## Split design

Cell files are in [`cells/`](campaign/cells/), counts in
[`cells/SPLIT_MANIFEST.json`](campaign/cells/SPLIT_MANIFEST.json). The frozen cells are
lr-cells-v5 (`SPLIT_MANIFEST.json` `calibration.cells_version`). A segment is the unit of
independence: cells that share a segment count as one workload.

| Family | Train | Val | Test |
|---|---|---|---|
| Mooncake: six disjoint 10-min slices, each a 4-min warm-up plus a 6-min window | w0, w2, w4 | w1 | w3, w5 |
| AgentX: 82 complete 128K plays in disjoint subsets | A1–A3 (11 plays each) | V1–V3 (7 each) | T1–T4 (7 each) |
| Synthetic sessions: generator seeds | s0–s3 | s4–s5 | s6–s9 |
| FAST25 conversation (aligned to w3/w5) and FAST25 synthetic (x0, x1) | — | — | test only |
| Toolagent | excluded: it's a relabeled copy of the Mooncake trace (setup audit r1 F1) | | |

**Cells and worker counts:**

| Split | Cells | Segments | N |
|---|---:|---:|---|
| Train | 34 | 10 (11 labels) | {4, 8} |
| Val | 14 | 6 | {4, 6, 8} |
| Test | 60 | 12 | {2, 4, 6, 8, 16, 32} |

Train has 10 segments in the split design (`SPLIT_MANIFEST.json` `segments_by_split`). The train
cells carry an 11th segment label, `agentx:A1+A2`, on the pooled cell
`agentx-A1_A2-base-n8-lanes-L3`. That cell draws from all 22 plays of A1 and A2, so it overlaps
both segments. `facts/noise_calibration.json` still counts it as a separate segment and reports
11 train segments.

**Test cells per N** (`cells/test.jsonl`):

| N | 2 | 4 | 6 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| Cells | 8 | 13 | 7 | 17 | 7 | 8 |

The 7 test cells at N = 6 are flagged `selection_exposed`, because validation also uses N = 6.

**Primary hold-out axis of the test cells:**

| Axis | Cells |
|---|---:|
| Worker count | 30 |
| Transform extrapolation, beyond the train ranges in `SPLIT_MANIFEST.json` `train_ranges` | 10 |
| Held-out Mooncake time window | 6 |
| Held-out family (FAST25) | 6 |
| Held-out AgentX plays | 4 |
| Held-out session seeds | 4 |

Calibration added 4 AgentX test cells at N = 16 and 32, made possible by A3.

**AgentX cells and their base manifests.** 21 of the 26 AgentX cells (5 train, 4 val, 12 test)
use only the `cap300` base. The other 5 each use one transformed base: train
`agentx-A2-think0.5-n4-lanes-L2` (`cap300_think0.5`), `agentx-A3-think2.0-n8-lanes-L1`
(`cap300_think2`) and `agentx-A3-osl1.5-n4-lanes-L2` (`cap300_osl1.5`); test
`agentx-T1-think4.0-n4-lanes-L2` (`cap300_think4`) and `agentx-T2-osl2.5-n8-lanes-L2`
(`cap300_osl2.5`). All six base manifests are mirrored as path-redacted copies (REPRODUCE step 8).

## Status

As of 2026-10-05 (`campaign/facts/STATE.md`; this page's results come from the report stage's
files, which a mirror sync copies under `campaign/`). Every PLAN step is done, and so are the live
finalist runs (A13.2, A20) and the paper draft.

| Stage (PLAN step) | State |
|---|---|
| 0 Setup | done at WT <commit-01>. Audits raised two major findings, both handled: the repeat-sd noise floor was false (fixed by A1 CRN replicates), and the toolagent trace is a relabeled Mooncake copy (excluded at build). |
| 1 Build: policies, workloads, harness, integration | done at WT <commit-07>. Audit rounds and fixes ran through <commit-14>; every lens passes. |
| A3 AgentX lowering (sidecar) | done (`status: ok`). Parity and isolation audits pass after one fix. |
| 2 Calibrate | frozen as lr-cells-v5 (`cells/test.jsonl` SHA-256 `7b998b8e…`) after fix r0, which lengthened the open-loop session warm-up and re-derived the sessions SLA and levels. The independent re-run audit r1 passed ([`campaign/audits/calibration/`](campaign/audits/calibration/)). |
| A9 remote CPU lane | done (`status: ok` in the CPU-cluster record `facts/remote.json`, which isn't published; `facts/compute_summary.md` summarizes it). Every node image used was parity-checked bit-exact against the workstation. |
| 3 Pilot and gate | done 2026-10-03: gate PASS, escalate (`facts/pilot.json`, `facts/gate.json`). On validation, pilot M1 led the best heuristic by +0.030, below the MDE of 0.038. |
| 4 Full tuning: tiers A, B, C and the AIS league | done 2026-10-04: 66 restarts × 400 CMA-ES evaluations, equal budget verified (`facts/tierA.json`, `facts/tierBC.json`). The phase2-mid audit found objective gaming by the unconstrained M1; every later learned arm is sign-constrained (A11.2). |
| 5 Select + test + robustness | done 2026-10-04: finalists frozen first (`facts/finalists.json`), then one test pass of 60 cells × 3 replicates × 33 policies with 0 errors (`facts/test_results.json`), plus lag and timing robustness (`facts/robustness.json`). |
| Final audit and fix | completeness and refute-headline-2 failed with one major finding each, both fixed by added facts and scoped wording; refute-headline-1 and -3 passed ([`campaign/audits/final/`](campaign/audits/final/)). |
| 6 Report | done: [`campaign/report/REPORT.md`](campaign/report/REPORT.md), figures in `campaign/report/fig/`. |
| A13.2 live finalist runs | done 2026-10-05, validation only (A20): 42 of 42 cell runs valid on 6 frozen N = 4 test cells, one 8×H100 SXM node per job. The live policy ranking agrees with simulation (pooled Kendall τ-b 1.00), and M1-v2 − ramjet is positive on 5 of 5 informative cells (mean +0.219 live, +0.162 in simulation). Live results never change a finalist, the headline or a frozen number (`facts/live_results.json`, [`campaign/report/LIVE.md`](campaign/report/LIVE.md), REPORT §15). |
| Paper (A20) | draft in [`paper/`](paper/), built from REPORT and the live results; `make` there builds it. The publication audit passed it for publication as a draft ([`campaign/audits/publication/`](campaign/audits/publication/)); the author list is still open. |

Campaign "phase 1" is PLAN steps 0–3, up to the pilot gate; "phase 2" is steps 4–6. STATE.md tags
such as `setup(phase0)` and `calibrate(phase2)` use PLAN step numbers, not these phases.

## Results

Everything below is **AIS-timed simulation**. Full tables, CIs, figures and caveats are in
[`campaign/report/REPORT.md`](campaign/report/REPORT.md).

**Headline (pre-registered, test split evaluated once).** M1-v2, the `learned-choice` policy with
feature set v2, **beats the branch's ported heuristics, each tuned with an equal budget.**

- Against the validation-selected best baseline (tuned ramjet), its windowed goodput on test is
  higher by a segment-mean clipped log-ratio of +0.0316 (95% CI [+0.0150, +0.0501]; geometric-mean
  ratio 1.032).
- It is ahead on 10 of 12 independent segments: all 8 informative ones, with the 4 sessions
  segments at ceiling.
- The one-sided exact Wilcoxon test gives p = 0.0017 (`facts/test_results.json` `headline`).
- Independent auditors reproduced it exactly and did not refute it under lag, timing, window or
  simulator-model perturbations (`campaign/audits/final/`).

**What the headline does not say:**

- **Magnitude.** Only the sign is a claim: the effect is below the MDE (0.038) and below the timing
  perturbation spread (0.060).
- **Not "learning beats every heuristic".** Faithful LMetric was kept out of the baseline pool by
  A2.5, yet it is inside M1-v2's class and was one of its inits.
  - Untuned, it beats every tuned baseline on validation.
  - Default + faithful LMetric reproduces 53–66% of M1-v2's fresh-validation margin.
  - The learned increment over it is about +0.011 per segment on validation and unmeasured on test
    (`facts/fairness_in_class_heuristic.json`).
- **SLO scale.** Significance holds at SLO scales 1–3 and is lost at 0.75 and 0.5.
- **Request count, not tokens.** The token-weighted A19 metric is mixed (−0.0092). M1-v2 gains on
  short and mid prompts. Prompts of 32K–64K tokens wait longer for their first token than under
  ramjet (TTFT p90 +14%, on 11 of 12 segments; at 64K and more TTFT is on par), and prompts of 32K
  tokens and more lose about 4 points of good fraction.
- **Concentration.** M1-v2 packs short requests onto a hot worker (share cap exceeded on 24 of 60
  cells, against 5 for ramjet).

**Test excerpt** (segment-mean clipped log-ratio; generated with the full tables in REPORT §3.1 by the
report stage's `report/scripts/build_report_data.py` in the campaign root):

| Policy | vs default@defaults: segment mean [95% CI] | vs ramjet: segment mean [95% CI], segments ahead, one-sided p |
|---|---|---|
| M1-v2 (headline learned arm) | +0.176 [+0.127, +0.222] | +0.0316 [+0.0150, +0.0501], 10/12, p 0.0017 |
| M2-ais (AIS league) | +0.173 [+0.124, +0.219] | +0.0286 [+0.0138, +0.0458], 11/12, p 0.0005 |
| M2 rank 2 | +0.165 [+0.118, +0.211] | +0.0211 [+0.0043, +0.0406], 7/12, p 0.0549 |
| M1 (sign-constrained; = A12 row) | +0.163 [+0.116, +0.207] | +0.0186 [+0.0013, +0.0378], 7/12, p 0.0549 |
| llm-d-precise-prefix, tuned (best baseline on test) | +0.146 [+0.099, +0.194] | +0.0022 [-0.0089, +0.0139], 4/12, p 0.4548 |
| ramjet, tuned (val-best baseline) | +0.144 [+0.098, +0.189] | (reference) |
| lmetric port, tuned | +0.136 [+0.086, +0.186] | -0.0086 [-0.0192, +0.0011], 6/12, p 0.8303 |
| M0: default cost fn, tuned | +0.116 [+0.078, +0.156] | -0.0281 [-0.0439, -0.0137], 1/12, p 0.9993 |
| round_robin | -0.563 [-0.715, -0.403] | -0.6657 [-0.8234, -0.5019], 0/12, p 1.0000 |

**Other findings** (REPORT sections in parentheses):

- **Ladder (§5).**
  - M1 beats M0 (+0.047).
  - M2's context term adds nothing over M1, and the rank 3–4 gate failed.
  - Feature set v2 is the step that passes: M1-v2 vs M1 +0.013, p 0.0034.
  - The originally planned M1 and M2 alone are not significant against ramjet (p 0.055 each).
- **Ablations (§5).**
  - A tuned queue threshold adds nothing measurable (ablation a).
  - The session-affinity feature is removable (A15).
  - Ablation b was skipped.
- **AIS league (§8).** AIS-derived runtime features add nothing over M1-v2 (M2-ais −0.0029). They
  do lift the tuned default cost function (+0.026).
- **Worker counts (§4).** Trained on N 4 and 8, M1-v2 stays ahead of ramjet at unseen N 2, 16 and
  32 (7 of 7 segments, uncorrected p 0.008). Per-N claims are descriptive.
- **Coefficients (§6).** M1-v2 is the default cost router plus a learned penalty on uncached
  prefill that scales with the worker's active requests: +1.275 − 0.525 × active requests per 8K
  tokens. Zeroing that term costs 0.018 on validation.
- **Robustness (§7).** Router-state lag of 10, 50 and 200 ms and timing perturbations of 0.8× and
  1.2× keep the sign. M1-v2 ranks first in every condition (Kendall τ-b 0.78–0.95).
- **Porting (§12).** One YAML, O(N·d) per request, no AIS at runtime. Keep the structural router
  knobs at their defaults, decide on host session affinity explicitly, and validate live before any
  rollout.
- **Live validation (§15).** On one 8×H100 node per cell (N = 4), the live ranking of the six
  finalists agrees with simulation, and M1-v2 stays ahead of ramjet on every informative cell. The
  live runs check the ranking and that sign only, not absolute goodput or the size of the gaps; the
  token-weighted A19 comparison does not hold live (nor in simulation on these cells).

## Directory map

| Path | Contents |
|---|---|
| `README.md`, `REPRODUCE.md` | this page; step-by-step reproduction |
| `sync_from_campaign.sh` | refreshes `campaign/` from the campaign root |
| `campaign/PLAN.md` | plan v2 and the operator's decisions |
| `campaign/CONTRACT.md` | shared contract, binding for every stage, with amendments A1–A20 |
| `campaign/config/engine.json` | the engine, the single source of truth (original SHA-256 `7b5ff164…`; the published copy is path-redacted, `4d509bd8…`) |
| `campaign/facts/STATE.md` | append-only stage log, one line per stage result |
| `campaign/facts/DEVIATIONS.md`, `UPSTREAM_FOLLOWUPS.md` | recorded deviations and the Dynamo/AISim follow-ups |
| `campaign/facts/*.json` | per-stage facts, from `setup`, `build_*`, `calibration` and `test_freeze` through `pilot`, `gate`, `tierA`, `tierBC`, `finalists`, `test_results`, `robustness` and `live_results`. The compute records `remote.json` (CPU cluster) and `live.json` (GPU cluster) aren't published; `compute_summary.md` summarizes them. |
| `campaign/cells/` | frozen lr-cells-v5 `{train,val,test}.jsonl`, the pre-calibration lr-cells-v3 `*.candidates.jsonl` and `SPLIT_MANIFEST.json` |
| `campaign/traces/` | the trace `MANIFEST.json` (SHA-256 of every source trace) and all six AgentX base-lowering manifests, all path-redacted |
| `campaign/audits/` | audit and fix reports by checkpoint: `setup/`, `build/`, `agentx-lowering/`, `calibration/`, `phase2-mid/`, `final/`, `live/`, `publication/` |
| `campaign/report/` | the final report `REPORT.md`, the live-validation report `LIVE.md` and the figures in `fig/` |
| `campaign/runs/` | decision tables and load curves of calibration r0 and every fix r0 round; gzipped `results.jsonl` of small runs, including the step-4 references `calibrate/step4_r4/` and `calibrate-fix-r0/step4_r6/`; `lr-train` smoke outputs; the CPU-cluster lane's evidence (`remote-lane/`, `remote/returned/`); both `cells_validation.json`; `policies.tar.gz` |
| `campaign/scripts/` | calibration drivers: `calibrate/` (r0, 21 files, 100,614 bytes) and `calibrate-fix-r0/` (29 files, 152,857 bytes); `setup/write_engine_config.py` (3,801 bytes); sizes of the published, path-redacted copies |
| `campaign/literature/` | `LESSONS.md` (LR-01 to LR-15), `BIBLIOGRAPHY.md` and `notes/` (four scout notes) |
| `campaign/WORKLOG.md` | the campaign worklog: the operator's decisions, workflow launches, incidents and status notes in time order, redacted like the rest of the mirror |
| `benchmarks/learned_routing/` | the harness package `learned_routing`: `lr-eval`, `lr-train`, `lr-report`, `workloads/`, `spaces/`, `tests/`, and the Rust helper `tools/agentx_lower` |
| `benchmarks/learned_routing/remote/` | the A9 CPU-cluster lane: `submit_eval.sh`, `submit_train.sh`, `fetch_ingest.sh`, `common.sh`, node scripts, helpers, the phase-2 orchestrator `p2orch.py` and its own `README.md` (REPRODUCE section 13). |
| `benchmarks/learned_routing/live/` | the A13 live GPU lane: AIPerf inputs, the live scorer and the deployment recipe under `deploy/`, each with its own `README.md` |
| `paper/` | the campaign paper (LaTeX; `make` builds `build/main.pdf`) |
| `lib/router-plugins/builtin/src/learned_choice/` | the `learned-choice` policy and `FEATURES.md` |
| `lib/router-plugins/builtin/src/sticky_session.rs` | the `sticky-session` policy, plus `choice.rs` and `session_map.rs`, which both new policies use |
| `lib/router-plugins/builtin/src/default/` | the `seed` parameter on `dynamo-default-cost-fn` (`parameters.rs`) |

`campaign/audits/build/` isn't tracked by a plain `git add`: the repository `.gitignore` rule
`[Bb][Uu][Ii][Ll][Dd]/` matches it. Commit it with `git add -f`.

## Key decisions and amendments

The full text is in [`CONTRACT.md`](campaign/CONTRACT.md); rationale and side effects are in
[`facts/DEVIATIONS.md`](campaign/facts/DEVIATIONS.md).

- **Operator decisions (PLAN):**
  - Goodput at fixed loadgen loads; no capacity-at-SLA search.
  - No AIS coupling at runtime.
  - AgentX at 128K context.
  - No hard stop.
  - One recalibration if the pilot gate fails, then stop with a diagnostic.
  - Bugs are patched on the campaign branch and listed as follow-ups.
  - CPU-cluster spillover is allowed.
- **A1, noise:**
  - `--repeats K` means common-random-number (CRN) workload replicates, protocol `crn-order-v1`,
    with policy seed k + 1 for seeded policy types. Identical re-runs are deterministic, so they
    never measure noise.
  - "Differs beyond noise" uses the paired per-replicate ratio sd against default, plus a
    two-sided |mean| > 2 × SE test over independent segments.
- **A2, objective:** as above. Baselines run as implemented.
- **A3, AgentX:** AgentX uses AISim's own Weka lowering, recycled into Agentic Mooncake v2 traces,
  so lanes stay loaded at N = 16 and 32.
- **A4, lengthening:** a workload may be lengthened by duplication, with disjoint hash ranges and
  split purity. Duplicates are not independent segments.
- **A5, staleness:** router-state lag {0, 50, 200} ms is a test-stage robustness check only.
  Worker-set extrapolation means worker counts only.
- **A6, upstream lane:** production-ready generic fixes go to a clean branch and a draft PR.
  Campaign agents only flag candidates in `UPSTREAM_FOLLOWUPS.md`.
- **A7, device tier:** the device-tier overlap fix is already in the base `2be8d9c43a`, so
  two-tier's replay behavior is valid.
- **A8, notes:** this notes mirror. Only the top-level session syncs it and pushes the branch,
  with no PR. Per-file cap: the CONTRACT text says 1 MiB. The operator then allowed files of a
  couple of MB ("if it's a couple MBs can also push it"), so `sync_from_campaign.sh` caps files at
  5 MiB with a 25 MiB total budget. The CONTRACT text hasn't been amended to match.
- **A9, CPU-cluster lane** (2026-10-03 ~01:40 PDT plus an ~01:50 addendum):
  - Once `facts/remote.json` reports `ok`, any batch or tuning job projected at more than about
    1 hour of local wall-clock runs on CPU-cluster nodes. This replaces the CONTRACT's 8-hour gate
    threshold.
  - Holding several nodes at once is fine, and node type doesn't matter.
  - Each distinct node OS or glibc image needs one bit-exact parity smoke against local.
  - Every allocation is recorded with its cancel command.
- **Notable deviations:**
  - Toolagent is excluded.
  - Synthetic sessions are generated multi-turn Mooncake traces, not replay's native
    `synthetic-session` source.
  - AgentX idle gaps are capped at 300 s.
  - Mooncake windows were redesigned to be disjoint.
  - Logged arrival bursts are spread per replicate (`crn-spread-v1`).
  - Session cells are think-time invariant: think time no longer changes with load or N.
  - Open-loop session cells warm up for two session-lifetime envelopes, 960–2160 s by transform,
    instead of 180 s, and the sessions knee is where doubling that history lowers default's good
    fraction by more than 5% (calibration fix r0).
  - Loads are knee-matched per N.

## Artifacts not in git

These live under the campaign root `CR = <campaign-root>`. All of them are
bulky or regenerable. [REPRODUCE.md](REPRODUCE.md) explains how to fetch or regenerate each one.

| Artifact | Location | Identity in git |
|---|---|---|
| Source traces: Mooncake, toolagent (excluded), FAST25, sessions, AgentX plays | `CR/traces/{mooncake,toolagent,fast25,synthetic_sessions,agentx}/` | `campaign/traces/MANIFEST.json`, SHA-256 per file |
| AgentX HF source (1,847,151,435 bytes) | `CR/traces/agentx/source/traces.jsonl` | `MANIFEST.json` `agentx.source` |
| AgentX lowered bases and recycled traces | `CR/traces/agentx_lowered/{weka_*,base_*,gen}/` | all six base manifests in `campaign/traces/`; `lowered.spec` in each AgentX cell |
| Derived (transformed) traces | `CR/traces/derived/<sha>.jsonl` and `.meta.json` | `trace_sha256` in each cell |
| Frozen test traces (39 files, 2,130,865,265 bytes) | as referenced by `cells/test.jsonl` | `facts/test_freeze.json` |
| Calibration intermediates: sweep results, scored items, derived-trace indexes, `final_r*/` cell sets, fix r0's `out/` diagnostics | `CR/runs/calibrate/`, `CR/runs/calibrate-fix-r0/` | the decision tables, curves and step-4 results are mirrored; the rest regenerates with the mirrored scripts (REPRODUCE step 10) |
| Replay cache, CRN replicate traces, run outputs | `CR/runs/{cache,replicates,…}/` | none; replay recomputes them |
| Literature: 56 PDFs (148,964,819 bytes) and 43 text extractions | `CR/literature/{pdfs,txt}/` | `campaign/literature/BIBLIOGRAPHY.md`, which still names the earlier location `/tmp/learned-routing-lit/` |

## Refreshing the mirror

On this public branch, only the maintainers' sanitizing publish pipeline refreshes `campaign/`. It
runs `sync_from_campaign.sh` on the internal campaign root, replaces infrastructure details with
placeholders, summarizes the compute records, and commits nothing unless a fail-closed denylist scan
of the result is clean. Never commit the raw output of the script run on an internal campaign root.

On your own campaign root, the script works as below. It never copies the per-allocation compute
records (`facts/remote.json`, `facts/live.json`, `facts/cluster_cleanup.json`, `*.jobs*.json`,
`*bundle_jobs*`, lock files and cleanup ledgers), because they name clusters, nodes, accounts and job
IDs.

```bash
cd <worktree>
CR=<campaign-root> notes/learned-routing/sync_from_campaign.sh
git add notes/learned-routing && git add -f notes/learned-routing/campaign/audits/build
git commit -s -m "docs(learned-routing): sync campaign records"
```

- `CR` is required: the campaign root.
- `PLAN_SRC` optionally names the plan document when it lives outside the campaign root.
- The script copies only small curated files. Its caps are set at the top of the script:
  - `MAX_BYTES` is 5 MiB per file, also applied to gzipped `results.jsonl`. Larger files are
    skipped with a message.
  - `BUDGET_BYTES` is 25 MiB total, and only produces a warning.
- The published mirror at this snapshot is 35,558,687 bytes in 593 files under `campaign/`,
  without the per-job compute records. Its largest file is `runs/phase2/local/stickyhard-s3/results.jsonl.gz` at
  4,405,645 bytes; the script skips any file over 5 MiB.
- After a sync, update the Status section above if the newest `STATE.md` entry changed it.
- Per A8, only the top-level session syncs the internal branch, and per A18 only the publish
  pipeline updates this one. Stage agents write to `CR` and don't edit this directory.
