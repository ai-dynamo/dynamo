<!-- # SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 -->

# Learned routing in AISim: campaign notes

This directory is the durable record of the learned-routing campaign on branch
`rupei/learned-routing-public`. Start here. To rebuild up to the current state, follow
[REPRODUCE.md](REPRODUCE.md). Every small file the reproduction needs is mirrored under
[`campaign/`](campaign/): the facts, the cells, all six AgentX base manifests, both sets of
calibration driver scripts and the engine-config script (REPRODUCE section 17). Off the original
host, git plus the public trace sources are enough; the bulky artifacts (traces, replay caches) are
regenerated. Every number on this page comes from a file under `campaign/`, and the file is named
next to it. There are two kinds of exception: newer state marked "live CR, not mirrored", and the
sizes of the mirror and of artifacts outside it, measured while writing this page.

**Public mirror.** This branch is the sanitized public copy of the internal campaign branch. The
internal history is regrouped into a few commits, so a commit named in these records appears as
`<commit-NN>`, its position in the internal history; where this page or REPRODUCE says to check
out a commit, use this branch's head, which holds the final code. Cluster, node, account, job and
path details are replaced by placeholders such as `<campaign-root>`, `<worktree>`, `<scratch>`,
`<node>` and `<job>`, the clusters are called "the CPU cluster" (a Slurm CPU cluster) and "the GPU
cluster" (8 x H100 SXM nodes), and the per-allocation records are replaced by
[`campaign/facts/compute_summary.md`](campaign/facts/compute_summary.md), which keeps the CPU
cluster's hardware classes, job counts, node-hours, throughput and parity evidence. GPU-cluster jobs
appear there only in snapshots that mirror the GPU lane's job record; this one doesn't. Nine
mirrored files that recorded host paths are path-redacted, so their SHA-256s differ from the values
the records cite; REPRODUCE ("Path-redacted files") lists both and what that changes.

**Mirror snapshot:** commit `<commit-19>`; the sync ran at 2026-10-03 02:40 PDT. Check out
this commit to get exactly the state this page documents (REPRODUCE step 1). Update these three
lines on every sync.

**Newest stage entry in it:** 2026-10-03 02:33:55 PDT, the calibration re-run audit r1: PASS
on `lr-cells-v5` (the last line of [`campaign/facts/STATE.md`](campaign/facts/STATE.md)). Phase 1 continues
with the pilot and gate.

**Live CR only, not mirrored:** the CPU-cluster lane (`facts/remote.json`) was still being provisioned at
sync time.

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
| `lmetric`, `ramjet`, `dualmap`, `chwbl`, `llm-d-precise-prefix` | ports from the operator's stack (#15450, #15453) |
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

| Stage (PLAN step) | State |
|---|---|
| 0 Setup | done at WT <commit-01>. Audits raised two major findings, both handled: the repeat-sd noise floor was false (fixed by A1 CRN replicates), and the toolagent trace is a relabeled Mooncake copy (excluded at build). |
| 1 Build: policies, workloads, harness, integration | done at WT <commit-07>. Audit rounds and fixes ran through <commit-14>; every lens passes. |
| A3 AgentX lowering (sidecar) | done (`status: ok`). Parity and isolation audits pass after one fix. |
| 2 Calibrate, r0 | frozen 2026-10-02 21:19 PDT at WT <commit-16> as lr-cells-v4 (`cells/test.jsonl` SHA-256 `b8648ac5…`), 13,532 replays. The independent re-run audit r0 FAILED with one major finding: the 180 s warm-up of open-loop session cells is shorter than session-population relaxation. |
| 2 Calibrate, fix r0 | done 2026-10-03 01:57 PDT, with no code change: the scripts are in [`campaign/scripts/calibrate-fix-r0/`](campaign/scripts/calibrate-fix-r0/). The open-loop session warm-up is now two lifetime envelopes, 960–2160 s by transform. The sessions knee, SLA and levels were re-derived, and only session cells changed (train 8, val 5, test 15). The test set was re-frozen as lr-cells-v5 (`cells/test.jsonl` SHA-256 `7b998b8e…`); no test cell had been evaluated. 9,792 replays. The binding decision table is the round-2 `runs/calibrate-fix-r0/decisions.json` ([`audits/calibration/fix-r0.md`](campaign/audits/calibration/fix-r0.md)). |
| 2 Calibrate, re-audit | PASS (independent re-run audit r1, 02:33:55 PDT; [`campaign/audits/calibration/`](campaign/audits/calibration/)). |
| A9 CPU-cluster lane | **in progress.** Workflow `wf_e37792bc-dd5` provisions and parity-checks CPU-cluster nodes. Its record `facts/remote.json` (not published; `facts/compute_summary.md` summarizes it) reported `status: in_progress` at 01:53:09 PDT. Runner scripts are in `benchmarks/learned_routing/remote/`, untracked in the WT so far (REPRODUCE section 13). |
| 3 Pilot and gate | TBD |
| 4 Full tuning, 5 Test, 6 Report (phase 2) | TBD |

Campaign "phase 1" is PLAN steps 0–3, up to the pilot gate; "phase 2" is steps 4–6. STATE.md tags
such as `setup(phase0)` and `calibrate(phase2)` use PLAN step numbers, not these phases.

No policy has been tuned or selected for the campaign, so nothing is known about which policy
wins. The only `lr-train` runs so far are smoke runs that test the tooling, not policies: the
build stage's, and the CPU-cluster lane's. The lane ran two smoke runs (`smoke-default-cost-fn-s1` and
`smoke-learned-choice-m1-s1`, 12 evaluations each on two train cells), locally and on the CPU cluster
(`campaign/runs/remote-lane/`, `campaign/runs/remote/returned/`). Expect
`facts/test_freeze.json` to change again if the re-audit forces another re-freeze.

## Directory map

| Path | Contents |
|---|---|
| `README.md`, `REPRODUCE.md` | this page; step-by-step reproduction |
| `sync_from_campaign.sh` | refreshes `campaign/` from the campaign root |
| `campaign/PLAN.md` | plan v2 and the operator's decisions |
| `campaign/CONTRACT.md` | shared contract, binding for every stage, with amendments A1–A9 |
| `campaign/config/engine.json` | the engine, the single source of truth (original SHA-256 `7b5ff164…`; the published copy is path-redacted, `4d509bd8…`) |
| `campaign/facts/STATE.md` | append-only stage log, one line per stage result |
| `campaign/facts/DEVIATIONS.md`, `UPSTREAM_FOLLOWUPS.md` | recorded deviations and the Dynamo/AISim follow-ups |
| `campaign/facts/*.json` | per-stage facts: `setup`, `build_rust`, `build_workloads`, `build_harness`, `integration`, `build_fix_r*`, `agentx_lowered`, `noise`, `noise_calibration`, `calibration`, `test_freeze` and `HEADLINE_TEST`. The CPU-cluster record `remote.json` isn't published; `compute_summary.md` summarizes it. |
| `campaign/cells/` | frozen lr-cells-v5 `{train,val,test}.jsonl`, the pre-calibration lr-cells-v3 `*.candidates.jsonl` and `SPLIT_MANIFEST.json` |
| `campaign/traces/` | the trace `MANIFEST.json` (SHA-256 of every source trace) and all six AgentX base-lowering manifests, all path-redacted |
| `campaign/audits/` | audit and fix reports by checkpoint: `setup/`, `build/`, `agentx-lowering/`, `calibration/` |
| `campaign/runs/` | decision tables and load curves of calibration r0 and every fix r0 round; gzipped `results.jsonl` of small runs, including the step-4 references `calibrate/step4_r4/` and `calibrate-fix-r0/step4_r6/`; `lr-train` smoke outputs; the CPU-cluster lane's evidence (`remote-lane/`, `remote/returned/`); both `cells_validation.json`; `policies.tar.gz` |
| `campaign/scripts/` | calibration drivers: `calibrate/` (r0, 21 files, 100,614 bytes) and `calibrate-fix-r0/` (29 files, 152,857 bytes); `setup/write_engine_config.py` (3,801 bytes); sizes of the published, path-redacted copies |
| `campaign/literature/` | `LESSONS.md` (LR-01 to LR-15), `BIBLIOGRAPHY.md` and `notes/` (four scout notes) |
| `benchmarks/learned_routing/` | the harness package `learned_routing`: `lr-eval`, `lr-train`, `lr-report`, `workloads/`, `spaces/`, `tests/`, and the Rust helper `tools/agentx_lower` |
| `benchmarks/learned_routing/remote/` | the A9 CPU-cluster lane: `submit_eval.sh`, `submit_train.sh`, `fetch_ingest.sh`, `common.sh`, node scripts, helpers and its own `README.md`. Untracked in the WT so far; its workflow is still building it (REPRODUCE section 13). |
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
- The published mirror at this snapshot is 19,715,598 bytes in 306 files under `campaign/`,
  without the per-job compute records. Its largest file is `runs/pilot/fullgen/results.jsonl.gz`
  at 1,030,030 bytes, so every file is also under the CONTRACT's original 1 MiB.
- After committing, put that commit's SHA in the "Mirror snapshot" line above. A commit can't
  name its own SHA, so this takes a follow-up commit that touches only this file.
- Per A8, only the top-level session syncs and pushes. Stage agents write to `CR` and don't edit
  this directory.
