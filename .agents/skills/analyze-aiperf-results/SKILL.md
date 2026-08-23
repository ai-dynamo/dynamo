---
name: analyze-aiperf-results
description: >-
  Validates and normalizes raw AIPerf outputs, then evaluates valid results against target SLOs and comparable prior
  candidates. Use after an AIPerf Job completes to produce benchmark audit, summary, and performance analysis artifacts.
license: Apache-2.0
metadata:
  author: NVIDIA
  tags:
    - dynamo
    - aiperf
    - benchmarking
---

# Analyze AIPerf Results

<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Audit benchmark evidence before interpreting performance. Preserve raw files unchanged and do not claim an unmeasured
server-side cause.

Read:

- `agent-docs/rules/benchmarking/benchmark-isolation.md`;
- `agent-docs/rules/benchmarking/comparison-uncertainty.md`;
- `agent-docs/rules/benchmarking/evidence-eligibility.md`;
- `agent-docs/rules/benchmarking/result-storage.md`;
- `agent-docs/rules/benchmarking/series-boundaries.md`;
- `agent-docs/rules/optimization/evidence-before-spend.md`;
- `agent-docs/rules/optimization/one-variable.md`;
- `agent-docs/rules/verification/config-engagement.md`;
- `agent-docs/rules/verification/implausible-speedup.md`;
- `agent-docs/rules/verification/overlap.md`; and
- `agent-docs/rules/verification/stack-verdict.md`.

Also read the user workload, active benchmark plan, execution ledger, AIPerf config, raw outputs, all prior
candidate audits and summaries, and the profile-export documentation matching the pinned AIPerf source or runtime.

## Audit And Normalize

- Parse per-request `profile_export.jsonl` with AIPerf's native Pydantic models when available. Record the parser and
  runtime version used.
- Parse `profile_export_aiperf.json` and multi-run aggregate/search artifacts when configured.
- Confirm files are readable, non-empty, and internally consistent.
- Confirm the executed config matches the active plan and candidate endpoint.
- Confirm trace hash or static-shape identity, schedule mode, endpoint type, model, tokenizer, warmup, seed, request
  count/duration, repetitions, and load controls.
- Separate warmup from profiling records and exclude only data the active plan says to exclude.
- Compare attempted, successful, failed, cancelled, and timed-out request counts.
- Check actual ISL/OSL distributions against the input workload and report any output-length shortfall.
- Check timestamps, benchmark duration, fixed-schedule coverage, duplicate/missing request ids, malformed metrics,
  NaN/inf values, units, and impossible negative latencies.
- Recompute user-requested percentiles from raw profiling records when AIPerf does not export them directly.

If the aggregate export is missing but complete raw records exist, reconstruct it once using the pinned AIPerf
models and metric definitions. Record `valid_with_recovery`, the missing file, method, and generated summary. Never
modify or replace the raw directory.

## Write Audit Artifacts And Gate Analysis

Write `benchmark_audit.json` with:

- `status`: `valid`, `valid_with_recovery`, or `invalid`;
- benchmark-series ID, plan path and SHA256, performance question, and workload identity;
- expected versus actual requests and phases;
- error/cancellation breakdown;
- integrity checks and recovery actions;
- parser/AIPerf versions;
- blockers and `next_action`: `continue_analysis`, `rerun_benchmark`, or `stop`.

Write `benchmark_summary.json` with normalized benchmark metadata and every numerical metric reported by AIPerf,
including units and available statistics. Include requested custom percentiles and per-GPU derived throughput with the
GPU-count source. Do not include gain/loss interpretation in the summary.

For `valid` or `valid_with_recovery`, set `next_action` to `continue_analysis` and continue below in the same
invocation.

For repairable `invalid` evidence, set `next_action` to `rerun_benchmark`. Return `benchmark_audit.json` and
`benchmark_execution.json` to `run-aiperf-benchmark`, preserve the invalid run as failed evidence, and rerun the active
series unchanged without overwriting its raw artifacts. Invoke `analyze-aiperf-results` again after the rerun.

If repair would change workload semantics or a bounded rerun repeats the same invalid result, set `next_action` to
`stop`. Do not write performance analysis or promote the candidate until a valid rerun exists. Never discard an
invalid run.

## Select Comparable History

Use only valid runs whose benchmark-series ID matches the active plan. From that set identify:

- `series_baseline`: earliest valid result in the series;
- `previous_valid`: most recent valid iteration before the current one;
- `best_prior`: best prior run for each objective, respecting metric direction and SLO feasibility;
- `history`: every valid same-series iteration.

Verify that every reference required by the plan is present before making a direct comparison. If no prior valid
same-series result exists, treat the current result as the series baseline and report absolute performance only.
Cross-series results may provide context but never a gain, loss, or Pareto calculation.

## Analyze

1. Evaluate every target SLO and report the observed statistic, threshold, pass/fail, and missing evidence.
2. Report goodput and good-request fraction when configured, including the attainment target.
3. Report throughput, output throughput per GPU, output throughput per user, TTFT, ITL, request latency, errors, and
   workload-shape metrics available for the run.
4. For trace workloads, include time-sliced behavior when it reveals warmup leakage, bursts, collapse, or instability.
5. When comparable references exist, compare current versus each highlighted prior and produce a compact same-series
   history table.
6. Calculate signed percent change as `(current - prior) / prior * 100`. Also state whether the value is higher or
   lower and whether that direction is an improvement or regression.
6a. When the series has no measured noise floor and the decision at hand rests on a small delta - or a
stop-request or final recommendation requires the series MDE per `optimize-loop.md` section 6 - return
   `repeat_decision: necessary` with the rationale "series noise-floor pilot (n=3 total)"; after the pilot,
   derive the run-to-run spread and minimum detectable effect and record both in `performance_analysis.json`
   (fields `series_noise_floor`, `minimum_detectable_effect`); copy both forward into every later same-series
   `performance_analysis.json`.
7. Classify an absolute performance change at or below the measured noise floor of the active benchmark series (see
   `comparison-uncertainty.md`) as noise and report it without recommending a repeat
   solely because the delta is small.
8. Analyze one valid run by default. A clear, substantial, plausible gain or loss may support a conclusion without a
   repeat; state that it is single-run evidence.
9. Use multi-run confidence intervals and coefficient of variation only when deliberate comparable repetitions exist
   and those statistics help resolve the decision. Do not treat degraded single-run intervals as confidence evidence.
10. Recommend another valid run only when the existing evidence cannot support a consequential decision, another run
    is likely to resolve the uncertainty, and the information value justifies the GPU cost. Record that rationale. If
    uncertainty remains but a repeat is not justified, report `inconclusive` and stop.
11. State whether the current candidate is characterized, SLO-feasible, Pareto-improving, mixed, regressed, or
    inconclusive, as supported by the available evidence.
12. Identify client-visible symptoms and missing measurements. Do not convert them into kernel, communication,
   scheduler, router, or backend root-cause claims.
13. Mark recipe-proxy results as proxy-scoped and carry the workload mismatch into limitations.

## Write Analysis Artifacts

Write `performance_analysis.json` containing:

- current candidate, performance question, benchmark-plan identity, and benchmark-series identity;
- target-SLO evaluation;
- absolute current metrics;
- comparisons to `series_baseline`, `previous_valid`, and per-objective `best_prior` when available;
- full valid same-series history and any required reference still missing;
- run count and confidence/uncertainty;
- `repeat_decision`: `not_needed`, `necessary`, or `not_justified`, with the GPU-cost rationale and the decision the
  repeat is expected to resolve;
- verdict, client-visible symptoms, missing evidence, and limitations.

Write `performance_analysis.md` with a concise executive verdict, SLO table, current metrics, applicable comparisons,
same-series history, insights, and limitations.

Append one compact record to `EXP_ROOT/analysis/performance_findings.jsonl` containing the iteration, verdict, primary
absolute metrics, applicable deltas, SLO status, and paths to the full artifacts. Preserve prior records.
