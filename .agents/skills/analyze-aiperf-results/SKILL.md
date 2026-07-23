---
name: analyze-aiperf-results
description: >-
  Evaluate an audited AIPerf benchmark against target SLOs and compare it with the original baseline, previous valid
  iteration, best prior result, and all prior valid same-series runs. Use to produce machine-readable and Markdown
  performance findings for downstream optimization agents and users.
---

# Analyze AIPerf Results

Interpret valid AIPerf evidence without claiming an unmeasured server-side cause.

Read `docs/rules/optimization.md`, `docs/rules/benchmarking.md`, `docs/guides/result_storage.md`, the target workload,
canonical benchmark plan, current audit/summary, and all prior candidate audits/summaries.

## Select Comparable History

Use only prior runs whose audit is valid and whose benchmark-series id matches the current plan. From that set identify:

- `original_baseline`: earliest valid iteration;
- `previous_valid`: most recent valid iteration before the current one;
- `best_prior`: best prior run for each objective, respecting metric direction and SLO feasibility;
- `history`: every valid same-series iteration.

Iteration 0 establishes the baseline. Do not describe it as a gain or loss.

## Analyze

1. Evaluate every target SLO and report the observed statistic, threshold, pass/fail, and missing evidence.
2. Report goodput and good-request fraction when configured, including the attainment target.
3. Report throughput, output throughput per GPU, output throughput per user, TTFT, ITL, request latency, errors, and
   workload-shape metrics available for the run.
4. For trace workloads, include time-sliced behavior when it reveals warmup leakage, bursts, collapse, or instability.
5. Compare current versus each highlighted prior and produce a compact all-history table.
6. Calculate signed percent change as `(current - prior) / prior * 100`. Also state whether the value is higher or
   lower and whether that direction is an improvement or regression.
7. Use multi-run confidence intervals/CV when available. If uncertainty prevents a defensible conclusion, mark the
   metric or overall comparison `inconclusive` if the performance deltas are less than 1% or are due to noise.
8. State whether the current candidate is SLO-feasible, Pareto-improving, mixed, regressed, or inconclusive.
9. Identify client-visible symptoms and missing measurements. Do not convert them into kernel, communication,
   scheduler, router, or backend root-cause claims.
10. Mark recipe-proxy results as proxy-scoped and carry the workload mismatch into limitations.

## Outputs

Write `performance_analysis.json` containing:

- current candidate and benchmark-series identity;
- target-SLO evaluation;
- absolute current metrics;
- comparisons to `original_baseline`, `previous_valid`, and per-objective `best_prior`;
- full valid history;
- confidence/uncertainty;
- verdict, client-visible symptoms, missing evidence, and limitations.

Write `performance_analysis.md` with a concise executive verdict, SLO table, current metrics, gain/loss comparison
table, history, insights, and limitations.

Append one compact record to `EXP_ROOT/analysis/performance_findings.jsonl` containing the iteration, verdict, primary
absolute metrics, highlighted deltas, SLO status, and paths to the full artifacts. Preserve prior records.
