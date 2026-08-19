<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conditional Agent Rules

These files define activity-scoped invariants for Dynamo agent workflows. They are not global instructions for every
agent. A role or skill loads the rule group, or the exact files from it, when it performs the corresponding activity.

Rules state what must remain true. Skills own procedures, commands, schemas, and artifact-writing mechanics. Guides
provide operational advice, and references define shared terms and contracts.

## Benchmarking

- [`benchmark-isolation.md`](benchmarking/benchmark-isolation.md) — prevent serving-path and benchmark-client
  contention from contaminating measurements.
- [`comparison-uncertainty.md`](benchmarking/comparison-uncertainty.md) — apply AIPerf confidence intervals and the
  `0.5%` noise floor.
- [`concurrency-grid.md`](benchmarking/concurrency-grid.md) — select a bounded concurrency set for capacity and Pareto
  experiments without changing trace-fidelity workloads.
- [`evidence-eligibility.md`](benchmarking/evidence-eligibility.md) — separate decision-grade benchmark evidence from
  contextual or ad hoc observations.
- [`proxy-workload-selection.md`](benchmarking/proxy-workload-selection.md) — choose and label a recipe-derived proxy
  only when the user supplied neither an exact trace nor an exact static workload.
- [`series-boundaries.md`](benchmarking/series-boundaries.md) — define which benchmark-contract changes start a new
  comparison series.

Agents that create or execute a benchmark need the workload-selection, concurrency, isolation, and series rules.
Agents that audit, analyze, generate hypotheses from, or challenge benchmark evidence need the evidence, uncertainty,
and series rules. Load additional files whenever an assignment crosses those boundaries.

## Optimization

- [`evidence-before-spend.md`](optimization/evidence-before-spend.md) — require measured headroom and an actionable
  lever before spending GPU time on a candidate.
- [`one-variable.md`](optimization/one-variable.md) — preserve attribution by changing one independently testable knob
  per candidate.

Apply optimization rules before generating, approving, deploying, or benchmarking a new candidate.

## Verification

- [`config-engagement.md`](verification/config-engagement.md) — prove the proposed DGD change reached the live serving
  process.
- [`implausible-speedup.md`](verification/implausible-speedup.md) — investigate gains beyond the changed knob's
  plausible impact.
- [`node-equivalence.md`](verification/node-equivalence.md) — prevent Kubernetes node differences from becoming false
  configuration wins.
- [`overlap.md`](verification/overlap.md) — require a gain to clear the noise floor and AIPerf confidence intervals.
- [`stack-verdict.md`](verification/stack-verdict.md) — retain small verified gains that are insufficient for
  promotion by themselves.

Apply verification rules after benchmarking and before accepting, stacking, or promoting a candidate.
