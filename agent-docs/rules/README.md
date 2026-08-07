<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Conditional Agent Rules

Load only the rules relevant to the current activity. Rules define invariants, guides provide advice, and references
define shared contracts.

## Execution

- [`user-workload.md`](execution/user-workload.md) — preserve user-grounded workload and DGD inputs.
- [`deployment.md`](execution/deployment.md) — preserve deployment ownership and bounded repair.
- [`logging.md`](execution/logging.md) — keep execution evidence reproducible.
- [`run-artifacts.md`](execution/run-artifacts.md) — use the canonical artifact layout.

## Benchmarking

- [`benchmark-isolation.md`](benchmarking/benchmark-isolation.md) — prevent resource contention from contaminating
  results.
- [`comparison-uncertainty.md`](benchmarking/comparison-uncertainty.md) — handle noise, repeats, and confidence.
- [`concurrency-grid.md`](benchmarking/concurrency-grid.md) — select bounded concurrency experiments when needed.
- [`evidence-eligibility.md`](benchmarking/evidence-eligibility.md) — identify decision-grade evidence.
- [`proxy-workload-selection.md`](benchmarking/proxy-workload-selection.md) — choose and label fallback workloads.
- [`result-storage.md`](benchmarking/result-storage.md) — preserve auditable results and comparison history.
- [`series-boundaries.md`](benchmarking/series-boundaries.md) — tailor benchmarks to the current question and freeze
  semantics only for direct comparisons.

## Optimization

- [`evidence-before-spend.md`](optimization/evidence-before-spend.md) — require evidence before testing a candidate.
- [`one-variable.md`](optimization/one-variable.md) — preserve attribution between candidates.

## Verification

- [`config-engagement.md`](verification/config-engagement.md) — prove the candidate change reached the serving process.
- [`implausible-speedup.md`](verification/implausible-speedup.md) — investigate implausible gains.
- [`overlap.md`](verification/overlap.md) — interpret overlap and optional confidence evidence.
- [`stack-verdict.md`](verification/stack-verdict.md) — retain verified gains that are too small to promote alone.
