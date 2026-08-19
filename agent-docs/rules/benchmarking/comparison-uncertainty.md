<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Comparison Uncertainty

Default to one measured AIPerf run per candidate. Configure each measured run to finish in 30 minutes or less. GPU
benchmarking is expensive, so do not collect repetitions only to produce confidence intervals.

## Default Decision Policy

- **Current result**: the configuration being evaluated.
- **Reference result**: the valid baseline or prior configuration used for comparison.
- Compare the same metric, statistic, unit, workload phase, and benchmark-series identity.
- Classify an absolute performance change of `0.5%` or less as noise. Report it without automatically repeating the
  benchmark.
- A clear, substantial improvement or regression may support a conclusion from one valid, isolated, plausible run.
  State that the comparison is single-run evidence.
- Repeat a valid benchmark only when the existing evidence cannot support a consequential decision, another run is
  likely to resolve that uncertainty, and the information value justifies the GPU cost. Record why the repeat is
  necessary before launching it.
- Use the same frozen workload and preserve each run's raw artifacts when a repeat is necessary.

## Confidence Statistics

Use AIPerf confidence intervals and coefficient of variation only when deliberate, comparable repetitions exist and
the statistics help resolve the decision:

- Do not treat AIPerf's degraded single-run output as confidence evidence, even when `ci_low`, `ci_high`, and the mean
  are equal.
- For a higher-is-better metric, the current result's `ci_low` must exceed the reference result's `ci_high` to claim
  separation from confidence intervals.
- For a lower-is-better metric, the current result's `ci_high` must be below the reference result's `ci_low`.
- When confidence intervals are needed and overlap, classify the comparison as `inconclusive`, not as noise. Inconclusive does not necessarily mean repeating the benchmark is required.
- Do not require confidence intervals or another run for a clear, substantial gain or loss solely because the current
  evidence contains one run.

Report the absolute values, signed delta, run count, repeat decision and rationale, confidence statistics when used,
and remaining limitations.
