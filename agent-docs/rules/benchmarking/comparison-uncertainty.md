<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Comparison Uncertainty

Use the multi-run confidence statistics reported by AIPerf.

## Evaluate Uncertainty

- **Current result**: the configuration being evaluated.
- **Reference result**: the valid baseline or prior configuration used for comparison.
- Treat an improvement of `0.5%` or less as noise.
- Use AIPerf's reported confidence intervals, not only the means.
- For a higher-is-better metric, the current result's `ci_low` must exceed the reference result's `ci_high`.
- For a lower-is-better metric, the current result's `ci_high` must be below the reference result's `ci_low`.
- Compare the same metric, statistic, unit, workload phase, and benchmark-series identity.
- Do not treat AIPerf's degraded single-run output as confidence evidence, even when `ci_low`, `ci_high`, and the mean
  are equal.

Classify an improvement of `0.5%` or less as noise. If it exceeds `0.5%` but fails the confidence-bound test, mark it
`inconclusive`. Report the absolute values, signed delta, AIPerf confidence interval, and coefficient of variation.
