<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overlap

A favorable mean is not enough. A real improvement must:

1. exceed the `0.5%` noise floor; and
2. clear the AIPerf confidence intervals in the improving direction.

For higher-is-better metrics, the current result's `ci_low` must exceed the reference result's `ci_high`. For
lower-is-better metrics, its `ci_high` must be below the reference result's `ci_low`.

Classify a change of `0.5%` or less as noise. Classify a larger change with overlapping intervals as `inconclusive`.
A degraded single-run interval cannot pass this gate. See
[`comparison-uncertainty.md`](../benchmarking/comparison-uncertainty.md).
