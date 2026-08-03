<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Overlap

Classify an absolute performance change of `0.5%` or less as noise and report it without automatically repeating the
benchmark. A clear, substantial gain or loss may support a conclusion from one valid, isolated, plausible run; identify
it as single-run evidence.

Use confidence intervals only when additional runs were necessary and deliberately collected. For higher-is-better
metrics, the current result's `ci_low` must exceed the reference result's `ci_high` to claim interval separation. For
lower-is-better metrics, its `ci_high` must be below the reference result's `ci_low`.

When confidence intervals are needed and overlap, classify the comparison as `inconclusive`. A degraded single-run
interval is not confidence evidence, but the absence of a multi-run interval is not by itself a reason to repeat a
clear benchmark or block a conclusion. See
[`comparison-uncertainty.md`](../benchmarking/comparison-uncertainty.md).
