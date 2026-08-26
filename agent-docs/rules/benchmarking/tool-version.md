<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Tool Version

The benchmark client is part of the measurement instrument; its version is part of benchmark-series identity.

- At benchmark-plan authoring time, resolve the LATEST STABLE AIPerf release, record the exact version in the
  immutable plan, and use that single version for every run in the engagement. Do not inherit a version pin from a
  repo recipe's perf manifest by default: that pin exists to reproduce THAT recipe's reference numbers, not to
  govern a new series, and it goes stale.
- Never change the tool version within a series. A version change is a series boundary
  (`series-boundaries.md`): results measured under different tool versions are not directly comparable, because
  releases may change measurement semantics (windowing, pacing, accounting), not just fix bugs.
- Exception 1: when the engagement's goal is to reproduce or compare against an external reference (for example a
  recipe's published perf numbers), match the reference's pinned version for that comparison and record the choice.
- Exception 2: an operator-specified version always wins; record it in the plan.
- The audit already records the tool version per run; the analyzer must reject a series whose runs disagree on it.
- RECORD THE RESOLUTION: the benchmark plan must state the mechanism used to resolve "latest stable" (for example
  the package-index query and its output) and the date. A version asserted from memory is not a resolution;
  models otherwise resolve "latest" from training priors and land releases behind.
