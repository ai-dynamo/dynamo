<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Benchmark Tool Version

The benchmark client is part of the measurement instrument; its version is part of benchmark-series identity.

- At benchmark-plan authoring time, resolve the LATEST STABLE AIPerf release, record the exact version in the
  immutable plan, and use that single version for every run in the series. Do not inherit a version pin from a
  repo recipe's perf manifest by default: that pin exists to reproduce THAT recipe's reference numbers, not to
  govern a new series, and it goes stale.
- Do not change the tool version within a series without accounting for it. Record every run's tool version in
  the audit and compare it to the plan's pin; when versions differ, respond in proportion to the difference rather
  than discarding history:
  - a PATCH difference is comparable; note the delta in the audit and the comparison;
  - a MINOR difference is comparable when either the release notes for the span show no measurement-affecting
    change (metric definitions, timing or windowing, request accounting, tokenizer handling) or one BRIDGING RUN
    exists: the current best-prior configuration re-measured once under the newer version, which yields a measured
    version delta instead of an assumed one; record which of the two justified the comparison;
  - a MAJOR difference, a release-notes-flagged measurement change, or a minor difference with neither
    justification is a series boundary (`series-boundaries.md`): report absolute results, no gain or loss claims,
    and open a new plan and series for the newer version.
  A bridging run costs one benchmark; re-measuring a whole series costs many. Prefer the bridging run. If a
  bridging run shows a shift beyond the series noise floor, treat that as a finding about the instrument and
  record it as an ask.
- Exception 1: when the engagement's goal is to reproduce or compare against an external reference (for example a
  recipe's published perf numbers), match the reference's pinned version for that comparison and record the choice.
- Exception 2: an operator-specified version always wins; record it in the plan.
- The audit records the tool version per run; the analyzer must check it against the plan before any same-series comparison and apply the graded response above, never compare across an unaccounted version difference.
- RECORD THE RESOLUTION: the benchmark plan must state the mechanism used to resolve "latest stable" (for example
  the package-index query and its output) and the date. A version asserted from memory is not a resolution;
  models otherwise resolve "latest" from training priors and land releases behind.
