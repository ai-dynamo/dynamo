<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Comparison Uncertainty

Default to one measured AIPerf run per candidate. Configure each measured run to finish in 30 minutes or less
unless the operator authorizes longer windows; record any such authorization in the benchmark plan. GPU benchmarking
is expensive, so do not collect repetitions only to produce confidence intervals.

## Default Decision Policy

- **Current result**: the configuration being evaluated.
- **Reference result**: the valid baseline or prior configuration used for comparison.
- Compare the same metric, statistic, unit, workload phase, and benchmark-series identity.
- Establish the noise floor of each benchmark series empirically before adopting or retiring any candidate on a
  small delta: run a pilot repetition (n=3) of one configuration at the decision point, compute the run-to-run
  spread, and derive the minimum detectable effect. This pilot runs ONCE per benchmark series and is amortized
  across every candidate measured in that series; candidates themselves stay single-run unless their delta falls
  inside the measured noise band. Treat any default percentage as a placeholder, never as a
  measured floor; uncontrolled state such as prefix-cache carryover can put the real floor an order of magnitude
  higher.
- Screening and categorical outcomes (OOM, error storms, wide-margin SLO results) remain valid at n=1.
- For adopt-or-retire decisions inside the noise band, use paired repetitions with controlled or deliberately
  randomized warmup and cache state (for example concurrent arms on separate nodes, or AB/BA ordering), analyze the
  paired differences rather than comparing two means, and add repetitions sequentially up to a declared maximum while
  the interval overlaps the decision boundary.
- Give the selected finalist a fresh confirmatory repetition after selection: adaptive search across many candidates
  inflates the best observed result.
- Keep practical significance separate from detectability: state the smallest delta that matters for the engagement
  and do not claim gains below it regardless of statistical separation.
- Record neighbour occupancy (what else runs on the node) with every measurement and compare only like with like:
  candidates measured under the same occupancy are like-for-like. A fleet or full-node projection from
  idle-neighbour measurements is invalid until confirmed by one co-located measurement with load generation
  verified unsaturated.
- Neighbour occupancy is a recorded condition, not an admission gate. Do not require idle or exclusive nodes to
  measure: on a shared cluster they may never exist, and an empirical noise floor measured under real occupancy is
  the decision floor. Reserve isolation requests for the finalist's confirmatory absolutes, and when isolation is
  unavailable even then, report the finalist with occupancy labeled as a limitation rather than blocking the
  engagement. A neighbour transition during a measurement invalidates that run only when the run's own time-series
  shows a corresponding performance shift; otherwise record the transition and keep the run — on a shared cluster
  transitions are constant, and a rule that discards every affected run converges to no eligible runs at all.
- Cache-state policy for reuse-heavy workloads: reset caches between points and rank candidates on cold, identical
  content (comparability), but before promising an absolute number — a floor-pick or a fleet projection — on a
  workload with substantial prefix reuse, confirm the finalist once at warm steady state. A cold, short window
  measures the cold-to-warm transient, not the regime production runs in.
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
