<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concurrency Grid

Perform a concurrency sweep during benchmarking only if you deem it is necessary - it is not always required.

## Selection Order

1. When the user workload specifies exact concurrency values, run only those values.
2. Otherwise, use a bounded powers-of-two grid such as `1, 2, 4, 8, ...` through the declared maximum.

Do not invent an unbounded sweep when the target and benchmark plan provide no safe maximum. Include `c=1` when the
goal is to characterize the full latency/throughput frontier, but do not add it to a user-constrained set merely to
complete a curve. Size the request count to the decision the point must support - a tail-percentile claim needs
enough observations in the tail, and thin samples at decision points cause expensive confirmation churn downstream:

- Screening a point on mean throughput or a categorical outcome (OOM, error storm, wide-margin SLO result): at
  least `max(4x concurrency, 32 completed requests)`. A screening point supports no percentile claim and cannot
  promote a candidate.
- Ranking candidates on an approximate p95: at least `max(8x concurrency, 100 completed requests)`.
- A final p95 SLO gate or finalist confirmation: at least `max(8x concurrency, 200 completed requests)` (about ten
  observations in the upper 5% tail); note that tighter exceedance guarantees need more (zero observed violations
  in ~59 requests only bounds the violation rate below 5% at 95% confidence; ~299 for a 1% bound).
- Never fewer requests than the concurrency itself - fewer requests than slots cannot even fill the batch.
- Each measured point must also cover a minimum steady-state interval after warmup; more requests inside a short
  transient do not establish stationary tail latency. Keep the measurement inside the standard 30-minute window per
  `comparison-uncertainty.md`; when the window and the sample floor conflict, shrink the sweep, not the sample at
  the decision point.

Use a non-power-of-two point only when it is required by the user workload, needed to reproduce a baseline, selected
by an AIPerf search method, or chosen as a bounded refinement around an SLO boundary or observed knee. Record the
reason.

## Execution

- Prefer one AIPerf Job with a native sweep or search against a stable server.
- Keep all non-load inputs fixed across the grid.
- Record planned and executed points, ordering, early stops, and omissions.
- Keep request-rate and concurrency experiments in separate series.
