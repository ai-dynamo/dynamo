<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Concurrency Grid

Apply this rule only to a capacity or throughput/interactivity Pareto series that deliberately varies concurrency. A
fixed-schedule trace-fidelity run does not become a concurrency sweep.

## Selection Order

1. When the target workload specifies exact concurrency values, run only those values.
2. Otherwise, use a bounded powers-of-two grid such as `1, 2, 4, 8, ...` through the declared maximum.

Do not invent an unbounded sweep when the target and benchmark plan provide no safe maximum. Include `c=1` when the
goal is to characterize the full latency/throughput frontier, but do not add it to a user-constrained set merely to
complete a curve.

Use a non-power-of-two point only when it is required by the target workload, needed to reproduce a baseline, selected
by an AIPerf search method, or chosen as a bounded refinement around an SLO boundary or observed knee. Record the
reason.

## Execution

- Prefer one AIPerf Job with a native sweep or search against a stable server.
- Keep all non-load inputs fixed across the grid.
- Record planned and executed points, ordering, early stops, and omissions.
- Keep request-rate and concurrency experiments in separate series.
