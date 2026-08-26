<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Measurement Mode

Closed-loop concurrency and open-loop request-rate benchmarks answer different production questions. Choosing the
wrong primary mode does not merely add noise; it can invert the engagement's verdict, because a closed-loop harness
admits each request the instant a slot frees, so arrivals synchronize with server state and a latency tail measured
that way includes the harness's own admission waves.

## Choose The Primary Mode At Iteration 0

Select the primary measurement mode from the engagement objective BEFORE baseline characterization, record it in the
benchmark plan, and keep it for every same-series comparison (cross-series comparisons are forbidden by
`series-boundaries.md`, so a late mode switch forces expensive re-reference runs):

- Latency-SLO capacity objectives (max throughput subject to a TTFT or end-to-end latency percentile; "how much
  traffic can this serve within the SLO"): the primary mode is OPEN-LOOP rate sweeps with independent (Poisson)
  arrivals. Sweep the offered request rate and find the highest rate at which the SLO percentile holds and the
  system is stable.
- Saturation and max-emission objectives (absolute peak tokens/sec, batch-depth characterization), and
  throughput-coupled ITL objectives where admission timing does not dominate the gated metric: closed-loop
  concurrency grids per `concurrency-grid.md`.
- When a throughput recommendation requires saturation evidence (per the optimize-loop stop rules), closed-loop
  saturation runs are a SEPARATE series complementing the open-loop primary, never a substitute for it.

## Open-Loop Validity Requirements

An open-loop rate point is decision-grade only when its delivery is verified. For every rate point record: offered
rate, achieved issue rate, completion rate, and issue-lag trend across the window. Classify the point
`client_limited` and exclude it from frontier claims when the achieved issue rate materially trails the offered
rate, issue lag grows through the window, or client-side saturation (CPU, event loop, connection pool) is detected;
verify once per engagement that the load generator can achieve the target rate against a fast endpoint. Require a
sustained post-warmup measurement interval, account for errors, timeouts, and unfinished requests explicitly, and
repeat at least one boundary point. Never report an offered rate the client could not deliver as achieved
throughput.

## Reporting

State the measurement mode next to every headline number. A latency-SLO capacity claim carries the open-loop rate
at which it was established; a closed-loop TTFT percentile at high concurrency is reported as a harness-coupled
observation, not as the SLO verdict.
