<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Measurement protocol

Invariants and pointers. **No counts, no thresholds, no flags** — those live in the harness that owns them, and restating them here guarantees drift.

The human-facing version of this material is the Performance Analysis Method page in the developer guide. Keep them consistent; that page is the fuller prose.

## Preconditions

A measurement taken while any of these is false describes the fault, not the system.

| Check | Owned by |
| --- | --- |
| A single request succeeds against the endpoint | skill `dynamo-troubleshoot` |
| KV transport is on the intended path (disaggregated only) | skill `dynamo-interconnect-check` |
| Router is in the mode you believe — KV-aware silently degrades when workers do not publish KV events | skill `dynamo-router-starter` |
| Nothing else on the machine: no build, no unrelated load | skill `dynamo-frontend-benchmark` § Pitfalls |
| The load generator is not itself saturated — client-side tokenization is expensive | skill `dynamo-frontend-benchmark` § Pitfalls |
| The first run after a fresh build is discarded as a cold-start outlier | — |

**Closed-loop load measures concurrency divided by latency, not maximum throughput.** A single concurrency point is not a saturation measurement. Idle server cores usually mean latency-bound, not fast. Sweep concurrency to find the knee.

## Run protocol

1. Tear down fully between runs — router and cache state accumulates and inflates later runs.
2. Warm up before the measured phase.
3. Run more than once. One sample has no error bar.
4. Compare medians, not means — a cold first run drags a mean.
5. Interleave arms (A, B, A, B), never all of A then all of B.
6. Change one variable per comparison.

Concrete teardown commands, warmup sizes, and repetition counts for the frontend harness are in `dynamo-frontend-benchmark` § "Running a throughput benchmark — methodology". Do not restate them here.

## A/B design

The design that holds up: **paired** runs, **arm order randomized within each pair**, compared by a **distribution-free interval on the median ratio** against an equivalence threshold, where an interval straddling the threshold is **inconclusive and blocks the claim** rather than passing it.

Dynamo has one fully worked instance, in `dynamo-kv-replay-parity` Stage 7. Follow that **design**.

Do **not** transplant its constants. They are calibrated to a fast, deterministic, single-process workload producing one elapsed value per run — which is what makes a large sample affordable and what licenses attributing the difference to overhead. None of that holds for a GPU sweep, a Kubernetes deployment, or any per-request distribution.

**Known gap:** no procedure exists for A/B-ing request-level distributions, where the unit of analysis is the request rather than the run. "Did p99 TTFT get worse under load" is not covered. Say so rather than borrowing a procedure that does not fit.

## Capture

| To learn | Capture |
| --- | --- |
| Where latency accumulates within a request | Per-request stage traces, converted to a timeline |
| Which component along the path is slow | Stage-duration and queue-depth metrics |
| Where CPU time goes | On-CPU profile and flame graph |
| What blocked threads wait on | Off-CPU profile. Parked async workers are benign idle; look for application lock frames |
| Whether the deployment is healthy at all | Dashboards, before anything else |

Metric names live in the observability reference. Environment variables live in the tracing docs.

## Fidelity

Simulation-backed tools report **computed** values. The default timing model is uncalibrated, several paths are not modelled at all (including multimodal encoder compute), and no calibration harness exists, so the gap to hardware is undocumented.

Rank candidates with simulation under one frozen configuration, validate the winner on hardware, and never present a simulated figure as measured.

## Reporting

State: workload, concurrency, substrate, revision, number of runs, spread across runs, whether arms were interleaved, and which preconditions were verified.

Published cross-configuration claims also follow the benchmark catalog contract, which binds a claim to the hardware, traffic definition, and the deployment and benchmark manifests of every arm. That contract covers **what was run**; the statistical qualifiers above cover **how well it was measured**, and it does not require them.
