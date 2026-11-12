<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Jev decision-engine experiment

The experimental `jev` adapter implements Dynamo Planner's `EngineProtocol`.
It replaces the decision engine inside Gym replay: Jev's selections directly
produce `PlannerEffects.scale_to`. The native Planner remains a separate baseline.
This does not configure a deployed Dynamo Planner service.

## Run the pilot

Use the same working Dynamo runtime as other offline Gym matches. From `gyms/planner-gym`:

```bash
python -m pip install -e '.[jev]'
# Supply TYPESAFE_API_KEY through your environment/secret manager.
python scripts/run_match_config.py configs/match.jev.example.yaml --validate-only --print-matrix
python scripts/run_match_config.py configs/match.jev.example.yaml
```

The pilot compares Jev, Planner, KEDA/HPA port, reactive, and static 4P4D on
flat, staircase, and flash-crowd synthetic workloads. It pins Jev to
`jev-1.13.0` and the serving performance model to vLLM `0.24.0` on H200.
Check that the installed AISimulate performance data supports that serving version. Validation
does not import the Jev adapter or contact TypeSafe.

There are three Jev cells, each capped at 100 calls with no retry. A missing API
key fails at construction. Hosted calls use TypeSafe's
[documented HTTP API](https://docs.typesafe.ai/api); the optional `jev` extra
provides the HTTP client. Keys are accepted only through `TYPESAFE_API_KEY`,
never the Match Config. Run directories are unique and logs cannot be overwritten.

## Decision contract

One Choice question per pool selects from the current expected replica count
and a bounded step up/down. Questions share one state and one HTTP request.
Code constructs valid targets; Match Config rejects combined pool maxima above
the GPU budget. The default step is one replica and the cadence is 15 simulated
seconds. Starting workers are included in the expected count.

State includes aggregate queue/token backlog, KV utilization, ready and expected
workers, traffic shape and rate, worker GPU costs/cold-start delay, SLO objectives,
and four prior snapshots. Missing measurements remain null. Request content,
worker identifiers, prefix hashes, future arrivals, and baseline decisions are
not sent. The tick interface supplies no measured TTFT/ITL; their limits are
objectives, not observations. Aggregated topology asks only for the decode pool.

Default `min_confidence: 0` exposes Jev's choices without an uncalibrated cutoff.
Setting a threshold explicitly makes low-confidence answers hold their pool's
current expected target. Default `failure_mode: raise` fails on a timeout, API
error, malformed answer, or pinned-model mismatch. `failure_mode: hold` is a
separate reliability experiment; it holds both targets and logs the error.
Exhausting `max_calls` always stops the run.

## Evidence and interpretation

Each Jev run writes `jev-decisions.jsonl` containing the exact request, request
digest, validated answers/probabilities/confidence, returned model, available token
usage, wall latency, selected targets, and error categories. Match results include
the artifact path and a `runtime.jev` overhead summary. Exception text and error
response bodies are excluded; no credentials enter these artifacts.

**API wall time does not advance replay time.** The first experiment evaluates
decision quality with ideal actuation timing. Controller latency is reported
separately and cannot be interpreted as already included in TTFT or GPU-hours.
Delayed-action simulation or a live deployment is needed for that comparison.

Read goodput/GPU together with good rate, latency, GPU-hours, and oscillation.
The native policies retain their documented defaults: Planner polls every five
seconds, KEDA/reactive/Jev every fifteen; dynamic policies start at 1P1D and
static starts at 4P4D. Jev/reactive/KEDA maxima are 16P8D within the common
32-GPU budget; Planner has its own allocation rules. These are policy-package
comparisons, not controlled ablations of the decision mechanism alone.

Freeze prompts and settings before testing held-out workloads. Repeat Jev cells
to measure variability, and report failed or gated decisions. Tests using local
HTTP fixtures validate integration only; they provide no evidence of Jev's
autoscaling quality.
