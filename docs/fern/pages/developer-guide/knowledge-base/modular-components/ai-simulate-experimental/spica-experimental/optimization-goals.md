---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Optimization Goals
subtitle: Objective metrics, SLA constraints, and Pareto-front scoring
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Spica provides no SLA, accuracy, or configuration
> optimality guarantees.

An `OptimizationGoal` (the `goal:` block of a `SmartSearchConfig`) declares **what
"better" means** plus the SLA constraint. It is pinned, never searched. It picks one
`OptimizationTarget` and — for `pareto` — the list of scalar objectives whose frontier to
trace. The target also derives the Dynamo Planner's `optimization_target`
(`OptimizationTarget.planner_optimization_target`; see search-space.md).

The whole `goal:` block is **optional**: it defaults to a `throughput` goal with no SLA
(`OptimizationGoal()` → `target = throughput`). `OptimizationGoal` itself is
`extra="forbid"`, so unknown goal keys are rejected.

The goal drives two things in `score.py`: the **objective** read from each replay
`trace_report` (`objective_value`), and **step 3** of scoring — `rank` (scalar) or
`pareto_front` (multi-objective).

## Targets

Every `OptimizationTarget` and the exact report metric it reads (`score.objective_value`):

| `target` | direction | report metric | needs SLA? |
|---|---|---|---|
| `throughput` | maximize | `output_throughput_tok_s` | no |
| `throughput_per_gpu` | maximize | `output_throughput_tok_s / avg_gpu` (tok/s/gpu) | no |
| `throughput_per_user` | maximize | `mean_output_token_throughput_per_user` (tok/s/user) | no |
| `e2e_latency` | **minimize** | `mean_e2e_latency_ms` | no |
| `goodput` | maximize | `goodput_output_throughput_tok_s` | **yes** |
| `goodput_per_gpu` | maximize | `goodput_output_throughput_tok_s / avg_gpu` (tok/s/gpu) | **yes** |
| `pareto` | per-objective | a vector — one value per `pareto_objectives` entry | iff an objective needs it |

`e2e_latency` is the only minimized target: `OptimizationTarget.maximize` returns `False`
for it (and raises for `pareto`, which has no single direction). `score_report` negates
minimized targets so **higher is always better** internally; for a Pareto goal the raw
(unsigned) value is kept and `_dominates` applies each objective's own direction.
Missing-key defaults differ by direction: a maximized target reads `0.0` when its key is
absent, but `e2e_latency` defaults to `+inf` — so a report missing `mean_e2e_latency_ms`
scores worst-possible (`-inf` after negation) rather than best.

The `*_per_user` metric is already a rate (mean of per-token-gap `1000/itl`), so it gets
**no** GPU/time normalization — it is the InferenceX x-axis (tok/s/user).

### `avg_gpu` — the per-GPU divisor

The two `*_per_gpu` targets divide a tok/s rate by the **time-averaged provisioned GPU
count** (`score._avg_gpu`):

```
avg_gpu = gpu_hours / e2e_hours        # e2e_hours = duration_ms / 3_600_000
```

This is the integral of provisioned GPUs over the run divided by its duration:

- **static deployment** — `avg_gpu` collapses to the fixed GPU count
  (`gpu_hours = gpu_count * e2e_hours`).
- **planner-scaled run** — `avg_gpu` averages provisioned GPUs over startup + serve +
  drain.

Dividing the rate by `gpu_hours` directly would be wrong: the rate already has time
divided out. `_avg_gpu` returns `0.0` when `gpu_hours` **or** `duration_ms` is `<= 0`
(missing report keys default to `0.0`, so the guard covers both missing and non-positive
values), and the `*_per_gpu` targets then return `0.0` (divide-by-zero guard).

### SLA requirement rule

Only the **goodput** targets need an SLA — their metric
(`goodput_output_throughput_tok_s`) counts only SLA-satisfying requests (the replay
bridge's per-request goodput SLA). `_SLA_TARGETS = {goodput, goodput_per_gpu}`.

`OptimizationGoal._validate_goal` computes the *effective* objective set — `{target}` for
a scalar goal, or the resolved `pareto_objectives` for a Pareto goal — and requires an SLA
iff that set intersects `_SLA_TARGETS`. So an SLA is mandatory when:

- `target` is `goodput` or `goodput_per_gpu`, **or**
- `target` is `pareto` **and** its objectives include one of those.

A satisfying SLA is `e2e_ms`, **or** both `ttft_ms` and `itl_ms`. Note SLA is *not* gated
during feasibility (`is_feasible` checks only the GPU budget) — it lives entirely inside
the goodput metric, so an aggregate latency gate would double-count it.

`SLATarget` shape (ms, each `> 0`, `extra="forbid"`):

| field | meaning |
|---|---|
| `ttft_ms` | time-to-first-token bound — pair with `itl_ms` |
| `itl_ms` | inter-token-latency bound — pair with `ttft_ms` |
| `e2e_ms` | end-to-end bound — standalone alternative |

## Pareto

`pareto` is the one **multi-objective** target. Instead of a scalar score it optimizes the
tradeoff between the scalar targets in `pareto_objectives`.

- **objectives** — `OptimizationGoal.pareto_objectives` (`list[OptimizationTarget] | None`).
  `resolved_pareto_objectives` returns it, or the default pair
  `_DEFAULT_PARETO_OBJECTIVES = (throughput_per_gpu, throughput_per_user)` only when it is
  unset (`None`) — the **InferenceX tok/s/gpu (y) vs tok/s/user (x) frontier**. An
  explicit list is kept as-is so the validator can reject it: `>= 2` entries, distinct, no
  `pareto` among them, and `pareto_objectives` is only legal under a `pareto` target.

- **Vizier study** — the sweep declares **one metric per objective** (each with its own
  MAXIMIZE/MINIMIZE goal): `run_smart_search` passes
  `sampler_objectives = [(t.value, t.maximize) for t in resolved_pareto_objectives]` to the
  `VizierBranchSampler`, which appends a `vz.MetricInformation` per objective
  (`sampler.py`). The Vizier algorithm is `"DEFAULT"` for **every** study (scalar or
  Pareto); what changes is the *number* of metrics. With `>= 2` metrics `DEFAULT`
  (GP-UCB-PE) behaves multi-objective — optimizing the Pareto tradeoff via **hypervolume
  scalarization** — and each `observe` reports every objective's raw value in one
  measurement. (Single-objective goals declare the sampler's default single maximized
  `"objective"` metric, pre-signed by the caller.)

- **front** — `score.pareto_front` returns the **non-dominated** subset.
  `_dominates(a, b)` is true iff `a` is at least as good as `b` on **every** objective (in
  that objective's own `maximize` direction) and strictly better on at least one. The
  front is **sorted by the last objective ascending** — the x-axis — so the list traces
  the frontier left-to-right (e.g. low→high per-user throughput). `run_smart_search`
  returns this front for a Pareto goal, and `rank` (best score, ties → fewer GPUs) for
  every scalar goal.

- **swept load dimension** — `workload.concurrency` is always one fixed in-flight cap.
  A Pareto workload may instead set `kv_load_ratio: [min, max]`, which Vizier models as a
  continuous parameter. Each ratio is converted to an absolute concurrency from that
  candidate's decode/agg KV capacity, so the model compares equivalent load pressure across
  different replica and parallel configurations. If a synthetic Pareto workload omits all
  load fields, the range defaults to `[0.0, 1.0]`; see [traffic.md](traffic.md).

Per-objective raw values are stored on `Candidate.objectives` (keyed by
`OptimizationTarget` value, e.g. `{"throughput_per_gpu": .., "throughput_per_user": ..}`)
by `make_candidate`; `Candidate.score` carries the first objective's value as a headline
number only (not used for Pareto ranking). `objectives` is `None` for a scalar goal.
