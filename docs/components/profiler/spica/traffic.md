---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Spica Traffic
subtitle: Trace, request-rate, concurrency, and KV-load workload definitions
---

> [!WARNING]
> **Experimental.** Spica is intended for evaluation and feedback, not production capacity
> planning. Its API, configuration schema, search results, and deployment output may change
> without a standard deprecation period. Spica provides no SLA, accuracy, or configuration
> optimality guarantees.

The `workload:` block of a `SmartSearchConfig` YAML is the traffic **every candidate is
replayed against**. Most workload fields are pinned. The exception is a ranged
`kv_load_ratio` under a `pareto` goal: Vizier searches it as a continuous load dimension.
The block maps to `Workload` in `components/src/dynamo/profiler/spica/config.py`.

A workload is **exactly one of four load shapes**. The shape is inferred from which field
is set (`Workload._validate_workload`), and each shape is either **open-loop** (requests
arrive on a clock, independent of how fast the system drains them) or **closed-loop** (a
fixed number of requests are kept in flight; a new one starts as one finishes).

## The four load shapes

| # | Shape | Set | Loop | Driven by |
|---|---|---|---|---|
| 1 | **mooncake trace** | `trace_path` | open-loop (default) | the trace's arrival timestamps, scaled by `arrival_speedup_ratio` |
| 1c | **mooncake trace, capped** | `trace_path` + `replay_concurrency` | closed-loop | cap N in flight; **trace timestamps ignored** |
| 2 | **synthetic request-rate** | `request_rate` (+ `isl`/`osl`/`num_request_ratio`) | open-loop | a fixed QPS (`synthetic_arrival_interval_ms = 1000 / request_rate`) |
| 3 | **synthetic concurrency** | `concurrency` (+ `isl`/`osl`/`num_request_ratio`) | closed-loop | cap N in flight |
| 4 | **synthetic KV load** | `kv_load_ratio` (+ `isl`/`osl`/`num_request_ratio`) | closed-loop | derive N from each candidate's aggregate decode/agg KV capacity |

`is_trace_based` is `trace_path is not None`; `is_synthetic` is its complement. Setting
`trace_path` selects shape 1; otherwise it is synthetic and **exactly one** of
`request_rate` (shape 2), `concurrency` (shape 3), or `kv_load_ratio` (shape 4) selects the
sub-shape.

The closed-loop in-flight cap is resolved by `effective_in_flight_cap()` (`None` = open-loop):

- trace -> `replay_concurrency` (so a trace is open-loop unless `replay_concurrency` is set);
- synthetic KV load -> the candidate-derived `concurrency_override`;
- synthetic fixed `concurrency` -> that value;
- request-rate -> `None` (open-loop).

## Fields

Every `Workload` field:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `isl` | `int \| None` | `None` | Synthetic input (prompt) sequence length, tokens. Required for synthetic. |
| `osl` | `int \| None` | `None` | Synthetic output sequence length, tokens. Required for synthetic. |
| `concurrency` | `int \| None` | `None` | Fixed positive closed-loop in-flight cap (shape 3). It is always scalar, including under a Pareto goal. |
| `kv_load_ratio` | `float \| list[float] \| None` | Pareto default: `[0.0, 1.0]` when no other load is set | Candidate-relative closed-loop load (shape 4). A scalar pins the load for any goal; a two-value `[min, max]` range is a continuous Vizier dimension and is allowed only under a Pareto goal. Values are non-negative; a user may set a maximum above `1` to search oversubscription. |
| `request_rate` | `float \| None` | `None` | Open-loop QPS (shape 2). Mutually exclusive with `concurrency` and `kv_load_ratio`. |
| `num_request_ratio` | `float \| None` | `None` | Synthetic request count relative to the load: `num_requests = round(num_request_ratio * load)`. Required for synthetic. See below. |
| `shared_prefix_ratio` | `float` | `0.0` | Fraction of shared prefix across requests (cache-locality / prefix sharing). |
| `num_prefix_groups` | `int` | `0` | Number of distinct shared-prefix groups. |
| `turns_per_session` | `int` | `1` | Turns per multi-turn session. |
| `inter_turn_delay_ms` | `float` | `0.0` | Think-time between turns in a multi-turn synthetic session, ms. |
| `trace_path` | `str \| None` | `None` | Path to a replay trace (shape 1). Its presence selects the trace shape and **forbids** all synthetic fields. |
| `trace_format` | `str` | `"mooncake"` | Replay-ready trace schema. Decoded but **not** forwarded by the evaluator — the trace path is read as mooncake regardless; effectively inert today. |
| `arrival_speedup_ratio` | `float` | `1.0` | Scales the trace's inter-arrival times (open-loop trace only). `>1` speeds arrivals up. |
| `replay_concurrency` | `int \| None` | `None` | Closed-loop in-flight cap **for a trace** (shape 1c); when set, trace timestamps are ignored. For synthetic closed-loop use `concurrency` instead. |

The synthetic fields are `isl`, `osl`, `request_rate`, `concurrency`, `kv_load_ratio`,
`num_request_ratio`;
`shared_prefix_ratio`, `num_prefix_groups`, `turns_per_session`, `inter_turn_delay_ms` are
shared synthetic knobs threaded into the replay (`ReplayEvaluator._synthetic_kwargs`).

## `kv_load_ratio` (candidate-relative concurrency)

> [!WARNING]
> `kv_load_ratio` requires an AI Configurator release that provides
> `aiconfigurator.sdk.memory`. The default Planner/Profiler image currently retains AI Configurator
> 0.9, so this workload mode fails fast before search starts in that image. Use a trace workload
> or fixed `concurrency` with the default image.

Spica resolves a KV-load trial after the backend, parallel shape, replicas, and batching
knobs have been selected. For every active role, it asks AI Configurator for the **per-rank** KV token
capacity using that candidate's `max_num_batched_tokens`, `max_num_seqs`, memory fraction,
parallel shape, and MTP setting. The scheduler-visible role capacity is:

```text
per_rank_usable_tokens = floor(per_rank_tokens / block_size) * block_size
role_capacity_tokens = per_rank_usable_tokens * attention_dp * replicas
```

Attention-DP ranks own independent sequence pools, so capacity is multiplied by
`attention_dp`; TP/EP ranks shard the same sequences and are not multipliers. For disagg,
both prefill and decode are checked for candidate-specific memory feasibility, but only
**decode** capacity drives load. For agg, **agg** capacity drives load.

The concrete closed-loop cap is:

```text
average_tokens_per_request = isl + floor(osl / 2)
capacity_concurrency = floor(role_capacity_tokens / average_tokens_per_request)
concurrency = max(1, floor(kv_load_ratio * capacity_concurrency))
```

`kv_load_ratio = 0` therefore maps to the minimum concurrency `1`; `1` means estimated
100% steady-state KV occupancy. It is an estimate, not a guarantee that replay sees no
temporary KV pressure or request retraction. Batching combinations that leave no KV budget
are reported to Vizier as infeasible before replay.

Every resulting candidate records `kv_load_ratio`, the derived `concurrency`,
`kv_load_concurrency_capacity`, `kv_load_capacity_tokens`, and per-role
`*_kv_capacity_tokens` for traceability.

## `num_request_ratio` (synthetic length scales with the load)

`resolved_request_count(concurrency_override=None)` computes the synthetic request count as

```
num_requests = max(1, round(num_request_ratio * load))
```

where `load` is, in precedence order: the candidate-derived `concurrency_override` (KV-load
mode), else fixed `concurrency` (closed-loop), else `request_rate` (open-loop).

So the synthetic trace length **scales with the swept load automatically**: with
`num_request_ratio = 10`, concurrency `256` yields `2560` requests, concurrency `512`
yields `5120`. Result is floored at `1`; `num_request_ratio` itself is treated as `0.0`
when unset (`max(1, …)` keeps at least one request).

## Validation (`Workload._validate_workload`)

- **Trace workload** (`trace_path` set): must **not** set any synthetic field
  (`isl`, `osl`, `request_rate`, `concurrency`, `kv_load_ratio`, `num_request_ratio`) — error lists the
  offenders. `replay_concurrency`, if set, must be a positive int.
- **Synthetic workload** (no `trace_path`): **exactly one** of `request_rate`,
  `concurrency`, or `kv_load_ratio` (none / multiple -> error); `isl`, `osl`,
  `num_request_ratio` are all
  **required**; `replay_concurrency` is rejected (it is trace-only — use `concurrency`).
  `concurrency` must be one positive int; `request_rate`, `isl`, `osl`, and
  `num_request_ratio` must be positive. KV-load values must be finite and non-negative.
- **Ranged KV load only under Pareto** — `[min, max]` must contain exactly two values with
  `min < max`; `SmartSearchConfig._validate_kv_load_ratio_range` rejects it for scalar
  goals. A scalar `kv_load_ratio` is valid for every goal. A synthetic Pareto config that
  omits all three load fields defaults to `kv_load_ratio: [0.0, 1.0]`.

## Replay routing (from `evaluator.py`)

Each shape × deployment case routes to a Dynamo Replay entry point; all emit the same flat
`trace_report` dict. `ReplayEvaluator.evaluate` branches on `is_trace_based`, then on
`plan.is_static`:

| Load | static (no planner) | planner-in-the-loop |
|---|---|---|
| **mooncake trace** | `dynamo.replay.api.run_trace_replay(..., planner_config=None)` | `run_trace_replay(..., planner_config=<dict>)` |
| **synthetic** (rate, fixed concurrency, or KV load) | `dynamo.replay.api.run_synthetic_trace_replay(..., planner_config=None)` | `run_synthetic_trace_replay(..., planner_config=<dict>)` |

Notes:

- The closed-loop cap passed as `replay_concurrency=` on every path is
  `effective_in_flight_cap()` — `replay_concurrency` for a trace, fixed `concurrency`, or
  the KV-load-derived per-trial `concurrency_override` for synthetic.
- The **goodput SLA** (`goal.sla`) is passed as `sla_ttft_ms` / `sla_itl_ms` / `sla_e2e_ms`
  on every path **only when an SLA is configured** — `_goodput_sla_kwargs` returns `{}` when
  `goal.sla is None`, so no `sla_*` kwargs are passed and no goodput is computed. It is
  independent of the planner's own scaling SLA.
- Under `kv_router` the searched router weights become a real `KvRouterConfig`;
  `round_robin` passes `router_config=None`.
