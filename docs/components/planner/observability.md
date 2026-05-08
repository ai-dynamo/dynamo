---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Observability Reference
---

Complete reference for Prometheus metrics emitted by the Planner under
the `dynamo_planner_*` prefix, organised by metric family. Use this
page when building dashboards, writing alert rules, or interpreting
metric output during incident response.

## Scrape configuration

The Planner exposes its metrics endpoint on `PLANNER_PROMETHEUS_PORT`
(default `0` = disabled). Set this env var (or
`metric_reporting_prometheus_port` in `PlannerConfig`) to a positive
port to enable scraping. Prometheus then reads from
`http://<planner-pod>:<port>/metrics`.

Plugin-framework metrics (everything below the legacy "Decision /
worker / observed" section) only populate when
`scheduling.use_orchestrator: true`. On the legacy PSM path these
series stay unregistered.

## Family 1 — Decision / worker / observed (legacy, both paths)

These existed pre-PR-8 and continue to fire on both paths.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_num_prefill_replicas` | Gauge | — | Current prefill replicas |
| `dynamo_planner_num_decode_replicas` | Gauge | — | Current decode replicas |
| `dynamo_planner_predicted_num_prefill_replicas` | Gauge | — | Decided prefill replicas (final EXECUTE input) |
| `dynamo_planner_predicted_num_decode_replicas` | Gauge | — | Decided decode replicas |
| `dynamo_planner_observed_ttft_ms` | Gauge | — | Observed TTFT from cluster Prometheus |
| `dynamo_planner_observed_itl_ms` | Gauge | — | Observed ITL |
| `dynamo_planner_observed_requests_per_second` | Gauge | — | Observed RPS |
| `dynamo_planner_load_scaling_decision` | Enum | state | Active load-scaling state name (see [Decision states](#decision-states)) |
| `dynamo_planner_throughput_scaling_decision` | Enum | state | Active throughput-scaling state name |
| `dynamo_planner_estimated_ttft_ms` | Gauge | — | Regression-estimated TTFT |
| `dynamo_planner_estimated_itl_ms` | Gauge | — | Regression-estimated ITL |
| `dynamo_planner_engine_queued_*` | Gauge | worker_id, dp_rank | Per-engine queue depths from FPM |
| `dynamo_planner_gpu_hours` | Gauge | — | Cumulative GPU hours consumed |

### Decision states

`dynamo_planner_load_scaling_decision` and
`dynamo_planner_throughput_scaling_decision` are Prometheus `Enum`
gauges — the active state is set to `1`, all others to `0`. The state
names form a stable contract; new values are appended (never inserted
or reordered) so older scrapers keep parsing.

**Load-scaling states** (`LOAD_DECISION_STATES`):

```
unset, disabled, no_fpm_data, scaling_in_progress, worker_count_mismatch,
insufficient_data, no_change, scale_up, scale_down,
scale_down_capped_by_throughput, override_by_user_plugin,
reconcile_clamped_to_floor, reconcile_clamped_to_ceiling,
held_over, rejected_by_plugin
```

**Throughput-scaling states** (`THROUGHPUT_DECISION_STATES`):

```
unset, disabled, no_traffic_data, predict_failed, model_not_ready,
set_lower_bound, scale, override_by_user_plugin, held_over,
circuit_open, rejected_by_plugin
```

> **Orchestrator-path parity for throughput accept-path** (PR 8 A2):
> pre-A2 only the PSM path populated
> `dynamo_planner_throughput_scaling_decision` with `set_lower_bound` /
> `scale` — the orchestrator path stayed at `unset` even when
> `BuiltinThroughputPropose` accepted with a real decision, making
> dashboards observability-blind on the path that PR 11 will leave as
> the only one. A2 wires the same vocabulary through
> `BuiltinThroughputPropose._last_throughput_diagnostics` →
> `OrchestratorEngineAdapter._project_throughput_diagnostics`.
> Decision outputs remain byte-equal across paths (locked by
> `tests/integration/test_dual_path_parity.py`); only the
> observability surface changed.

## Family 2 — Plugin lifecycle (orchestrator path only)

Per-plugin invocation accounting. One series per (plugin, stage)
combination so dashboards can decompose pipeline cost per plugin.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_plugin_evaluations_total` | Counter | `plugin_id`, `stage`, `result` | Plugin call count, classified by outcome |
| `dynamo_planner_plugin_latency_seconds` | Histogram | `plugin_id`, `stage` | RPC latency for successful calls. Buckets: 1ms, 5ms, 10ms, 50ms, 100ms, 500ms, 1s, 5s |
| `dynamo_planner_plugin_circuit_state` | Gauge | `plugin_id` | Circuit breaker state: `0`=closed, `0.5`=half-open, `1`=open |
| `dynamo_planner_plugin_held_over_total` | Counter | `plugin_id`, `stage` | HOLD_LAST cache replays (plugin not due, prior result inherited) |
| `dynamo_planner_plugin_cache_age_seconds` | Gauge | `plugin_id` | Age of the HOLD_LAST cached result |
| `dynamo_planner_plugin_override_active` | Gauge | `plugin_id`, `stage`, `override_type` | `1` if the plugin contributed an override of the named type this tick, else `0`. `override_type ∈ {SET, AT_LEAST, AT_MOST, REJECT}` |

### `result` label values for `plugin_evaluations_total`

| Value | Meaning |
|---|---|
| `accept` | Plugin returned ACCEPT — no opinion, pipeline continues |
| `set` / `at_least` / `at_most` | Plugin returned an OverrideResult of the named type |
| `reject` | Plugin returned RejectResult — stage short-circuits |
| `held_over` | Cache replay (no actual call this tick) |
| `error` | Transport error / unhandled exception during call |
| `timeout` | Per-plugin `request_timeout_seconds` exceeded |

## Family 3 — Reconcile / Constrain behaviour

Tracks merge-stage clamping and short-circuits. Useful for catching
"plugin A's output keeps getting overridden by plugin B".

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_reconcile_clamped_total` | Counter | `sub_component_type`, `component_name`, `source` | RECONCILE result was clamped by a floor (AT_LEAST) or ceiling (AT_MOST) plugin. `source` is the plugin_id of whichever bound actually won the clamp |
| `dynamo_planner_constrain_capped_total` | Counter | `sub_component_type`, `component_name`, `source` | Same semantics, fired by the CONSTRAIN stage instead. Expected primary contributor: `builtin_budget_constrain` |
| `dynamo_planner_reject_short_circuited_total` | Counter | `plugin_id` | A REJECT triggered a stage short-circuit; pipeline aborted before EXECUTE |

## Family 4 — GlobalPlanner

Emitted by both the Planner-side `GlobalPlannerConnector` (client) and
the GlobalPlanner-side `ScaleRequestHandler` (server). The `reason`
label distinguishes the two viewpoints.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_global_scale_request_total` | Counter | `result`, `reason` | Scale-request RPC count |
| `dynamo_planner_global_scale_request_latency_seconds` | Histogram | `result` | RPC latency. Buckets up to 60s for loaded-cluster K8s apiserver tail |
| `dynamo_planner_global_managed_dgd_gpus` | Gauge | `dgd_name` | GPUs currently allocated to each tracked DGD |

### `reason` label values

| Value | Source | Meaning |
|---|---|---|
| `client` | Planner connector | Client-side RTT view (success/error from caller's POV) |
| `success` | GlobalPlanner handler | Scale applied successfully |
| `not_authorized` | GlobalPlanner handler | Caller namespace not in `--managed-namespaces` |
| `no_operation` | GlobalPlanner handler | Handler running with `--no-operation` (logged-only) |
| `budget_exceeded` | GlobalPlanner handler | Request would exceed `--max-total-gpus` |
| `exception` | GlobalPlanner handler | Unhandled error during scaling |

## Family 5 — EXECUTE stage (both paths)

These fire from `NativePlannerBase._apply_scaling_targets`, which both
PSM and orchestrator paths funnel through, so dashboards see scaling
activity regardless of which engine produced the decision.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_execute_total` | Counter | `result` | EXECUTE outcomes: `success`, `error`, `advisory`, `skipped_no_change`, `skipped_rejected`, `in_progress`, `skipped_connector_blocked` |
| `dynamo_planner_execute_latency_seconds` | Histogram | `result` | Connector call latency. Buckets up to 15s for loaded apiserver |
| `dynamo_planner_execute_skip_reason_total` | Counter | `reason` | Finer breakdown when `result` is a skip kind. Values: `no_scale_to`, `advisory`, `no_change`, `rejected`, `in_progress`, `deployment_not_ready` (and any other `ConnectorBusyError.reason` strings) |

> **`skipped_connector_blocked` watch tip**: a sustained non-zero rate
> here means the *backend*, not the planner, is wedged. Pre-A1, the
> K8s connector silently `return`-ed when the DGD reported
> `Ready=False` — `execute_total{result=success}` stayed at 100%
> while no scaling actually happened. The connector now raises
> `ConnectorBusyError`, the funnel records this label, and ops can
> alert directly on it. Common cause: stale Ready condition on the
> DGD object due to the operator-cache lag.

## Family 6 — Tick scheduling (orchestrator path only)

Tick-loop health metrics. Use these to size `tick_max_duration_seconds`
and detect plugin-induced lag.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `dynamo_planner_tick_skipped_total` | Counter | `plugin_id` | Plugin skipped this tick (execution_interval not elapsed) |
| `dynamo_planner_tick_lag_seconds` | Gauge | `plugin_id` | Seconds past the plugin's scheduled "due" moment when it actually evaluated |
| `dynamo_planner_tick_duration_seconds` | Histogram | — | Total tick time (PREDICT + PROPOSE + RECONCILE + CONSTRAIN). Buckets up to 30s (default `tick_max_duration_seconds`) |
| `dynamo_planner_tick_timeout_total` | Counter | — | Ticks that exceeded `tick_max_duration_seconds` |

## Pre-built Grafana dashboard

A ready-to-import dashboard covering all six families above lives at
[`deploy/observability/grafana_dashboards/dynamo-planner.json`](https://github.com/ai-dynamo/dynamo/blob/main/deploy/observability/grafana_dashboards/dynamo-planner.json)
(repo path
`deploy/observability/grafana_dashboards/dynamo-planner.json`). It
ships 27 timeseries / stat / state-timeline panels organised by
family, with templated `namespace` / `pod` / `plugin_id` variables so
the same dashboard works across deployments.

Recommended use during the `scheduling.use_orchestrator` flag-flip
canary: the **EXECUTE outcomes** row is the primary regression
detector — pre-A1 deployments would show 100% `result=success` even
when the K8s connector was silently dropping every command, so a
sustained non-zero `skipped_connector_blocked` rate indicates the
backend (operator / DGD Ready cache / apiserver) is wedged, not the
planner. The **Plugin circuit state** row similarly catches external
plugin failures before they manifest as decision drift.

Import: `Dashboards → New → Import → Upload JSON file`. Pick a
Prometheus datasource when prompted; no other configuration needed.

## Suggested PromQL recipes

**P99 plugin latency by plugin**:
```promql
histogram_quantile(0.99,
  sum by (plugin_id, stage, le) (rate(dynamo_planner_plugin_latency_seconds_bucket[5m]))
)
```

**Plugin error rate**:
```promql
sum by (plugin_id) (
  rate(dynamo_planner_plugin_evaluations_total{result=~"error|timeout"}[5m])
)
/ sum by (plugin_id) (
  rate(dynamo_planner_plugin_evaluations_total[5m])
)
```

**Tick budget headroom** (1.0 = at deadline):
```promql
histogram_quantile(0.99,
  sum by (le) (rate(dynamo_planner_tick_duration_seconds_bucket[5m]))
) / 30
```

**GPU budget utilisation across managed DGDs**:
```promql
sum(dynamo_planner_global_managed_dgd_gpus)
```

**Active circuit breakers**:
```promql
dynamo_planner_plugin_circuit_state == 1
```

## Migration notes (PSM → orchestrator path)

When operators flip `scheduling.use_orchestrator: true`, the metric
surface changes as follows:

- **Family 1 fields stay stable** — `num_*_replicas`, `observed_*`,
  `predicted_num_*_replicas`, `gpu_hours`, the two scaling-decision
  Enum gauges. PSM and orchestrator both populate these.
- **`estimated_ttft_ms` / `estimated_itl_ms`**: PSM populates from
  regression output; orchestrator populates the same way as long as
  the regression accumulated enough samples. Pre-deployment profiling
  (`pre_deployment_sweeping_mode` ≠ `none`) seeds both paths
  equivalently.
- **Family 2/3/6 are new** — orchestrator-only. PSM path leaves these
  series unregistered. Build alerts on `absent()` if you need to
  detect a flag flip back to PSM.
- **Family 4 (GlobalPlanner)** fires on both paths when `environment:
  global-planner` is set and the connector is wired to send scale
  requests. PSM path emits only `reason=client`; orchestrator emits
  `reason=client` plus the server-side reasons after the handler
  receives the request.
- **Family 5 (EXECUTE)** fires on both paths uniformly — the funnel
  is in `NativePlannerBase`, not in the engine.

## See also

- [Audit events catalog](audit-events.md) — structured log events with
  stable field names, complementary to these counters
- [Orchestrator rollout runbook](orchestrator-rollout.md) — staged
  flag-flip procedure including a "what new metrics to watch" section
