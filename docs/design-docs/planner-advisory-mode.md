<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Planner Advisory Mode Design

## 1. Background and Motivation

In production inference environments, enabling Planner auto-scaling directly carries risk.
Customers need a way to observe the Planner's decisions, evaluate their
correctness, and build confidence before turning on automatic execution.

The current `no_operation` mode simply skips scaling with no output — it provides
zero observability into what the Planner *would have done*. This makes it useless
for pre-production validation.

### Use Cases

| Use Case | Description |
|----------|-------------|
| **Pre-production validation** | Deploy Planner in advisory mode alongside a new model. Observe recommendations for days/weeks before enabling active scaling. |
| **Ongoing auditing** | Run advisory output in active mode so every executed scaling action is accompanied by its full decision context for post-incident review. |
| **Customer demo / PoC** | Show customers how the Planner would behave on their traffic without touching their running workloads. |

---

## 2. Three-Mode Architecture

Replace the boolean `no_operation` flag with a `ScalingMode` enum.

```python
class ScalingMode(str, Enum):
    ACTIVE = "active"
    ADVISORY = "advisory"
    NOOP = "noop"
```

### Mode Behavior Matrix

| Capability | active | advisory | noop |
|------------|--------|----------|------|
| Initialize Connector (read cluster state) | Y | Y | N |
| Execute scaling (`set_component_replicas`) | Y | N | N |
| Emit Advisory Prometheus metrics | Y | Y | N |
| Emit structured advisory logs | Y | Y | N |
| Write advisory JSONL file | Y | Y | N |

**Why advisory mode initializes the Connector**: Advisory mode must call
`get_workers_info()` and `get_actual_worker_counts()` via `KubernetesConnector` to
know the current replica counts. Without this, `advisory_current_p/d` and all delta
metrics would be zero, making the output meaningless. The Connector is used in
read-only mode — `_apply_scaling()` and `_apply_scaling_blocking()` are gated on
`scaling_mode == ACTIVE`.

**Why active mode also emits advisory metrics and structured logs**: Every executed
scaling decision should be traceable. When an incident occurs, SREs can query the
advisory metrics and search structured logs to see the full decision context
(predicted load, estimated SLA impact, GPU budget constraints) that led to the
scaling action.

### Connector Initialization Logic

```python
# In BasePlanner.__init__ and DisaggPlanner.__init__
init_connector = config.scaling_mode in (ScalingMode.ACTIVE, ScalingMode.ADVISORY)

if init_connector:
    if config.environment == "global-planner":
        self.connector = GlobalPlannerConnector(...)
    elif config.environment == "kubernetes":
        self.connector = KubernetesConnector(...)
    elif config.environment == "virtual":
        self.connector = VirtualConnector(...)
```

### Scaling Execution Gate

```python
# In BasePlanner._apply_scaling / _apply_scaling_blocking
async def _apply_scaling(self, desired_replicas: int) -> None:
    if self.config.scaling_mode != ScalingMode.ACTIVE:
        return
    # ... existing K8s scaling logic
```

### Backward Compatibility

```python
# In PlannerConfig model_validator
@model_validator(mode="after")
def _validate_config(self) -> "PlannerConfig":
    if self.no_operation and self.scaling_mode == ScalingMode.ACTIVE:
        logger.warning(
            "DEPRECATION: no_operation=True is deprecated. "
            "Use scaling_mode='noop' instead. "
            "Automatically mapping no_operation=True to scaling_mode='noop'."
        )
        self.scaling_mode = ScalingMode.NOOP
    # ...
```

Mapping rules:
- `no_operation: true` + no `scaling_mode` set → `scaling_mode: "noop"` (with deprecation warning)
- `no_operation: false` (default) + no `scaling_mode` set → `scaling_mode: "active"` (current behavior, no warning)
- Explicit `scaling_mode` always takes precedence over `no_operation`

---

## 3. Advisory Engine — Three-Layer Output

The Advisory Engine is a set of methods on `BasePlanner` that run in `active` and
`advisory` modes. It is invoked after replica computation but before scaling execution.

```
┌──────────────────────────────────────────────────────────┐
│              Planner Core (all modes share)               │
│                                                          │
│  Prometheus ──→ Observe Traffic ──→ Predict Load          │
│                                       │                  │
│                              Compute Replicas            │
│                                       │                  │
│                            Apply GPU Budget              │
│                                       │                  │
│                         ┌─────────────┴─────────────┐    │
│                         ▼                           ▼    │
│              Advisory Engine               Scaling Gate  │
│          (active + advisory)            (active only)    │
│                   │                           │          │
│        ┌──────────┼──────────┐                ▼          │
│        ▼          ▼          ▼       set_component_      │
│    Layer 1    Layer 2    Layer 3     replicas()           │
│   Metrics     Logs      SLA Est                          │
└──────────────────────────────────────────────────────────┘
```

### Layer 1: Scaling Recommendation (Prometheus Metrics)

Updated every adjustment interval. These metrics are the primary input for
the Grafana Advisory Dashboard.

**Gauges (12):**

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_planner_advisory_recommended_p` | Gauge | Final recommended prefill replicas (after GPU budget) |
| `dynamo_planner_advisory_recommended_d` | Gauge | Final recommended decode replicas (after GPU budget) |
| `dynamo_planner_advisory_current_p` | Gauge | Current actual prefill replicas from DGD |
| `dynamo_planner_advisory_current_d` | Gauge | Current actual decode replicas from DGD |
| `dynamo_planner_advisory_delta_p` | Gauge | Prefill delta (recommended - current). Positive = scale up |
| `dynamo_planner_advisory_delta_d` | Gauge | Decode delta (recommended - current). Positive = scale up |
| `dynamo_planner_advisory_scaling_action` | Gauge | Aggregate action: 1=scale up, 0=hold, -1=scale down |
| `dynamo_planner_advisory_action_reason` | Gauge | Reason code (see table below) |
| `dynamo_planner_advisory_est_ttft` | Gauge | Estimated TTFT after applying recommendation (ms). NaN if no profiling data |
| `dynamo_planner_advisory_est_itl` | Gauge | Estimated ITL after applying recommendation (ms). NaN if no profiling data |
| `dynamo_planner_advisory_ttft_headroom` | Gauge | TTFT SLA target - estimated TTFT (ms). Positive = safe |
| `dynamo_planner_advisory_itl_headroom` | Gauge | ITL SLA target - estimated ITL (ms). Positive = safe |

**Counters (3):**

| Metric | Type | Description |
|--------|------|-------------|
| `dynamo_planner_advisory_scaleup_total` | Counter | Cumulative scale-up recommendations |
| `dynamo_planner_advisory_scaledown_total` | Counter | Cumulative scale-down recommendations |
| `dynamo_planner_advisory_hold_total` | Counter | Cumulative hold recommendations |

**Total: 15 new metrics.**

Note: Prometheus counters reset to zero on Pod restart. Grafana queries must use
`rate()` or `increase()`, never raw counter values.

**Action Reason Codes:**

| Code | Meaning |
|------|---------|
| 0 | No traffic (metrics invalid, skipped) |
| 1 | Throughput prediction: predicted load exceeds current capacity |
| 2 | Throughput prediction: predicted load below current capacity |
| 3 | Load-based: all workers above SLA threshold |
| 4 | Load-based: all workers below scale-down boundary |
| 5 | GPU budget constraint applied (raw recommendation was higher) |
| 6 | Hold: no change needed |

In v1, reason codes provide a quick signal in dashboards. Detailed reasoning
is in the structured logs (Layer 2).

> **Note (v1 implementation status):** Reason codes 0 (no traffic) and 5 (GPU
> budget constraint) are reserved for future use and are not emitted in the
> current implementation. Only codes 1–4 and 6 are active.

**Naming convention**: Advisory metrics use `dynamo_planner_advisory_*` (underscore-separated)
following the Dynamo naming guideline in `lib/runtime/src/metrics/prometheus_names.rs`.
Existing Planner metrics (`planner:num_p_workers`, etc.) retain their current naming
for backward compatibility and will be migrated in a follow-up PR.

### Layer 2: Execution Path (Structured Logs)

Output only in `advisory` mode. Provides human-readable and machine-parseable
decision context.

**Throughput-based recommendation log:**

```json
{
  "event": "advisory_recommendation",
  "scaling_mode": "advisory",
  "source": "throughput",
  "action": "scale_up",
  "current": {"prefill": 1, "decode": 1},
  "recommended_raw": {"prefill": 3, "decode": 1},
  "recommended_final": {"prefill": 3, "decode": 1},
  "gpu_budget_applied": false,
  "predicted_load": {"req_rate": 15.2, "isl": 3200, "osl": 180},
  "engine_capacity": {"prefill_thpt_per_gpu": 5800, "decode_thpt_per_gpu": 1205},
  "path": ["1P1D", "2P1D (observe 1 interval)", "3P1D"],
  "est_ttft_ms": 85.2,
  "est_itl_ms": 12.1,
  "ttft_headroom_ms": 414.8,
  "itl_headroom_ms": 37.9,
  "risk": "During scale-up only 1 prefill GPU serves traffic; TTFT may degrade"
}
```

**Load-based recommendation log:**

```json
{
  "event": "advisory_recommendation",
  "scaling_mode": "advisory",
  "source": "load",
  "action": "scale_up",
  "current": {"prefill": 2, "decode": 1},
  "recommended_final": {"prefill": 3, "decode": 1},
  "trigger": "all_workers_above_target",
  "worker_metrics": {
    "worker_0": {"active_prefill_tokens": 12800},
    "worker_1": {"active_prefill_tokens": 13100}
  },
  "target_threshold": 10240,
  "regression": {"slope": 0.0042, "intercept": 15.3, "observations": 47}
}
```

**GPU budget constraint log (appended when budget clips the recommendation):**

```json
{
  "event": "advisory_gpu_budget",
  "raw_recommendation": {"prefill": 5, "decode": 3, "total_gpus": 8},
  "budget_limit": 6,
  "final_recommendation": {"prefill": 3, "decode": 2, "total_gpus": 5},
  "action_needed": "Increase max_gpu_budget or add GPUs to serve predicted load"
}
```

All logs use Python's `logging` with `extra` dict fields. This is compatible
with Dynamo's existing `configure_dynamo_logging()` framework and integrates
with ELK, Grafana Loki, and other log aggregation systems via JSON formatters.

### Layer 3: SLA Estimation

Uses `PrefillInterpolator.interpolate_ttft()` and
`DecodeInterpolator.find_best_throughput_per_gpu()` to estimate what TTFT and ITL
would be if the recommended replica count were in effect.

**Graceful degradation when profiling data is unavailable:**

```python
def _estimate_sla_with_replicas(self, num_p: int, num_d: int) -> dict:
    if not self.enable_throughput:
        # Load-only mode: no profiling data available for estimation
        if self.prometheus_metrics:
            self.prometheus_metrics.advisory_est_ttft.set(float('nan'))
            self.prometheus_metrics.advisory_est_itl.set(float('nan'))
            self.prometheus_metrics.advisory_ttft_headroom.set(float('nan'))
            self.prometheus_metrics.advisory_itl_headroom.set(float('nan'))
        logger.info(
            "[ADVISORY] SLA estimation unavailable: "
            "throughput scaling disabled (no profiling data)"
        )
        return {"est_ttft": None, "est_itl": None}
    # ... estimation using interpolators
```

Grafana panels handle NaN values by displaying "N/A" instead of broken lines.

All SLA estimation outputs carry a disclaimer in logs:
`"note": "Estimation based on profiling data; actual values may differ due to runtime conditions"`

---

## 4. Robustness and Safety Guards

These safeguards are essential for production deployment where metrics may be
intermittent, Planner Pods may restart, and cluster state may be inconsistent.

### Metrics Validity Gate

```python
def _emit_advisory_metrics(self, recommended_p, recommended_d, source, reason_code):
    if not self.last_metrics.is_valid():
        logger.info(
            "[ADVISORY] Skipping advisory output: "
            "metrics contain None/NaN (no active traffic or Prometheus unreachable)"
        )
        return
```

**Why**: At Planner startup or during Prometheus outages, `last_metrics` contains
None/NaN values. Emitting advisory recommendations based on invalid data would
mislead operators.

### Non-Negative Replica Guard

```python
    recommended_p = max(0, recommended_p)
    recommended_d = max(0, recommended_d)
```

**Why**: Edge cases in prediction (negative predictions from ARIMA, division
issues) must never produce negative replica recommendations.

### Anomaly Detection

```python
    delta_p = recommended_p - self.shared_state.num_p_workers
    delta_d = recommended_d - self.shared_state.num_d_workers
    threshold = self.config.advisory_anomaly_threshold

    if abs(delta_p) > threshold or abs(delta_d) > threshold:
        logger.warning(
            f"[ADVISORY] Unusually large delta: delta_p={delta_p}, delta_d={delta_d} "
            f"(threshold={threshold}). Possible metrics jump or configuration error."
        )
```

**Why**: A recommendation to go from 1 to 50 replicas is almost certainly a bug
or a metrics anomaly. The warning gives operators an early signal to investigate.
The threshold is configurable via `advisory_anomaly_threshold` (default: 10).

### NaN Protection on Gauge Writes

```python
def _safe_gauge_set(gauge, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        gauge.set(float('nan'))
    else:
        gauge.set(value)
```

**Why**: Some estimation paths may produce None (e.g., SLA estimation without
profiling data). Direct `.set(None)` would raise an exception.

### Load-Based Blocking Prevention

In advisory mode, the load-based loop must NOT call `_apply_scaling_blocking()`
(which waits for deployment readiness). This would stall the entire load loop
indefinitely since no actual scaling occurs.

```python
# In _load_loop and DisaggPlanner._load_loop:
if self.config.scaling_mode == ScalingMode.ACTIVE:
    await self._apply_scaling_blocking(desired_replicas)
# advisory mode: emit metrics only, no blocking
```

---

## 5. Data Output Channels

Four output channels ensure advisory data is accessible regardless of the
customer's monitoring infrastructure.

| Channel | Format | Retention | Use Case |
|---------|--------|-----------|----------|
| Prometheus metrics | Gauge / Counter | 15-30 days (cluster config) | Real-time Grafana dashboards, alerting |
| Structured logs | JSON in Python `logging` `extra` | Depends on log pipeline | Searchable via ELK / Grafana Loki |
| JSONL file | Append-only at `log_dir` path | Permanent (disk) | Long-term audit trail, offline analysis |
| HTTP endpoint | JSON on Prometheus HTTP port | Current snapshot only | Quick `curl` check, custom tool integration |

### JSONL File Output

When `advisory_file_output: true` and `log_dir` is set, each advisory cycle
appends one JSON line:

```
File: {log_dir}/advisory_history.jsonl

{"ts":1741234567,"mode":"advisory","action":"scale_up","source":"throughput",
 "current":{"p":1,"d":1},"recommended":{"p":3,"d":1},"raw":{"p":3,"d":1},
 "est_ttft":85.2,"est_itl":12.1,"reason_code":1}
```

> **Note (v1 implementation status):** The `raw` block (pre-budget-clipping
> recommendation) is not yet included in the JSONL output. Currently only
> `current` and `recommended` (post-budget) are emitted.

**Why a separate file instead of just logs**: Prometheus has a retention window
(typically 15-30 days). Log pipelines may also rotate. The JSONL file provides a
permanent, simple-to-parse record that survives both. It reuses the existing
`log_dir` config field (currently defined in `PlannerConfig` but unused).

### HTTP Endpoint

A lightweight JSON endpoint served on the existing Planner Prometheus HTTP port:

```
GET /advisory/status

Response:
{
  "scaling_mode": "advisory",
  "last_update": "2026-03-05T10:30:00Z",
  "current": {"prefill": 1, "decode": 1},
  "recommended": {"prefill": 3, "decode": 1},
  "delta": {"prefill": 2, "decode": 0},
  "action": "scale_up",
  "reason": "throughput_prediction",
  "sla_estimation": {
    "est_ttft_ms": 85.2,
    "est_itl_ms": 12.1,
    "ttft_headroom_ms": 414.8,
    "itl_headroom_ms": 37.9
  }
}
```

**Why**: Not all customers have Prometheus/Grafana. A `curl` command against the
Planner Pod gives instant visibility. Also enables integration with customer-side
monitoring tools via simple HTTP polling.

---

## 6. Disagg Mode Specifics

`DisaggPlanner` manages both prefill and decode in a single loop and applies
`_apply_global_gpu_budget()` as a joint constraint. The advisory engine must
respect this coupling.

### Advisory Output Placement

```
DisaggPlanner._throughput_loop():
  1. observe_traffic_stats()
  2. prefill_planner.plan_adjustment() -> raw_p
  3. decode_planner.plan_adjustment()  -> raw_d
  4. _apply_global_gpu_budget(raw_p, raw_d) -> final_p, final_d
  5. >>> _emit_disagg_advisory(raw_p, raw_d, final_p, final_d)  <<<
  6. if scaling_mode == ACTIVE: set_component_replicas(final_p, final_d)
```

Step 5 emits advisory metrics with both raw and final values. If budget caused
clipping (`raw != final`), the GPU budget constraint log is emitted.

### `no_operation` Replacement Points

All `if not self.config.no_operation:` checks become mode-aware:

| File | Lines | Change |
|------|-------|--------|
| `base.py` | 276, 759, 772, 926, 951 | `scaling_mode != NOOP` for init; `scaling_mode == ACTIVE` for execute |
| `disagg.py` | 64, 85, 159, 244 | Same pattern |
| `agg.py` | 86, 107, 325 | Same pattern |

Pattern:

```python
# Before (boolean)
if not self.config.no_operation:
    await self.connector.set_component_replicas(...)

# After (three-way)
if self.config.scaling_mode == ScalingMode.ACTIVE:
    await self.connector.set_component_replicas(...)
```

For initialization blocks:

```python
# Before
if not self.config.no_operation:
    self.connector = KubernetesConnector(...)

# After
if self.config.scaling_mode != ScalingMode.NOOP:
    self.connector = KubernetesConnector(...)
```

---

## 7. Grafana Advisory Dashboard

An independent dashboard with 7 panels. Includes a dashboard-level link to the
existing "Dynamo Planner - SLA & Scaling" dashboard for cross-reference.

### Panel 1: Recommended vs Actual Replicas

- Dual-line chart per component (prefill / decode)
- Recommended = dashed line, Actual = solid line
- Delta region shaded (green = headroom, red = under-provisioned)
- Queries: `dynamo_planner_advisory_recommended_p`, `dynamo_planner_advisory_current_p`

### Panel 2: Scaling Action Timeline

- State timeline visualization
- Values: 1 (scale up / green), 0 (hold / gray), -1 (scale down / blue)
- Query: `dynamo_planner_advisory_scaling_action`

### Panel 3: SLA Headroom

- Dual-line: `dynamo_planner_advisory_ttft_headroom` and `dynamo_planner_advisory_itl_headroom`
- Horizontal reference line at y=0 (boundary between safe and violation)
- Area below 0 shaded as warning zone
- Handles NaN display as "N/A" via Grafana's "Connect null values" = off

### Panel 4: Estimated vs Actual TTFT

- Two lines overlaid:
  - `dynamo_planner_advisory_est_ttft` (what Planner predicted)
  - `rate(dynamo_frontend_time_to_first_token_seconds_sum[3m]) / rate(dynamo_frontend_time_to_first_token_seconds_count[3m]) * 1000` (what actually happened)
- Divergence between lines indicates estimation accuracy
- Useful for building trust in advisory recommendations before enabling active mode

### Panel 5: GPU Delta Over Time

- Single line: `dynamo_planner_advisory_delta_p * prefill_gpu + dynamo_planner_advisory_delta_d * decode_gpu`
- Positive = needs more GPUs, Negative = can release GPUs
- Visualizes resource pressure over time

### Panel 6: Recommendation Statistics

- Stacked rate chart:
  - `rate(dynamo_planner_advisory_scaleup_total[5m])`
  - `rate(dynamo_planner_advisory_scaledown_total[5m])`
  - `rate(dynamo_planner_advisory_hold_total[5m])`
- Shows recommendation frequency distribution
- A healthy Planner should mostly "hold" with occasional scale-up/down

### Panel 7: Decision Context Timeline

- Multi-row panel combining:
  - Top: `planner:observed_request_rate` + `planner:predicted_request_rate`
  - Middle: `dynamo_planner_advisory_recommended_p` vs `dynamo_planner_advisory_current_p`
  - Bottom: `dynamo_planner_advisory_scaling_action`
- Grafana Annotations overlay marking actual scaling events (from K8s events or DGD status changes)
- Provides a single view of the full decision chain: traffic → prediction → recommendation → execution

---

## 8. Configuration Reference

### New Config Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `scaling_mode` | `"active"` / `"advisory"` / `"noop"` | `"active"` | Controls scaling execution and advisory output |
| `advisory_max_step_size` | int | 1 | Max replicas per step in path recommendations. Only affects Layer 2 log text, not the `advisory_recommended_*` metric values |
| `advisory_anomaly_threshold` | int | 10 | Warn when `abs(delta)` exceeds this value |
| `advisory_file_output` | bool | False | Enable JSONL file output to `log_dir` |

### Validation Rules

```python
@model_validator(mode="after")
def _validate_config(self) -> "PlannerConfig":
    # Backward compat: no_operation -> noop
    if self.no_operation and self.scaling_mode == ScalingMode.ACTIVE:
        logger.warning("DEPRECATION: no_operation is deprecated. Use scaling_mode='noop'.")
        self.scaling_mode = ScalingMode.NOOP

    # File output requires log_dir
    if self.advisory_file_output and not self.log_dir:
        raise ValueError(
            "advisory_file_output=True requires log_dir to be set"
        )

    # advisory_max_step_size must be positive
    if self.advisory_max_step_size < 1:
        raise ValueError("advisory_max_step_size must be >= 1")
```

### Config Interaction Notes

- `scaling_mode: "advisory"` works with all `mode` values (`disagg`, `prefill`, `decode`, `agg`)
- `scaling_mode: "advisory"` works with all `environment` values (`kubernetes`, `virtual`, `global-planner`)
- In `global-planner` environment, advisory mode initializes the `GlobalPlannerConnector` but never sends `ScaleRequest` messages
- `scaling_mode: "noop"` preserves exact current `no_operation: true` behavior — no Connector, no metrics, no logs

---

## 9. File Change Summary

### Modified Files

| File | Scope of Change |
|------|----------------|
| `components/src/dynamo/planner/config/defaults.py` | Add `ScalingMode` enum. Add `scaling_mode`, `advisory_max_step_size`, `advisory_anomaly_threshold`, `advisory_file_output` to `SLAPlannerDefaults` |
| `components/src/dynamo/planner/config/planner_config.py` | Add new fields to `PlannerConfig`. Add backward compat validation. Add file output validation |
| `components/src/dynamo/planner/core/base.py` | Add 15 metrics to `PlannerPrometheusMetrics`. Add `_emit_advisory_metrics()`, `_estimate_sla_with_replicas()`, `_build_path_recommendation()`, `_safe_gauge_set()`, `_write_advisory_jsonl()` to `BasePlanner`. Replace `no_operation` checks with `scaling_mode` branches. Add `/advisory/status` HTTP handler |
| `components/src/dynamo/planner/core/disagg.py` | Replace 4 `no_operation` checks. Add advisory emission in `_throughput_loop` and `_load_loop` after GPU budget application |
| `components/src/dynamo/planner/core/agg.py` | Replace 3 `no_operation` checks. Add advisory emission in `_load_loop` |
| `components/src/dynamo/planner/core/prefill.py` | No structural changes. Advisory metrics emitted from parent `DisaggPlanner` |
| `components/src/dynamo/planner/core/decode.py` | No structural changes. Advisory metrics emitted from parent `DisaggPlanner` |

### New Files

| File | Description |
|------|-------------|
| `deploy/observability/grafana_dashboards/planner_advisory.json` | Grafana dashboard JSON (7 panels) |
| `docs/design-docs/planner-advisory-mode.md` | This design document |

### No Changes Required

| File | Reason |
|------|--------|
| `kubernetes_connector.py` | Already supports read operations. No new methods needed |
| `global_planner_connector.py` | Advisory mode skips `set_component_replicas()` call. No changes needed |
| `scale_protocol.py` | No protocol changes |
| `prometheus.py` | Metric collection logic unchanged |
| `perf_interpolation.py` | Used as-is for SLA estimation |
| `load_based_regression.py` | Used as-is for load-based decisions |

---

## 10. Hierarchical Planner Compatibility

In hierarchical deployments (multiple DGDs with GlobalPlanner):

- Advisory metrics are emitted by **each Pool Planner** independently
- Each Pool Planner's metrics are scoped to its own namespace (e.g., `prefill-pool-0`)
- **GlobalPlanner requires no changes** — in advisory mode, Pool Planners never send `ScaleRequest` messages to GlobalPlanner
- `throughput_metrics_source: "router"` is fully compatible with advisory mode since metric collection logic is shared across all modes
- The Grafana Advisory Dashboard can filter by Planner Pod using the `instance` label

---

## 11. Future Iterations (v2)

### Confidence Score

Add `dynamo_planner_advisory_confidence` Gauge (0.0 - 1.0) based on:
- Metrics stability (low variance in recent observations)
- NaN frequency (fewer NaN gaps = higher confidence)
- Prediction consistency (consecutive recommendations agreeing)
- Regression model quality (R-squared value for load-based)

### Accuracy Backtesting

In active mode, compare advisory recommendations against actual post-scaling
outcomes:
- Record `(recommended_replicas, est_ttft)` at decision time
- After scaling completes, record `(actual_ttft)` at steady state
- Compute `accuracy = 1 - abs(est_ttft - actual_ttft) / est_ttft`
- Expose as `dynamo_planner_advisory_accuracy` Gauge

### Webhook and Alert Integration

- Fire webhook when `advisory_ttft_headroom < 0` for N consecutive intervals
- Configurable webhook URL and payload template
- Integration with PagerDuty, Slack, DingTalk

### Grafana Alert Rules

- Alert: "SLA headroom negative for >5 minutes" (critical)
- Alert: "Advisory recommends scale-up for >10 minutes without action" (warning, advisory mode only)
- Alert: "Anomalous delta detected" (info)
