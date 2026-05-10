# Power Planner Design

**Status:** Draft
**Author:** Kai Ma
**Date:** 2026-05-08
**Based on:** [PR #5280](https://github.com/ai-dynamo/dynamo/pull/5280) (unmerged, `power-planner-dev` branch)
**Target:** Reimplementation on current ToT, phased into five delivery phases

---

## 1. Overview

The **Power Planner** adds power-aware intelligence to the Dynamo planner in three complementary layers:

| Layer | What it does | Delivery |
|-------|-------------|----------|
| **Infrastructure** | Config fields, pod annotation (`dynamo.nvidia.com/gpu-power-limit`), Power Agent DaemonSet, Prometheus metrics | Phase 1 (PR 1) |
| **Budget enforcement** | `_apply_power_budget()` on the state machine clamps replica scaling within a watt budget | Phase 2 (PR 2) |
| **AIC optimizer** | AIConfigurator sweep picks the (replica, power) config that maximises throughput within the watt budget; correction coefficients + drift detection close the loop | Phase 3 (PR 3) |
| **Power as sweep dimension** | AIC models power–throughput curves; optimizes asymmetric P/D power targets | Phase 4 (PR 4, AIC team dependency) |
| **Hardware validation** | End-to-end SLA + throughput verification on real silicon | Phase 5 |

### 1.1 Why reimplementation (not rebase of PR #5280)

The planner codebase was architecturally refactored after PR #5280:

| PR #5280 target | Current ToT location | Status |
|---|---|---|
| `planner/defaults.py` | `planner/config/defaults.py` | Moved + rewritten (Pydantic-based) |
| `planner/kube.py` | `planner/connectors/kubernetes_api.py` | Moved + rewritten |
| `planner/kubernetes_connector.py` | `planner/connectors/kubernetes.py` | Moved + heavily extended |
| `planner/utils/planner_argparse.py` | **Deleted** | Replaced by `planner/config/planner_config.py` (Pydantic) |
| `planner/utils/planner_core.py` | **Deleted** | Split into `core/budget.py`, `core/base.py`, `core/state_machine.py`, etc. |
| `planner/utils/prometheus.py` | **Deleted** | Replaced by `monitoring/traffic_metrics.py`, `monitoring/perf_metrics.py`, etc. |

The entire `planner/utils/` directory no longer exists. A rebase would conflict on every touched file. What carries forward directly (~60% of the PR): `components/power_agent/`, `deploy/power_agent/`, and `examples/deployments/powerplanner/` — all new content with no conflicts.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Planner (existing + new)                      │
│                                                                      │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────┐        │
│  │ Traffic   │───▶│ State Machine│───▶│ Scaling Decision    │        │
│  │ Metrics   │    │ (throughput/ │    │ (num_p, num_d)      │        │
│  └──────────┘    │  load logic) │    └─────────┬───────────┘        │
│                  └──────────────┘              │                     │
│  ┌──────────────────────────┐    ┌────────────▼────────────┐       │
│  │ AIC Power Optimizer       │◀NEW▶│ GPU Budget Constraint   │       │
│  │ (aic_power_optimizer)     │ (P3)│ _apply_global_budget()  │       │
│  │ c_ttft, c_itl,            │    └────────────┬────────────┘       │
│  │ c_power_p, c_power_d      │                  │                    │
│  │ (c_power_agg in agg mode) │                  │                    │
│  └──────────────────────────┘                  │                     │
│                                   ┌────────────▼────────────┐       │
│                                   │ Power Budget Constraint  │ ◀─P2 │
│                                   │ _apply_power_budget()   │        │
│                                   └────────────┬────────────┘       │
│                        ┌───────────────────────▼──────────┐         │
│                        │ Pod Annotation              ◀─P1  │         │
│                        │ dynamo.nvidia.com/gpu-power-limit │         │
│                        └───────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ (K8s API watch)
┌──────────────────────────────────────────────────────────────────────┐
│                  Power Agent DaemonSet (NEW, P1)                     │
│                                                                      │
│  ┌────────────┐   ┌──────────────────┐   ┌─────────────────────┐   │
│  │ Watch Pods  │──▶│ For each phys GPU:│──▶│ NVML Power Enforce  │   │
│  │ Annotations │   │  NVML→PID list,   │   │ SetPowerMgmtLimit() │   │
│  │             │   │  cgroup→pod_uid   │   │                     │   │
│  └────────────┘   └──────────────────┘   └─────────────────────┘   │
│  Reconciliation loop: every 15s                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**AIC optimizer context (Phase 3+):**

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Layer 3: Closed-Loop Runtime Controller (NativePlannerBase.run())      │
│  - Observes live TTFT/ITL/power via PrometheusAPIClient                 │
│  - Updates per-request EMAs c_ttft, c_itl + per-component c_power_p /  │
│    c_power_d (disagg) or c_power_agg (agg). Latency gated on num_req>0; │
│    power gated on per-side scheduled tokens. See §5.3.                  │
│  - Triggers re-optimization on SLA miss or capacity-exceeded drift only │
│                               │ re-invokes on drift                      │
│  Layer 2: AIC Power Optimizer (monitoring/aic_power_optimizer.py)       │
│  - Calls AIC cli_default with SLA constraints (TP pinned to cluster)    │
│  - Filters by SLA gate: aic_ttft × max(1, c_ttft) ≤ target              │
│                          aic_itl  × max(1, c_itl)  ≤ target              │
│  - Filters by budget gate (disagg, called twice):                       │
│        cap_p = ceil(aic_power_w_p × max(c_power_p, 1.0))                │
│        cap_d = ceil(aic_power_w_d × max(c_power_d, 1.0))                │
│        n_p × cap_p × p_gpu + n_d × cap_d × d_gpu ≤ budget               │
│  - Ranks by aic_throughput; ties broken by lower GPU count              │
│                               │ uses                                     │
│  Layer 1: AIC Core Engine (aiconfigurator pip package)                  │
│  - Performance model from collected silicon data                        │
│  - Config → (throughput, TTFT, TPOT, power_w) estimator                │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Configuration

### 3.1 PlannerConfig fields

All configuration is in `components/src/dynamo/planner/config/` via Pydantic — no argparse flags.

**`config/defaults.py` — `SLAPlannerDefaults` additions:**

```python
# Power-aware (disabled by default for backward compat).
# total_gpu_power_limit and power_agent_safe_default_watts are required when
# enable_power_awareness=True (validator enforces). Defaults are None — not
# placeholder integers — so the type itself signals "operator must set".
enable_power_awareness: bool = False
total_gpu_power_limit: Optional[int] = None
prefill_engine_gpu_power_limit: int = 300  # watts per GPU, prefill replicas
decode_engine_gpu_power_limit: int = 300   # watts per GPU, decode replicas
power_agent_safe_default_watts: Optional[int] = None

# AIC optimizer (disabled by default)
enable_aic_optimizer: bool = False
aic_reoptimize_interval: int = 300       # seconds between AIC sweeps
aic_drift_relative_threshold: float = 0.15  # fraction before re-sweep triggers
aic_drift_consecutive_ticks: int = 3     # ticks of sustained signal needed
# Cold-start values for the per-component power coefficients (B1).
# Disagg uses c_power_p / c_power_d separately because prefill is compute-bound
# (SM-clock-driven) and decode is memory-bound (HBM-driven); a single scalar
# c_power averaged the two regimes and produced wrong caps. mode=agg uses one.
aic_initial_c_power_prefill: float = 1.0  # cold-start c_power_p before live data
aic_initial_c_power_decode:  float = 1.0  # cold-start c_power_d before live data
aic_initial_c_power_agg:     float = 1.0  # cold-start c_power_agg (mode=agg only)
aic_initial_c_ttft: float = 1.0          # cold-start c_ttft before live data
aic_initial_c_itl:  float = 1.0          # cold-start c_itl  before live data
aic_max_consecutive_failures: int = 5   # failures before optimizer auto-disables
aic_throughput_regression_warn_threshold: float = 0.10  # drop fraction for WARNING
```

**`config/planner_config.py` — `PlannerConfig` additions:**

```python
class PlannerConfig(BaseModel):
    # --- Power-aware scaling (Phase 1 + 2) ---
    enable_power_awareness: bool = SLAPlannerDefaults.enable_power_awareness
    total_gpu_power_limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Total GPU power budget in watts for this DGD. Required when "
            "enable_power_awareness=True. Recommended formula: "
            "(rack_capacity_W × headroom_factor) − non_gpu_overhead with "
            "headroom_factor ≈ 0.85–0.9. See §3.3."
        ),
    )
    prefill_engine_gpu_power_limit: int = Field(
        default=SLAPlannerDefaults.prefill_engine_gpu_power_limit,
        description="Per-GPU power cap (watts) applied to prefill replicas via NVML.",
    )
    decode_engine_gpu_power_limit: int = Field(
        default=SLAPlannerDefaults.decode_engine_gpu_power_limit,
        description="Per-GPU power cap (watts) applied to decode replicas via NVML.",
    )
    power_agent_safe_default_watts: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Per-GPU fail-closed cap (watts) the Power Agent applies on cold-start "
            "GPUs (no prior cap) when annotation parsing fails. Required when "
            "enable_power_awareness=True; the validator emits a clear error if unset. "
            "Recommended: ~70%% of SKU TDP (e.g. 500W on H200 SXM, 490W on H100 SXM)."
        ),
    )

    # --- AIC optimizer (Phase 3+) ---
    enable_aic_optimizer: bool = Field(default=False)
    aic_system: Optional[str] = Field(
        default=None,
        description="AIC system identifier (e.g. 'h200_sxm'). Falls back to aic_interpolation.system.",
    )
    aic_reoptimize_interval: int = Field(default=300)
    aic_drift_relative_threshold: float = Field(default=0.15, ge=0.0)
    aic_drift_consecutive_ticks: int = Field(default=3, ge=1)
    # Per-component cold-start power coefficients (B1). Disagg uses prefill +
    # decode separately; mode=agg uses agg. The cross-mode unused fields are
    # silently ignored at runtime.
    aic_initial_c_power_prefill: float = Field(
        default=1.0,
        description="Cold-start c_power_p (disagg prefill side). H200 dense: ~1.05.",
    )
    aic_initial_c_power_decode: float = Field(
        default=1.0,
        description="Cold-start c_power_d (disagg decode side). H200 dense: ~1.15.",
    )
    aic_initial_c_power_agg: float = Field(
        default=1.0,
        description="Cold-start c_power_agg (aggregated mode only).",
    )
    aic_initial_c_ttft: float = Field(
        default=1.0,
        description="Cold-start c_ttft before live data arrives.",
    )
    aic_initial_c_itl: float = Field(
        default=1.0,
        description="Cold-start c_itl before live data arrives.",
    )
    aic_max_consecutive_failures: int = Field(default=5, ge=1)
    aic_throughput_regression_warn_threshold: float = Field(default=0.10, ge=0.0, le=1.0)

    # --- Frontend admission coupling (Phase 3+); see §5.7 ---
    admission_mode: Literal["off", "inherit", "autoset"] = Field(
        default="off",
        description=(
            "Frontend /busy_threshold coupling. 'off' never POSTs; 'inherit' "
            "computes implied thresholds and emits Prometheus gauges only; "
            "'autoset' POSTs implied thresholds after every config change."
        ),
    )
    admission_safety_margin: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Multiplier applied to implied thresholds before POST in autoset mode. "
            "Lower values shed earlier (more headroom for queueing variance). "
            "Also passed through to AIC TaskRunner as busy_threshold_safety_margin "
            "when admission_mode != 'off'."
        ),
    )
    frontend_http_port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description=(
            "TCP port the frontend pod's HTTP server listens on. Used by the "
            "planner to POST /busy_threshold in autoset admission mode. Must "
            "match the containerPort declared in the frontend's pod spec. "
            "Default 8000 matches the operator's default and the existing "
            "`get_frontend_metrics_url(port=8000)` convention. Override only "
            "if the DGD spec configures a non-default frontend port. See §13 "
            "Open Question #13 for the planned auto-discovery successor."
        ),
    )
```

**Validation (`_validate_config`):**

```python
if self.enable_power_awareness:
    if self.total_gpu_power_limit is None:
        raise ValueError(
            "total_gpu_power_limit is required when enable_power_awareness=True. "
            "Recommended: (rack_capacity_W × headroom_factor) − non_gpu_overhead "
            "with headroom_factor ≈ 0.85–0.9 (see §3.3). Setting this incorrectly "
            "could silently cap your cluster — there is no safe default."
        )
    if self.power_agent_safe_default_watts is None:
        raise ValueError(
            "power_agent_safe_default_watts is required when "
            "enable_power_awareness=True. This is the cold-start fail-closed cap "
            "the Power Agent applies on a fresh GPU (no prior cap) when "
            "annotation parsing fails. Recommended: ~70%% of SKU TDP "
            "(e.g. 500W on H200 SXM)."
        )

# model_validator: when enable_aic_optimizer=True, require aic_interpolation
# with valid prefill_pick and decode_pick (pins TP to the deployed cluster).
# When mode=disagg, c_power_p and c_power_d are tracked separately. When
# mode=agg, only c_power_agg is used. The cross-mode unused initial-coefficient
# fields are silently ignored at runtime — no validation error if the operator
# sets aic_initial_c_power_agg in a disagg deployment, just dead config.
```

### 3.2 Naming conventions

**Config field prefixes:**
- `aic_*` — consumed by the AIC optimizer.
- `*_engine_gpu_power_limit` / `total_gpu_power_limit` — power-budget settings for the autoscaling layer.
- `enable_*` — boolean feature toggles (matches existing `enable_throughput_scaling`).

No single-word generic names (no bare `system`, `threshold`, `interval`).

**Threshold fields encode units:** `aic_drift_relative_threshold` (fraction), `aic_drift_consecutive_ticks` (count).

**Counter-style fields encode cardinality:** `aic_max_consecutive_failures` (count of consecutive events).

**Prometheus metric conventions:**

| Suffix | Meaning | Example |
|--------|---------|---------|
| `_total` | Cumulative counter | `dynamo_aic_optimizer_exceptions_total`, `dynamo_power_agent_multi_pod_gpu_total` |
| `_watts` | Gauge in watts | `dynamo_power_agent_applied_limit_watts` |
| `_utilization` | Gauge 0.0–1.0+ ratio | `dynamo_planner_power_budget_utilization` |
| (no suffix) | Current-state gauge | `dynamo_aic_consecutive_failures` (resets on success) |

**Pod annotation keys:** `dynamo.nvidia.com/<concept>` domain, kebab-case (K8s convention). Today: `dynamo.nvidia.com/gpu-power-limit`.

**Complete names registry:**

| Kind | Name | Phase |
|------|------|-------|
| Config | `total_gpu_power_limit` | P1 |
| Config | `prefill_engine_gpu_power_limit` | P1 |
| Config | `decode_engine_gpu_power_limit` | P1 |
| Config | `power_agent_safe_default_watts` | P1 |
| Config | `enable_aic_optimizer` | P3 |
| Config | `aic_system` | P3 |
| Config | `aic_reoptimize_interval` | P3 |
| Config | `aic_drift_relative_threshold` | P3 |
| Config | `aic_drift_consecutive_ticks` | P3 |
| Config | `aic_initial_c_power_prefill` | P3 |
| Config | `aic_initial_c_power_decode` | P3 |
| Config | `aic_initial_c_power_agg` | P3 |
| Config | `aic_initial_c_ttft` | P3 |
| Config | `aic_initial_c_itl` | P3 |
| Config | `aic_max_consecutive_failures` | P3 |
| Config | `aic_throughput_regression_warn_threshold` | P3 |
| Config | `admission_mode` | P3 |
| Config | `admission_safety_margin` | P3 |
| Config | `frontend_http_port` | P3 |
| Annotation | `dynamo.nvidia.com/gpu-power-limit` | P1 |
| Metric (gauge, watts) | `dynamo_power_agent_applied_limit_watts` | P1 |
| Metric (counter, label `disposition={"agree","conflict"}`) | `dynamo_power_agent_multi_pod_gpu_total` | P1 |
| Metric (counter) | `dynamo_power_agent_apply_failures_total` | P1 |
| Metric (counter) | `dynamo_power_agent_safe_default_applied_total` | P1 |
| Metric (counter, label `direction={"min","max"}`) | `dynamo_power_agent_cap_clamped_total` | P1 |
| Metric (gauge) | `dynamo_planner_power_budget_total_watts` | P1 |
| Metric (gauge) | `dynamo_planner_power_projected_watts` | P1 |
| Metric (gauge) | `dynamo_planner_power_budget_utilization` | P1 |
| Metric (gauge, info) | `dynamo_aic_optimizer_disabled_reason` | P3 |
| Metric (gauge, count) | `dynamo_aic_consecutive_failures` | P3 |
| Metric (gauge, label `component={"prefill","decode","agg"}`) | `dynamo_aic_c_power` | P3 |
| Metric (gauge) | `dynamo_aic_c_ttft` | P3 |
| Metric (gauge) | `dynamo_aic_c_itl` | P3 |
| Metric (counter) | `dynamo_aic_optimizer_exceptions_total` | P3 |
| Metric (counter) | `dynamo_aic_throughput_regression_total` | P3 |
| Metric (counter, label `coefficient={"ttft","itl","power_prefill","power_decode","power_agg"}`) | `dynamo_aic_correction_pegged_total` | P3 |
| Metric (gauge, ratio) | `dynamo_aic_advisory_alternative_tp_speedup` | P3 |
| Metric (counter) | `dynamo_planner_admission_partial_success_total` | P3 |
| Metric (gauge, ratio) | `dynamo_planner_admission_implied_theta_decode` | P3 |
| Metric (gauge, ratio) | `dynamo_planner_admission_implied_theta_prefill_frac` | P3 |
| Metric (gauge, ratio) | `dynamo_planner_admission_set_theta_decode` | P3 |
| Metric (gauge, ratio) | `dynamo_planner_admission_set_theta_prefill_frac` | P3 |
| Metric (gauge, count) | `dynamo_planner_admission_set_theta_prefill_abs` | P3 |
| Metric (counter) | `dynamo_planner_admission_max_batched_tokens_unavailable_total` | P3 |

### 3.3 Budget scope

`total_gpu_power_limit` covers **GPU board power as enforced by NVML `nvmlDeviceSetPowerManagementLimit()`**. The cap is a hard hardware bound — the on-chip power controller throttles SM and HBM clocks to stay under the cap. `projected_power = N × cap_per_gpu` is therefore an exact upper bound, not an estimate.

**Not included** (must be accounted for when sizing against rack/PDU capacity): CPU/DRAM power, NIC/PCIe, PSU efficiency losses, cooling overhead, non-worker GPU consumers.

In practice: `total_gpu_power_limit = (rack_capacity_W × headroom_factor) − non_gpu_overhead`, where `headroom_factor ≈ 0.9`.

**Scope: per-DGD, not per-cluster.** One planner manages one DGD and enforces one budget across its prefill/decode workers.

| Topology | Supported? | Operator responsibility |
|---|---|---|
| One DGD per cluster (or disjoint node pools) | Yes — typical | None beyond sizing budget vs. available rack capacity. |
| Multiple DGDs, disjoint GPUs, shared nodes | Yes | Sum of all `total_gpu_power_limit` values must fit facility power with margin. Planner does not coordinate across DGDs. |
| Multiple DGDs sharing one physical GPU (MIG/MPS) | **Out of scope v1** | Annotation race — last-writer wins. Use one DGD per GPU. |

---

## 4. Data Flow

```
User Config (JSON/YAML)
  │
  ▼
PlannerConfig (Pydantic)
  ├─ enable_power_awareness: true
  ├─ total_gpu_power_limit: 4000 W
  ├─ prefill_engine_gpu_power_limit: 500 W  ← set by AIC optimizer in P3+ (uses c_power_p)
  └─ decode_engine_gpu_power_limit: 425 W   ← set by AIC optimizer in P3+ (uses c_power_d)
  │
  ▼
AICPowerOptimizer.optimize()              ◀── Phase 3+ (on startup + drift)
  ├─ AIC cli_default sweep (TP pinned)
  ├─ Apply §5.1 bridge per side:
  │    ├─ cap_p = ceil(aic_power_w_p × max(c_power_p, 1.0))
  │    └─ cap_d = ceil(aic_power_w_d × max(c_power_d, 1.0))
  └─ Write cap values into PlannerConfig.*_engine_gpu_power_limit
  │
  ▼
PlannerStateMachine.on_tick()
  ├─ Load/throughput scaling → desired_p, desired_d
  ├─ self._apply_global_budget()  → clamped by GPU count
  └─ self._apply_power_budget()   → clamped by watt budget   ◀── Phase 2
  │
  ▼
NativePlannerBase._apply_effects()
  └─ connector.set_component_replicas([...])
  │
  ▼
NativePlannerBase._apply_power_annotations()               ◀── Phase 1
  ├─ connector.get_component_pods(PREFILL) → [pod1, pod2]
  ├─ kube_api.patch_pod_annotation(pod1, "dynamo.nvidia.com/gpu-power-limit", "480")
  └─ (reads actual annotation from Pod object; PATCHes only on mismatch)
  │
  ▼
Power Agent DaemonSet (per node)                           ◀── Phase 1
  ├─ Watch annotated pods via K8s API (every 15s)
  ├─ For each physical GPU: nvmlDeviceGetComputeRunningProcesses(handle) → PID list
  ├─ For each PID: read /proc/{pid}/cgroup → extract pod_uid (cAdvisor pattern)
  ├─ Look up pod_uid in K8s pod cache → get annotation value (watts)
  └─ pynvml.nvmlDeviceSetPowerManagementLimit(gpu, 480_000)  # milliwatts
  │
  ▼
AICPowerOptimizer.update_correction()                      ◀── Phase 3+
  ├─ if traffic.num_req > 0:
  │    ├─ c_ttft = EMA(observed_ttft_avg / aic_ttft)
  │    └─ c_itl  = EMA(observed_itl_avg  / aic_itl)
  ├─ disagg: per-side gates (compute-bound vs memory-bound regimes)
  │    ├─ if traffic.scheduled_prefill_tokens > 0:
  │    │    └─ c_power_p = EMA(observed_power_w_prefill / aic_power_w_p)
  │    └─ if traffic.scheduled_decode_kv_tokens > 0:
  │         └─ c_power_d = EMA(observed_power_w_decode / aic_power_w_d)
  └─ agg: c_power_agg = EMA(observed_power_w_agg / aic_power_w)
```

---

## 5. Algorithms

### 5.1 AIC cap bridge formula

This formula ties the AIC optimizer to the autoscaling layer. When the AIC optimizer is enabled, it sets `prefill_engine_gpu_power_limit` and `decode_engine_gpu_power_limit` so that the autoscaling layer's invariant — *"cap × N is an exact upper bound on draw"* — holds with realistic numbers, not nameplate TDP.

```python
def aic_to_planner_cap(aic_power_w: float, c_power: float) -> int:
    """
    Convert AIC's predicted per-GPU draw into the NVML cap for PlannerConfig.

    Asymmetric clamp: never lowers cap below AIC's estimate (that would force
    NVML throttling on every tick); inflates when live data shows AIC under-predicts.

    Called twice per AIC sweep in disagg mode — once with (aic_power_w_p, c_power_p)
    for the prefill cap, once with (aic_power_w_d, c_power_d) for the decode cap.
    In aggregated mode (mode=agg) it is called once with (aic_power_w, c_power_agg).
    See §5.3 for why power coefficients are split per-component.
    """
    safety = max(c_power, 1.0)
    return math.ceil(aic_power_w * safety)
```

**Worked example (H200 SXM, Llama-3-8B, disagg):** AIC predicts prefill draw 480W and decode draw 360W. After 3 minutes of live data:

- `c_power_p = 1.04` (prefill compute-bound; AIC tracks dense GEMM power closely)
- `c_power_d = 1.18` (decode memory-bound; KV-cache traffic adds power AIC's static model misses)

```
cap_p = ceil(480 × max(1.04, 1.0)) = ceil(499.2) = 500W
cap_d = ceil(360 × max(1.18, 1.0)) = ceil(424.8) = 425W
```

- Power Agent applies `nvmlDeviceSetPowerManagementLimit(500_000)` per prefill GPU and `(425_000)` per decode GPU.
- Worst-case burst (e.g., 4 prefill GPUs + 4 decode GPUs): `4 × 500 + 4 × 425 = 3700W < 4000W` budget.
- A *single* fleet-wide `c_power = (1.04 + 1.18)/2 ≈ 1.11` would have produced caps of 533W (prefill — wasteful) and 400W (decode — forces throttling). The split coefficients prevent both errors.
- At nameplate TDP (700W per GPU): `8 × 700 = 5600W > 4000W` — would over-commit budget.

When the AIC optimizer is **not** enabled, the operator hand-tunes the cap fields directly; the same `cap × N` invariant applies.

### 5.2 Power budget enforcement (`_apply_power_budget`)

**Location:** Method on `PlannerStateMachine` in `core/state_machine.py`, next to the existing `_apply_global_budget` method.

```python
def _apply_power_budget(self, num_p: int, num_d: int) -> tuple[int, int]:
    """Apply power budget constraint to replica counts.

    Mirrors _apply_global_budget so both budgets share the same scaling semantics:
      1. Projected power fits budget → return unchanged.
      2. Even min_endpoint replicas of both components don't fit → return (0, 0) + warn.
      3. Otherwise: reserve decode floor, proportionally scale prefill, give
         remaining budget to decode (capped at original demand — never up-scales).
    """
    if not self._config.enable_power_awareness:
        return num_p, num_d

    p_gpu = self._capabilities.prefill.num_gpu if self._capabilities.prefill else None
    d_gpu = self._capabilities.decode.num_gpu if self._capabilities.decode else None
    if p_gpu is None or d_gpu is None:
        return num_p, num_d

    budget  = self._config.total_gpu_power_limit
    p_watts = self._config.prefill_engine_gpu_power_limit * p_gpu  # per-replica
    d_watts = self._config.decode_engine_gpu_power_limit  * d_gpu  # per-replica

    projected = num_p * p_watts + num_d * d_watts
    if projected <= budget:
        return num_p, num_d

    min_p_watts = self._config.min_endpoint * p_watts
    min_d_watts = self._config.min_endpoint * d_watts
    if budget < min_p_watts + min_d_watts:
        logger.warning(
            f"total_gpu_power_limit ({budget}W) < minimum required "
            f"({min_p_watts + min_d_watts}W) for min_endpoint replicas; "
            f"enforcing zero replicas."
        )
        return 0, 0

    scale   = budget / projected
    max_p   = math.floor((budget - min_d_watts) / p_watts)
    scaled_p = max(self._config.min_endpoint, min(max_p, math.floor(num_p * scale)))
    remaining = budget - scaled_p * p_watts
    scaled_d  = max(self._config.min_endpoint, min(num_d, math.floor(remaining / d_watts)))

    logger.warning(
        f"Power budget: projected {projected}W > limit {budget}W. "
        f"Scaled ({num_p}p, {num_d}d) → ({scaled_p}p, {scaled_d}d)."
    )
    return scaled_p, scaled_d
```

**Algorithm properties:**

| Input | Result | Notes |
|---|---|---|
| `(4,4)`, both 250W, `p_gpu=d_gpu=1`, budget=1500W | `(3, 3)` | Exact: 1500W used. |
| `(4,4)`, same but budget=1499W | `(2, 3)` | 1250W used; `(3,3)` and `(2,4)` both = 1500W > 1499W. |
| `(2,4)`, `p_gpu=8, d_gpu=1`, both 300W, budget=4000W | `(1, 4)` | 3600W; asymmetric TP cost-per-replica correctly accounted. |
| `(4,4)`, both 250W, budget=400W, `min_endpoint=1` | `(0, 0)` | min_required=500W > 400W → explicit zero + warning. |
| `(10,2)`, `p_gpu=4, d_gpu=1`, both 250W, budget=5000W | Decode capped at 2 | `min(num_d, …)` prevents latent up-scale-beyond-demand bug in `_apply_global_budget`. |

**Integration — call sites:**

```python
# core/load_scaling.py — after existing _apply_global_budget call
final_p, final_d = self._apply_global_budget(final_p, final_d)
final_p, final_d = self._apply_power_budget(final_p, final_d)   # NEW (Phase 2)

# core/throughput_scaling.py — same pattern
num_p, num_d = self._apply_global_budget(num_p, num_d)
num_p, num_d = self._apply_power_budget(num_p, num_d)           # NEW (Phase 2)
```

**Order is a correctness property.** GPU budget first, then power budget. The two do not commute on cost-asymmetric inputs (`p_gpu ≠ d_gpu`). The unit test in §6.3 case 4 pins this.

**`POWER_ANNOTATION_KEY` constant** — shipped in Phase 1 in `core/budget.py` (stable, dependency-free):
```python
POWER_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"
```

### 5.3 AIC correction coefficients

EMA-smoothed coefficients bridge AIC offline estimates to live serving behaviour. **All use per-request signals** (TTFT, ITL, per-GPU power) — never a total-throughput ratio. This design choice is deliberate; see "Why not a throughput ratio" below.

**Power coefficients are split per-component in disagg mode.** Prefill is compute-bound (SM-clock-driven GEMM power); decode is memory-bound (HBM-bandwidth-driven, KV-cache traffic). A single fleet-wide `c_power` averages the two regimes and produces simultaneously over-capped decode and under-capped prefill (or vice versa). Aggregated mode (`mode=agg`) keeps a single `c_power_agg` because the engine itself mixes regimes via chunked prefill — splitting is not physically meaningful for a single engine type.

```python
# Disaggregated mode — separate prefill and decode power coefficients.
c_ttft    = α × (observed_ttft_avg / aic_ttft) + (1 − α) × c_ttft_prev
c_itl     = α × (observed_itl_avg  / aic_itl)  + (1 − α) × c_itl_prev
c_power_p = α × (observed_power_w_prefill / aic_power_w_p) + (1 − α) × c_power_p_prev
c_power_d = α × (observed_power_w_decode  / aic_power_w_d) + (1 − α) × c_power_d_prev

# Aggregated mode (mode=agg) — single engine type, single coefficient.
c_power_agg = α × (observed_power_w_agg / aic_power_w) + (1 − α) × c_power_agg_prev

# α = 0.3 (smoothing factor).
# All four (or five) coefficients are clamped to [0.5, 2.0] after each update.

# Applied during feasibility filtering. Asymmetric clamps preserve the
# "conservative-in-the-same-direction" property: we never loosen SLA or
# under-cap power based on noisy live data.
ttft_safety    = max(1.0, c_ttft)    # never tightens below AIC's estimate
itl_safety     = max(1.0, c_itl)
power_safety_p = max(1.0, c_power_p)   # disagg prefill;  in agg, use max(1.0, c_power_agg)
power_safety_d = max(1.0, c_power_d)   # disagg decode;   same in agg

corrected_ttft  = aic_ttft    × ttft_safety              # SLA gate
corrected_itl   = aic_itl     × itl_safety               # SLA gate
corrected_cap_p = ceil(aic_power_w_p × power_safety_p)   # budget gate (§5.1 bridge)
corrected_cap_d = ceil(aic_power_w_d × power_safety_d)   # budget gate

# Ranking (after feasibility) uses raw aic_throughput — there is no live
# "total throughput" measurement to correct it with that wouldn't be
# load-confounded (see below).
best = max(feasible, key=lambda c: (c.aic_throughput,
                                    -(c.n_p * c.tp_p + c.n_d * c.tp_d),       # fewer GPUs
                                    -(sum_cap_power(c))))                       # lower cap power
```

| Coefficient | Captures | Typical range | Gate it acts on |
|-------------|----------|---------------|-----------------|
| `c_ttft`  | Prefill scheduling overhead, KV transfer, AIC vs real prefill gaps | 1.0–1.3 (AIC under-predicts latency) | SLA gate — filters near-TTFT-edge configs |
| `c_itl`   | Decode batching effects, attention overhead, framework cost | 1.0–1.4 (AIC under-predicts latency) | SLA gate — filters near-ITL-edge configs |
| `c_power_p` | **Prefill power** (compute-bound regime): SM-clock-driven GEMM draw, AIC vs real attention/MLP power gaps | 1.00–1.10 (small bias — compute-bound power tracks AIC's analytic model closely) | Budget gate — inflates `cap_p` so optimizer doesn't over-pack prefill replicas |
| `c_power_d` | **Decode power** (memory-bound regime): HBM bandwidth-driven, KV-cache traffic, attention re-reads, framework overhead AIC's static model misses | 1.05–1.20 (larger bias — AIC under-models decode-side memory traffic) | Budget gate — inflates `cap_d` so optimizer doesn't over-pack decode replicas |
| `c_power_agg` | **Aggregated mode only**: chunked-prefill mixed regime; weighted average of prefill and decode power profiles | 1.05–1.15 | Budget gate — inflates the single cap shared by both regimes |

`c_ttft` is the prefill-stage latency coefficient (TTFT is determined by prefill scheduling). `c_itl` is the decode-stage latency coefficient (ITL is determined by decode batching). They are not split further because each is already stage-specific by definition.

All coefficients are **conservative-in-the-same-direction**: when AIC is optimistic, they shrink the feasible set. They protect orthogonal invariants (power-prefill vs. power-decode vs. prefill-latency vs. decode-latency) but none loosen the optimizer when AIC drifts.

**EMA update gates** — when the update is ill-defined we skip it (no EMA push toward the floor):

```python
# Latency: gated on overall num_req > 0 (TTFT/ITL are end-to-end signals).
if traffic.num_req > 0 and observed_ttft_avg > 0:
    c_ttft = α * (observed_ttft_avg / aic_ttft) + (1 - α) * c_ttft

if traffic.num_req > 0 and observed_itl_avg > 0:
    c_itl = α * (observed_itl_avg / aic_itl) + (1 - α) * c_itl

# Power (disagg): per-component, gated on per-component scheduled work.
# Prevents idle-side EMA drag — when prefill is busy and decode is idle (or
# vice versa), only the working side updates, and the idle side keeps its
# prior calibration instead of being pulled toward idle-watt / aic-watt ≈ 0.06.
if traffic.scheduled_prefill_tokens > 0 and observed_power_w_prefill > 0:
    c_power_p = α * (observed_power_w_prefill / aic_power_w_p) + (1 - α) * c_power_p

if traffic.scheduled_decode_kv_tokens > 0 and observed_power_w_decode > 0:
    c_power_d = α * (observed_power_w_decode / aic_power_w_d) + (1 - α) * c_power_d

# Power (agg): chunked prefill mixes regimes; gate on total tokens scheduled.
if (traffic.scheduled_prefill_tokens + traffic.scheduled_decode_kv_tokens) > 0 \
        and observed_power_w_agg > 0:
    c_power_agg = α * (observed_power_w_agg / aic_power_w) + (1 - α) * c_power_agg
```

The latency gate uses an existing `TrafficObservation.num_req` field. The per-side power gates require `TrafficObservation.scheduled_prefill_tokens` and `TrafficObservation.scheduled_decode_kv_tokens` — already collected by the FPM event plane (per `planner-design.md` §"Load-Based Scaling"); plumbing them through `TrafficObservation` is a small Phase-3 deliverable. See §13 Open Question #11. When traffic is sparse on a side, the SLA / budget gates run against the prior-tick coefficients — AIC's own predictions corrected by the last live calibration — which is the safe behavior.

**Cold-start:** `c_power_p`, `c_power_d` initialized to `aic_initial_c_power_prefill`, `aic_initial_c_power_decode` (defaults 1.0); `c_ttft` and `c_itl` initialized to `aic_initial_c_ttft` and `aic_initial_c_itl` (defaults 1.0). In aggregated mode, `c_power_agg` is initialized to `aic_initial_c_power_agg`. If actual values exceed AIC's estimates during the first interval, configs near the SLA edge are briefly tighter and the GPU may throttle briefly until the EMA catches up (~2–3 intervals). For H200 dense models the recommended cold-start values are `aic_initial_c_power_prefill = 1.05`, `aic_initial_c_power_decode = 1.15`, `aic_initial_c_ttft = 1.15`, `aic_initial_c_itl = 1.15` — these reflect the typical compute-bound vs. memory-bound asymmetry. **Recommended values to be validated in Phase 5**; before that they remain conservative guesses.

**Why not a throughput ratio.** A previous draft used `c_throughput = observed_tokens_per_sec / aic_estimated_throughput`. That formula is **load-confounded**: when offered traffic is below capacity, `observed_tokens_per_sec` reflects *demand*, not *capability*, and the ratio collapses toward zero through no fault of the hardware. With the asymmetric clamp `max(1.0, 1/c_throughput)` it would make `safety_factor` skyrocket, falsely tightening every SLA gate. The fix is to **never measure capability against demand**: TTFT and ITL are demand-invariant per-request properties (they describe how fast served requests are served, regardless of how many arrive), so per-request ratios are the right signal to compare AIC against. Total throughput remains the *ranking* metric (§5.4) because there it is compared between AIC predictions of different *configs*, not between an AIC prediction and a load-dependent observation.

**Aggregated mode (`mode=agg`).** A single engine type runs both prefill and decode via chunked prefill on the same GPU, so a per-component power split is not physically meaningful — there is only one engine to measure. `c_power_agg` is used for the single per-GPU cap. The split-coefficient design exists for disaggregated mode where prefill and decode workloads run on separate GPUs with separate caps.

### 5.4 AIC optimizer selection logic (Phase 1: TDP-only)

```
# Disaggregated mode (per-component power coefficients)
For each candidate config from cli_default():
    aic_power_w_p, aic_power_w_d = config.power_w  (or nameplate TDP if missing)
    cap_p = ceil(aic_power_w_p × max(c_power_p, 1.0))   # §5.1 bridge — uses c_power_p
    cap_d = ceil(aic_power_w_d × max(c_power_d, 1.0))   # §5.1 bridge — uses c_power_d
    total_power = n_p × cap_p × p_gpu + n_d × cap_d × d_gpu   # cap, not raw estimate

    if total_power > total_gpu_power_limit:  skip
    if aic_ttft × max(1.0, c_ttft) > target_ttft:  skip   # per-request TTFT correction
    if aic_itl  × max(1.0, c_itl)  > target_itl:   skip   # per-request ITL  correction
    record (config, aic_throughput, cap_p, cap_d)

Return config with max aic_throughput (tie-break: lower GPU count, then lower cap power)
Write cap_p → prefill_engine_gpu_power_limit, cap_d → decode_engine_gpu_power_limit

# Aggregated mode (mode=agg) — single engine type, single coefficient
For each candidate config:
    cap_agg = ceil(aic_power_w × max(c_power_agg, 1.0))
    total_power = n_agg × cap_agg × gpu_per_engine
    (SLA gates and ranking unchanged from disagg.)
```

The budget check uses the **cap** (not raw AIC power estimate) so the number the optimizer uses for feasibility is exactly the number `_apply_power_budget()` will enforce on every tick. The two layers never disagree.

**Why total tokens/s is the ranking metric** (not tokens/J or tokens/GPU): under per-DGD scope the GPUs are dedicated — they're paid for whether idle or saturated. Total tokens/s is the directly user-visible quantity and matches the stated objective. `tokens/J` and `tokens/GPU` sort the same way as `total tokens/s` when the corresponding constraint is binding, but pick worse answers when the other constraint binds.

### 5.5 AIC optimizer selection logic (Phase 2: power as sweep dimension)

Phase 2 requires new AIC APIs (§8). The optimizer loop gains a `power_p, power_d` search dimension:

```python
# tp_p, tp_d are CONSTRAINED (read from AICInterpolationSpec.{prefill,decode}_pick)
# Optimizer sweeps only: (n_p, n_d, power_p, power_d, bs)
for (n_p, n_d, power_p, power_d, bs):
    cap_p = ceil(power_p × max(c_power_p, 1.0))   # disagg: c_power_p
    cap_d = ceil(power_d × max(c_power_d, 1.0))   # disagg: c_power_d
    throughput, ttft, tpot = AIC.estimate_perf(..., power_w_p=power_p, power_w_d=power_d)
    total_power = n_p × tp_p × cap_p + n_d × tp_d × cap_d
    if feasible:  record
# Aggregated mode: single c_power_agg replaces both c_power_p and c_power_d.
```

**Power-performance regimes (why asymmetric P/D power is the key insight):**
- **Compute-bound (prefill GEMMs):** throughput scales ~linearly with SM clock. Cutting prefill power by 30% costs ~30% prefill throughput.
- **Memory-bound (decode attention, KV-cache reads):** throughput is governed by HBM bandwidth, nearly insensitive to SM-clock reductions — until the cap forces HBM downclocking. The minimum cap where HBM stays at full clock is the **decode power floor**.

Operational implication: cap decode tightly (near its floor) to free budget for prefill; cap prefill only as much as the budget requires. Phase 2 finds this optimal asymmetric split automatically.

### 5.6 Drift detection and re-optimization

**Definition of `self._estimated_throughput`.** Total predicted tokens-per-second across the entire DGD at the currently-applied AIC config. Computed once in `_apply_aic_config()` from the picked AIC row:

```
self._estimated_throughput = (
    picked.aic_seq_per_s_per_replica * picked.n_d
    * (picked.isl + picked.osl)
)
```

Units must match `traffic.total_tokens_per_sec` (tokens/s aggregate, not per-replica, not seq/s). The factor `n_d` is the steady-state-determining replica count for token throughput in disaggregated mode (decode is the bottleneck for OSL ≫ chunk size); aggregated mode uses `n_agg` instead.

**Cold-start behavior.** Initialized to `0` in `AICPowerOptimizer.__init__()`. With `_estimated_throughput == 0`, `capacity_exceeded` cannot fire (the `> 0` guard short-circuits) and only `sla_violated` can trigger re-optimization. This is correct: before the first successful sweep, the optimizer has no reference throughput to compare live load against. Failure modes #1 and #4 leave `_estimated_throughput` at 0 permanently after auto-disable, which is also correct since the optimizer is no longer driving config changes.

**Drift reaction time floor.** The composition of `aic_reoptimize_interval` (rate-limit) and `aic_drift_consecutive_ticks` (hysteresis) bounds how fast the optimizer can react:

```
T_react ≥ adjustment_interval × aic_drift_consecutive_ticks
```

At default values (`adjustment_interval=180s`, `aic_drift_consecutive_ticks=3`), the floor is **9 minutes** from drift onset to sweep firing. Operators tuning these knobs should size them together against their tolerance for sustained SLA misses.

```python
def should_reoptimize(self, traffic: TrafficObservation) -> bool:
    if self._time_since_last_optimize < self._config.aic_reoptimize_interval:
        return False

    # SLA-miss triggers: latency above target is always a problem.
    sla_violated = (
        traffic.ttft_avg > self._config.ttft or
        traffic.itl_avg  > self._config.itl
    )

    # Throughput drift trigger: ONLY fires when demand exceeds predicted
    # capacity. The downward case (demand below capacity) is "user sent
    # fewer requests" — never a planner problem and would otherwise force
    # spurious sweeps in lightly-loaded clusters. See "Direction matters"
    # below.
    capacity_exceeded = (
        self._estimated_throughput > 0 and
        traffic.total_tokens_per_sec > self._estimated_throughput * (
            1.0 + self._config.aic_drift_relative_threshold
        )
    )

    needs_reopt = sla_violated or capacity_exceeded

    self._consecutive_violation_ticks = (
        self._consecutive_violation_ticks + 1 if needs_reopt else 0
    )
    return self._consecutive_violation_ticks >= self._config.aic_drift_consecutive_ticks
```

**Hysteresis** (`aic_drift_consecutive_ticks`, default 3): a single transient avg spike does not trigger a 6-second AIC sweep. The EMA smooths the coefficients; hysteresis controls when we re-sweep. Both mechanisms compose: a blip doesn't move coefficients much (EMA) and doesn't trigger a sweep (hysteresis).

**Direction matters.** A previous draft used a symmetric throughput-drift trigger (`abs(observed − estimated) / estimated > threshold`). That fires on the *under-load* case too: if a cluster is sized for 1000 tok/s but currently serves 100 tok/s, `|100−1000|/1000 = 0.9 > 0.15`, so after 3 ticks the optimizer would re-sweep — and would land on a config sized for the lower load, which is wrong because *the observed throughput is bounded by demand, not by capacity*. The next time real load arrives, the optimizer would be under-provisioned and have to re-sweep again. We avoid this entire failure mode by only triggering on the upward direction: capacity exceeded means more replicas/looser caps are warranted; capacity-with-headroom means do nothing.

**Note on metric source.** This design uses `ttft_avg` and `itl_avg` (matching the existing `PrometheusAPIClient.get_avg_*` methods in `monitoring/traffic_metrics.py`), not `*_p95`. The current planner does not collect p95 — adding it is out of scope here. If post-Phase-3 production data shows averages are too forgiving (a small fraction of bad requests doesn't move the average enough), we can add p95 queries to `PrometheusAPIClient` and switch the trigger over. Tracked as a follow-up.

### 5.7 Frontend admission coupling (`/busy_threshold`)

The Dynamo frontend has an admission-control primitive: when a worker's KV-cache utilization or prefill-token utilization crosses a configured threshold, the frontend rejects new requests with HTTP 503 (see `docs/fault-tolerance/request-rejection.md` and `lib/runtime/src/pipeline/network/egress/push_router.rs::generate_with_fault_detection`). Thresholds are per-model and runtime-mutable via `POST /busy_threshold`. The power planner uses this as the **admission control surface** that pairs with replica scaling and power capping.

**Why this matters for power-aware control.** Replica scaling and power capping change the *capacity* a deployment can serve. Admission thresholds change the *demand* the frontend lets through. A power-aware loop that adjusts capacity without adjusting admission produces one of two failure modes:

| Capacity vs admission | Symptom |
|---|---|
| Capacity drops, admission stays open | Engine queue builds; ITL inflates; drops fire chaotically at high tail latency. |
| Capacity rises, admission stays tight | Replicas idle; deployment pays power for unused work. |

The fix: derive admission thresholds from the chosen operating point so the frontend sheds at exactly the QPS the engines are sized for.

**Implied threshold from the operating point.** Each row of an AIC `pareto_df` (or any picked configuration) implies a steady-state utilization on each side of the engine. By Little's law on the chosen `concurrency` and per-replica `seq/s`:

```
θ_decode_impl       = concurrency × (ISL + OSL/2)  /  kv_total_tokens
θ_prefill_frac_impl ≈ achieved_seq_s_per_replica   /  peak_seq_s_per_replica
                     (peak = 1000 / TTFT_isolated_ms)
```

Both quantities are dimensionless and bounded in [0, 1] by construction (the engine cannot exceed its KV pool or its prefill capacity), and both are computable from the row data plus `AIConfiguratorPerfEstimator.get_max_kv_tokens(...)` for `kv_total_tokens` per the picked TP/PP. The full derivation (lifetime KV occupancy, queueing-variance margin, prefill saturation behavior) lives in §5.8 alongside the AIC TaskRunner spec — no new math is added in this section.

**Absolute prefill threshold (defense-in-depth).** The fractional check on the frontend evaluates as `active_prefill_tokens > frac × max_num_batched_tokens`, where `max_num_batched_tokens` is reported per-worker via MDC. When a worker's runtime config does not report this value, the frontend falls back to `DEFAULT_MAX_TOKENS = 10_000_000` (see `lib/llm/src/discovery/worker_monitor.rs:55-58`), which silently disables the fractional threshold on that worker until MDC re-syncs. To close this hole, the planner also POSTs an absolute prefill threshold derived from the same operating point:

```
active_prefill_tokens_threshold = ceil(theta_prefill_frac_impl × min_M)
```

where `min_M = min over prefill workers of WorkerInfo.max_num_batched_tokens` (already populated from MDC at `components/src/dynamo/planner/monitoring/worker_info.py:44`). Min across workers because prefill workers in a DGD are normally homogeneous (same TP and engine config), and min is the conservative-correct cap when they are not. The frontend evaluates both prefill checks under OR logic — whichever fires first wins — so the absolute serves as a hard backstop while the fractional self-corrects under engine-config changes.

If `min_M` cannot be derived (no prefill worker has reported `max_num_batched_tokens` yet), the planner sets `active_prefill_tokens_threshold = null`, emits a CRITICAL log, and increments `dynamo_planner_admission_max_batched_tokens_unavailable_total`. The fractional threshold is also broken in that case, but the failure is an MDC outage, not an admission-control bug — alerting on this counter surfaces it for the operator. See §8 failure mode #12.

**Three planner modes** (new `PlannerConfig.admission_mode`, default `"off"`):

| Mode | Planner behavior | When to use |
|---|---|---|
| `"off"` | Never POST `/busy_threshold`. Frontend uses operator-set values (or none). Current default. | Phase 1–2 deployments; manual control. |
| `"inherit"` | Compute `θ_*_impl` for the picked config every tick; emit Prometheus gauges; do NOT push to frontend. | Observability with operator override. |
| `"autoset"` | Compute `θ_*_impl`; POST `min(1.0, admission_safety_margin × θ_*_impl)` to `/busy_threshold` after every config change. | Phase 3+ closed-loop control. |

`autoset` closes the power-aware loop: when AIC re-optimizes for a new power cap, the new `θ_*_impl` reflects the new physical capacity, and the frontend's shed point follows it. When AIC reverts to a previous config (failure mode #5 in §8), the implied threshold of the *currently-applied* config is re-POSTed — `_apply_aic_config()` idempotently pushes every time it runs. There is no separate revert path.

**Multi-replica frontend.** `/busy_threshold` writes a single frontend pod's manager state (the threshold registry isn't yet backed by etcd). When multiple frontend replicas serve the same model, the planner must POST to each. The connector method `connector.list_frontend_pods(model)` returns the set; the planner fans out POSTs and treats partial-success as ERROR (`dynamo_planner_admission_partial_success_total`). Per-pod retry budget is 5s. Tracked as Open Question #7 — a future direction is to back the threshold with etcd in the frontend, removing the fanout.

**Worked example (extending §5.1's H200 SXM, Llama-3-8B, ISL=2048, OSL=512).** AIC picks `prefill TP=1 × 1 replica` (cap_p=500W per §5.1), `decode TP=2 × 4 replicas` (cap_d=425W per §5.1), predicting concurrency=12 per decode replica with TTFT=620 ms, TPOT=18 ms. With `kv_total_tokens ≈ 200,000` for TP=2 H200 and isolated TTFT=400 ms, and prefill workers reporting `max_num_batched_tokens = 8192` via MDC:

```
θ_decode_impl              = 12 × (2048 + 512/2) / 200,000  = 0.138
peak_seq_s_per_replica     = 1000 / 400                     = 2.50
achieved_seq_s_per_replica = 1000 / 620                     = 1.61
θ_prefill_frac_impl        = 1.61 / 2.50                    ≈ 0.65

# Absolute prefill threshold (defense-in-depth, B6):
min_M                              = 8192   # min across prefill workers
active_prefill_tokens_threshold    = ceil(0.65 × 8192)   = 5325
```

`autoset` POSTs `{model, active_decode_blocks_threshold: 0.14, active_prefill_tokens_threshold: 5325, active_prefill_tokens_threshold_frac: 0.65}` to every frontend replica. The frontend's OR-of-three evaluation: a worker is busy when `active_prefill_tokens > 5325` OR `active_prefill_tokens > 0.65 × max_num_batched_tokens`. If the worker reports 8192 normally, both checks fire at the same point (5325 ≈ 0.65 × 8192). If the worker fails to report (DEFAULT_MAX_TOKENS=10M fallback), the fractional check sleeps until ~6.5M tokens but the absolute still fires at 5325 — defense-in-depth holds.

If a later tick caps to 360W on prefill (lower R_p → higher `θ_pf_impl` for the same QPS), AIC re-optimizes; the new implied thresholds (both fractional and absolute) are POSTed; the frontend now sheds earlier — exactly tracking the new physical capacity. Numbers above are illustrative; production values come from the AIC sweep, not hand calculation.

### 5.8 AIC TaskRunner extensions (admission + power)

A small set of additions to AIC's `TaskRunner` makes the picker multi-constraint: **GPUs × Latency × Admission × Power**. All new constraints default to "off"; output is identical to today at defaults.

**`TaskConfig` — new optional fields (all backward-compatible):**

```python
@dataclass
class TaskConfig:
    # ... existing ...
    total_gpus: int
    isl: int
    osl: int
    ttft: float
    tpot: float
    request_latency: float | None = None

    # NEW — admission constraints. 1.0 = off (no extra constraint beyond SLA).
    active_decode_blocks_threshold: float = 1.0
    active_prefill_tokens_threshold_frac: float = 1.0
    busy_threshold_safety_margin: float = 1.0

    # NEW — power budget. 0.0 = off.
    total_power_budget_w: float = 0.0

    # NEW (advanced) — sweep over GPU power-cap variants. None = system TDP only.
    candidate_per_gpu_power_caps_w: list[float] | None = None

    # NEW (optional) — alternate ranking key.
    picker_objective: Literal["throughput", "efficiency"] = "throughput"
```

**`pareto_df` — new always-on columns** (computed for every Pareto row regardless of whether the constraint is active; the planner reads them in `inherit` mode):

| Column | Definition |
|---|---|
| `theta_decode_impl` | per-replica steady-state decode utilization for this row |
| `theta_prefill_frac_impl` | per-replica steady-state prefill utilization for this row |
| `theta_decode_set_recommended` | `min(1.0, margin × theta_decode_impl)` — autoset target |
| `theta_prefill_frac_set_recommended` | autoset target for prefill |
| `prefill_power_w_per_gpu` | from AIC perf DB; falls back to system TDP if unpopulated |
| `decode_power_w_per_gpu` | same |
| `prefill_total_power_w` / `decode_total_power_w` / `total_power_w` | sums |
| `seq/s/W` | efficiency metric; ranking key when `picker_objective="efficiency"` |

**Filter cascade** (order is irrelevant — every constraint is a hard filter):

```python
df = pareto_df_full

# Existing constraints
df = df[df["ttft"] <= task.ttft * 1000.0]
df = df[df["tpot"] <= task.tpot * 1000.0]
df = df[df["total_gpus_needed"] <= task.total_gpus]

# NEW — admission (off when threshold == 1.0)
if task.active_decode_blocks_threshold < 1.0:
    df = df[df["theta_decode_impl"]
            <= task.active_decode_blocks_threshold * task.busy_threshold_safety_margin]
if task.active_prefill_tokens_threshold_frac < 1.0:
    df = df[df["theta_prefill_frac_impl"]
            <= task.active_prefill_tokens_threshold_frac * task.busy_threshold_safety_margin]

# NEW — power (off when budget == 0)
if task.total_power_budget_w > 0:
    df = df[df["total_power_w"] <= task.total_power_budget_w]

# Rank by max seq/s (or seq/s/W if objective=efficiency); tiebreaks per §5.4.
sort_key = "seq/s" if task.picker_objective == "throughput" else "seq/s/W"
df = df.sort_values(sort_key, ascending=False)
```

**Theta derivation (precise math).** For decode: under Little's law on KV occupancy, time-averaged active decode tokens per replica = `concurrency × (ISL + OSL/2)`; dividing by `kv_total_tokens = AIConfiguratorPerfEstimator.get_max_kv_tokens(isl, osl, **picked_kwargs)` yields `theta_decode_impl`. For prefill: peak per-replica prefill throughput at zero queueing is `1000 / TTFT_ms_isolated` sequences/s; `achieved_seq_s_per_replica = row["prefill_seq/s"] / max(1, row["prefill_dp"])`; the ratio gives `theta_prefill_frac_impl`. Both quantities are demand-invariant per-row properties — they describe the picked operating point's fingerprint on engine state, independent of arrival pattern.

**Why default off (1.0 / 0.0 / None).** `theta_*_impl ≤ 1.0` and `total_power_w ≥ 0` by construction, so default values are no-ops on the filter cascade. Pre-existing call sites (today's `run_rapid`, today's `_run_autoscale_sim`) compose unchanged. Operators opt in by setting non-default values in the DGDR spec, or via `AICPowerOptimizer` calling AIC with non-default fields. Matches the broader doc convention (§12 backward-compat).

**Power data prerequisite — TDP fallback.** The `power_w` column in the AIC perf database is currently zeroed for all rows (`components/src/dynamo/profiler/utils/aic_dataframe.py::build_prefill_row` and `build_decode_row`). Per-row power must therefore fall back to system TDP × num_gpus until the AIC team backfills measured power per (system × parallelism × batch × ISL/OSL). Fallback emits a one-time WARNING per AIC sweep so operators know they are running on TDP estimates rather than measured numbers. This lets the constraint API ship in parallel with the AIC backfill (Phase 4 deliverable).

```python
def per_gpu_power_w(row, system_spec) -> float:
    p = row.get("power_w", 0.0)
    return p if p > 0.0 else system_spec["gpu"]["tdp_w"]
```

**Effort delta vs Phase 4 baseline:** the new TaskConfig fields, output columns, and filter cascade are ~150 LOC in `aiconfigurator/sdk/task.py` and `aiconfigurator/sdk/picking.py`. The dynamo profiler `rapid.py` passes the values through; the planner consumes them in `_apply_aic_config()`. Backward-compat is preserved at every layer — every new field has an explicit "off" default.

**Worked example (illustrative — Llama-3-8B, 8× H200 SXM, ISL=2048, OSL=512, TTFT=1s, TPOT=20ms):**

| Scenario | Inputs | Rank-1 picked | total_power_w | θ_decode_impl | θ_prefill_frac_impl |
|---|---|---|---|---|---|
| **Defaults** | budget=0, thresholds=1.0 | `prefill TP=1×1`, `decode TP=2×4` (480W cap) | 6300 W (TDP fallback) | 0.14 | 0.65 |
| **5kW power budget** | budget=5000, thresholds=1.0 | `prefill TP=1×1`, `decode TP=2×3` (480W cap) | 3580 W | 0.14 | 0.65 |
| **Budget + tight prefill admission** | budget=5000, prefill_frac=0.5 | `prefill TP=2×1`, `decode TP=2×3` | 3840 W | 0.14 | 0.40 |

Property by property: power budget reshapes the topology (fewer replicas), admission reshapes it again (deeper TP buys headroom by halving TTFT), and the implied admission threshold tracks the chosen operating point automatically. Numbers above are illustrative; real picks come from the AIC sweep against the live perf database.

**Picker objective — `throughput` vs `efficiency`.** Default `throughput` matches today's `_run_autoscale_sim` ranking and is the right choice when the budget is binding. `efficiency` (`seq/s/W`) is the right choice during partial-load operation when the budget is non-binding and the deployment should leave power slack on the table — useful for the power planner during off-peak windows. The selector is exposed as a TaskConfig field rather than a separate AIC entry point so the picker stays single-pass.

---

## 6. Component Implementation

### 6.1 Kubernetes API layer

**`connectors/kubernetes_api.py`** — add `CoreV1Api` and pod-patching:

```python
class KubernetesAPI:
    def __init__(self, k8s_namespace=None):
        self.custom_api = client.CustomObjectsApi()
        self.core_api   = client.CoreV1Api()   # NEW (Phase 1)
        self.current_namespace = k8s_namespace or get_current_k8s_namespace()

    def patch_pod_annotation(self, pod_name: str, key: str, value: str) -> None:
        self.core_api.patch_namespaced_pod(
            name=pod_name,
            namespace=self.current_namespace,
            body={"metadata": {"annotations": {key: value}}},
        )

    def list_pods_by_label(self, label_selector: str) -> list:
        return self.core_api.list_namespaced_pod(
            namespace=self.current_namespace,
            label_selector=label_selector,
        ).items
```

### 6.2 Kubernetes connector layer

**`connectors/kubernetes.py`** — expose pod objects:

```python
class KubernetesConnector(PlannerConnector):
    def get_component_pods(self, sub_component_type: SubComponentType) -> list:
        service = get_service_from_sub_component_type_or_name(
            self._graph_deployment, sub_component_type=sub_component_type
        )
        if service is None:
            return []
        label_selector = (
            f"nvidia.com/dynamo-graph-deployment={self._graph_deployment_name},"
            f"nvidia.com/dynamo-service={service[0]}"
        )
        return self.kube_api.list_pods_by_label(label_selector)
```

### 6.3 Pod annotation loop

**`core/base.py`** — `NativePlannerBase._apply_power_annotations()`:

```python
async def _apply_power_annotations(self) -> None:
    """Annotate worker pods with per-GPU power limits.

    Reads the actual annotation from each Pod object returned by
    get_component_pods(). Only PATCHes when annotation is missing or wrong.
    K8s is the source of truth — no local cache.
    """
    if not self.config.enable_power_awareness:
        return
    if not isinstance(self.connector, KubernetesConnector):
        return

    pods_and_limits: list[tuple] = []
    if self.require_prefill:
        for pod in self.connector.get_component_pods(SubComponentType.PREFILL):
            pods_and_limits.append((pod, str(self.config.prefill_engine_gpu_power_limit)))
    if self.require_decode:
        for pod in self.connector.get_component_pods(SubComponentType.DECODE):
            pods_and_limits.append((pod, str(self.config.decode_engine_gpu_power_limit)))

    for pod, limit_str in pods_and_limits:
        current = (pod.metadata.annotations or {}).get(POWER_ANNOTATION_KEY)
        if current == limit_str:
            continue
        try:
            self.connector.kube_api.patch_pod_annotation(
                pod.metadata.name, POWER_ANNOTATION_KEY, limit_str,
            )
        except Exception as e:
            logger.warning(f"Failed to patch pod {pod.metadata.name}: {e}")
```

Called in the main loop after `_apply_effects()`:
```python
await self._apply_effects(effects)
await self._apply_power_annotations()   # Phase 1
```

**Why per-tick reconciliation (not annotate-on-create):**

| Phase | What changes per tick | Per-tick LIST-and-PATCH does |
|---|---|---|
| Phase 1–2 | Pods created/destroyed; caps are static `PlannerConfig` fields. | Annotates newly-created pods; re-patches if admin removed annotation; no-op in steady state. |
| Phase 3+ | Caps become dynamic (AIC re-optimization may change them ~every 5 min). | Pushes new cap to all running pods on the next tick. **This is the case the per-tick pattern is necessary for.** |

One LIST call per tick — already happening for worker-count reconciliation; zero marginal cost. Only mismatches trigger PATCH (in-memory string comparison). Alternative "annotate at create-time only" rejected: breaks forward-compat with AIC and doesn't self-heal external annotation removal.

**Worst-case PATCH rate to the K8s API server.** Three layers compose to bound the rate:
1. AIC re-optimization is gated by `aic_reoptimize_interval` (default 300s) — caps cannot change more than once every 5 min, ever.
2. Drift detection requires `aic_drift_consecutive_ticks` (default 3) of sustained signal, so a transient blip never triggers a sweep.
3. Per-pod string equality check before PATCH — a no-op tick produces zero K8s calls.

Combining these: for a DGD with `N` pods, the maximum sustained PATCH rate is `N / aic_reoptimize_interval`. For `N=1000` (well above typical Dynamo deployments) that is `~3.3 PATCH/s`; for typical `N=10–100` it is `<1 PATCH/s`. Default kube-apiserver QPS budgets per controller (5–50 QPS sustained, 10× burst) absorb this with multiple orders of magnitude headroom. No additional `min_delta_watts` rate-limit is required.

### 6.4 AIC optimizer integration

**`monitoring/aic_power_optimizer.py`** (new, Phase 3):

```python
class AICPowerOptimizer:
    def __init__(self, config: PlannerConfig): ...

    def optimize(self) -> PowerAwareConfig:
        """AIC sweep (~6s). Run via asyncio.to_thread to avoid blocking."""
        # Constrain TP to deployed cluster (from aic_interpolation.{prefill,decode}_pick)
        prefill_kwargs = picked_to_aic_model_config_kwargs(self._config.aic_interpolation.prefill_pick)
        decode_kwargs  = picked_to_aic_model_config_kwargs(self._config.aic_interpolation.decode_pick)
        # Sweep (n_p, n_d, [power_p, power_d in Phase 2]) with the §5.1 bridge
        ...

    def update_correction(
        self,
        traffic: TrafficObservation,           # for num_req / scheduled_*_tokens gates
        observed_ttft_avg: float,              # seconds; matches PrometheusAPIClient.get_avg_time_to_first_token
        observed_itl_avg: float,               # seconds; matches PrometheusAPIClient.get_avg_inter_token_latency
        # Disagg power signals (None when mode=agg):
        observed_power_w_prefill: Optional[float] = None,
        observed_power_w_decode:  Optional[float] = None,
        # Agg power signal (None when mode=disagg):
        observed_power_w_agg:     Optional[float] = None,
    ) -> None: ...
    def should_reoptimize(self, traffic: TrafficObservation) -> bool: ...  # §5.6
```

**Wired into `NativePlannerBase`:**

```python
# __init__ (after _initialize_gpu_counts populates prefill/decode_engine_num_gpu):
if self.config.enable_aic_optimizer:
    self._aic_optimizer = AICPowerOptimizer(self.config)
    initial_config = self._aic_optimizer.optimize()   # startup sweep
    self._apply_aic_config(initial_config)

# run() loop — after _apply_effects() and _apply_power_annotations():
if self._aic_optimizer:
    if self.config.mode == "disagg":
        self._aic_optimizer.update_correction(
            traffic=tick_input.traffic,
            observed_ttft_avg=self._prom_client.get_avg_time_to_first_token(...),
            observed_itl_avg=self._prom_client.get_avg_inter_token_latency(...),
            observed_power_w_prefill=self._prom_client.get_avg_per_gpu_power_by_component(
                interval=self._tick_interval_str, component="prefill",
                k8s_namespace=self._k8s_namespace,
                dgd_name=self.config.dgd_name,
                service_key=self._prefill_service_key,
            ),
            observed_power_w_decode=self._prom_client.get_avg_per_gpu_power_by_component(
                interval=self._tick_interval_str, component="decode",
                k8s_namespace=self._k8s_namespace,
                dgd_name=self.config.dgd_name,
                service_key=self._decode_service_key,
            ),
        )
    else:  # mode == "agg"
        self._aic_optimizer.update_correction(
            traffic=tick_input.traffic,
            observed_ttft_avg=self._prom_client.get_avg_time_to_first_token(...),
            observed_itl_avg=self._prom_client.get_avg_inter_token_latency(...),
            observed_power_w_agg=self._prom_client.get_avg_per_gpu_power_by_component(
                interval=self._tick_interval_str, component="agg",
                k8s_namespace=self._k8s_namespace,
                dgd_name=self.config.dgd_name,
                service_key=self._agg_service_key,
            ),
        )
    if self._aic_optimizer.should_reoptimize(tick_input.traffic):
        new_config = await asyncio.to_thread(self._aic_optimizer.optimize)
        self._apply_aic_config(new_config)
```

**`_apply_aic_config()` writes the drift-comparison reference (B8):**

```python
async def _apply_aic_config(self, new_config: PowerAwareConfig) -> None:
    self._set_replicas(new_config.n_p, new_config.n_d)
    self._set_caps(new_config.cap_p, new_config.cap_d)

    # Pin §5.6's drift-comparison reference to the new operating point.
    # Units: tokens/s aggregate across the DGD (matches traffic.total_tokens_per_sec).
    self._aic_optimizer._estimated_throughput = (
        new_config.aic_seq_per_s_per_replica * new_config.n_d
        * (new_config.isl + new_config.osl)
    )
    # ... (admission fanout below; see §6.7)
```

**What the optimizer can and cannot change at runtime:**

| Quantity | Optimizer can change? | Notes |
|----------|----------------------|-------|
| `n_p`, `n_d` (replica count) | Yes | Via existing `_apply_effects()` path |
| `prefill_engine_gpu_power_limit` (cap_p) | Yes | Driven by `c_power_p`; via `_apply_aic_config` → `_apply_power_annotations` picks it up next tick |
| `decode_engine_gpu_power_limit` (cap_d)  | Yes | Driven by `c_power_d`; same path |
| `tp_p`, `tp_d` (tensor parallelism) | **No** | Requires pod restart; TP is frozen from `AICInterpolationSpec` picks |
| `prefill/decode_engine_num_gpu` | **No** | Mechanically tied to TP |
| `bs` (batch size hint) | Phase 2 only | Out of scope for v1; likely via DGD `extraPodSpec` |

### 6.5 Power Agent DaemonSet

**`components/power_agent/power_agent.py`** — net-new file on ToT (the path does not exist on ToT today, so there is no merge conflict). Initial content is adapted from PR #5280's `power_agent.py` as a code-text source, with the improvements below. The implementation plan does not depend on PR #5280 being merged.

**Key behaviors:**
1. Privileged DaemonSet with `hostPID` (one agent per node).
2. Every 15s, for each physical GPU on the node:
   - `nvmlDeviceGetComputeRunningProcesses(handle)` — NVML returns the host PIDs of every compute context attached to **this physical GPU**. The PID→GPU mapping comes from NVML directly; cgroups are *not* used for that step.
   - For each PID, parse `/proc/{pid}/cgroup` to recover the **pod UID** (standard cAdvisor pattern). This is the only role of cgroup parsing.
   - Look up the pod UID in the per-tick `list_pod_for_all_namespaces(field_selector=spec.nodeName=...)` snapshot, read its `dynamo.nvidia.com/gpu-power-limit` annotation, and call `nvmlDeviceSetPowerManagementLimit(handle, watts × 1000)`.

   Because the agent runs `hostPID: true` + `privileged: true`, NVML always sees physical (host) GPU indices regardless of CDI/CRI namespacing on worker pods. Reading `CUDA_VISIBLE_DEVICES` from `/proc/{pid}/environ` would be **incorrect** — it is a container-local logical index that does not map to `nvmlDeviceGetHandleByIndex(physical)`.
3. Restores GPU to default TGP when annotation disappears (pod removed) or on SIGTERM.
4. Exposes `dynamo_power_agent_applied_limit_watts{gpu_index="N"}` so operators can verify NVML calls succeeded (separate from DCGM's measured `DCGM_FI_DEV_POWER_USAGE`).

**cgroup parser — all QoS × driver × runtime combinations:**

Original PR #5280 only matched Guaranteed-QoS on the systemd driver, silently failing for Burstable-QoS pods (the default for any pod with `requests < limits`). The updated parser iterates `/proc/{pid}/cgroup` lines (cgroup v1 has one line per controller; cgroup v2 has a single unified line; cri-containerd / cri-o wrap the pod slice with a container-scope segment) and uses `.search()` so the pod-slice substring matches even when it appears mid-path:

```python
# Handles the full set of K8s cgroup path variants:
#   - cgroup v1 (multi-line, one per controller) and cgroup v2 (single unified line)
#   - systemd / cgroupfs drivers
#   - Guaranteed / Burstable / BestEffort QoS classes
#   - cri-containerd, cri-o, dockershim wrappers (the pod slice may be embedded
#     within a container-scope or container-id segment along the path)
_SYSTEMD_RE  = re.compile(r"kubepods-(?:burstable-|besteffort-)?pod([a-fA-F0-9_]+)\.slice")
_CGROUPFS_RE = re.compile(r"/kubepods(?:/burstable|/besteffort)?/pod([a-fA-F0-9-]+)(?:/|$)")


def _extract_pod_uid_from_cgroup(pid: int) -> Optional[str]:
    """Recover the pod UID from /proc/{pid}/cgroup.

    Iterates lines because:
      - cgroup v1: one line per controller hierarchy; only some lines carry
        the pod-slice path (e.g., the cpu/memory controller on most setups)
      - cgroup v2: a single unified line, but cri-containerd wraps the pod
        slice with `cri-containerd-<container-id>.scope` inside the slice
      - mixed setups: hybrid kernels with both v1 and v2 mounts

    The substring search (.search not .match) handles the wrapper segments.
    Returns None if no line matches — process is non-K8s, skip silently.
    """
    try:
        with open(f"/proc/{pid}/cgroup") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for line in lines:
        m = _SYSTEMD_RE.search(line)
        if m:
            return m.group(1).replace("_", "-")  # systemd encodes dashes as underscores
        m = _CGROUPFS_RE.search(line)
        if m:
            return m.group(1)
    return None  # non-K8s process — skip
```

**Multi-pod-per-GPU policy** (misconfig fallback, not a supported topology — see §3.3):

| Situation | Action |
|---|---|
| 1 pod on GPU N | Apply that pod's annotation. |
| 2+ pods, all annotations agree | Apply the agreed value, log WARNING (multi-pod-per-GPU is misconfig even when caps match), increment `dynamo_power_agent_multi_pod_gpu_total{disposition="agree"}`. |
| 2+ pods, annotations differ | Apply `power_agent_safe_default_watts`, log ERROR, increment `dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"}` and `dynamo_power_agent_safe_default_applied_total`. |
| Annotated PIDs but no annotation parseable, GPU has a prior agent-applied cap | Keep the prior cap (NVML caps are persistent across processes), log ERROR, increment `dynamo_power_agent_apply_failures_total`. |
| Annotated PIDs but no annotation parseable, GPU is at TDP (cold start, no prior cap) | Apply `power_agent_safe_default_watts` (config field, per-SKU operator-set), log ERROR, increment `dynamo_power_agent_apply_failures_total` and `dynamo_power_agent_safe_default_applied_total`. |

**Unified fallback principle.** Every "agent cannot determine the correct cap" case — conflicting annotations, parse failure on a fresh GPU — converges to the same action: apply `power_agent_safe_default_watts`. This is deterministic, auditable (single counter to alert on), and avoids the prior MIN-cap policy's failure mode where two unsupported pods could drag a GPU below its decode floor and break inference for both. The agree-case is the only multi-pod situation that still applies a pod-derived value, because there the value isn't ambiguous — but it's still WARNING-logged because the topology itself is unsupported.

**Why fail-closed only on cold start (parse-failure case).** NVML power caps persist across processes until reboot or another `SetPowerManagementLimit()` call. So a parse failure on a *running* cluster is not safety-critical — every previously-managed GPU stays at its last-applied cap, and only log spam degrades. The genuine exposure is the cold-start path: a fresh node (or a GPU that was never capped, e.g. just-installed silicon) starts at BIOS/TDP and a parse failure on the first reconcile would leave it there. `power_agent_safe_default_watts` is the floor for that case. It is required (no default) when `enable_power_awareness=True`; the validator emits a clear error if it is unset, mirroring the `total_gpu_power_limit` placeholder check in §3.1.

**NVML cap clamping to SKU-defined constraints.** `nvmlDeviceSetPowerManagementLimit()` raises `NVML_ERROR_INVALID_ARGUMENT` when the requested cap is outside the GPU's hardware-defined min/max bounds (`nvmlDeviceGetPowerManagementLimitConstraints()`). These bounds are SKU-specific (e.g., H200 SXM ≈ [200W, 700W]; H100 PCIe ≈ [100W, 350W]). If the planner — driven by AIC's analytic model — emits a cap below `min_w` (e.g., a tiny decode replica predicted at 180W on H200), the call would error and the agent would silently leave the GPU uncapped. Defense: every NVML write goes through `_clamp_to_constraints()`, which clamps to `[min_w, max_w]`, logs a WARNING when clamping occurs, and increments `dynamo_power_agent_cap_clamped_total{direction=...}` so operators can spot when caps are being silently saturated.

```python
def _clamp_to_constraints(handle, requested_w: int, gpu_idx: int) -> int:
    """Clamp `requested_w` to the SKU-defined NVML power-cap range.

    Returns the value actually applicable. Increments dynamo_power_agent_cap_clamped_total
    when clamping occurs so operators can alert on it.
    """
    try:
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
    except pynvml.NVMLError:
        # Constraints query failed — apply requested as-is. SetPowerManagementLimit
        # will surface its own NVML_ERROR_INVALID_ARGUMENT if it is out of range.
        return requested_w
    min_w, max_w = min_mw // 1000, max_mw // 1000
    if requested_w < min_w:
        logger.warning(
            "Requested cap %d W below SKU min %d W on GPU %d; clamping up.",
            requested_w, min_w, gpu_idx,
        )
        _metrics.power_agent_cap_clamped_total.labels(direction="min").inc()
        return min_w
    if requested_w > max_w:
        logger.warning(
            "Requested cap %d W above SKU max %d W on GPU %d; clamping down.",
            requested_w, max_w, gpu_idx,
        )
        _metrics.power_agent_cap_clamped_total.labels(direction="max").inc()
        return max_w
    return requested_w


def _apply_cap(handle, gpu_idx: int, requested_w: int, pod_uid: Optional[str]) -> None:
    """All NVML cap writes go through here."""
    effective_w = _clamp_to_constraints(handle, requested_w, gpu_idx)
    pynvml.nvmlDeviceSetPowerManagementLimit(handle, effective_w * 1000)
    _managed_gpus.add(gpu_idx)                                    # in-process
    _record_managed_gpu_uuid(handle)                               # B12 — persistent
    _metrics.power_agent_applied_limit_watts.labels(gpu=gpu_idx).set(effective_w)
```

The clamp direction is split into `direction={"min","max"}` so an operator alert on the `min` direction (cap raised because AIC under-estimated the SKU floor) does not look the same as the `max` direction (cap saturated to nameplate, AIC over-estimated). The min-direction case does cost a small amount of budget headroom in `_apply_power_budget` because the actual draw can be up to `effective_w > requested_w` on those replicas — operators sizing tight to budget should treat sustained `direction="min"` increments as a signal to widen `total_gpu_power_limit` or to set the AIC offline coefficient floor higher.

**Graceful shutdown — SIGTERM handler:**

NVML caps are persistent across processes until reboot or another `SetPowerManagementLimit()` call. An ungraceful agent termination leaves GPUs capped indefinitely.

```python
_managed_gpus: set[int] = set()
_shutdown = threading.Event()

def _handle_sigterm(signum, frame):
    for gpu in _managed_gpus:
        try:
            handle   = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            default  = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default)
        except Exception:
            logger.exception("Failed to restore TGP on GPU %d", gpu)
    pynvml.nvmlShutdown()
    _shutdown.set()

signal.signal(signal.SIGTERM, _handle_sigterm)
```

DaemonSet spec sets `terminationGracePeriodSeconds: 30`.

**Cold-start orphan-cap restoration (UUID-gated).** On agent startup (after a SIGKILL, OOM, kernel panic, or first deployment), the agent restores default TDP only on GPUs that **this agent previously managed**. The list of managed GPU UUIDs is persisted to a host-bind-mounted file so it survives agent crashes:

```python
# Persistent state — survives agent restart. Path is bind-mounted from a
# host directory (the DaemonSet manifest declares the volume; see below).
_MANAGED_STATE_PATH = "/var/lib/dynamo-power-agent/managed_gpus.json"

# Module-level mirror of the persisted set; loaded once at agent startup,
# updated atomically (in-memory + on-disk) whenever _apply_cap() succeeds.
_previously_managed: set[str] = set()


def _load_previously_managed_gpus() -> set[str]:
    """Return the persisted set of GPU UUIDs previously capped by this agent."""
    try:
        with open(_MANAGED_STATE_PATH) as f:
            return set(json.load(f).get("managed_uuids", []))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()


def _persist_managed_gpus(uuids: set[str]) -> None:
    os.makedirs(os.path.dirname(_MANAGED_STATE_PATH), exist_ok=True)
    tmp = _MANAGED_STATE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"managed_uuids": sorted(uuids)}, f)
    os.replace(tmp, _MANAGED_STATE_PATH)  # atomic


def _record_managed_gpu_uuid(handle) -> None:
    """Called from _apply_cap() after every successful NVML write.

    Updates both the in-memory mirror (_previously_managed) and the persisted
    file. The atomic os.replace() in _persist_managed_gpus keeps the file
    coherent even on agent SIGKILL between writes.
    """
    uuid = pynvml.nvmlDeviceGetUUID(handle).decode("ascii")
    if uuid not in _previously_managed:
        _previously_managed.add(uuid)
        _persist_managed_gpus(_previously_managed)


def _restore_orphaned_gpus_on_startup() -> None:
    """Restore default TDP only on GPUs we (a previous incarnation of this
    agent) capped, AND that are now idle. Never touch a GPU we don't recognize:
    it might be capped by a different operator workflow (different DGD,
    nvidia-smi -pl from a node bootstrap script, vendor firmware setting)."""
    global _previously_managed
    _previously_managed = _load_previously_managed_gpus()
    for gpu_idx in range(self.device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
        uuid   = pynvml.nvmlDeviceGetUUID(handle).decode("ascii")
        if uuid not in _previously_managed:
            continue  # never managed by us — leave alone
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        if procs:
            continue  # workload running — let normal reconcile handle it
        current = pynvml.nvmlDeviceGetPowerManagementLimit(handle) // 1000
        default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) // 1000
        if current < default:
            pynvml.nvmlDeviceSetPowerManagementLimit(handle, default * 1000)
            _previously_managed.discard(uuid)
    _persist_managed_gpus(_previously_managed)
```

This handles the common SIGKILL recovery case: agent dies → pod is later deleted (cap is now orphaned) → agent restarts → orphan is restored. **No external bootstrap script is required for this case.** UUID-gating means the agent will never revert a cap it didn't apply — including caps applied by a co-tenant DGD, a manual `nvidia-smi -pl`, or vendor firmware defaults.

**DaemonSet manifest — persistent state volume.** The persistent state directory is host-bind-mounted so it survives container restart:

```yaml
volumes:
  - name: power-agent-state
    hostPath:
      path: /var/lib/dynamo-power-agent
      type: DirectoryOrCreate
volumeMounts:
  - name: power-agent-state
    mountPath: /var/lib/dynamo-power-agent
```

The file is small (a JSON list of UUIDs) and host-local. No PVC required. Worst case loss of the file (host disk wipe) reduces the agent to its pre-B12 behavior — i.e., it will not revert orphan caps until it has applied at least one — which is strictly safer than the alternative.

**SIGKILL recovery — UUID-gated behavior matrix:**

| Scenario | Behavior on agent restart | Why this is correct |
|---|---|---|
| Agent SIGKILL'd, pod still running with the prior cap | Skip the GPU (it has running processes); on the next reconcile tick, re-derive the cap from the pod's annotation (no-op if unchanged). | The running pod still wants its cap. Reverting to TGP would let it suddenly draw nameplate power. |
| Agent SIGKILL'd, pod deleted, GPU now idle, **GPU UUID in `managed_gpus.json`** | Restore to default TGP. Next pod scheduled here starts fresh and gets capped on its first reconcile (≤15s). | Standard orphan recovery; the agent owned this GPU previously, so reverting is safe. |
| Agent SIGKILL'd, pod deleted, GPU now idle, **GPU UUID NOT in `managed_gpus.json`** | Skip. Whatever cap is in place (firmware default, operator-set, co-tenant DGD) stays. | Defense-in-depth: agent only touches GPUs it has previously managed. Prevents inadvertently overwriting another workflow's cap. |
| Agent SIGKILL'd, pod deleted, but a *non-Dynamo* workload now runs on the GPU | Skip (the GPU is no longer idle). Cap remains at the prior Dynamo value. | Rare, but a hard case for any automated recovery. The non-Dynamo workload sees the stale cap; operator-side `nvidia-smi -pl <default>` in node-bootstrap remains the recommended belt-and-braces fix for shared-tenant nodes. |
| `managed_gpus.json` lost (host disk wipe / first deployment on the node) | Restore nothing on the cold cycle. After the first cap-application tick, the file is recreated; subsequent restarts behave normally. | Strictly safer than the alternative — the agent never reverts a cap it doesn't *know* it applied. |

A *separate* node-bootstrap DaemonSet was considered and rejected: it would race with the Power Agent itself, would need the same NVML-write privilege (doubling the trust-boundary surface), and would not handle the third row above any better than the operator-side script does.

**Power convergence window:** ≤15s after pod scale-up before the agent applies its cap.

For LLM inference workloads this gap is operationally bounded:

| Startup phase | Power profile | Safety |
|---|---|---|
| Weight load (disk → HBM) | Memory-bandwidth-bound; ~150–250W on H100/H200. Far below TGP. | Trivially safe. |
| NCCL init / topology discovery | Communication-bound (small buffers timing the interconnect). SM occupancy near zero. | Safe. |
| Engine warm-up / CUDA-graph capture (vLLM, TRT-LLM, SGLang) | Real GEMMs at small batch sizes / warm-up shapes. **Empirically draws ~300–500W on H100/H200, i.e. up to ~70% of TDP.** | Below the cap that the planner has *budgeted* for, but **above** the cap the agent will eventually apply. |
| Steady-state serving | Bounded by the applied NVML cap. | Safe. |

The warm-up draw (~70% of TDP for ≤15s) is the actual operator-facing concern, *not* a TDP-level spike. Three structural protections keep this from becoming a rack-PDU event:

1. **The pod count is already budget-bounded.** `_apply_power_budget()` (§5.2) sizes new replica counts assuming the cap is in effect; the planner never asks K8s to start more pods than `total_gpu_power_limit / cap_per_pod`. So even if every newly-scheduled pod briefly draws warm-up power, the *count* of such pods is bounded.
2. **Pods are not routed traffic until they register with the frontend/router**, which happens after engine warm-up completes. So a "cap-not-yet-applied" pod is also pre-routing — its draw is bounded by warm-up shapes, not full prod load.
3. **Operator sizing guidance.** The §3.3 formula `total_gpu_power_limit = (rack_capacity_W × headroom_factor) − non_gpu_overhead` should size the headroom against the *warm-up* power profile of a simultaneous-restart event, not steady-state cap. Recommended `headroom_factor ≤ 0.85` when the deployment is allowed to scale up multiple pods per tick on the same rack.

What the 15s window does **not** protect against: SIGKILL of the agent on a node where uncapped pods are already serving (covered in §6.5's SIGKILL/OOM operator-side `nvidia-smi -pl <default>` boot-script recommendation), and pre-Phase-1 deployments where no agent runs at all (those simply have no cap enforcement; orthogonal to this design).

What about an InitContainer that sets the cap before the main container starts? Considered and rejected for v1: (a) device assignment from the NVIDIA device plugin happens at *container* start, so the InitContainer cannot know which physical GPU index to cap; (b) it would require privileged + NVML write capability in *every* worker pod, multiplying the trust-boundary surface that the single-DaemonSet design deliberately confines to one place per node; (c) it does not help the SIGKILL/OOM recovery case, which is the residual hardware-safety hole. Future work could move enforcement into the NVIDIA k8s-device-plugin or a CDI hook — tracked as open question #3, deferred until operator feedback.

### 6.6 Multi-DGD operator playbook

Per-DGD scope (§3.3) means the planner sizes power for the workers it owns and *does not coordinate with other DGDs*. Operators running multi-DGD clusters need a small set of practices to keep the per-DGD model safe.

**Recommended deployment patterns** (in order of preference):

1. **One DGD per node pool.** Each DGD targets a disjoint set of GPU nodes (label/taint-based). Power budgets are per-pool and never overlap. This is the only configuration where the planner's invariants hold without operator-side coordination.
2. **Multiple DGDs, disjoint GPUs, shared nodes.** Supported. Operator must ensure `Σ total_gpu_power_limit ≤ facility_capacity − non_gpu_overhead − headroom`. The planner cannot detect over-commitment across DGDs.
3. **Multiple DGDs sharing one physical GPU (MIG / MPS / time-slicing).** **Out of scope v1.** Two DGDs would race to set the same pod's annotation; the Power Agent's last-writer-wins behavior is not a sound contract. If you need shared-GPU multi-tenancy, consolidate into a single DGD or wait for v2.

**Detection alerts for misconfiguration:**

| Symptom | PromQL alert | What it means |
|---|---|---|
| Conflicting annotations on a GPU (multi-DGD writing to the same pod's GPU, or pattern #3 above) | `increase(dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"}[5m]) > 0` | An unsupported topology has been deployed. The agent has applied `power_agent_safe_default_watts` on the affected GPUs; investigate before pods miss SLA. |
| Safe-default fallback is being hit at all | `increase(dynamo_power_agent_safe_default_applied_total[15m]) > 0` | Either a multi-pod conflict or a cold-start parse failure. Both warrant an operator look. |
| Cluster-wide power approaching budget across all DGDs | `sum(DCGM_FI_DEV_POWER_USAGE) > facility_capacity_w * 0.9` | Multi-DGD over-commitment. Recompute `Σ total_gpu_power_limit` against facility capacity. |
| A single DGD's projected power exceeds its declared budget | `dynamo_planner_power_projected_watts > dynamo_planner_power_budget_total_watts` | `_apply_power_budget` clamping is active; either traffic outgrew sizing or budget was set too tight. |

**Escalation steps when `multi_pod_gpu_total{disposition="conflict"}` increments:**

1. `kubectl get pods --field-selector spec.nodeName=<node> -o json` — find which pods landed on the same GPU; check their `nvidia.com/dynamo-graph-deployment` labels.
2. If the pods belong to different DGDs: confirm whether they share a physical GPU (CDI selector, device plugin assignment). If yes, this is the unsupported MIG/MPS pattern — separate the DGDs onto disjoint GPUs.
3. If the pods belong to the same DGD: inspect annotations directly with the command in `components/power_agent/README.md` Troubleshooting section. A planner bug or an external mutator is the likely cause.
4. While diagnosing, the affected GPUs are at `power_agent_safe_default_watts`, which is conservative-correct for any single DGD on that GPU. No urgent intervention is required to protect hardware; SLA recovery may need scale-out.

A separate operator playbook (in `examples/deployments/powerplanner/MULTI_DGD.md`) is tracked as Phase 1 deliverable scope.

### 6.7 `/busy_threshold` connector integration

A thin connector method enables `autoset` mode without adding a new HTTP client to the planner. Wired only when `admission_mode != "off"`.

**`connectors/kubernetes.py` — additions:**

```python
class KubernetesConnector(PlannerConnector):
    def list_frontend_pods(self) -> list[V1Pod]:
        """Frontend pods carry nvidia.com/dynamo-service=frontend by convention."""
        label_selector = (
            f"nvidia.com/dynamo-graph-deployment={self._graph_deployment_name},"
            f"nvidia.com/dynamo-service=frontend"
        )
        return self.kube_api.list_pods_by_label(label_selector)

    async def post_busy_threshold(
        self,
        pod: V1Pod,
        model: str,
        port: int,
        active_decode_blocks_threshold: Optional[float],
        active_prefill_tokens_threshold: Optional[int],         # absolute tokens — B6
        active_prefill_tokens_threshold_frac: Optional[float],
    ) -> None:
        url = f"http://{pod.status.pod_ip}:{port}/busy_threshold"
        body: dict = {"model": model}
        if active_decode_blocks_threshold is not None:
            body["active_decode_blocks_threshold"] = active_decode_blocks_threshold
        if active_prefill_tokens_threshold is not None:
            body["active_prefill_tokens_threshold"] = active_prefill_tokens_threshold
        if active_prefill_tokens_threshold_frac is not None:
            body["active_prefill_tokens_threshold_frac"] = active_prefill_tokens_threshold_frac
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
```

**`core/base.py` helper — minimum `max_num_batched_tokens` across prefill workers (B6):**

```python
def _min_prefill_max_num_batched_tokens(self) -> Optional[int]:
    """Return min over prefill workers of WorkerInfo.max_num_batched_tokens.

    Used by the planner to derive an absolute prefill admission threshold
    that the frontend can evaluate even when individual workers fail to
    report max_num_batched_tokens to MDC (frontend's DEFAULT_MAX_TOKENS
    fallback would otherwise silently disable the fractional check).

    Returns None when no prefill worker has reported a value yet.
    """
    workers = self._worker_registry.list_workers(component="prefill")
    values = [w.max_num_batched_tokens for w in workers if w.max_num_batched_tokens]
    return min(values) if values else None
```

**Wired into `NativePlannerBase._apply_aic_config()`** (extending §6.4):

```python
async def _apply_aic_config(self, new_config: PowerAwareConfig) -> None:
    self._set_replicas(new_config.n_p, new_config.n_d)
    self._set_caps(new_config.cap_p, new_config.cap_d)

    # Pin §5.6's drift reference (B8).
    self._aic_optimizer._estimated_throughput = (
        new_config.aic_seq_per_s_per_replica * new_config.n_d
        * (new_config.isl + new_config.osl)
    )

    # Always record implied θ as gauges (inherit + autoset paths).
    if self.config.admission_mode != "off":
        self._metrics.admission_implied_theta_decode.set(new_config.theta_decode_impl)
        self._metrics.admission_implied_theta_prefill_frac.set(new_config.theta_prefill_frac_impl)

    # autoset: fan out POSTs to every frontend replica.
    if self.config.admission_mode == "autoset":
        margin = self.config.admission_safety_margin
        theta_d_set  = min(1.0, margin * new_config.theta_decode_impl)
        theta_pf_set = min(1.0, margin * new_config.theta_prefill_frac_impl)

        # B6: derive absolute prefill threshold from the same operating point.
        # Defense-in-depth against frontend's DEFAULT_MAX_TOKENS=10M fallback
        # when a prefill worker fails to report max_num_batched_tokens.
        min_M = self._min_prefill_max_num_batched_tokens()
        if min_M is not None:
            theta_p_abs_set: Optional[int] = math.ceil(theta_pf_set * min_M)
            self._metrics.admission_set_theta_prefill_abs.set(theta_p_abs_set)
        else:
            theta_p_abs_set = None
            self._metrics.admission_max_batched_tokens_unavailable_total.inc()
            logger.critical(
                "No prefill worker reports max_num_batched_tokens; cannot derive "
                "absolute prefill admission threshold. Fractional threshold may "
                "also be ineffective on workers that haven't synced MDC. "
                "Investigate worker MDC plumbing."
            )

        self._metrics.admission_set_theta_decode.set(theta_d_set)
        self._metrics.admission_set_theta_prefill_frac.set(theta_pf_set)

        pods = self.connector.list_frontend_pods()
        results = await asyncio.gather(*(
            self.connector.post_busy_threshold(
                pod, self.config.served_model_name,
                port=self.config.frontend_http_port,                       # B5
                active_decode_blocks_threshold=theta_d_set,
                active_prefill_tokens_threshold=theta_p_abs_set,           # B6 — absolute
                active_prefill_tokens_threshold_frac=theta_pf_set,
            ) for pod in pods
        ), return_exceptions=True)
        failed = [p.metadata.name for p, r in zip(pods, results) if isinstance(r, Exception)]
        if failed:
            self._metrics.admission_partial_success_total.inc(len(failed))
            logger.error(
                "Failed to set busy_threshold on %d/%d frontends: %s",
                len(failed), len(pods), failed,
            )
```

`admission_mode == "inherit"` skips the POST step but still records `theta_*_impl` to Prometheus for dashboards. `admission_mode == "off"` skips both gauges and POSTs.

**Idempotency.** Every `_apply_aic_config()` call POSTs the implied threshold of the *currently-applied* config — no separate revert path. If AIC reverts to a previous config (failure mode #5 in §8), the previous-config implied θ is re-POSTed automatically.

**Per-tick cost.** One LIST call (cached by the existing `connector` machinery) plus one POST per frontend replica. Frontend replica counts are typically 1–5, so per-tick POST volume is bounded. Frequency is gated upstream by `aic_reoptimize_interval` (default 300s) and `aic_drift_consecutive_ticks` (default 3) — same guards as §6.3 Phase-1 pod annotation rate analysis.

---

## 7. Prometheus Observability

These metrics are for **dashboard observability only** — they do NOT feed into `_apply_power_budget()` enforcement, which uses only static config values. If DCGM goes down, dashboard metrics go stale but budget enforcement is unaffected.

**`monitoring/traffic_metrics.py` — `PrometheusAPIClient` additions:**

> **Selector note (DCGM workload attribution):** the DCGM exporter
> rewrites kubelet pod-info into `exported_pod` and `exported_namespace`
> labels.  The bare `pod` and `namespace` labels identify the *DCGM
> exporter's own* pod and namespace, not the workload — using them here
> would silently match nothing once attribution is enabled.  The pod-name
> regex must accept the operator's `<dgd>-<replica-idx>-<service-key-lc>-<hash>`
> form (e.g. `qwen3-quickstart-0-vllmworker-86nvj`).

```python
def get_total_dgd_power(self, k8s_namespace: str, dgd_name: str) -> Optional[float]:
    """Sum of DCGM-reported per-GPU watts across this DGD (all components).

    Used for the planner_power_projected_watts dashboard gauge.
    """
    result = self.prom.custom_query(
        'sum(DCGM_FI_DEV_POWER_USAGE'
        f'{{exported_namespace="{k8s_namespace}",'
        f'  exported_pod=~"^{dgd_name}-[0-9]+-.*"}})'
    )
    ...

def get_avg_per_gpu_power_by_component(
    self,
    interval: str,                  # e.g. "180s" — same as adjustment_interval
    k8s_namespace: str,             # NOT the dynamo logical namespace
    dgd_name: str,
    component: str,                 # "prefill" | "decode" | "agg" — used for logging
    service_key: str,               # e.g. "VllmDecodeWorker" — lowercased into pod regex
) -> Optional[float]:
    """Per-GPU average draw over the last `interval`, restricted to one
    component (prefill / decode / agg). Drives the per-component c_power
    EMA update in §5.3."""
    if not service_key:
        return None  # cannot build a meaningful pod regex
    result = self.prom.custom_query(
        f'avg_over_time('
        f'  avg(DCGM_FI_DEV_POWER_USAGE{{exported_namespace="{k8s_namespace}",'
        f'      exported_pod=~"^{dgd_name}-[0-9]+-{service_key.lower()}-.*"}})'
        f'[{interval}:])'
    )
    ...
```

The `pod=~"{dgd_name}-{component}-.*"` selector relies on the standard naming convention emitted by the Dynamo K8s operator. See §13 Open Question #12 for the operator-naming-override case (label-based fallback path).

**`monitoring/planner_metrics.py` — `PlannerPrometheusMetrics` additions:**

```python
self.power_budget_total_watts = Gauge(f"{PREFIX}_power_budget_total_watts", ...)
self.power_projected_watts    = Gauge(f"{PREFIX}_power_projected_watts", ...)
self.power_budget_utilization = Gauge(f"{PREFIX}_power_budget_utilization", ...)

# Phase 3+: AIC latency coefficients
self.aic_c_ttft                 = Gauge(f"{PREFIX}_aic_c_ttft", ...)
self.aic_c_itl                  = Gauge(f"{PREFIX}_aic_c_itl", ...)

# B1: Per-component power coefficients. Single labeled gauge instead of three
# separate metrics; queries can filter by component={"prefill","decode","agg"}.
# Disagg deployments emit prefill+decode samples; agg deployments emit agg.
self.aic_c_power                = Gauge(
    f"{PREFIX}_aic_c_power",
    "EMA-smoothed AIC power correction coefficient.",
    labelnames=("component",),
)

# Pegged-at-clamp counter — labeled by which coefficient saturated (B1).
self.aic_correction_pegged_total = Counter(
    "dynamo_aic_correction_pegged_total",
    "Times an AIC correction coefficient pegged at its [0.5, 2.0] clamp.",
    labelnames=("coefficient",),  # "ttft" | "itl" | "power_prefill" | "power_decode" | "power_agg"
)

self.aic_consecutive_failures   = Gauge("dynamo_aic_consecutive_failures", ...)
self.aic_optimizer_exceptions   = Counter("dynamo_aic_optimizer_exceptions_total", ...)

# B6: Admission control — absolute prefill threshold + MDC-availability counter.
self.admission_set_theta_prefill_abs = Gauge(
    f"{PREFIX}_admission_set_theta_prefill_abs",
    "Absolute prefill admission threshold (tokens) POSTed to the frontend.",
)
self.admission_max_batched_tokens_unavailable_total = Counter(
    f"{PREFIX}_admission_max_batched_tokens_unavailable_total",
    "Times the planner could not derive an absolute prefill threshold because "
    "no prefill worker had reported max_num_batched_tokens via MDC.",
)

# B10: NVML cap clamping counter (Power Agent side).
self.power_agent_cap_clamped_total = Counter(
    "dynamo_power_agent_cap_clamped_total",
    "Times the Power Agent clamped a requested cap to the SKU NVML constraints.",
    labelnames=("direction",),  # "min" | "max"
)
```

**Useful PromQL.** Surfaces operators will want when debugging:

```promql
# Disagg: spread between prefill and decode power coefficients.
# Large positive number means decode draws more than prefill relative to AIC's
# estimates — i.e. the workload is more memory-bound than AIC modelled.
dynamo_aic_c_power{component="decode"} - dynamo_aic_c_power{component="prefill"}

# Pegged-coefficient alert (calibration signal #6 in §8). Per-coefficient.
increase(dynamo_aic_correction_pegged_total{coefficient=~"power_.*"}[10m]) > 0

# B6 alert: prefill workers stop reporting max_num_batched_tokens.
increase(dynamo_planner_admission_max_batched_tokens_unavailable_total[5m]) > 0

# B10 alert: NVML caps are being clamped to SKU bounds (operator should widen
# total_gpu_power_limit or set the AIC offline coefficient floor higher).
increase(dynamo_power_agent_cap_clamped_total[10m]) > 0
```

---

## 8. Failure Modes (Phase 3+)

Principle: **fail-open at runtime, fail-closed at startup**. Once serving traffic, never crash on AIC bugs — the autoscaling layer keeps working without AIC. At startup, misconfiguration is loud.

| # | Failure | When | Behaviour | Signal |
|---|---------|------|-----------|--------|
| 1 | AIC empty feasible set at **startup** | First `optimize()` call | Scale to `min_endpoint`, disable optimizer for this run, fall back to static `_apply_power_budget`. Planner still serves. | ERROR log; `dynamo_aic_optimizer_disabled_reason{reason="infeasible_at_startup"}=1` |
| 2 | AIC empty feasible set on **re-optimization** | Drift-triggered sweep | Keep `_last_optimal_config`. Increment `dynamo_aic_consecutive_failures`. If ≥ `aic_max_consecutive_failures` → promote to #1. | WARNING per occurrence; ERROR + state change at disable |
| 3 | AIC sweep **exception** at runtime | Inside `optimize()` | Catch, log traceback, keep `_last_optimal_config`. Same consecutive-failure counter as #2. | ERROR with traceback; `dynamo_aic_optimizer_exceptions_total` |
| 4 | AIC sweep **exception at startup** | Inside `__init__` | Catch, log, disable optimizer for this run. Do not crash pod — autoscaling layer stays alive. | ERROR + traceback; `dynamo_aic_optimizer_disabled_reason{reason="startup_exception"}=1` |
| 5 | Re-optimization produces **lower predicted throughput** | After `optimize()` | Apply config (optimizer is authority — correction coefficients already factored in). Emit WARNING with old/new numbers. | WARNING; `dynamo_aic_throughput_regression_total` |
| 6 | Correction coefficients **pegged at clamp** for >5 min | `update_correction()` EMA | Do NOT auto-disable. Emit CRITICAL — AIC calibration is far off modelled silicon. | CRITICAL log; `dynamo_aic_correction_pegged_total` |
| 7 | Power Agent NVML apply failure | Power Agent metrics | Out of scope here — owned by Power Agent. AIC-side signal: `c_power_p` / `c_power_d` (or `c_power_agg`) drift → re-optimize to more conservative config. | `dynamo_power_agent_apply_failures_total` |
| 8 | `aic_interpolation` ConfigMap malformed | At startup (`model_validator`) | Fail fast — Pydantic validation error. Pod crash-loops with clear message. | Pod NotReady; validation error in logs |
| 9 | All frontend `/busy_threshold` POSTs fail | `_apply_aic_config()` autoset path | Log ERROR. Replica/cap changes still apply; frontend continues with stale thresholds (the prior implied values), which is conservative-correct because they were derived from a valid prior operating point. The fanout converges on the next `_apply_aic_config()`. | ERROR with failed pod list; `dynamo_planner_admission_partial_success_total` |
| 10 | Partial frontend POST success | autoset path, multi-replica frontend | Same as #9 — no rollback, loud counter, eventual convergence. | `dynamo_planner_admission_partial_success_total` (incremented by failure count) |
| 11 | Frontend returns 404 (model not yet discovered) | First POST after fresh deploy | Frontend's `/busy_threshold` requires the model to be registered first. Treat as transient — same path as #9 (log + counter), no special handling. Subsequent ticks succeed once the model registers (typically <30s). | `dynamo_planner_admission_partial_success_total` |
| 12 | No prefill worker reports `max_num_batched_tokens` to MDC | `_apply_aic_config()` autoset path (B6) | Cannot derive absolute prefill admission threshold. POST decode + fractional thresholds only. Frontend's `DEFAULT_MAX_TOKENS = 10_000_000` fallback may also defang the fractional check on those workers, so admission is effectively wide-open until MDC re-syncs. CRITICAL log + counter; subsequent ticks self-heal once MDC populates. | CRITICAL log; `dynamo_planner_admission_max_batched_tokens_unavailable_total` |
| 13 | NVML cap clamped to SKU constraints | Power Agent `_apply_cap()` (B10) | Requested cap is outside `nvmlDeviceGetPowerManagementLimitConstraints()` range. Agent clamps to `[min_w, max_w]`, applies the clamped value, logs WARNING. `direction="min"` consumes a small amount of `_apply_power_budget` headroom (actual draw can exceed AIC estimate); `direction="max"` means AIC predicted above SKU TDP. Sustained increments → operator should widen `total_gpu_power_limit` or raise the AIC offline floor. | WARNING; `dynamo_power_agent_cap_clamped_total{direction=...}` |

**Disable recovery:** no automatic re-enable. Operator must restart the pod with the underlying issue fixed. Prevents flapping.

---

## 9. Testing Strategy

### 9.0 Verified test pass — clean from-scratch repro (2026-05-10)

The patchset described in this design has been validated end-to-end by tearing
down the dev workload, redeploying via
[`docs/components/planner/dpp-dev-env.md`](../components/planner/dpp-dev-env.md)
Step 1–7, and running the full test suite against a live Azure AKS dev
cluster.

| Suite | Pass | Skip | Fail | Time |
|-------|-----:|-----:|-----:|------|
| `planner/tests/unit/` (22 modules) | 465 | 0 | 0 | ~10 s |
| `planner/tests/integration/test_aic_power_e2e_sim.py` | 15 | 0 | 0 | ~5 s |
| `planner/tests/integration/test_aic_power_optimizer.py` | 34 | 0 | 0 | ~5 s |
| `planner/tests/integration/test_metric_paths_live.py` (no traffic) | 22 | 3 | 0 | ~2 s |
| `planner/tests/integration/test_metric_paths_live.py` (after one chat completion) | 23 | 2 | 0 | ~2 s |
| `planner/tests/integration/test_actuation_knobs_live.py` | 10 | 1 | 0 | ~6 s |
| `planner/tests/integration/test_actuation_knobs_live.py::TestScalingRealMutation` (`RUN_DISRUPTIVE_TESTS=1`) | 1 | 0 | 0 | ~6 s |
| `power_agent/tests/` | 43 | 0 | 0 | <1 s |
| **Total (cold deploy, no traffic)** | **590** | **4** | **0** | **~35 s** |
| **Total (after sanity-check chat completion)** | **591** | **3** | **0** | **~35 s** |

**Documented skips (not bugs):**

1. `test_metric_paths_live.py::TestPrometheusFrontendMetrics::test_frontend_metric_series_exists` — skipped on a cold deploy with no traffic; passes after even one chat completion has been sent (see the sanity-check snippet in [`docs/components/planner/dpp-dev-env.md`](../components/planner/dpp-dev-env.md)).
2-3. `test_metric_paths_live.py::TestDirectRouterMetricsClientLive::*` — the `qwen3-quickstart` DGD uses the default LocalRouter, which does not register the `dynamo_frontend_worker_*` gauges (see Open Question #14). These pass on KV-router-mode frontends.
4. `test_actuation_knobs_live.py::TestScalingRealMutation` — opt-in disruptive test, gated by `RUN_DISRUPTIVE_TESTS=1`. Passes when enabled.

**Test fixes landed in this validation pass (test bugs, not product bugs):**

| File | Fix |
|------|-----|
| `tests/unit/test_kubernetes_connector.py::test_kubernetes_connector_no_env_var` | Test assumed `DYN_PARENT_DGD_K8S_NAME` was unset, but the dev pod always sets it. Added `monkeypatch.delenv("DYN_PARENT_DGD_K8S_NAME", raising=False)` to make the test hermetic. |
| `tests/integration/test_aic_power_optimizer.py::_make_config` | Added `aic_throughput_regression_warn_threshold` as a kwarg (was hardcoded to 0.20, blocking `test_regression_increments_counter`); widened `total_gpu_power_limit` type to `int \| None`. |
| `tests/integration/test_aic_power_optimizer.py::TestOptimizeBudgetConstraint::test_unbounded_budget` | Replaced `total_gpu_power_limit=None` with `200_000` (validator rejects None when `enable_power_awareness=True`). |
| `tests/integration/test_aic_power_optimizer.py::TestUpdateCorrection::test_power_decode_ema_gated_independently` | Changed `observed_power_w_decode` from `700.0` to `850.0`; the synthetic `last_optimal_config` has `aic_power_w_decode=700.0`, so the original observation produced ratio=1.0 and the EMA correctly didn't move — defeating the assertion. |

**Reproducing this test pass:** see the "From-Scratch Repro Script" section in
[`docs/components/planner/dpp-dev-env.md`](../components/planner/dpp-dev-env.md).
It is the canonical recipe and it produced exactly the numbers above on a
clean AKS namespace.

### Phase 1 tests

| Level | What | Where |
|-------|------|-------|
| Unit | `PlannerConfig` validation with power fields | `tests/unit/test_planner_config.py` (extend) |
| Unit | Prometheus power query mocking | `tests/unit/test_prometheus.py` (extend) |
| Unit | `_extract_pod_uid_from_cgroup` — line iteration over `/proc/<pid>/cgroup`, all QoS × driver × runtime variants: cgroup v1 multi-line; cgroup v2 single-line; systemd / cgroupfs drivers; Guaranteed / Burstable / BestEffort QoS; cri-containerd / cri-o / dockershim wrapper segments; mixed v1+v2 hybrid hosts; non-K8s process (returns None) | `components/power_agent/tests/test_cgroup_parser.py` (new) |
| Unit | Multi-pod-per-GPU policy: 1 pod (apply), 2 agreeing (apply + WARN), 2 conflicting (safe-default + ERROR), parse failure on warm GPU (keep prior), parse failure on cold GPU (safe-default) | `components/power_agent/tests/test_multi_pod_policy.py` (new) |
| Unit | SIGTERM handler: restores TGP for all managed GPUs, calls `nvmlShutdown` exactly once | `components/power_agent/tests/test_shutdown.py` (new) |
| Integration | Pod annotation via `KubernetesConnector` | `tests/unit/test_kubernetes_connector.py` (extend) |
| E2E | 17-test verification suite | `examples/deployments/powerplanner/verify_poweraware.bash` |
| E2E | DaemonSet rollout: old pod restores TGP during grace period, new pod re-applies | extend `verify_poweraware.bash` |

### Phase 2 tests — `_apply_power_budget`

```python
def _make_sm(config, p_gpu=1, d_gpu=1):
    sm = PlannerStateMachine(config)
    sm.update_capabilities(WorkerCapabilities(
        prefill=ComponentCapabilities(num_gpu=p_gpu),
        decode=ComponentCapabilities(num_gpu=d_gpu),
    ))
    return sm

# 1. No-op when feature flag off.
assert _make_sm(config_disabled)._apply_power_budget(4, 4) == (4, 4)

# 2. No-op when capabilities not yet populated (early ticks). Must not crash.
assert PlannerStateMachine(config_enabled)._apply_power_budget(4, 4) == (4, 4)

# 3. No-op when projected ≤ budget.
assert _make_sm(config_large_budget)._apply_power_budget(2, 2) == (2, 2)

# 4. Even-budget: 4×250+4×250=2000W vs 1500W budget → (3,3) = 1500W exactly.
assert _make_sm(config_1500w)._apply_power_budget(4, 4) == (3, 3)

# 5. Tight-budget: 2000W vs 1499W → (2,3) = 1250W. (3,3) and (2,4) both = 1500 > 1499.
assert _make_sm(config_1499w)._apply_power_budget(4, 4) == (2, 3)

# 6. Asymmetric TP: p_gpu=8, d_gpu=1, both 300W, (2,4), budget=4000W.
#    p_watts=2400, d_watts=300, projected=6000 > 4000.
#    max_p = floor((4000-300)/2400) = 1. scaled_p=1, remaining=1600, scaled_d=min(4,5)=4.
assert _make_sm(cfg, p_gpu=8, d_gpu=1)._apply_power_budget(2, 4) == (1, 4)

# 7. Budget below min_endpoint sum → (0, 0) with warning, not silent over-allocation.
assert _make_sm(config_tiny_budget)._apply_power_budget(4, 4) == (0, 0)

# 8. Budget exactly equal to min_endpoint sum → (min_endpoint, min_endpoint).
assert _make_sm(config_min_budget)._apply_power_budget(4, 4) == (1, 1)

# 9. Demand-respect: decode must NOT up-scale above original num_d.
p, d = _make_sm(cfg, p_gpu=4, d_gpu=1)._apply_power_budget(10, 2)
assert d <= 2   # min(num_d, …) cap
assert p == 4

# Budget-pipeline composition tests:
# 10. GPU zero-out propagates: power budget is no-op on (0,0) input.
# 11. Power zero-out: GPU passes (4,4) but power can't fit min_endpoint → (0,0).
# 12. Both budgets bite: GPU→(3,3), then power→(2,3).
# 13. Order matters: GPU-then-power ≠ power-then-GPU on asymmetric inputs (pins convention).
# 14. Power-disabled passthrough: pipeline collapses to GPU-budget-only.
# 15. Capabilities-not-yet-populated passthrough: both budgets are no-ops.
```

### Phase 3 tests — AIC optimizer (mocker backend)

**What the mocker test validates:**

| Concern | How exercised |
|---------|--------------|
| Optimizer wires into `__init__()` + `run()` correctly | Startup sweep writes caps to `PlannerConfig`; mocker pods get annotations |
| §5.1 bridge plumbs through to `_apply_power_budget()` | Budget enforcement sees AIC-set caps, enforces `cap × N ≤ budget` |
| TP constraint fires | Same budget, two different `prefill_pick.tp` values → different `n_p`, same TP |
| Hysteresis suppresses transient spikes | 1-tick TTFT spike via mocker latency knob → `should_reoptimize` does not fire |
| Drift detection with sustained signal | 5-tick sustained TTFT regression → fires after `aic_drift_consecutive_ticks` |
| Validators reject misconfiguration | `enable_aic_optimizer=True` without `aic_interpolation` → crash-loop with clear error |
| §8 failure modes rows 1–5 | Unsupported system → auto-disable; tight budget at startup → `min_endpoint` fallback |

**What the mocker test does NOT validate:** silicon-vs-AIC drift (mocker is a closed AIC→AIC loop, so `c_ttft`, `c_itl`, and the power coefficients `c_power_p` / `c_power_d` / `c_power_agg` all converge to ~1.0 by construction); NVML enforcement (Power Agent is not exercised); power burst dynamics (hardware property the on-chip controller enforces).

Test location: `planner/tests/integration/test_aic_power_optimizer.py` (new).

---

## 10. Roadmap

### Phase 1 — Infrastructure pipeclean (PR 1)

| Task | File(s) | Effort |
|------|---------|--------|
| Power config fields + validation | `config/defaults.py`, `config/planner_config.py` | 0.5d |
| `CoreV1Api` + pod-patch methods | `connectors/kubernetes_api.py`, `connectors/kubernetes.py` | 0.5d |
| Prometheus power queries + gauges | `monitoring/traffic_metrics.py`, `monitoring/planner_metrics.py` | 0.5d |
| `_apply_power_annotations()` in planner loop | `core/base.py`, `core/budget.py` | 0.5d |
| Port `power_agent.py` (all improvements from §6.5) | `components/power_agent/` | 0.5d |
| DaemonSet manifests + RBAC | `deploy/power_agent/` | 0.25d |
| Example configs + scripts + bug fixes | `examples/deployments/powerplanner/` | 1d |
| Pre-commit, integration testing, docs | — | 0.5d |
| **PR 1 total** | | **~4.25d** |

**PR 1 end-to-end flow (no scaling changes):**
```
PlannerConfig (prefill/decode_engine_gpu_power_limit: 250W)
  → Pod annotation (dynamo.nvidia.com/gpu-power-limit: "250")
  → Power Agent NVML enforce (250W)
  → Prometheus metrics (dynamo_power_agent_applied_limit_watts, DCGM_FI_DEV_POWER_USAGE)
```

**Quickstart recipes shipped in `examples/deployments/powerplanner/`:**

| Recipe | What it sets | When to use |
|---|---|---|
| `agg.yaml`, `disagg.yaml` | Defaults, no power awareness. | Baseline / functional smoke. |
| `disagg-power-aware.yaml` | `enable_power_awareness=True`, explicit `total_gpu_power_limit`, explicit `power_agent_safe_default_watts`, hand-tuned `prefill/decode_engine_gpu_power_limit`. | Normal Phase 1+2 deployment. |
| `disagg-conservative-cold-start.yaml` | Above, plus `aic_initial_c_power_prefill=1.05`, `aic_initial_c_power_decode=1.15`, `aic_initial_c_ttft=1.15`, `aic_initial_c_itl=1.15` (Phase 3+ only). | First production deployment of a new model/SKU combination, where the operator wants zero throttling during the first few minutes before live calibration converges. The doc-recommended values for H200 dense models reflect the compute-bound vs memory-bound asymmetry (decision #31). Once `c_power_p`, `c_power_d`, `c_ttft`, `c_itl` settle (~2–3 reoptimize intervals = 10–15 min), values can be reset to 1.0 on next deploy. |

The conservative cold-start recipe trades a small amount of static over-provisioning during the warm-up window for predictable latency from the first request — useful for deployments with strict SLO commitments at launch.

### Phase 2 — Budget enforcement (PR 2)

| Task | File(s) | Effort |
|------|---------|--------|
| `_apply_power_budget()` method on `PlannerStateMachine` + unit tests | `core/state_machine.py`, `tests/unit/test_state_machine.py` | 0.5d |
| Wire calls in load/throughput scaling | `core/load_scaling.py`, `core/throughput_scaling.py` | 0.25d |
| E2E validation of budget-constrained scaling | `examples/deployments/powerplanner/` | 0.5d |
| **PR 2 total** | | **~1.25d** |

**Prerequisite:** PR 1 merged.

### Phase 3 — AIC closed-loop optimizer (PR 3)

| Task | File(s) | Effort |
|------|---------|--------|
| `AICPowerOptimizer` wrapper (TDP-only, TP-constrained) | `monitoring/aic_power_optimizer.py` | Drafted |
| `PlannerConfig` fields + `aic_interpolation` validator | `config/planner_config.py`, `config/defaults.py` | 1d |
| Wire into `NativePlannerBase.__init__()` + `run()` | `core/base.py` | 3d |
| Correction coefficients (`c_ttft`, `c_itl`, `c_power_p`, `c_power_d`, `c_power_agg`) with EMA + clamps + per-side gates (`num_req>0` for latency; `scheduled_*_tokens>0` for per-component power) | `monitoring/aic_power_optimizer.py` | 2d |
| Drift detection + hysteresis (`should_reoptimize`) | `monitoring/aic_power_optimizer.py` | 3d |
| Graceful transitions (already atomic via PR 1 path) | — | 1d |
| Failure-mode handling (§8 rows 1–6) | `monitoring/aic_power_optimizer.py` | 2d |
| Admission coupling: `admission_mode`, `list_frontend_pods`, `post_busy_threshold`, `_apply_aic_config` fanout (§5.7, §6.7) | `connectors/kubernetes.py`, `core/base.py`, `monitoring/planner_metrics.py` | 2d |
| Implied-θ derivation from picked config (§5.7 math, no AIC API change) | `monitoring/aic_power_optimizer.py` | 1d |
| Failure-mode handling (§8 rows 9–11: admission POST failures) | `core/base.py` | 0.5d |
| E2E test: mocker backend (7 wire-up + 5 failure-mode cases + 3 admission cases: off/inherit/autoset) | `tests/integration/test_aic_power_optimizer.py` | 7.5d |
| **PR 3 total** | | **~22d** |

**Prerequisite:** PR 2 merged. `aiconfigurator ≥ 0.8.0` (version with `power_w` column from PR#153).

### Phase 4 — Power as AIC sweep dimension (PR 4)

**AIC team deliverables (external dependency):**

| Task | Owner | Effort |
|------|-------|--------|
| Multi-power-level data collection (H200, H100 SXM) with HBM-clock tracking | AIC team | 1–2 weeks |
| Power-aware perf model (per-op power-scaling curves) | AIC team | 2–3 weeks |
| `cli_power_optimize` API + `estimate_perf(..., power_w=X)` | AIC team | 1 week |
| `TaskConfig` extensions for admission + power budget (§5.8): `active_decode_blocks_threshold`, `active_prefill_tokens_threshold_frac`, `busy_threshold_safety_margin`, `total_power_budget_w`, `picker_objective` | AIC team | 0.5 week |
| `pareto_df` output enrichment (§5.8): `theta_*_impl`, `theta_*_set_recommended`, `*_power_w_per_gpu`, `total_power_w`, `seq/s/W` | AIC team | 0.5 week |
| `power_w` perf-DB backfill (replace zeroed column in `aic_dataframe.py::build_*_row`) | AIC team | 1–2 weeks (overlaps with multi-power data collection) |

**Planner deliverables (after AIC APIs land):**

| Task | File(s) | Effort |
|------|---------|--------|
| Integrate `cli_power_optimize` (still TP-constrained, still §5.1 bridge) | `monitoring/aic_power_optimizer.py` | 3d |
| Regression reset + re-bootstrap on `power_p`/`power_d` change (§6.4) | `core/perf_model/base.py`, `monitoring/aic_interpolation.py`, `monitoring/aic_power_optimizer.py` | 3d |
| Pass `total_power_budget_w` and admission constraints into `TaskRunner` calls; consume `theta_*_impl` from `pareto_df` rows in `_apply_aic_config()` | `monitoring/aic_power_optimizer.py` | 1d |
| Switch from TDP fallback to measured `power_w` once perf DB is backfilled | `monitoring/aic_power_optimizer.py` | 0.5d |
| **PR 4 planner total** | | **~7.5d** |

**Prerequisite:** PR 3 merged. AIC Phase 4 APIs available.

**M2 regression reset protocol** — When AIC changes per-GPU power caps:

```python
def _apply_aic_config(self, new_config: PowerAwareConfig) -> None:
    old = self._last_optimal_config
    if old is None or new_config.prefill_power_w != old.prefill_power_w:
        self._state_machine.prefill_perf_model.reset()
        prefill_fpms = run_aic_interpolation(
            self._config.aic_interpolation, SubComponentType.PREFILL,
            power_w_override=new_config.prefill_power_w,
        )
        self._state_machine.prefill_perf_model.load_benchmark_fpms(prefill_fpms)
    if old is None or new_config.decode_power_w != old.decode_power_w:
        # Same for decode
        ...
    # Apply replicas + caps as before
```

Reset is per-component: a `power_p`-only change rebuilds prefill regression alone, leaving decode's accumulated live calibration intact. Phase 3 (TDP-only) is staleness-free — caps don't change between sweeps.

### Phase 5 — Hardware validation

| Validation | Setup | Effort |
|-----------|-------|--------|
| PR 1+2 end-to-end (NVML enforcement, Prometheus) | Real K8s node with NVML | 1 week |
| Phase 3 closed-loop (EMA convergence: c_ttft, c_itl, c_power_p, c_power_d) and validation of decision #31 cold-start values for H200 dense | 8× H200 SXM, Llama-3-8B | 1–2 weeks |
| Phase 4 power-sweep validation (<10% error on throughput, <15% on latency) | 3 models × 5 power levels | 1–2 weeks |
| **Simultaneous-restart hardware-safety test:** trigger SIGKILL of Power Agent on N nodes at once, force pod restarts, measure PDU draw vs. `headroom_factor × rack_capacity` over the 15s convergence window. Verify `_restore_orphaned_gpus_on_startup` correctly restores idle GPUs and skips active ones. Validate that the §6.5 sizing formula with `headroom_factor ≤ 0.85` keeps the rack under capacity for the worst-case warm-up profile. | Single rack, ≥4 nodes with shared PDU instrumentation | 1 week |

---

## 11. Design Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | **`_apply_power_budget` lives on `PlannerStateMachine`** (not module-level in `core/budget.py`) | Module-level functions in `core/budget.py` are unused dead code from an earlier refactor. The production scaling pipeline calls `self._apply_global_budget` on the state machine; `_apply_power_budget` mirrors it exactly. Single source of truth for GPU counts via `self._capabilities`. |
| 2 | **Per-GPU caps from `self._capabilities` (live DGD)**, not `config.*_engine_num_gpu` | `config.*_engine_num_gpu` can be `None` in K8s mode (only used as CLI fallback). `_capabilities.*.num_gpu` is the live value from the DGD. Mirrors `_apply_global_budget` which already uses `_capabilities`. |
| 3 | **GPU budget first, then power budget** | The two do not commute on asymmetric inputs. GPU-first is the existing convention for `_apply_global_budget`; power budget slots in immediately after. |
| 4 | **Read-from-pod annotation, not local write-cache** | Stale if external actor removes annotation. `pod.metadata.annotations` from the per-tick LIST is the K8s source of truth — no local state can diverge. |
| 5 | **No software-side cap ramping** | NVML `SetPowerManagementLimit` is atomic; the on-chip controller responds in milliseconds. PSUs/PDUs tolerate atomic cap changes at our magnitudes (~200W max). Ramping would require multi-tick planner state for a non-problem; if ever needed, it belongs inside the Power Agent, not the planner. |
| 6 | **Asymmetric `max(c_power_*, 1.0)` clamp** | Never lower the NVML cap below AIC's estimate — that would force throttling on every tick. Only inflate when live measurements show AIC under-predicts. Applies independently to `c_power_p`, `c_power_d`, `c_power_agg` so each side's cap responds to its own regime (compute-bound vs memory-bound). The `max(1.0, c_ttft)` and `max(1.0, c_itl)` clamps on the SLA gate (§5.3) follow the same "conservative-in-the-same-direction" property: never loosen SLA, only tighten when live data says AIC under-predicts latency. |
| 7 | **Latency corrections applied at the SLA gate, not throughput ranking** | The corrections live where they constrain feasibility: `c_ttft` and `c_itl` tighten the SLA filter (`aic_ttft × max(1, c_ttft) > target` skips the config). Applying them to a hypothetical `corrected_throughput = aic_throughput × scalar` would be sort-invariant (multiplying by a positive scalar doesn't change ranking). The SLA gate is where the correction actually changes which configs survive. |
| 8 | **AIC TP is fixed; optimizer sweeps only `(n_p, n_d, power_p, power_d, bs)`** | TP changes require pod recreation — too disruptive for automated closed-loop control. The optimizer reads TP from `AICInterpolationSpec.{prefill,decode}_pick` and pins it for every sweep. An advisory gauge can suggest TP changes to operators. |
| 9 | **`aic_interpolation` required when `enable_aic_optimizer=True`** | Without it the optimizer has no TP to constrain its sweep. Fail loud at config-load time — cheaper than silently picking a TP that doesn't match the cluster. |
| 10 | **Rank by total tokens/s** | Under per-DGD scope, GPUs are dedicated — tokens/J and tokens/GPU pick worse answers when the other constraint is binding. Tie-break by lower GPU count (frees rack space cluster-wide), then lower total cap power. |
| 11 | **No result cache for `optimize()`** | `should_reoptimize` rate-limits to once per `aic_reoptimize_interval`. There is no other call site. A TTL cache would never serve a hit and adds dead code. `_last_optimal_config` is for drift comparison, not caching. |
| 12 | **Hysteresis: `aic_drift_consecutive_ticks` (default 3)** | Suppresses transient p95 spikes from cache misses, GC pauses, model warm-up — which would otherwise cause spurious 6-second AIC sweeps. EMA smooths coefficients; hysteresis controls when to re-sweep. |
| 13 | **Fail-open on runtime AIC failures** | The autoscaling layer doesn't depend on AIC. Crashing the pod on an AIC bug kills `_apply_power_budget` too. Fail-open keeps the autoscaling layer alive while loudly advertising the AIC issue via logs and Prometheus. |
| 14 | **No automatic re-enable after auto-disable** | Prevents flapping. Operator must restart with the underlying issue fixed. |
| 15 | **Safe-default policy for multi-pod-per-GPU conflicts** | Conflicting annotations on the same GPU mean the agent cannot determine the right cap. The previous draft applied the *minimum* of the two — deterministic but failure-prone: two unsupported pods could drag a GPU below its decode floor and break inference for both. Switched to `power_agent_safe_default_watts`, which converges this case to the same fallback used for cold-start parse failures (§6.5). One counter (`dynamo_power_agent_safe_default_applied_total`) for operators to alert on. The agree-case still applies the agreed value because that value isn't ambiguous, but it's WARNING-logged because multi-pod-per-GPU is unsupported topology. |
| 16 | **cgroup parser iterates lines and handles all QoS × driver × runtime combinations** | Original PR #5280 only matched Guaranteed-QoS on systemd driver, silently failing for Burstable-QoS pods (the default for any pod with `requests < limits`). The parser also failed on cgroup v1 (per-controller multi-line files where only some lines carry the pod slice) and on cri-containerd / cri-o-wrapped paths where the pod slice appears mid-string. The current implementation reads `/proc/{pid}/cgroup` line-by-line, uses `.search()` so wrapper segments don't defeat the match, and accepts both `[a-fA-F0-9_]+` (systemd) and `[a-fA-F0-9-]+` (cgroupfs) UID alphabets. |
| 17 | **`total_gpu_power_limit` and `power_agent_safe_default_watts` are `Optional[int] = None`** | The previous design used integer placeholders (`2000` and `0`) and validated against equality with the placeholder. That coupled the validation logic to specific magic numbers — operator setting `total_gpu_power_limit = 2000` for an actual 2000W budget would falsely trip the validator, and `power_agent_safe_default_watts = 0` is structurally invalid and could not be a meaningful sentinel. Both fields now default to `None` and the validator is `is None` checks. The Pydantic type itself signals "operator must set"; validation failure messages explain why. |
| 18 | **SIGTERM handler restores TGP + `terminationGracePeriodSeconds: 30`** | NVML caps persist across processes. Without a handler, DaemonSet rollouts leave GPUs capped indefinitely. SIGKILL/OOM is documented as a known limitation with a recommended `nvidia-smi -pl <default>` boot script. |
| 19 | **Per-request EMAs (`c_ttft`, `c_itl`, `c_power_p`, `c_power_d`, `c_power_agg`) — no `c_throughput` ratio** | A throughput ratio (`observed_tokens_per_sec / aic_estimated_throughput`) is load-confounded: under-load pulls the ratio toward zero through no fault of the hardware, then `safety_factor = max(1.0, 1/c_throughput)` falsely tightens every SLA gate. TTFT, ITL, and per-GPU power (per-component in disagg, fleet-wide in agg) are demand-invariant per-request signals — they describe how *served* requests are served (or what *running* GPUs draw), regardless of arrival rate. Total throughput remains the *ranking* metric (AIC vs. AIC) where it isn't load-confounded. See §5.3 "Why not a throughput ratio". |
| 20 | **Throughput drift trigger fires only on `observed > estimated × (1+threshold)`** | Symmetric drift (`abs(observed − estimated) / estimated > threshold`) fires under-load too — when a 1000 tok/s cluster serves 100 tok/s, drift = 0.9 > 0.15 and after 3 ticks the optimizer would re-sweep to a smaller config that under-provisions for the next demand surge. Capacity-exceeded is the only direction that warrants action; capacity-with-headroom is exactly what we want. SLA-miss triggers (TTFT/ITL above target) remain unchanged. |
| 21 | **EMA updates gated by traffic — latency by `num_req > 0`, power by per-side scheduled tokens** | Latency: `observed_ttft_avg` and `observed_itl_avg` are undefined when no requests are in flight; the gate uses the existing `TrafficObservation.num_req` field. Power (B1): in disagg, prefill and decode are independent regimes — gating the per-component update on the *fleet-wide* `num_req` would poison `c_power_p` toward the idle ratio when only the decode side is busy (and vice versa). The fix is per-component scheduled-token gates: `c_power_p` updates iff `traffic.scheduled_prefill_tokens > 0`, `c_power_d` iff `traffic.scheduled_decode_kv_tokens > 0`. Aggregated mode is gated on the sum. The new fields are already collected by the FPM event plane; plumbing them through `TrafficObservation` is tracked as Open Question #11. |
| 22 | **Cold-start fail-closed: `power_agent_safe_default_watts`** | Steady-state, NVML caps persist across processes — a parse failure on a running cluster keeps the prior cap, so the only real exposure is the cold-start path (fresh node, fresh GPU at TDP). The Power Agent applies `power_agent_safe_default_watts` on first encounter when annotation parsing fails, instead of leaving the GPU at TDP. The field is `Optional[int] = None` and required when `enable_power_awareness=True` (per decision #17) — avoids picking a safe default for the operator that may be wildly wrong for their SKU. |
| 23 | **No `min_delta_watts` rate-limit on annotation PATCH** | Three independent guards already bound the API-server PATCH rate: (1) `aic_reoptimize_interval` (default 300s) caps sweep frequency; (2) `aic_drift_consecutive_ticks` (default 3) suppresses transient triggers; (3) per-pod string equality before PATCH no-ops steady state. Worst-case for `N=1000` pods is ~3.3 PATCH/s — multiple orders of magnitude under kube-apiserver QPS budgets. A `min_delta_watts` threshold would be a fourth guard solving an already-solved problem. |
| 24 | **Admission control = `/busy_threshold`, not a new mechanism** | The frontend already has KV/prefill-utilization-based load shedding (`PushRouter::generate_with_fault_detection` in `lib/runtime/src/pipeline/network/egress/push_router.rs`). Building a separate planner-side admission controller would duplicate engine-state observation. Reusing the existing primitive ties admission to the same physical signals AIC predicts and keeps a single source of truth for the shed point (the engine's own metrics). |
| 25 | **Admission thresholds derived from chosen operating point, not statically configured** | A static threshold is independent of the deployment's actual capacity; it cannot follow capacity changes from replica scaling or power capping. The picked AIC config defines the operating point; the implied θ is a deterministic function of that point and is correct by construction. When power caps change, the implied θ moves with the new physical capacity automatically. |
| 26 | **Three modes (off / inherit / autoset), default off** | Backward compatibility (§12) and operator agency: existing deployments keep manual control, observability deployments see the implied θ in Prometheus without any policy change, and autoset is opt-in for closed-loop power-aware control. Same default-off pattern as `enable_power_awareness` and `enable_aic_optimizer`. |
| 27 | **Multi-replica frontend fanout, no rollback on partial failure** | The threshold registry isn't yet backed by etcd, so a single POST reaches one frontend. Partial failure is loud (counter + ERROR), not an error path that rolls back replicas — stale thresholds on one frontend are conservative-correct (the prior implied values were derived from a valid prior operating point). The fanout converges on next `_apply_aic_config()`. Tracked as Open Question #7 — etcd-backing is a frontend-side improvement, orthogonal to this design. |
| 28 | **Power budget input at picker, default 0 (off)** | Symmetric with admission thresholds (default 1.0 = off): `total_power_budget_w == 0` short-circuits the new filter, preserving today's `_run_autoscale_sim` behavior exactly. The filter ships in parallel with AIC's `power_w` backfill (Phase 4 deliverable) using TDP × num_gpus as a conservative fallback so the API isn't blocked on data collection. |
| 29 | **Default picker objective is `throughput`, not `efficiency`** | Today's `_run_autoscale_sim` ranks by `seq/s` and the AIC team has tuned the picker around that signal. `efficiency` (`seq/s/W`) is a useful alternate for the power planner during partial-load operation but a poor default for the budget-binding case (it prefers under-utilizing the budget when a slightly-better-throughput config exists slightly above the efficiency knee). Operators who want `efficiency` opt in. |
| 30 | **Implied θ always recorded as gauges in `inherit`/`autoset`** | Operators running closed-loop `autoset` need to see what value was pushed to the frontend. Operators running `inherit` need to see what value `autoset` *would have* pushed before opting in. Two gauges (`*_implied_theta_*`, `*_set_theta_*`) cover both cases at zero marginal cost. The `inherit` mode exists primarily for this observability staircase: it lets operators validate the math against their own SLO observations before flipping to `autoset`. |
| 31 | **Power coefficients split per-component (`c_power_p` / `c_power_d`) in disagg; single `c_power_agg` in agg** | Prefill is compute-bound (SM-clock-driven GEMM power) and decode is memory-bound (HBM-bandwidth-driven, dominated by KV-cache traffic). A single fleet-wide `c_power` averages the two regimes and produces simultaneously over-capped decode and under-capped prefill (or vice versa), losing budget headroom on one side and forcing throttling on the other. The split is only physically meaningful when prefill and decode run on separate GPUs — i.e. disagg mode. Aggregated mode (`mode=agg`) keeps a single `c_power_agg` because chunked prefill mixes regimes within the same engine. Per-component EMAs are gated on per-side scheduled tokens (decision #21) so an idle side doesn't drag the busy side toward the idle ratio. The §5.1 `aic_to_planner_cap` bridge is called twice in disagg with the per-side coefficient and once in agg. |
| 32 | **Frontend port via `frontend_http_port` config (default 8000), not auto-discovery** | The DGD spec doesn't yet standardize a named port (`name: http`) on the frontend container, so the planner can't read `containerPort` from `V1Pod.spec.containers[].ports[].name == "http"` without ambiguity. Hardcoding `:8000` was wrong — the operator might serve on a non-default port; failing autoset silently when the port mismatched would be a stealth failure. A config field with default 8000 (matches the operator's default and the existing `get_frontend_metrics_url(port=8000)` convention) ships now and lets operators override. Open Question #13 tracks the named-port-based auto-discovery successor; until then, the config field is the deterministic surface. |
| 33 | **POST all three `/busy_threshold` fields (decode blocks + prefill fraction + prefill absolute)** | The frontend evaluates `is_busy` as the OR of three independent threshold checks (`worker_monitor.rs:189-202`). Setting only the fractional prefill threshold has a defense-in-depth hole: when a worker fails to report `max_num_batched_tokens` to MDC, the frontend falls back to `DEFAULT_MAX_TOKENS = 10_000_000` (`worker_monitor.rs:55-58`), which silently turns the fractional check into a no-op (`active_prefill_tokens > 0.65 × 10M = 6.5M` essentially never fires). The absolute prefill threshold (B6) is computed from the same operating point as `ceil(theta_prefill_frac_impl × min_M)` where `min_M` is the minimum `max_num_batched_tokens` across prefill workers (already populated by MDC into `WorkerInfo`). Both checks fire at the same point under normal operation; the absolute serves as a hard backstop when MDC sync is broken. Failure mode #12 covers the `min_M` unavailable case. |
| 34 | **`_estimated_throughput` = total predicted tokens-per-second across the DGD at the picked AIC config** | §5.6's drift-detection trigger compares `traffic.total_tokens_per_sec` against `_estimated_throughput`, but the previous draft did not pin the definition. The value is set in `_apply_aic_config()` from the picked row as `aic_seq_per_s_per_replica × n_d × (isl + osl)` (factor `n_d` because decode is the steady-state-determining replica count for token throughput in disagg; agg uses `n_agg`). Cold-start initial value is `0`, which makes `capacity_exceeded` short-circuit until the first sweep — only `sla_violated` can trigger re-optimization before that. Failure modes #1 and #4 leave the value at `0` permanently after auto-disable, which is correct because the optimizer is no longer driving config changes. |
| 35 | **NVML cap clamping to `nvmlDeviceGetPowerManagementLimitConstraints()`** | The NVML SDK rejects `SetPowerManagementLimit()` with `NVML_ERROR_INVALID_ARGUMENT` when the requested cap is outside the SKU-specific hardware bounds. Without an explicit clamp, the agent would silently leave the GPU uncapped on every such error — defeating the entire enforcement mechanism. The `_clamp_to_constraints()` helper (B10) reads `[min_w, max_w]` per-SKU, clamps the requested value, applies the clamped value, and increments `dynamo_power_agent_cap_clamped_total{direction="min"|"max"}` so operators can spot when caps are being silently saturated. The `direction` label is essential — `min` direction means actual draw can exceed the planner's intended cap (small budget-headroom cost), while `max` direction means AIC predicted above SKU TDP (calibration-bug signal). Failure mode #13 documents the runtime-visible behavior. |
| 36 | **UUID-gated orphan-cap restoration on Power Agent startup** | The previous draft restored default TDP on every idle GPU during `_restore_orphaned_gpus_on_startup()`. That implicitly assumed the agent was the only entity capping GPUs on the node — false in any shared-tenant configuration (different DGD on the same physical node, manual `nvidia-smi -pl` from a node bootstrap script, vendor firmware default that already differs from `GetPowerManagementDefaultLimit`). The fix (B12) persists the set of GPU UUIDs the agent has previously capped to `/var/lib/dynamo-power-agent/managed_gpus.json` (host-bind-mounted volume, atomic write via tmp+rename). On startup, the agent reverts default TDP only on idle GPUs whose UUID is in that set. State-file loss (host disk wipe) reduces the agent to its pre-B12 behavior — strictly safer than the alternative — and is recovered after the first cap-application tick. The §6.5 SIGKILL-recovery table documents the full state-machine. Closes Open Question #3. |

---

## 12. Backward Compatibility

- `enable_power_awareness` defaults to `False` — zero change for existing deployments.
- No existing config fields modified or removed.
- Power Agent is a standalone DaemonSet — not deployed unless explicitly installed.
- RBAC patch is additive.
- `CoreV1Api` added to `KubernetesAPI` but unused unless power features are enabled.
- `enable_aic_optimizer` defaults to `False`.

---

## 13. Open Questions

1. **cgroup parser CI matrix** — Unit tests cover the full QoS × driver × runtime matrix (cgroup v1 multi-line, cgroup v2 single-line, systemd / cgroupfs drivers, Guaranteed / Burstable / BestEffort QoS, cri-containerd / cri-o / dockershim wrappers, mixed v1+v2 hybrid hosts, non-K8s process). Integration tests should cover at least: cgroup v2 + systemd (K8s 1.25+ / Ubuntu 22.04+ default) and cgroup v1. Does CI provision both, or do we accept "v1 unit tests + v2 integration tests"? Tracking with CI team.

2. **Latent up-scale-beyond-demand in `_apply_global_budget`** — The existing method can return `num_d > num_d_input` when prefill is the binding constraint and budget slack remains. `_apply_power_budget` has an explicit `min(num_d, …)` cap that prevents this. A separate PR should add the same fix to `_apply_global_budget`. Out of scope here.

3. **SIGKILL recovery for shared-tenant nodes — RESOLVED in B12.** `_restore_orphaned_gpus_on_startup` (§6.5) now uses a persistent UUID list (`/var/lib/dynamo-power-agent/managed_gpus.json`) to track previously-managed GPUs and reverts default TDP only on idle GPUs the agent itself has capped. The residual hole — agent SIGKILL'd, pod deleted, then a *non-Dynamo* workload claims the GPU before the agent restarts — still needs operator-side `nvidia-smi -pl <default>` in node-bootstrap, or eventual integration with the NVIDIA k8s-device-plugin / a CDI hook. The UUID-gated approach also means a co-tenant DGD's caps are never inadvertently reverted. See §6.5 SIGKILL recovery table and decision #36.

4. **Switch to p95 latency triggers if averages prove too forgiving** — §5.6 uses `ttft_avg` and `itl_avg` to match the existing `PrometheusAPIClient.get_avg_*` methods. If post-Phase-3 production data shows a small fraction of bad requests doesn't move the average enough to trigger drift detection, add p95 queries to `PrometheusAPIClient` (matches the histogram metrics the frontend already exports) and switch the trigger over. The EMA-of-ratios mechanism in §5.3 is unchanged either way.

5. **Multi-tenant power coordination** — When multiple DGDs share a GPU pool, `optimization_target = "tokens_per_joule"` mode would minimize capacity instead of maximizing throughput. Out of scope for v1; would add a new ranking branch.

6. **Sizing-helper CLI for `total_gpu_power_limit`** — A small operator-side script that reads rack/PDU specs, the §3.3 formula, and the SKU's warm-up power profile, then emits a recommended `total_gpu_power_limit` (and warns when summed across multiple DGDs it exceeds facility capacity). Properly belongs in `examples/deployments/powerplanner/sizing/` rather than the planner itself; tracked here so it's not lost. No core-design impact.

7. **Multi-replica frontend `/busy_threshold` fanout vs etcd-backed registry** — `/busy_threshold` writes a single frontend pod's manager state. The planner fans out POSTs to all frontend replicas in `autoset` mode (§6.7), which is correct but O(N) on each `_apply_aic_config()`. A future direction is to back the threshold registry with etcd in the frontend so a single POST propagates to all replicas. Tracked as a frontend-side improvement; the planner-side fanout is the right thing to ship in v1.

8. **AIC perf-DB `power_w` backfill** — The `power_w` column in AIC's per-row schema is currently zeroed (`components/src/dynamo/profiler/utils/aic_dataframe.py::build_*_row`). Until the AIC team backfills measured per-(system × parallelism × batch × ISL/OSL) power numbers, the `total_power_budget_w` filter in §5.8 falls back to system TDP × num_gpus, which is conservative. Fallback emits a one-time WARNING per sweep. Tracked in PR#153 (AIC team) and Phase 4 roadmap.

9. **`max_num_batched_tokens` exposure in `pareto_df` — RESOLVED in B6.** The absolute prefill threshold (§5.7) is now derived planner-side from `min` of `WorkerInfo.max_num_batched_tokens` across prefill workers — already populated by MDC at `components/src/dynamo/planner/monitoring/worker_info.py:44`. No AIC `pareto_df` schema change required. Failure mode #12 covers the "no worker has reported yet" case via a CRITICAL log + counter. The cross-backend sweep use case (configs comparing different chunked-prefill caps) is unaddressed — defer until it surfaces.

10. **Open-loop arrival pattern variance margin** — The §5.7/§5.8 `theta_*_impl` math uses Little's-law steady-state means. For Poisson-arrival traffic, the instantaneous `active_decode_blocks` and `active_prefill_tokens` fluctuate around those means — setting threshold = mean ⇒ ~50% time-in-busy. The `admission_safety_margin` field (default 1.0) is the lever for queueing variance; values <1.0 push the shed point below the steady-state mean. Empirical guidance for production traffic (Poisson vs bursty vs closed-loop) should be added to the operator playbook once Phase 5 hardware data is available.

11. **`TrafficObservation` fields for per-side scheduled tokens (B1)** — The per-component power EMA gates in §5.3 require `TrafficObservation.scheduled_prefill_tokens` and `TrafficObservation.scheduled_decode_kv_tokens` (and `scheduled_decode_kv_tokens` for the agg sum). The FPM event plane already collects these at the engine boundary (`planner-design.md` §"Load-Based Scaling" describes `sum_prefill_tokens` / `sum_decode_kv_tokens` flowing through `LoadPredictor`). Plumbing the existing values through `TrafficObservation` is a small Phase-3 deliverable: it requires updating the `TrafficObservation` dataclass, the FPM-side serializer, and a new `LoadPredictor.predict_scheduled_tokens()` method (or repurposing `predict_load`). Until then, `update_correction()` falls back to a fleet-wide gate (the previous design's `num_req > 0` semantics) and emits a one-time INFO log saying so. Tracked for closeout in PR 3 alongside the rest of the AIC closed-loop work.

12. **DCGM selector when operator overrides pod naming (B1)** — `get_avg_per_gpu_power_by_component()` (§7) now uses `exported_pod=~"^{dgd_name}-[0-9]+-{service_key.lower()}-.*"` to filter DCGM_FI_DEV_POWER_USAGE samples.  This matches the operator's default `<dgd>-<replica-idx>-<service-key-lc>-<hash>` pod-name format.  When the operator is configured with a non-default pod-name template (overridable via the DGD spec), the regex won't match and `c_power_*` updates will silently NaN-out.  The robust fix is to label DCGM samples with `dynamo_service` (the same label the planner already uses for component identification elsewhere) — but that requires a kube-state-metrics or DCGM-exporter relabeling rule.  Phase-3 ships the regex form; Phase-5 hardware validation should test the operator's name-override path and add a relabeling rule if needed.

    **Resolved sub-issues** (Phase-3 hardware validation, May 2026):
    - The original implementation used the bare `pod` label (DCGM exporter's own pod) instead of `exported_pod` (workload pod).
    - The original implementation passed the dynamo logical namespace to `exported_namespace`, which is labeled with the K8s namespace.
    - The original regex `<dgd>-<component>-.*` did not accommodate the `<replica-idx>` segment in the operator's default pod naming.
    - **Cluster-side gap (infrastructure, not planner)**: on the dev cluster, two `nvidia-dcgm-exporter` pods had been crash-looping for 30+ days because the manually-managed `gpu-operator/dcgm-metrics-config` ConfigMap referenced `DCGM_FI_DEV_CLOCK_THROTTLE_REASONS`, a field renamed in `dcgm-exporter` v4.5.x+ (`unknown ExporterCounter field` startup error). One of the crashing pods was scheduled on the GPU node hosting the qwen3-quickstart workers, producing the symptom that `DCGM_FI_DEV_POWER_USAGE{exported_pod=~"qwen3-quickstart-..."}` returned zero series even though the planner queries were correct.  Fix: comment out the offending field in the ConfigMap and delete the crashing exporter pods so the DaemonSet rebuilds them.  After the fix, `get_total_dgd_power` returned ~166 W and `get_avg_per_gpu_power_by_component` returned ~83 W per VllmWorker GPU, and all 5 `TestDcgmPerPodPower` live tests pass with their `if attributed:` watt-value branches now exercised.  Action item for productionization: pin a DCGM exporter chart version that is consistent with the metrics-config ConfigMap shipped alongside it, and add a synthetic test that fails CI when any `nvidia-dcgm-exporter` pod is `CrashLoopBackOff` on a GPU node we expect to attribute power for.

13. **Frontend port auto-discovery via named ports (B5)** — Now resolved (Phase-3 hardware validation, May 2026).  The operator already emits a named `http` port on the frontend container (`deploy/operator/internal/dynamo/component_frontend.go:34-40`, with `DynamoContainerPortName = "http"` and `DynamoServicePort = 8000` in `deploy/operator/internal/consts/consts.go:13-15`); the live `qwen3-quickstart-0-frontend-*` pod confirms `containers[0].ports = [{name: "http", containerPort: 8000, protocol: "TCP"}]`.  The planner now reads each frontend pod's actual port via `KubernetesConnector.resolve_frontend_http_port(pod, fallback)` (a static method on the connector), and the admission-control fanout in `_apply_admission_thresholds()` calls it per-pod so a DGD that overrides the operator default is honored without a config mirror.  The `frontend_http_port` config field is retained as a fallback for legacy manifests authored before the named-port standardization, hand-rolled DGDs that omit `containers[].ports`, and unit-test fixtures with minimal `V1Pod` mocks; its description has been updated to flag it as the legacy-fallback path.

    **Resolved sub-issues** (Phase-3 hardware validation, May 2026):
    - The operator-side change was already in `main` — `FrontendDefaults.GetBaseContainer` has emitted the named port since the consts.go constants were added; no operator diff needed.
    - Planner-side: added `KubernetesConnector.resolve_frontend_http_port(pod, fallback)` (static method) — reads `containers[].ports[name=http].container_port`, coerces to `int`, and defends against malformed pod specs (`spec is None`, missing `ports` attribute, string container_port from hand-rolled YAML) by falling back rather than raising.
    - The single callsite in `NativePlannerBase._apply_admission_thresholds()` was updated to compute the port per-pod (rather than using `self.config.frontend_http_port` directly) so DGD-spec port overrides flow through automatically.
    - 10 new unit tests pin the resolver behavior: named port present, named port absent (other named ports only), `ports is None`, `ports == []`, multi-port containers (filter by name not position), operator port override (e.g. 8443), `container_port` coerced from string, malformed pod with `spec is None`, container without `ports` attribute, multi-container pod where only the main container has the `http` port.
    - Live cluster verification: `KubernetesConnector.resolve_frontend_http_port(pod, fallback=9999)` returns `8000` for the live `qwen3-quickstart-0-frontend-q629n` pod (using the fallback `9999` as a sentinel makes the named-port path unambiguously distinguishable from the fallback).  All 463 planner unit tests and 33 live integration tests still pass.

14. **`DirectRouterMetricsClient` endpoint scope** — The class scrapes a Prometheus exposition for the `dynamo_frontend_worker_*` per-worker gauges (`active_prefill_tokens`, `active_decode_blocks`, `last_ttft`/`last_isl`/`last_itl`).  These gauges are populated only on KV-router code paths: `KvWorkerMonitor` writes the `WORKER_LOAD_METRICS` `IntGaugeVec` from ActiveLoad events, and `observe_first_token_gauges()` / `observe_finish_gauges()` in `lib/llm/src/protocols/common/timing.rs` are invoked only from the Python KV-router binding (`lib/bindings/python/rust/llm/kv.rs`).  RoundRobin / Random / PowerOfTwoChoices / LeastLoaded frontends register the gauge families on startup but never write a single labeled instance, so Prometheus omits them from the exposition entirely (a `GaugeVec` family with zero labeled samples is invisible).  Two deployment topologies expose these gauges in practice: (a) the Global-Planner topology with a dedicated `LocalRouter` pod (`python3 -m dynamo.router`, port 9090) — Service DNS `<dgd>-localrouter:9090`, and (b) frontends launched with `--router-mode kv` (in-process KV router) — Service DNS `<dgd>-frontend:8000`.  The `DirectRouterMetricsClient` docstring now enumerates these supported endpoints; the live test (`TestDirectRouterMetricsClientLive`) discovers them at runtime by probing both URLs and verifying at least one target gauge is present, and skips with a clear topology-aware reason when the active DGD has neither.

    **Resolved sub-issue** (Phase-3 hardware validation, May 2026):
    - The original test fixture hard-coded `http://<dgd>-vllmworker:9090/metrics` (the worker's own component-metrics endpoint), which never exposes the `dynamo_frontend_worker_*` gauges.  The test then skipped with the generic message "metric names not found", masking the fact that the URL was wrong.  The replacement discovery probe checks the LocalRouter pod first, then a KV-mode frontend, and only skips after both come up empty — and when an endpoint *is* discovered, an empty parse result is now a hard failure (not a skip) to catch future parser regressions.  This is purely a test/documentation improvement; production behavior is unchanged.

    **Validation note** — closing the loop on this cluster (a KV-routed deployment that would let the test transition skip → pass) was attempted by patching the `qwen3-quickstart` DGD to launch the frontend with `--router-mode kv`.  The KV-mode frontend started cleanly and registered the gauge families on `/metrics` (i.e. the endpoint discovery probe correctly picked it up), but `/v1/models` returned an empty list and every `/v1/chat/completions` request returned 404.  Frontend logs showed `KubeDiscoveryClient::list returning 0 instances for query=AllEndpoints` — the KV router's discovery loop never observed the vLLM worker as a KV-aware backend.  Conclusion: getting an aggregated DGD to actually *serve* KV-routed traffic requires worker-side wiring (the vLLM worker must publish KV-indexer events through the NATS event plane and advertise its endpoints in a KV-router-aware shape), which is separate from the planner work and not enabled by default in the standard `vllm-runtime` aggregated image.  The DGD was reverted to the round-robin default and the planner suite continues to pass with the new topology-aware skip on `TestDirectRouterMetricsClientLive`.

    **Follow-up — Global-Planner LocalRouter does NOT expose these gauges either** (Phase-3 hardware validation continued, May 2026): a minimal Global-Planner topology was stood up to retest the `LocalRouter` scrape path — `gp-ctrl` DGD (Frontend in `agg` mode + GlobalRouter, no GlobalPlanner) plus `gp-pool-0` DGD (LocalRouter + 1 vLLM worker, no per-pool Planner), running Qwen3-0.6B.  After resolving disk-pressure evictions on the control-plane pods (cordoned the offending node) and the LocalRouter's `WorkerKvQueryResponse` deserialization noise (LocalRouter launched with `--no-router-kv-events --no-router-track-active-blocks --router-kv-overlap-score-weight 0` so it tolerates a non-KV-aware vLLM worker), the full chain Frontend → GlobalRouter → LocalRouter → Worker served `/v1/chat/completions` cleanly (verified by 20-request burst, all 200s).  The LocalRouter's `/metrics` endpoint exposed `dynamo_component_router_*` and `dynamo_work_handler_*` series — but **none** of the `dynamo_frontend_worker_*` gauges, even with traffic flowing.  Root cause: `register_worker_load_metrics` and `register_worker_timing_metrics` are called only from `lib/llm/src/http/service/service_v2.rs`; the standalone router (`python3 -m dynamo.router`) starts a separate `system_status_server` (`lib/runtime/src/system_status_server.rs`) whose Prometheus registry never sees these registrations.  The global `WORKER_LOAD_METRICS` static *is* updated in-process by the LocalRouter's KvScheduler (`lib/llm/src/kv_router/sequence.rs`), but the metric family isn't bound to the registry exposed at `:9090/metrics`, so it never appears.  This is a real implementation gap, not a deployment misconfiguration.  Implications for the planner: pool planners running under a Global-Planner topology cannot use `DirectRouterMetricsClient` — they must drive their per-worker view from the `PrometheusAPIClient` router-source histograms (filtered by `dynamo_namespace` so each pool reads only its own LocalRouter's series).  The `DirectRouterMetricsClient` docstring and the `_discover_kv_router_metrics_url` test helper were updated to drop the false `<dgd>-localrouter:9090` claim; the live test now skips on Global-Planner DGDs with a topology-aware reason.  Real coverage of `DirectRouterMetricsClient` will land when the planner is exercised against an aggregated DGD with KV-aware worker wiring (separate work item).  Closing the LocalRouter exposition gap itself — adding `register_worker_load_metrics` to the runtime's `system_status_server` registry when KV routing is in use — is tracked as a future enhancement; it would let pool planners use `DirectRouterMetricsClient` directly and make the per-worker view symmetric across both topologies.

---

## 14. Code Pointers

| Component | Path |
|-----------|------|
| Planner Config | `components/src/dynamo/planner/config/planner_config.py` |
| Planner Defaults | `components/src/dynamo/planner/config/defaults.py` |
| State Machine | `components/src/dynamo/planner/core/state_machine.py` |
| Base Planner (run loop) | `components/src/dynamo/planner/core/base.py` |
| Budget Constants | `components/src/dynamo/planner/core/budget.py` |
| Load Scaling | `components/src/dynamo/planner/core/load_scaling.py` |
| Throughput Scaling | `components/src/dynamo/planner/core/throughput_scaling.py` |
| K8s API | `components/src/dynamo/planner/connectors/kubernetes_api.py` |
| K8s Connector | `components/src/dynamo/planner/connectors/kubernetes.py` |
| AIC Power Optimizer (new) | `components/src/dynamo/planner/monitoring/aic_power_optimizer.py` |
| AIC Perf Estimator (existing) | `components/src/dynamo/planner/monitoring/aic_estimator.py` |
| AIC Interpolation (existing) | `components/src/dynamo/planner/monitoring/aic_interpolation.py` |
| Prometheus Client | `components/src/dynamo/planner/monitoring/traffic_metrics.py` |
| Planner Metrics | `components/src/dynamo/planner/monitoring/planner_metrics.py` |
| Power Agent | `components/power_agent/power_agent.py` |
| Power Agent DaemonSet | `deploy/power_agent/daemonset.yaml` |
| Example configs + scripts | `examples/deployments/powerplanner/` |
| AIC power estimation design | [PR#153](https://github.com/ai-dynamo/aiconfigurator/pull/153) |
| Original implementation | [PR #5280](https://github.com/ai-dynamo/dynamo/pull/5280) |

### 14.1 Admission control (§5.7, §6.7) — net-new vs extended

| Symbol | Status | Path |
|---|---|---|
| Frontend HTTP endpoint `POST /busy_threshold` | Existing (Rust, mounted on the OpenAI service) | `lib/llm/src/http/service/busy_threshold.rs` |
| Frontend drop site (decides the actual reject) | Existing (Rust) | `lib/runtime/src/pipeline/network/egress/push_router.rs::generate_with_fault_detection` |
| Worker busy-state monitor (feeds the drop site) | Existing (Rust) | `lib/llm/src/discovery/worker_monitor.rs::KvWorkerMonitor` |
| `KubernetesConnector.get_frontend_metrics_url` (Service URL for /metrics scraping) | Existing — **not reusable** for `/busy_threshold` (Service load-balances to one pod; threshold registry is per-pod state) | `components/src/dynamo/planner/connectors/kubernetes.py:291` |
| `KubernetesConnector.list_frontend_pods()` → `list[V1Pod]` | **Net-new** (Phase 3) | `components/src/dynamo/planner/connectors/kubernetes.py` |
| `KubernetesConnector.post_busy_threshold(pod, model, θ_d, θ_pf)` | **Net-new** (Phase 3) | `components/src/dynamo/planner/connectors/kubernetes.py` |
| `AICPowerOptimizer.compute_implied_thresholds(picked_config, isl, osl)` (uses existing `get_max_kv_tokens`) | **Net-new** (Phase 3) | `components/src/dynamo/planner/monitoring/aic_power_optimizer.py` |
| `AIConfiguratorPerfEstimator.get_max_kv_tokens(...)` | Existing — consumed unchanged | `components/src/dynamo/planner/monitoring/aic_estimator.py:219` |
| `NativePlannerBase._apply_aic_config()` autoset fanout | **Extended** (Phase 3) — adds the §6.7 fanout block | `components/src/dynamo/planner/core/base.py` |
| `PlannerPrometheusMetrics.admission_*` gauges + counter | **Net-new** (Phase 3) | `components/src/dynamo/planner/monitoring/planner_metrics.py` |
| AIC `TaskConfig` admission/power fields and `pareto_df` `theta_*_*` columns (§5.8) | **Net-new** (Phase 4, AIC team) — until they land, the planner derives θ itself via the row above | external `aiconfigurator` SDK |

**Phase 3 viability checkpoint.** Every Phase-3-marked row above is implementable today against the current `aiconfigurator` SDK and the current frontend Rust API. The only Phase-4-gated row (the AIC `TaskConfig` extensions) is a *replacement* for planner-side θ derivation — not a prerequisite. Phase 3 lands a complete admission-enforcement path; Phase 4 simplifies it.
