# Perf Model Ownership Design Discussion Draft

## Background

We currently have two possible integration approaches:

1. **Planner directly calls the mocker shim / AIC FPM perf model**
   - The planner consumes FPM state directly, or consumes aggregated FPM state.
   - The planner calls `RustEnginePerfModel` directly from load scaling and throughput scaling.

2. **Router owns the perf model, and planner queries router**
   - The router owns worker runtime state, FPMs, KV cache overlap / reuse views, and the perf model.
   - The planner does not interpret FPMs directly. Instead, it queries the router for semantic load and capacity results.

The two options are described together below to make ownership boundaries and migration paths easier to discuss.

---

## Option 1: Planner Directly Calls the Shim

### Current State

The current PR already exposes the Rust shim:

- Rust: `EnginePerfModel`
- Python: `RustEnginePerfModel`
- Backed by AIC `ForwardPassPerfModel`
- Supports:
  - `best_available(...)`
  - `estimate_forward_pass_time(...)`
  - `tune_with_fpms(...)`
  - `get_queued_prefill_time(...)`
  - `get_scheduled_decode_itl(...)`
  - `find_engine_capacity_rps(...)`

However, the planner is not yet wired to the shim. The current planner still uses Python regression models directly:

- `PrefillRegressionModel`
- `DecodeRegressionModel`
- `AggRegressionModel`

### Flow Diagram

```mermaid
flowchart TD
  FPM["FPM stream<br/>ForwardPassMetrics"] --> SM["PlannerStateMachine"]

  SM --> OLD["Current planner perf models<br/>Python regression"]
  OLD --> P["PrefillRegressionModel"]
  OLD --> D["DecodeRegressionModel"]
  OLD --> A["AggRegressionModel"]

  P --> L1["load_scaling.py<br/>estimate_next_ttft"]
  D --> L2["load_scaling.py<br/>estimate_next_itl"]
  A --> L3["load_scaling.py<br/>estimate_next_ttft / estimate_next_itl"]

  P --> T1["throughput_scaling.py<br/>find_best_engine_prefill_rps"]
  D --> T2["throughput_scaling.py<br/>find_best_engine_decode_rps"]
  A --> T3["throughput_scaling.py<br/>find_best_engine_agg_rps"]

  SHIM["New shim<br/>RustEnginePerfModel"] -. "Not yet integrated with planner" .-> SM
  SHIM --> AIC["AIC ForwardPassPerfModel<br/>native + regression fallback + online tuning"]
```

### Target Integration

```mermaid
flowchart TD
  INIT["Planner startup / worker discovery"] --> BA["RustEnginePerfModel.best_available"]
  BA --> CFG["Inputs:<br/>engine_args / aic_config / limits / options"]
  CFG --> MODEL["EnginePerfModel per worker type<br/>prefill / decode / aggregated"]

  FPM["FPM updates<br/>one per attention-DP rank"] --> TUNE["model.tune_with_fpms(iterations)"]
  TUNE --> MODEL

  MODEL --> LOAD["Load scaling<br/>reactive"]
  MODEL --> THPT["Throughput scaling<br/>predictive"]

  LOAD --> QP["get_queued_prefill_time(fpms)"]
  LOAD --> SD["get_scheduled_decode_itl(fpms)"]

  QP --> LD1["TTFT / queued prefill pressure"]
  SD --> LD2["ITL / scheduled decode pressure"]
  LD1 --> DECIDE_LOAD["scale up/down decision"]
  LD2 --> DECIDE_LOAD

  THPT --> CAP["find_engine_capacity_rps(request)"]
  CAP --> FLOOR["throughput worker lower bound"]

  FLOOR --> FINAL["final ScalingDecision"]
  DECIDE_LOAD --> FINAL
```

### Planner Internal Decision Layer

```mermaid
flowchart TD
  TICK["Planner tick"] --> UPDATE["Update state / metrics"]

  UPDATE --> LQ{"enable_load_scaling?"}
  UPDATE --> TQ{"enable_throughput_scaling?"}

  LQ -->|yes| LOAD["LoadScalingMixin"]
  TQ -->|yes| THPT["ThroughputScalingMixin"]

  THPT --> PRED["Predict traffic for next time window"]
  PRED --> CAPACITY["Compute per-engine capacity under SLA"]
  CAPACITY --> LOWER["Set worker lower bound"]

  LOAD --> LAT["Estimate per-worker latency / load"]
  LAT --> UP{"Do all relevant workers<br/>exceed SLA / threshold?"}
  LAT --> DOWN{"Would it remain safe<br/>after scale-down?"}

  UP -->|yes| SCALEUP["load scale up"]
  DOWN -->|yes| SCALEDOWN["load scale down"]
  DOWN -->|no| NOLOAD["no load scale-down"]

  LOWER --> MERGE["Merge decisions"]
  SCALEUP --> MERGE
  SCALEDOWN --> MERGE
  NOLOAD --> MERGE

  MERGE --> RULE["Load decision takes priority;<br/>throughput lower bound constrains scale-down"]
  RULE --> EFFECTS["PlannerEffects.scale_to"]
```

### Semantic Notes

With this option, the planner is responsible for preparing the shim inputs.

Key differences:

- The old planner `estimate_next_ttft()` treats `queued_prefill_tokens + avg_isl` as the work in front of the next request.
- The new shim `get_queued_prefill_time()` only estimates the queued prefill drain time provided by the caller. It does not automatically add a hypothetical next request.
- The old planner `estimate_next_itl()` adds learned average decode length / request count.
- The new shim `get_scheduled_decode_itl()` only estimates the scheduled decode workload provided by the caller.
- The current FPM queued-prefill fields contain raw prompt tokens only. They do not include KV reuse information.
- If the planner wants to account for prefix cache / KV reuse, it must adjust queued prefill tokens before calling the shim.

### Benefits

- Short implementation path. The planner can directly replace Python regression models.
- Planner load / throughput policy context stays local, making incremental migration easier.
- The shim API is already close to the three main query types the planner needs today.

### Risks

- The planner continues to own a significant amount of runtime workload interpretation logic.
- Details such as KV reuse, preempted decode, waiting KV transfer, and attention DP may leak into planner logic.
- The router is already closer to routing and cache state. Having the planner interpret these states may duplicate responsibility.

---

## Option 2: Router Owns Perf Model, Planner Queries Router

### Core Idea

The router owns the information closest to real-time scheduling:

- worker list and worker state
- in-flight / queued request views
- FPM stream
- KV cache events
- prefix overlap / cache reuse estimates
- routing-policy-related state

Therefore, the router can own the perf model and expose semantic queries to the planner. The planner does not interpret FPMs directly; it only owns scaling policy.

### High-Level Flow Diagram

```mermaid
flowchart TD
  subgraph Engines["Engines / Workers"]
    W1["Prefill / Decode / Agg Workers"]
    FPM["FPM stream<br/>scheduled + queued workload"]
    KV["KV cache events / overlap state"]
  end

  subgraph Router["Router owns runtime state + perf model"]
    RS["Router state<br/>workers, queues, inflight, KV overlap"]
    PM["RustEnginePerfModel<br/>AIC native / regression / tuning"]
    ADJ["Workload adapter<br/>apply KV reuse / routing policy"]
    API["Planner query API"]
  end

  subgraph Planner["Planner"]
    PT["Planner tick"]
    L["Load decision"]
    T["Throughput decision"]
    SD["ScalingDecision"]
  end

  W1 --> FPM
  W1 --> KV

  FPM --> RS
  KV --> RS
  FPM --> PM

  RS --> ADJ
  ADJ --> PM

  PT --> API
  API --> Q1["query queued prefill time"]
  API --> Q2["query scheduled decode ITL"]
  API --> Q3["query engine capacity RPS"]
  Q1 --> PM
  Q2 --> PM
  Q3 --> PM

  PM --> API
  API --> L
  API --> T
  L --> SD
  T --> SD
```

### Sequence Diagram

```mermaid
sequenceDiagram
  participant Worker
  participant Router
  participant Perf as Router-owned PerfModel
  participant Planner

  Worker->>Router: FPM updates
  Worker->>Router: KV cache events / routing state
  Router->>Perf: tune_with_fpms(iterations)

  Planner->>Router: get_load_snapshot()
  Router->>Router: build effective workload view
  Router->>Router: apply KV reuse estimate to queued prefill
  Router->>Perf: get_queued_prefill_time(adjusted FPMs)
  Router->>Perf: get_scheduled_decode_itl(FPMs)
  Perf-->>Router: TTFT / ITL estimates
  Router-->>Planner: load signals per worker

  Planner->>Router: find_capacity(request shape + SLA)
  Router->>Perf: find_engine_capacity_rps(request)
  Perf-->>Router: rps + ttft/itl/e2e + eligible
  Router-->>Planner: capacity result

  Planner->>Planner: combine load + throughput policy
  Planner-->>Router: scaling decision / desired replicas
```

### Ownership Boundary

```mermaid
flowchart LR
  RouterOwns["Router owns:<br/>FPM grouping<br/>worker state<br/>KV overlap<br/>effective queued workload<br/>perf model tuning"]
  PlannerOwns["Planner owns:<br/>policy<br/>SLA thresholds<br/>scale up/down rules<br/>replica lower bounds<br/>final scaling decision"]

  RouterOwns -->|"query results"| PlannerOwns
  PlannerOwns -->|"desired capacity / scaling action"| RouterOwns
```

### Possible Router API

```text
RouterPerfService
  # Router continuously consumes FPM / KV events internally,
  # updates state, and tunes the perf model.

  get_worker_load(worker_id)
    -> queued_prefill_time
    -> scheduled_decode_itl
    -> diagnostics

  find_engine_capacity(worker_type, request_shape, sla, optimization_target)
    -> rps
    -> ttft / itl / e2e
    -> eligible

  get_cluster_capacity(request_shape, sla)
    -> per-worker capacity
    -> aggregate capacity
    -> bottleneck worker type

  get_scaling_inputs()
    -> load signals
    -> throughput lower-bound inputs
    -> model diagnostics
```

### Benefits

- The router is better positioned to handle KV reuse because it already has routing and cache-overlap views.
- The planner does not need to understand FPM schema details. Differences between FPM v0/v1/v2 can remain in the router/perf layer.
- Runtime details such as attention DP, preempted decode, and waiting KV transfer can be normalized at the router layer.
- The planner stays a policy engine: it scales based on semantic load and capacity signals.

### Risks

- The router API must be designed carefully; otherwise, it may become a remote view of planner-internal state.
- The planner scaling tick becomes dependent on router query latency and consistency.
- The router becomes heavier because it owns perf model tuning and capacity queries.
- In multi-router / multi-replica deployments, the scope of perf model state must be defined clearly: per-router, locally aggregated, or globally aggregated.

---

## Comparison of the Two Options

| Dimension | Planner directly calls shim | Router owns perf model |
| --- | --- | --- |
| Short-term implementation cost | Low | Medium to high |
| FPM schema complexity | Handled by planner | Handled by router/perf layer |
| KV reuse | Planner adjusts inputs before calling shim | Router adjusts using cache/routing state |
| Attention DP / rank grouping | Planner must pass the correct FPM list | Router normalizes internally |
| Planner responsibility | Policy + workload interpretation | Policy only |
| Router responsibility | Routing/cache state | Routing/cache state + perf model |
| Distance from current code | Closest | Requires new query API |
| Long-term ownership boundary | Easier to blur | Cleaner |

---

## Discussion Questions

1. Should the planner consume FPMs directly, or should it only consume semantic load signals exposed by the router?
2. Where should KV reuse be applied?
   - Planner-side input adjustment
   - Router workload adapter
   - Future FPM schema extension
3. What should the state scope of a router-owned perf model be?
   - per worker
   - per router process
   - per deployment / global
4. When the planner queries the router, is snapshot consistency required?
   - Should all worker state within a single planner tick come from the same point in time?
   - Is an eventually consistent signal acceptable?
5. Should capacity queries return per-worker capacity from the router, or should the router return aggregate cluster capacity directly?
6. Do we need an intermediate state?
   - First, planner directly calls the shim.
   - Later, FPM grouping / KV adjustment moves down into the router.
   - Finally, planner switches to querying the router.

---

## Suggested Migration Path

```mermaid
flowchart TD
  S1["Step 1<br/>Planner directly calls RustEnginePerfModel<br/>Replace Python regression"]
  S2["Step 2<br/>Introduce planner perf query interface<br/>Separate load/throughput policy from perf queries"]
  S3["Step 3<br/>Router implements the same query interface<br/>Internally owns FPM + KV-adjusted workload"]
  S4["Step 4<br/>Planner switches to querying router<br/>Planner no longer interprets FPMs directly"]

  S1 --> S2 --> S3 --> S4
```

In the short term, Option 1 can reduce integration risk by validating the shim and AIC native/fallback behavior. In the medium to long term, if the team agrees that the router is the right owner for runtime and cache state, Option 2 can become the target architecture.
