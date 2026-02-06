# Planner Design

> **Tier 3 design documentation** for contributors and architects. For user-facing docs, see [docs/planner/](/docs/planner/).

## Overview

The Planner is Dynamo's autoscaling controller. It observes system metrics, predicts future load, and adjusts prefill/decode worker replica counts to proactively meet SLA targets. This document covers the internal architecture, algorithms, and design trade-offs.

## Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                    Planner Component                     │
│                                                          │
│  ┌───────────────┐ ┌───────────────┐ ┌────────────────┐  │
│  │    Metric     │ │     Load      │ │  Performance   │  │
│  │   Collector   │ │   Predictor   │ │  Interpolator  │  │
│  │  (Prometheus) │ │ (ARIMA/etc.)  │ │  (JSON data)   │  │
│  └───────┬───────┘ └───────┬───────┘ └───────┬────────┘  │
│          │                 │                  │          │
│          ▼                 ▼                  ▼          │
│  ┌───────────────────────────────────────────────────┐   │
│  │              Scaling Algorithm                    │   │
│  └───────────────────────┬───────────────────────────┘   │
│                          │                               │
│  ┌───────────────────────▼───────────────────────────┐   │
│  │               Connector Layer                     │   │
│  │  ┌───────────────────┐  ┌───────────────────────┐ │   │
│  │  │ KubernetesConn.   │  │   VirtualConn.        │ │   │
│  │  │ (PATCH DGD)       │  │   (Runtime bridge)    │ │   │
│  │  └───────────────────┘  └───────────────────────┘ │   │
│  └───────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

## Scaling Algorithm

### Step 1: Metric Collection

Every `adjustment_interval` seconds, the planner queries Prometheus for:

- Average TTFT and ITL over the interval
- Total request count
- Average input sequence length (ISL) and output sequence length (OSL)

The Prometheus query targets the Frontend's `/metrics` endpoint, which exposes histograms and counters.

### Step 2: Correction Factor Calculation

The planner maintains correction factors that adapt profiling-based predictions to real-world behavior:

```text
prefill_correction = actual_ttft / expected_ttft
decode_correction  = actual_itl  / expected_itl
```

These factors account for hard to model factors such as:

- **Request queueing**: Bursty traffic causes higher TTFT than profiled steady-state
- **Prefix cache hits**: KV reuse reduces effective prefill tokens, lowering actual TTFT
- **Chunked prefill in decode**: Small prefills processed in decode engine affect ITL
- **Metric variance**: Average ISL/OSL may not represent the actual distribution

The correction factors are applied as multipliers to the next scaling decision. Setting `--no-correction` disables this for debugging or when cold-start artifacts dominate.

### Step 3: Load Prediction

The planner forecasts three values for the next interval:

- `next_num_req`: Number of requests
- `next_isl`: Average input sequence length
- `next_osl`: Average output sequence length

Four predictor implementations are available:


| Predictor    | Algorithm                                | Best For                         |
| ------------ | ---------------------------------------- | -------------------------------- |
| **Constant** | `next = current`                         | Stable workloads, long intervals |
| **ARIMA**    | Auto-ARIMA with optional log1p transform | Trending/seasonal patterns       |
| **Kalman**   | Local linear trend Kalman filter         | Bursty traffics                  |
| **Prophet**  | Facebook Prophet time-series model       | Complex seasonality              |


All predictors support warm-starting from trace files (`--load-predictor-warmup-trace`).

### Step 4: Replica Calculation

**Prefill replicas:**

```python
predicted_load = next_requests * next_isl / interval * min(1, prefill_correction)
prefill_replicas = ceil(predicted_load / interpolated_throughput / gpus_per_engine)
```

The prefill correction factor has a linear effect on throughput because prefill is single-batched.

**Decode replicas:**

```python
# Apply correction to the ITL SLA target
corrected_itl = target_itl / decode_correction_factor

# Find best throughput/GPU that achieves corrected ITL at predicted context length
throughput_per_gpu = decode_interpolator.find_best_throughput_per_gpu(
    itl=corrected_itl,
    context_length=next_isl + next_osl / 2
)

# Calculate required replicas
decode_replicas = ceil(next_num_req * next_osl / interval / throughput_per_gpu / gpus_per_engine)
```

### Step 5: Scaling Execution

The planner calls `connector.set_component_replicas()` with the calculated targets. Scaling is non-blocking by default: the planner continues monitoring while replicas are adjusting.

## Connector Design

### Interface

```python
class PlannerConnector(ABC):
    async def add_component(self, component_name)
    async def remove_component(self, component_name)
    # Extended interface (not on ABC, but implemented by both connectors):
    async def set_component_replicas(self, targets, blocking)
    async def validate_deployment(self, ...)
    async def wait_for_deployment_ready(self)
```

### KubernetesConnector

Directly PATCHes the DGD resource to update replica counts. The operator watches for DGD changes and reconciles component deployments.

**Design decisions:**

- Uses `DYN_PARENT_DGD_K8S_NAME` to find its parent DGD (injected by operator)
- Resolves services by `subComponentType` field (prefill/decode), with fallback to legacy component names
- Validates deployment structure on startup: checks that prefill and decode services exist and model names match

### VirtualConnector

For non-native environments (e.g., custom orchestrators). Writes scaling decisions to the distributed runtime via `VirtualConnectorCoordinator` (Rust binding). External systems use `VirtualConnectorClient` to poll decisions and report completion.

**Scaling decision flow:**

1. Planner writes `(num_prefill, num_decode, decision_id)` to runtime
2. External system reads decision via `client.wait()`
3. External system executes scaling
4. External system reports completion via `client.complete(decision)`
5. Planner sees `scaled_decision_id >= decision_id` and proceeds

**Timeout**: If scaling isn't acknowledged within 1800s (configurable), the planner proceeds with new decisions anyway.

## Performance Interpolation

The planner uses pre-deployment profiling data (NPZ files) to map (throughput, ISL/OSL, context_length) -> (TTFT, ITL). This data comes from the SLA-driven profiling process (either online GPU profiling or AI Configurator estimation).

Two interpolators are maintained:

- **Prefill interpolator**: Maps (throughput_per_gpu, ISL) -> TTFT
- **Decode interpolator**: Maps (throughput_per_gpu, context_length) -> ITL

The interpolators use the profiling sweep granularity to determine precision. Finer granularity means more profiling samples but more accurate interpolation.

## Initialization

The planner starts with a 30-second delay (`INIT_PLANNER_START_DELAY`) to allow other components (frontend, workers) to register and stabilize. This is a known workaround (marked TODO in code) that should be replaced with a proper readiness check.

After the delay:

1. Initialize the connector (K8s or Virtual based on `--environment`)
2. Validate deployment structure
3. Load profiling results
4. Build interpolators
5. Initialize load predictor
6. Enter main scaling loop

## Performance Considerations

- **Adjustment interval sizing**: The interval must be long enough for scaling operations to complete. If `adjustment_interval` is shorter than the time to add/remove a worker (which includes pod scheduling, model loading, and registration), scaling decisions will overlap. Default of 180s is conservative; workloads with fast model loading can use shorter intervals.
- **Correction factor stability**: Correction factors are recalculated each interval. During traffic transitions (e.g., ramp-up), they can oscillate. The `--no-correction` flag disables correction for scenarios where cold-start artifacts dominate and distort the factor.
- **Interpolation accuracy vs profiling cost**: Higher `prefillInterpolationGranularity` and `decodeInterpolationGranularity` in the profiling sweep produce more accurate interpolation but increase profiling time linearly. Default granularity (16 prefill, 6 decode) balances accuracy with profiling duration.
- **Predictor warm-up period**: All predictors need observation history before making reliable forecasts. ARIMA and Prophet need multiple adjustment intervals of data. Kalman starts forecasting after `--kalman-min-points` observations. During warm-up, the planner uses the constant predictor as fallback.

## Known Limitations

1. **30-second startup delay**: Hardcoded wait for component registration. It should be replaced with runtime readiness probing.
2. **Adjustment interval vs scaling latency**: If `adjustment_interval` < time to scale, scaling decisions can pile up. The planner logs warnings but doesn't queue.
3. **Average-based interpolation**: The planner uses average ISL/OSL, which may not represent bimodal or heavy-tailed distributions well.
4. **Single DGD scope**: Each planner instance manages exactly one DGD. Multi-model/multi-DGD coordination is not supported.
5. **Load-based planner deprecated**: The load-based code path exists but is non-functional with current backends (no prefill queue metrics).

## Future Work

- Support aggregated (non-disaggregated) scaling mode for single-worker deployments
- Multi-DGD coordination for shared-cluster scenarios
- Distribution-aware interpolation (beyond mean ISL/OSL)
- Adaptive adjustment interval based on observed scaling latency

## File Map


| File                         | Size | Purpose                                               |
| ---------------------------- | ---- | ----------------------------------------------------- |
| `planner_core.py`            | 36k  | Main scaling loop, algorithm implementation           |
| `perf_interpolation.py`      | 13k  | NPZ data loading and throughput/latency interpolation |
| `load_predictor.py`          | 16k  | ARIMA, Prophet, Kalman, Constant predictors           |
| `pre_swept_results_utils.py` | 12k  | Pre-computed H100/H200 profiling data loader          |
| `kubernetes_connector.py`    | 11k  | K8s API integration for DGD scaling                   |
| `kube.py`                    | 7.4k | Low-level K8s client wrapper                          |
| `exceptions.py`              | 7.2k | Custom exception hierarchy                            |
| `prometheus.py`              | 7.3k | Prometheus query builder and client                   |
| `defaults.py`                | 8.1k | Default configs, backend name mappings                |
| `planner_argparse.py`        | 6.2k | CLI argument definitions                              |


