# Compass: System-Level Bottleneck Attribution

Compass answers the question: **"Which component is responsible for my SLO violation, by how much, and what should I change?"**

## Quick Start

```bash
# Run attribution with mock data (no live deployment required)
compass diagnose --mock --human

# JSON output for programmatic use
compass diagnose --mock -o report.json

# With counterfactual analysis: "what if my SLO is 400ms?"
compass diagnose --mock --human --slo-ttft-p99-ms 400

# Sensitivity sweep
compass replay --mock --sweep kvbm-allocate-ms=0.5,1.0,2.0,4.0 --concurrency 16,32,64 --human

# Compare two reports
compass diff --report-a before.json --report-b after.json --human

# Calibrate mocker vs real measurements
compass calibrate --trace mocker_output.json --real-run real_output.json
```

## Architecture

```
L1: Sub-component probes (feature-gated)  -> KVBM phases, Router phases
L2: Collection (existing Dynamo infra)     -> Prometheus, Tempo, Pyroscope
L3: Attribution Engine                     -> Critical-path + Saturation + Floor
L4: Sensitivity Harness                    -> Sweep perturbation + Counterfactual
L5: Surfaces                               -> CLI, Grafana dashboard
```

## Attribution Engine

The engine combines three independent signals:

### Critical-Path Analysis
Walks distributed trace spans to find the longest causal chain per request, then aggregates across requests by percentile. Each component gets a contribution percentage of the overall critical path.

### Saturation Detection (Little's Law)
For each component queue: computes L (queue length), lambda (arrival rate), W = L/lambda. Detects growing queues at steady arrival rates as saturation.

### Theoretical Floor Comparison
Compares observed latency against physics-limited minimums:
- **KVBM allocate**: hash_time + radix_depth * pointer_chase_ns
- **Prefill compute**: model_flops / gpu_tflops
- **NIXL transfer**: block_size / bandwidth

### Verdict Formula

```
score(c) = 0.6 * CP_pct(c) + 0.3 * Sat_score(c) + 0.1 * (1 - 1/FloorRatio(c))
primary_bottleneck = argmax(score)
confidence = high if gap > 0.20, medium if 0.05..0.20, low otherwise
```

Weight profiles: `default` (0.6/0.3/0.1), `conservative` (0.8/0.15/0.05), `aggressive` (0.4/0.4/0.2).

## CLI Commands

### `compass diagnose`
Run bottleneck attribution on a deployment.

| Flag | Description |
|------|-------------|
| `--deployment NAME` | Deployment name (default: "default") |
| `--window MINS` | Analysis window in minutes (default: 15) |
| `--mock` | Use synthetic data for demo/dev |
| `--human` | Human-readable terminal output |
| `--weights PROFILE` | Weight profile: default, conservative, aggressive |
| `-o PATH` | Write report to file |
| `--slo-ttft-p99-ms MS` | Run counterfactual analysis against SLO |

### `compass replay`
Run sensitivity sweep with mocker-based perturbation.

| Flag | Description |
|------|-------------|
| `--sweep SPEC` | Component=multipliers, e.g. `kvbm-allocate-ms=0.5,1,2,4` |
| `--concurrency LEVELS` | Concurrency levels to test (default: 16,32,64) |
| `--human` | Human-readable output |
| `-o PATH` | Write results to file |

### `compass diff`
Compare two attribution reports side-by-side.

### `compass calibrate`
Compare mocker predictions against real measurements. Reports residual percentage and calibration status.

## Probes

Sub-component probes are feature-gated behind `compass-probes` and runtime-gated behind `DYN_COMPASS_PROBES=1`.

### KVBM Probes
- `dynamo_compass_kvbm_hash_ms` - Hash computation time
- `dynamo_compass_kvbm_lookup_ms` - Radix tree lookup time
- `dynamo_compass_kvbm_allocate_ms` - Block allocation (labels: total, on_cpu, lock_wait)
- `dynamo_compass_kvbm_evict_ms` - Block eviction time
- `dynamo_compass_kvbm_return_ms` - Block return time

### Router Probes
- `dynamo_compass_router_cost_compute_ms`
- `dynamo_compass_router_kv_overlap_score_ms`
- `dynamo_compass_router_worker_select_ms`
- `dynamo_compass_router_dispatch_ms`

## Grafana Dashboard

Import `deploy/observability/grafana_dashboards/compass-attribution.json`.

Panels:
1. Floor ratio gauge (observed/theoretical for KVBM allocate)
2. Per-component p99 contribution bar chart
3. Utilization timeseries with saturation coloring
4. Queue depth trends
5. KVBM phase breakdown (stacked area: hash, lookup, allocate on_cpu/lock_wait, evict, return)
6. Router phase breakdown
7. Floor ratio stat panel across components

## Configuration

Environment variables:
- `DYN_COMPASS_PROBES=1` - Enable sub-component probes at runtime
- `DYN_COMPASS_SAMPLE_RATE=0.01` - Probe sampling rate (default: 1%)

## Report Schema

See [schema-reference.md](schema-reference.md) for the full JSON schema of `AttributionReport`.
