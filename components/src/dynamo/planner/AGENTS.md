# AGENTS.md - Planner Component

This file provides guidance to AI agents when working with the Planner component.

## Overview

The Planner is Dynamo's autoscaling controller. It monitors system performance metrics via Prometheus and adjusts prefill/decode worker replica counts to meet SLA targets. It runs as a Dynamo component inside the inference graph (deployed via DGD) and communicates scaling decisions through connectors.

## Code Location

```
components/src/dynamo/planner/
├── __init__.py                 # Public API (PlannerConnector, KubernetesConnector, VirtualConnector, etc.)
├── _version.py                 # Version info
├── planner_sla.py              # Entry point: @dynamo_worker init_planner()
├── planner_connector.py        # Abstract base class (PlannerConnector)
├── kubernetes_connector.py     # K8s connector: scales DGD replicas via K8s API
├── virtual_connector.py        # Virtual connector: communicates via runtime for non-native envs
├── kube.py                     # Low-level Kubernetes API wrapper (KubernetesAPI)
├── defaults.py                 # Default configs, backend component name mappings
└── utils/
    ├── planner_core.py         # Core SLA planner loop (start_sla_planner, main scaling logic) [36k]
    ├── planner_argparse.py     # CLI argument definitions
    ├── load_predictor.py       # Load forecasting (ARIMA, Prophet, Kalman, Constant)
    ├── perf_interpolation.py   # Performance interpolation from profiling data
    ├── prometheus.py           # Prometheus query client
    ├── exceptions.py           # Custom exception hierarchy
    ├── dryrun.py               # Dry-run simulation utilities
    ├── dryrun_plot_utils.py    # Matplotlib plotting for dry-run results
    ├── trace_data_extractor.py # Trace data extraction for warm-starting predictors
    └── pre_swept_results_utils.py  # Pre-computed profiling results loader
        └── pre_swept_results/  # NPZ files for H100/H200 GPU profiles
```

## Key Commands

### Running Tests
```bash
# Planner unit tests
pytest tests/planner/

# Specific test file
pytest tests/planner/test_sla_planner.py -v

# With coverage
pytest tests/planner/ --cov=dynamo.planner
```

### Running the Planner
```bash
# As a standalone process (requires distributed runtime)
python -m dynamo.planner.planner_sla \
    --namespace dynamo \
    --backend vllm \
    --ttft 200 \
    --itl 20 \
    --profile-results-dir /path/to/profiling/results

# Via Kubernetes (normal deployment path)
# The planner is deployed as a component in the DGD, not run manually
```

### Testing Connectors
```bash
# KubernetesConnector CLI (for debugging)
python -m dynamo.planner.kubernetes_connector \
    --dynamo_namespace dynamo \
    --k8s_namespace default \
    --action add \
    --component prefill \
    --blocking
```

## Architecture

### Entry Flow
```
planner_sla.py::init_planner()
    ├── sleep 30s (wait for other components)
    ├── start_sla_planner(runtime, args)  [planner_core.py]
    │   ├── Initialize connector (KubernetesConnector or VirtualConnector)
    │   ├── Validate deployment (check subComponentType prefill/decode exist)
    │   ├── Load profiling results (NPZ/JSON from pre-deployment profiling)
    │   ├── Initialize performance interpolators (prefill + decode)
    │   ├── Initialize load predictor (ARIMA/Prophet/Kalman/Constant)
    │   └── Main loop (every adjustment_interval):
    │       ├── Query Prometheus for TTFT, ITL, request count, ISL, OSL
    │       ├── Calculate correction factors (actual vs expected performance)
    │       ├── Predict next interval load
    │       ├── Calculate optimal replica counts
    │       └── connector.set_component_replicas(targets)
    └── Register dummy endpoint (component registration requirement)
```

### Connector Pattern
```
PlannerConnector (ABC)
    ├── KubernetesConnector
    │   ├── Uses KubernetesAPI to PATCH DGD replicas
    │   ├── Waits for deployment readiness
    │   └── Reads model name from DGD spec (args parsing)
    └── VirtualConnector
        ├── Uses VirtualConnectorCoordinator (Rust binding)
        ├── Writes scaling decisions to runtime
        └── Waits for external environment to acknowledge
```

### Backend Component Name Mapping
The planner must know the K8s service names for each backend's prefill/decode workers. This mapping lives in `defaults.py`:

| Backend | Prefill K8s Name | Decode K8s Name |
|---------|-----------------|-----------------|
| vLLM | `VllmPrefillWorker` | `VllmDecodeWorker` |
| SGLang | `prefill` | `decode` |
| TRT-LLM | `TRTLLMPrefillWorker` | `TRTLLMDecodeWorker` |
| Mocker | `prefill` | `decode` |

These are being deprecated in favor of `subComponentType` field on services.

## Key Design Decisions

1. **SLA planner is the primary path.** Load-based planner (`load_planner.md`) is deprecated/inoperable. All development should target SLA planner.

2. **Kubernetes-only.** Bare metal/local deployment is deprecated. The planner requires `DYN_PARENT_DGD_K8S_NAME` environment variable (injected by operator).

3. **Pre-deployment profiling required.** The SLA planner needs profiling data (NPZ files) to make scaling decisions. Without profiling data, it cannot calculate optimal replica counts.

4. **Correction factors adapt to reality.** The planner compares actual TTFT/ITL from Prometheus against interpolated expected values and adjusts future predictions. This handles queueing effects, cache hit rates, and other real-world deviations.

5. **Non-blocking scaling.** The planner issues scale commands and continues monitoring. If `adjustment_interval` is too short, previous scaling may not complete before new decisions are made.

## Common Modification Patterns

### Adding a New Load Predictor
1. Add predictor class in `utils/load_predictor.py`
2. Register it in the predictor factory (same file)
3. Add CLI argument in `utils/planner_argparse.py`
4. Add default in `defaults.py::SLAPlannerDefaults`

### Adding a New Backend
1. Add component name mapping class in `defaults.py` (e.g., `NewBackendComponentName`)
2. Add entry to `WORKER_COMPONENT_NAMES` dict
3. Ensure the backend workers expose compatible metrics at `/metrics`

### Adding a New Connector
1. Implement `PlannerConnector` interface (at minimum: `add_component`, `remove_component`)
2. Also implement `set_component_replicas`, `validate_deployment`, `get_model_name`
3. Wire it up in `planner_core.py` connector initialization

## Error Handling

The planner uses a custom exception hierarchy in `utils/exceptions.py`:
- `PlannerError` - Base class
- `SubComponentNotFoundError` - DGD missing prefill/decode service
- `DuplicateSubComponentError` - Multiple services with same subComponentType
- `DeploymentValidationError` - Deployment config invalid
- `ModelNameNotFoundError` - Cannot determine model name from DGD
- `EmptyTargetReplicasError` - No scaling targets provided

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `DYN_NAMESPACE` | Dynamo logical namespace | `dynamo` |
| `DYN_PARENT_DGD_K8S_NAME` | Parent DGD name for K8s connector | (required) |
| `PROMETHEUS_ENDPOINT` | Prometheus URL for metric queries | `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090` |
| `PLANNER_PROMETHEUS_PORT` | Port for planner's own metrics | `0` (disabled) |
| `SCALING_CHECK_INTERVAL` | Virtual connector poll interval (seconds) | `10` |
| `SCALING_MAX_WAIT_TIME` | Virtual connector max wait (seconds) | `1800` |

## Testing Notes

- Planner tests are in `tests/planner/`
- Use `--no-operation` flag for observation-only mode (no actual scaling)
- Use `VirtualConnector` for testing without a real K8s cluster
- Dry-run utilities in `utils/dryrun.py` simulate planner behavior with recorded trace data
- Pre-swept results in `utils/pre_swept_results/` provide H100/H200 profiling data for tests
