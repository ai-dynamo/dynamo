# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Overview

Dynamo is a distributed inference framework for serving large language models at scale. It orchestrates disaggregated prefill/decode pipelines, intelligent KV-cache routing, SLA-driven autoscaling, and multi-backend support (vLLM, SGLang, TensorRT-LLM) across GPU clusters. The framework spans Rust (runtime, router, KV management), Python (components, planner, frontends, backends), and Go (Kubernetes operator).

## Repository Structure

```
dynamo/
├── components/src/dynamo/     # Python components (frontend, router, planner, backends)
├── lib/                       # Rust libraries (runtime, kv-router, llm, memory, parsers)
├── deploy/operator/           # Go Kubernetes operator (DGD/DGDR controllers)
├── deploy/helm/               # Helm charts
├── examples/                  # Deployment examples per backend
├── recipes/                   # Production-ready model recipes
├── benchmarks/                # Profiling, benchmarking, load generation
├── tests/                     # Integration and unit tests
├── docs/                      # Documentation (Sphinx + Fern)
└── container/                 # Container build definitions
```

## Key Commands

### Building (Rust)
```bash
# Check compilation
cargo check

# Development build (default workspace members only)
cargo build

# Full workspace build (includes dynamo-run, engines)
cargo build --workspace

# Release build
cargo build --release

# Build Python bindings (generates Rust->Python bridge)
cd lib/bindings/python/codegen && cargo build
```

### Building (Python)
```bash
# Install in development mode
pip install -e ".[dev]"

# Install with backend extras
pip install -e ".[vllm]"
pip install -e ".[sglang]"
pip install -e ".[trtllm]"
```

### Building (Go Operator)
```bash
cd deploy/operator
make build
make docker-build IMG=<image>
```

### Testing
```bash
# Rust tests
cargo test

# Python tests (planner)
pytest tests/planner/

# Python tests (router)
pytest tests/router/

# Python tests (frontend)
pytest tests/frontend/

# Operator tests (Go)
cd deploy/operator && make test

# Integration tests (requires GPU + running cluster)
pytest tests/basic/
pytest tests/serve/
```

### Running Locally
```bash
# Start infrastructure (NATS + etcd)
docker compose -f deploy/docker-compose.yml up -d

# Run a basic example
cd examples/backends/vllm && python -m dynamo.vllm --model Qwen/Qwen3-0.6B

# Run with dynamo-run launcher
cargo run --package dynamo-run -- --model <model-name>
```

### Kubernetes Deployment
```bash
# Install platform
export NAMESPACE=dynamo-system
export RELEASE_VERSION=0.9.0
helm install dynamo-crds dynamo-crds-${RELEASE_VERSION}.tgz --namespace default
helm install dynamo-platform dynamo-platform-${RELEASE_VERSION}.tgz --namespace ${NAMESPACE} --create-namespace

# Deploy a model (DGD - direct)
kubectl apply -f examples/backends/vllm/deploy/agg.yaml -n ${NAMESPACE}

# Deploy a model (DGDR - SLA-driven)
kubectl apply -f benchmarks/profiler/deploy/profile_sla_aic_dgdr.yaml -n ${NAMESPACE}
```

## Architecture Overview

### Core Runtime (Rust)

**Distributed Runtime (`lib/runtime/`)**
- Service discovery via etcd and NATS
- Request plane (point-to-point RPC) and event plane (pub/sub)
- Component lifecycle management and graceful shutdown
- Cancellation token propagation

**KV Router (`lib/kv-router/`)**
- Routes requests based on KV cache locality
- Prefix-aware scheduling to maximize cache hits
- Load-balanced and KV-aware routing modes

**LLM Library (`lib/llm/`)**
- Token management and sequence handling
- Sampling parameters and chat template processing

**Memory Management (`lib/memory/`)**
- GPU memory tracking and allocation
- KV cache buffer management (KVBM)

### Python Components (`components/src/dynamo/`)

**Frontend (`frontend/`)**
- OpenAI-compatible HTTP API server (FastAPI)
- Request validation, preprocessing, streaming responses
- Metrics exposure at `/metrics` for Prometheus

**Router (`router/`)**
- Request routing with multiple strategies (round-robin, KV-aware)
- Prefix-aware routing for cache optimization

**Planner (`planner/`)**
- SLA-based autoscaling (primary) and load-based scaling (deprecated)
- Predictive load forecasting (ARIMA, Prophet, Kalman, constant)
- Performance interpolation from profiling data
- Connectors: KubernetesConnector, VirtualConnector

**Backends (`vllm/`, `sglang/`, `trtllm/`)**
- Framework-specific worker implementations
- Disaggregated prefill/decode separation
- Engine lifecycle and model loading

### Kubernetes Operator (Go, `deploy/operator/`)

**Custom Resources:**
- `DynamoGraphDeployment` (DGD) - Direct pipeline specification
- `DynamoGraphDeploymentRequest` (DGDR) - SLA-driven deployment intent
- `DynamoComponentDeployment` - Per-component resource (managed by operator)
- `DynamoGraphDeploymentScalingAdapter` - Autoscaling integration

**Controllers:**
- DGD controller: Translates graph spec into component deployments
- DGDR controller: Orchestrates profiling -> DGD generation -> deployment
- Component controller: Manages individual component lifecycle
- Scaling adapter: Bridges external autoscalers

**Graph Construction (`internal/dynamo/`):**
- `graph.go` - Main graph topology builder
- `grove.go` - Grove (PodClique) integration for gang scheduling
- `component_frontend.go`, `component_worker.go`, `component_planner.go` - Component-specific logic
- `backend_vllm.go`, `backend_sglang.go`, `backend_trtllm.go` - Backend-specific configuration

## Configuration Patterns

### DynamoGraphDeployment (DGD)
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
    Worker:
      componentType: worker
      replicas: 2
      resources:
        limits:
          gpu: "1"
```

### DynamoGraphDeploymentRequest (DGDR)
```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeploymentRequest
spec:
  model: meta-llama/Llama-3-70b
  backend: vllm
  profilingConfig:
    profilerImage: nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.6.1
    config:
      sla:
        isl: 3000
        osl: 150
        ttft: 200
        itl: 20
  autoApply: true
```

### Environment Variables
- `DYN_NAMESPACE` - Dynamo logical namespace (injected by operator)
- `DYN_PARENT_DGD_K8S_NAME` - Parent DGD name (for planner)
- `HF_TOKEN` - Hugging Face authentication token
- `PROMETHEUS_ENDPOINT` - Prometheus URL for planner metrics
- `PLANNER_PROMETHEUS_PORT` - Port for planner's own metrics

## Development Patterns

### Error Handling
- Rust: `anyhow` for application errors, `thiserror` for library errors
- Python: Custom exception hierarchy in `planner/utils/exceptions.py`
- Go operator: Kubernetes controller-runtime error patterns with status conditions

### Async Architecture
- Rust: Tokio runtime with cancellation tokens, `async-nats` for messaging
- Python: `asyncio` with `@dynamo_worker()` decorator for component lifecycle
- Go: controller-runtime reconciliation loops

### Connector Pattern (Planner)
The planner uses a connector abstraction (`PlannerConnector`) to decouple scaling logic from infrastructure:
- `KubernetesConnector` - Scales DGD replicas via K8s API
- `VirtualConnector` - Communicates scaling decisions via runtime for non-native environments

### Service Discovery
Components register via etcd leases and discover peers through the distributed runtime. The operator injects `DYN_NAMESPACE` to scope discovery. Workers auto-register on startup and are removed on graceful shutdown.

## Testing Strategy

- **Unit tests**: Per-component in `tests/<component>/`
- **Integration tests**: `tests/basic/`, `tests/serve/` (require GPU)
- **Fault tolerance**: `tests/fault_tolerance/` (graceful shutdown, request migration)
- **Planner tests**: `tests/planner/` (SLA algorithm, connectors, load predictors)
- **Operator tests**: `deploy/operator/` with envtest and mock K8s API
- **Benchmarks**: `benchmarks/` (profiler, router, multimodal, load generators)

## Key Abstractions

- **`DistributedRuntime`** (Rust/Python): Central runtime managing service discovery, request plane, event plane
- **`@dynamo_worker()`** (Python): Decorator that bootstraps a component with runtime access
- **`PlannerConnector`** (Python): Interface for planner scaling actions
- **`DynamoGraphDeploymentSpec`** (Go): Top-level spec defining the inference graph
- **`ComponentKind`** (Go): Enum for underlying K8s resource type (PodClique, Deployment, LeaderWorkerSet)

## Per-Component AGENTS.md

Each major component has its own AGENTS.md with component-specific guidance:
- `components/src/dynamo/planner/AGENTS.md` - Planner development guide
