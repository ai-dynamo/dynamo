# STRUCTURE.md - Directory Layout and Organization

## Root Layout

```
/home/ryan/repos/dynamo-workspaces/ryan-velo-messenger/
├── Cargo.toml              ← Rust workspace root (all lib/ crates)
├── Cargo.lock
├── pyproject.toml          ← Python package config (pytest, hatch)
├── rust-toolchain.toml     ← Pinned Rust toolchain version
├── deny.toml               ← cargo-deny dependency audit config
├── lib/                    ← Rust library crates
├── components/             ← Python inference backend components
├── deploy/                 ← Kubernetes operator, Helm, Docker configs
├── tests/                  ← Top-level Python integration/E2E tests
├── examples/               ← Usage examples
├── benchmarks/             ← Benchmark code
├── recipes/                ← Deployment recipe templates
├── docs/                   ← Documentation
├── fern/                   ← Fern API docs site
├── container/              ← Container build configs
└── target/                 ← Rust build output (gitignored)
```

---

## `lib/` — Rust Crates

```
lib/
├── runtime/                ← dynamo-runtime: Component model, service discovery, pipelines
│   └── src/
│       ├── component/      ← Component trait and registry
│       ├── pipeline/       ← network/local/distributed pipelines
│       │   └── network/tcp/
│       ├── protocols/      ← Protocol definitions
│       ├── discovery/      ← etcd-based service discovery
│       ├── compute/        ← Compute resource tracking
│       ├── storage/        ← Storage abstractions
│       └── metrics/        ← Prometheus metrics helpers
├── kvbm-logical/           ← Core KV block lifecycle manager
│   └── src/
│       ├── blocks/         ← RAII state-machine block types
│       ├── registry/       ← PositionalRadixTree block registry
│       ├── pools/          ← Reset/Active/Inactive pool system
│       │   └── inactive/backends/  ← LRU, MultiLRU, Lineage eviction
│       ├── events/         ← Block event pipeline
│       ├── metrics/        ← Atomic Prometheus counters
│       ├── manager/        ← BlockManager orchestrator (entry point)
│       └── testing/        ← Test utilities (feature-gated)
├── kvbm-physical/          ← Physical block transfer (G1-G4 tiers)
│   └── src/
│       ├── layout/         ← Block memory layout
│       └── transfer/       ← RDMA/CUDA IPC transfer logic
├── kvbm-engine/            ← KVBM engine orchestration
├── kvbm-common/            ← Shared KVBM types/interfaces
├── kvbm-config/            ← KVBM configuration structs
├── kvbm-kernels/           ← CUDA kernel wrappers
├── kvbm-connector/         ← Connector between inference backends and KVBM
│   └── src/
│       ├── connector/      ← Connector implementation
│       │   └── worker/     ← Per-worker connector logic
│       ├── config.rs
│       └── vllm/           ← vLLM-specific connector
├── kv-router/              ← KV-prefix-aware request routing
│   └── src/
│       └── indexer/        ← PositionalRadixTree for prefix indexing
├── llm/                    ← Core LLM types, KV event schemas
│   └── tests/              ← Integration tests
├── tokens/                 ← TokenBlock and tokenization primitives
├── parsers/                ← Tool-calling format parsers (per-model)
│   └── src/tool_calling/
├── memory/                 ← Memory pool abstractions
├── async-openai/           ← OpenAI-compatible API client
├── mocker/                 ← Mock inference backend for testing
├── config/                 ← DYN_* env var configuration (Figment)
├── velo-common/            ← Velo transport common types (branch-only)
├── velo-transports/        ← Velo NATS/ZMQ transport impl (branch-only)
├── velo-events/            ← Velo event bus (branch-only)
├── bench/                  ← Benchmark crate
└── bindings/               ← FFI bindings
    ├── c/                  ← C bindings
    └── python/             ← Python bindings (via PyO3/codegen)
```

---

## `components/` — Python Inference Backends

```
components/src/dynamo/
├── vllm/                   ← vLLM integration
│   └── tests/
├── trtllm/                 ← TensorRT-LLM integration
│   └── tests/
│       └── request_handlers/
│       └── multimodal/
├── sglang/                 ← SGLang integration
│   └── tests/
├── frontend/               ← Frontend request processing
│   └── tests/
├── common/                 ← Shared Python utilities
│   ├── tests/
│   │   ├── configuration/
│   │   ├── multimodal/
│   │   └── memory/
│   └── utils/
│       └── tests/
├── router/                 ← Local request router component
├── planner/                ← Local scaling planner
├── global_planner/         ← Cluster-wide planning
├── global_router/          ← Cluster-wide routing
├── profiler/               ← Performance profiling component
│   └── tests/
└── mocker/                 ← Mock backend component
```

---

## `deploy/` — Deployment Infrastructure

```
deploy/
├── Kubernetes/
│   ├── operator/           ← Go Kubernetes operator (DynamoGraphDeployment CRD)
│   ├── helm/               ← Helm chart for production deployment
│   ├── inference-gateway/  ← HTTP inference gateway service
│   ├── observability/      ← Prometheus, Grafana configs
│   ├── nats-server.conf    ← NATS server configuration
│   ├── docker-compose.yml  ← Local dev docker-compose
│   └── docker-observability.yml
```

---

## `tests/` — Top-Level Integration Tests

```
tests/
├── conftest.py             ← Global fixtures (ports, processes, models)
├── utils/                  ← Shared test utilities
│   ├── constants.py        ← TEST_MODELS, DefaultPort
│   ├── managed_process.py  ← Subprocess lifecycle manager
│   ├── port_utils.py       ← Dynamic port allocation
│   └── test_output.py
├── serve/                  ← Serving integration tests
├── frontend/               ← Frontend API tests
├── planner/                ← Planner tests
├── deploy/                 ← Deployment tests
└── fault_tolerance/        ← Fault tolerance scenarios
    ├── gpu_memory_service/
    └── deploy/
```

---

## Key File Locations

| Purpose | Path |
|---------|------|
| Rust workspace config | `Cargo.toml` |
| Python package config | `pyproject.toml` |
| Block manager entry | `lib/kvbm-logical/src/manager/` |
| Runtime component API | `lib/runtime/src/component.rs` |
| KV router logic | `lib/kv-router/src/` |
| Tool call parsers | `lib/parsers/src/tool_calling/` |
| Kubernetes operator | `deploy/Kubernetes/operator/` |
| E2E test fixtures | `tests/conftest.py` |
| Rust test utilities | `lib/kvbm-logical/src/testing/` |

---

## Naming Conventions

### Rust
- Crate names: `dynamo-{name}` or `kvbm-{name}` (kebab-case)
- Package names: `dynamo_{name}` (snake_case)
- Modules: snake_case, test modules as `#[cfg(test)] mod tests {}`
- Test files: `tests.rs` alongside source, or `tests/` directory

### Python
- Packages: `dynamo.{backend}` namespace
- Test files: `test_{module_name}.py` in `tests/` subdirectories
- Fixtures: `conftest.py` per test directory level

### Go (Operator)
- Standard Go conventions: PascalCase types, camelCase functions
- Controller files: `{resource}_controller.go`

---

## Adding New Code

### New Rust crate
1. Create `lib/{name}/` with `Cargo.toml` and `src/lib.rs`
2. Add to `[workspace.members]` in root `Cargo.toml`
3. Add dependency entry to `[workspace.dependencies]` if shared

### New Python component
1. Create `components/src/dynamo/{name}/` package
2. Add `tests/` subdirectory with `conftest.py`
3. Register in `pyproject.toml`

### New inference backend connector
1. Add Rust connector in `lib/kvbm-connector/src/{backend}/`
2. Add Python component in `components/src/dynamo/{backend}/`
