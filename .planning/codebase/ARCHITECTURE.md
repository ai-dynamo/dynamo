# ARCHITECTURE.md - System Design and Patterns

## Overview

Dynamo is a distributed LLM inference framework with disaggregated prefill/decode architecture. It uses a layered Rust library system coordinated by a Python component layer and managed by a Kubernetes operator.

---

## Architectural Pattern

**Disaggregated Inference with KV Block Management**

The system separates prefill (prompt processing) from decode (token generation) workloads, enabling independent scaling. A shared KV block manager (`KVBM`) coordinates GPU memory across workers.

Core pattern: **Actor/Component model** where Rust `Component` trait implementations communicate via NATS pub/sub or Velo Messenger transport, coordinated by etcd for discovery.

---

## Layers

### Layer 1: Hardware Abstraction (`lib/kvbm-kernels/`, `lib/kvbm-physical/`)
- CUDA kernel wrappers for GPU memory operations
- Physical block transfer via RDMA (NIXL) and CUDA IPC
- Storage tiers: G1=GPU, G2=CPU, G3=Disk, G4=S3/Object Storage

### Layer 2: Logical Block Management (`lib/kvbm-logical/`)
- Type-state block lifecycle: `Reset → Staged → Registered → Inactive`
- Block registry with `PositionalRadixTree` indexed by `SequenceHash`
- Pluggable eviction backends: LRU, MultiLRU (frequency-aware), Lineage-aware
- TinyLFU frequency tracking, Prometheus metrics
- Event pipeline for block lifecycle events

### Layer 3: KV Block Manager Engine (`lib/kvbm-engine/`, `lib/kvbm-config/`, `lib/kvbm-common/`)
- Orchestrates logical + physical block management
- Configuration via `kvbm-config`
- Shared types and interfaces in `kvbm-common`

### Layer 4: KV Router (`lib/kv-router/`)
- Routes inference requests to workers with matching KV cache prefix hits
- `PositionalRadixTree` for prefix-aware routing decisions
- Integrates with runtime for discovery of workers and their cache states

### Layer 5: Core Runtime (`lib/runtime/`)
- `Component` trait: base abstraction for all distributed components
- Service discovery via etcd
- Pipeline abstraction: network (TCP), local, and distributed pipelines
- Engine routes, health checks, system status server
- OpenTelemetry tracing and Prometheus metrics integration

### Layer 6: LLM Abstractions (`lib/llm/`, `lib/tokens/`, `lib/parsers/`, `lib/memory/`, `lib/async-openai/`)
- `lib/llm/` — core inference request/response types, KV event schemas
- `lib/tokens/` — `TokenBlock` and tokenization primitives
- `lib/parsers/` — tool-calling format parsers (per-model, Rust)
- `lib/memory/` — memory pool abstractions
- `lib/async-openai/` — OpenAI-compatible API client (BYOT — bring your own transport)

### Layer 7: Velo Transport (`lib/velo-common/`, `lib/velo-transports/`, `lib/velo-events/`)
- Unified transport abstraction over NATS/ZMQ
- RPC framework for component-to-component calls
- Event bus for async communication
- **Status**: In active development on `ryan/velo-messenger` branch

### Layer 8: Python Components (`components/src/dynamo/`)
- Inference backend wrappers: `vllm/`, `trtllm/`, `sglang/`
- Common utilities: `common/` (storage, multimodal, memory)
- Frontend API processing: `frontend/`
- Routing/planning: `router/`, `planner/`, `global_planner/`, `global_router/`
- Profiler: `profiler/`

### Layer 9: Operator & Deployment (`deploy/Kubernetes/`)
- Go-based Kubernetes operator managing `DynamoGraphDeployment` CRDs
- Helm charts for production deployment
- NATS server, observability stack (Prometheus/Grafana)
- Inference gateway (HTTP → internal routing)

---

## Data Flow

### Aggregated (Collocated Prefill+Decode)
```
Client HTTP → Inference Gateway → Frontend Component
  → Router (KV-aware, picks worker with prefix hit)
    → vLLM/TRT-LLM/SGLang Worker
      → KVBM (manages GPU G1 blocks, evicts to G2/G3/G4)
        → Response back via OpenAI-compatible stream
```

### Disaggregated (Separate Prefill/Decode)
```
Client HTTP → Inference Gateway → Frontend Component
  → Global Router → Prefill Worker (processes prompt, generates KV blocks)
    → KV blocks transferred via RDMA/CUDA IPC to Decode Worker
      → Decode Worker (generates tokens from cached KV)
        → Response stream to client
```

### KV Block Transfer Flow
```
Prefill Worker → KVBM Engine → kvbm-physical (G1 GPU blocks)
  → RDMA/NIXL transfer → Decode Worker kvbm-physical
    → kvbm-logical (registers blocks in registry)
      → KV Router (updates routing table with new cache state)
```

---

## Key Abstractions

### Rust Traits
- `Component` (`lib/runtime/`) — base for all distributed service components
- `InactivePoolBackend<T>` (`lib/kvbm-logical/`) — eviction strategy interface
- `BlockMetadata` — marker trait for storage tier type parameters
- Transport traits in `lib/velo-transports/` — unified NATS/ZMQ/TCP abstraction

### Python Protocols
- Request handler base classes in `trtllm/` and `sglang/` components
- Aggregated/disaggregated handler split (`test_trtllm_aggregated_handler.py`)

### Type-State Pattern (KVBM)
Blocks use compile-time states to prevent invalid transitions:
```rust
MutableBlock<T> → CompleteBlock<T> → ImmutableBlock<T> → WeakBlock<T>
   (Reset)            (Staged)          (Registered)      (Non-owning)
```

---

## Entry Points

### Rust Libraries
- `lib/runtime/src/lib.rs` — runtime public API
- `lib/kvbm-logical/src/lib.rs` — block manager public API
- `lib/kv-router/src/lib.rs` — router public API

### Python Components
- `components/src/dynamo/vllm/` — vLLM integration entrypoint
- `components/src/dynamo/trtllm/` — TRT-LLM integration
- `components/src/dynamo/sglang/` — SGLang integration

### Kubernetes Operator
- `deploy/Kubernetes/operator/` — Go operator main package

### Tests
- `tests/` — top-level integration/E2E tests
- Per-crate `#[cfg(test)]` modules in Rust

---

## Cross-Cutting Concerns

### Configuration
- Rust: `DYN_*` environment variables via Figment (`lib/config/`)
- Python: pydantic settings models, per-component config classes

### Error Handling
- Rust: `anyhow::Result` and `thiserror` for typed errors
- Python: standard exception hierarchy with logging

### Observability
- Tracing: OpenTelemetry (OTLP gRPC export) via `tracing` + `tracing-opentelemetry`
- Metrics: Prometheus scrape endpoint per component
- Logs: structured via `tracing` crate with subscriber config
