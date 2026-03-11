# Technology Stack

**Analysis Date:** 2026-03-11

## Languages

**Primary:**
- Rust 1.93.1 - Core inference framework, distributed runtime, KV behavior models, event systems
- Python 3.10+ - High-level API, integrations with vLLM, TensorRT-LLM, SGLang, Kubernetes
- Go - Service discovery and deployment tooling (secondary)

**Secondary:**
- TypeScript/JavaScript - Frontend components
- C - FFI bindings from Python/Rust

## Runtime

**Environment:**
- Tokio 1.48.0 - Async runtime for all Rust services
- Linux/POSIX systems required
- CUDA-compatible GPUs (via cudarc 0.19.2)

**Package Manager:**
- Cargo (Rust) - Primary dependency manager
- pip/Poetry (Python) - Python package management
- Lockfiles: `Cargo.lock` (present and required), `pyproject.toml`

## Frameworks

**Core Inference:**
- vLLM 0.16.0 (optional) - Vector LLM backend with flashinfer and runai features
- TensorRT-LLM 1.3.0rc5.post1 (optional) - NVIDIA runtime for optimized inference
- SGLang 0.5.9 (optional) - Structured generation language backend
- Transformers 4.56.0+ - Model loading and tokenization

**Web/API:**
- Axum 0.8.4 - Async HTTP server framework (with macros feature)
- FastAPI 0.115.0+ - Python async API framework
- Tower-HTTP 0.6 - HTTP middleware and tracing support
- Hyper 1.7.0 - HTTP primitives
- Tonic 0.13.1 (optional) - gRPC server framework

**Testing:**
- pytest - Python test runner with multiple marker categories (gpu_1, gpu_8, e2e, integration, unit)
- pytest-benchmark - Performance benchmarking
- pytest-xdist - Parallel test execution
- rstest 0.23.0 - Parametrized tests (Rust)
- Criterion 0.5 - Rust benchmarking

**Build/Dev:**
- Hatchling - Python wheel building
- Tokio-console - Async runtime debugging (optional feature)
- Pre-commit hooks (configured via `.pre-commit-config.yaml`)
- Cargo-deny - Dependency audit and license checking

## Key Dependencies

**Critical:**
- Tokio 1.48.0 - Async runtime with full feature set
- Axum 0.8.4 - HTTP server with routing and extractors
- Serde 1 + serde_json 1 - Serialization (with rc feature for Arc/Rc serialization)
- DashMap 6.1 - Concurrent hashmap for event/request tracking
- Tracing 0.1 + tracing-subscriber 0.3 - Structured logging with json output

**Infrastructure:**
- async-nats 0.45.0 - NATS pub/sub messaging with service mesh features
- async_zmq 0.4.0 + zmq 0.10 - ZeroMQ message queue support (dual implementation)
- etcd-client 0.17.0 - Service discovery via etcd with TLS
- Prometheus 0.14 - Metrics collection
- OpenTelemetry 0.31.0 + opentelemetry-otlp 0.31.0 - Distributed tracing (gRPC backend)

**Model/Data:**
- hf-hub 0.4.2 - Hugging Face Hub integration (model downloading)
- ModelExpress 0.2.0 - Alternative model downloading client
- nixl-sys 0.10.1 - GPU memory management system bindings
- cudarc 0.19.2 - CUDA runtime bindings with fallback to latest

**Kubernetes:**
- Kubernetes 32.0.1 - Python K8s client
- Kube 2.0.1 - Rust K8s client (with derive, runtime, and rustls-tls features)
- k8s-openapi 0.26.0 - K8s API types for v1_32

**Velo Distributed Systems:**
- velo (git branch: ryan/velo-messenger) - Event-based messaging framework
- velo-common (git branch: ryan/velo-messenger) - Common distributed types
- velo-events (git branch: ryan/velo-messenger) - Event system with RAII guards and poison history
- velo-transports (git branch: ryan/velo-messenger) - Transport layer (HTTP, NATS, gRPC, optional UCX)

**Serialization:**
- rmp-serde 1.1 - MessagePack binary serialization
- serde_bytes 0.11 - Efficient bytes serialization
- bincode 1 - Fast binary codec

**Utilities:**
- UUID 1.18.1 - Unique ID generation (v4 with serde)
- Chrono 0.4 - Date/time handling (with serde, clock, now)
- Parking_lot 0.12.5 - Faster Mutex implementation
- Validator 0.20.0 - Data validation with derive macros
- Blake3 1 - Fast cryptographic hashing
- xxhash-rust 0.8 - Fast non-crypto hashing
- Figment 0.10.19 - Configuration from env, JSON, TOML, and test sources

## Configuration

**Environment:**
- Environment variables use prefixed namespace: `DYN_*` (e.g., `DYN_RUNTIME_NUM_WORKER_THREADS`)
- Figment library for merging environment, TOML, and default configs
- Core config locations: `RuntimeConfig`, `DistributedConfig`, `HealthStatus`
- File: `lib/runtime/src/config.rs`

**Key Environment Variables:**
- `DYN_RUNTIME_NUM_WORKER_THREADS` - Tokio worker threads (defaults to num_cpus)
- `DYN_RUNTIME_MAX_BLOCKING_THREADS` - Blocking thread pool size (default 512)
- `DYN_SYSTEM_PORT` - Health/metrics server port (default -1 = disabled)
- `DYN_EVENT_PLANE` - Event transport: 'nats' (default) or 'zmq'
- `DYN_EVENT_PLANE_CODEC` - Event serialization: 'json' or 'msgpack'
- Kubernetes discovery enabled via cluster config

**Build:**
- Cargo.toml workspace resolver: "3" (new cohesive resolver)
- Release profile: thin LTO + codegen-units=1 for binary size/performance tradeoff
- Tokio unstable features enabled via rustflags: `--cfg tokio_unstable`
- Rust toolchain: 1.93.1

## Platform Requirements

**Development:**
- Linux/POSIX operating system
- Rust 1.93.1 (via rust-toolchain.toml)
- Python 3.10+ with pip
- Docker (for containerized builds)
- CUDA toolkit (for GPU development)

**Production:**
- Deployment targets: Kubernetes clusters
- Container images for vLLM, TensorRT-LLM, SGLang runtimes
- NVIDIA GPU support (A100, H100)
- Persistent storage for model caching (via KV store)

---

*Stack analysis: 2026-03-11*
