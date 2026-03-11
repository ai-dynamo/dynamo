# INTEGRATIONS.md - External Services and APIs

## Overview

Dynamo integrates with distributed coordination, messaging, storage, GPU, observability, and model-serving ecosystems. All integrations are configured via `DYN_*` environment variables using Figment.

---

## Distributed Coordination

### etcd
- **Version**: `etcd-client = "0.17.0"` (with TLS feature)
- **Purpose**: Service discovery, distributed configuration, leader election
- **Usage**: Runtime component registration, worker discovery, cluster coordination
- **Config**: `DYN_ETCD_ENDPOINTS` environment variable
- **Location**: `lib/runtime/` — etcd client wrappers

---

## Messaging / Event Plane

### NATS
- **Version**: `async-nats = "0.45.0"` (with `service` feature)
- **Purpose**: Primary pub/sub event bus for distributed component communication
- **Usage**: KV cache events, block lifecycle events, inference scheduling messages
- **Config**: `DYN_NATS_SERVER` environment variable
- **Location**: `lib/runtime/`, `lib/velo-events/`

### ZeroMQ
- **Versions**: `async_zmq = "0.4.0"`, `zmq`, `tmq`
- **Purpose**: Alternative high-performance transport for point-to-point messaging
- **Usage**: Low-latency inference request routing, disaggregated prefill/decode transport
- **Location**: `lib/velo-transports/` (commented out of main workspace — on branch)

---

## Model Storage / Downloading

### Hugging Face Hub
- **Version**: `hf-hub = "0.4.2"` (tokio async feature, no default features)
- **Purpose**: Model weights downloading and caching
- **Usage**: Automatic model downloading by name/revision before inference
- **Config**: `HF_TOKEN`, `HF_HOME` environment variables

### ModelExpress
- **Purpose**: NVIDIA internal model delivery/caching service (alternative to HF Hub)
- **Usage**: Enterprise model distribution in NVIDIA environments

### S3 / MinIO (G4 Object Storage Tier)
- **Version**: `aws-sdk-s3 = "1.120.0"`
- **Purpose**: G4 (external) KV cache block storage tier
- **Usage**: `kvbm-physical` G4 backend — offloads KV blocks to object storage
- **Location**: `lib/kvbm-physical/`

---

## Kubernetes / Orchestration

### Kubernetes API
- **Version**: `kube = "2.0.1"`
- **Purpose**: Pod discovery, node metadata, operator management
- **Usage**:
  - Operator (`deploy/Kubernetes/operator/`) manages `DynamoGraphDeployment` CRDs
  - Runtime discovers co-located pods via K8s API for topology-aware routing
- **Location**: `deploy/Kubernetes/operator/`, `lib/runtime/`

---

## GPU / Hardware Acceleration

### CUDA / cudarc
- **Version**: `cudarc = "0.19.2"` (with `cuda-version-from-build-system` feature)
- **Purpose**: GPU memory management, kernel execution, device context
- **Usage**: KV cache block allocation on GPU (G1 tier), CUDA IPC for zero-copy transfer
- **Location**: `lib/kvbm-kernels/`, `lib/kvbm-physical/`

### NCCL (via cudarc)
- **Purpose**: GPU collective operations for multi-GPU coordination
- **Usage**: Tensor parallelism coordination between GPUs

### NIXL
- **Version**: `nixl-sys = "=0.10.1"`
- **Purpose**: RDMA (Remote Direct Memory Access) for cross-node KV block transfers
- **Usage**: `kvbm-physical` — ultra-low-latency G2/G3 tier KV block migration
- **Location**: `lib/kvbm-physical/`

---

## HTTP / Transport

### Axum
- **Version**: `axum = "=0.8.4"` (with macros)
- **Purpose**: HTTP server framework for OpenAI-compatible REST API
- **Usage**: Frontend inference gateway, health endpoints, metrics endpoints
- **Location**: `lib/runtime/`, `deploy/Kubernetes/inference-gateway/`

### Velo Messenger
- **Source**: `git = "https://github.com/ai-dynamo/dynamo", branch = "ryan/velo-messenger"`
- **Purpose**: Distributed RPC and transport abstraction layer
- **Usage**: Replaces direct NATS/ZMQ usage with unified transport API
- **Local workspace crates** (partially active): `lib/velo-common/`, `lib/velo-transports/`, `lib/velo-events/`
- **Status**: In development on `ryan/velo-messenger` branch; some members commented out of main `Cargo.toml`

### reqwest
- **Version**: `reqwest = "0.12.24"` (no default features)
- **Purpose**: HTTP client for outbound API calls (HF Hub, model APIs)

---

## Observability

### OpenTelemetry
- **Version**: `opentelemetry = "0.31.0"` (trace + logs features)
- **SDK**: `opentelemetry_sdk = "0.31.0"` (rt-tokio)
- **Exporter**: `opentelemetry-otlp = "0.31.0"` (gRPC via tonic)
- **Purpose**: Distributed tracing and structured logging
- **Config**: `OTEL_EXPORTER_OTLP_ENDPOINT` or similar OTEL env vars
- **Location**: `lib/runtime/` — tracing setup

### Prometheus
- **Version**: `prometheus = "0.14"`
- **Purpose**: Metrics collection and export
- **Usage**: Inference throughput, KV cache hit rates, block pool utilization, queue depths
- **Endpoint**: `/metrics` HTTP endpoint on each component
- **Location**: `lib/kvbm-logical/src/metrics.rs`, component-level prometheus utils

---

## Inference Backends (Python Components)

### vLLM
- **Purpose**: GPU-accelerated LLM inference engine
- **Integration**: `components/src/dynamo/vllm/` — Python wrapper + Dynamo hooks
- **KV events**: Hooks into vLLM KV cache for block-level eviction tracking

### TensorRT-LLM
- **Purpose**: NVIDIA TensorRT-optimized inference
- **Integration**: `components/src/dynamo/trtllm/` — request handlers, autodeploy

### SGLang
- **Purpose**: Structured generation language model server
- **Integration**: `components/src/dynamo/sglang/` — memory handlers, processor

---

## Authentication

No dedicated auth provider integration found. Access control relies on:
- Kubernetes RBAC for operator resources
- Network-level isolation for internal services
- HF token for model downloads
