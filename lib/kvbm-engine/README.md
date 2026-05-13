# kvbm-engine

Distributed coordination primitives for KV cache block management (KVBM).

This crate implements the leader/worker architecture for managing KV cache blocks across a tiered storage hierarchy:

**G1** (GPU HBM) → **G2** (Pinned DRAM) → **G3** (NVMe/SSD) → **G4** (S3/MinIO)

Leaders own block metadata and make placement decisions. Workers execute data transfers (RDMA, NVMe, object storage). Sessions coordinate multi-instance block transfers.

## Feature Flags


| Flag           | Purpose                                  |
| -------------- | ---------------------------------------- |
| `s3` (default) | S3/MinIO object storage (G4 tier)        |
| `testing`      | Test utilities and mock infrastructure   |
| `nats`         | NATS-based pub/sub transport             |
| `collectives`  | NIXL + NCCL multi-GPU collectives        |
| `nccl`         | NCCL via cudarc                          |
| `nvtx`         | NVIDIA Tools Extension profiling markers |


## Documentation

Detailed module documentation lives in `[docs/](docs/)`:

- [Architecture](docs/architecture.md) — Overall system design
- [Leader](docs/leader.md) — Block coordination and metadata management
- [Session](docs/session.md) — Distributed onboarding protocol
- [Worker](docs/worker.md) — Transfer execution
- [Worker Group](docs/worker-group.md) — SPMD parallel workers
- [Offload](docs/offload.md) — Async tier-demotion pipeline
- [Offload Developer Guide](docs/offload-developer.md) — Contributing to the offload module
- [Object Storage](docs/object.md) — S3/MinIO integration
- [Runtime](docs/runtime.md) — Runtime bundle (tokio, Velo, NIXL)
- [Testing](docs/testing.md) — Test utilities and fixtures

