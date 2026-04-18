# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

This is a Rust crate (`kvbm-engine`) in the dynamo workspace. Rust edition 2024, requires rustc 1.93.1+.

```bash
# Build
cargo build -p kvbm-engine
cargo build -p kvbm-engine --features s3,testing,nats

# Test (most tests require the `testing` feature)
cargo test -p kvbm-engine --features testing
cargo test -p kvbm-engine --features testing -- test_name  # single test

# Lint
cargo clippy -p kvbm-engine --all-features
cargo fmt
cargo machete
```

## Feature Flags

| Flag | Purpose |
|------|---------|
| `s3` (default) | S3/MinIO object storage (G4 tier) |
| `testing` | Test utilities, mock infrastructure, fixtures |
| `testing-s3` | S3 integration tests (requires live S3/MinIO endpoint) |
| `testing-nixl-obj` | NIXL OBJ integration tests (requires NIXL OBJ backend + S3 endpoint) |
| `nats` | NATS-based pub/sub transport |
| `collectives` | NIXL + NCCL multi-GPU collectives |
| `nccl` | NCCL via cudarc |
| `nvtx` | NVIDIA Tools Extension profiling markers |

## Architecture

kvbm-engine implements distributed coordination for KV cache block management across a tiered storage hierarchy:

- **G1** (GPU HBM) → **G2** (Pinned DRAM) → **G3** (NVMe/SSD) → **G4** (S3/MinIO)

Leaders own block metadata and make placement decisions. Workers execute data transfers (RDMA, NVMe, object storage). Sessions coordinate multi-instance block transfers between leaders and workers.

### Key Modules

- **`leader/`** — `InstanceLeader` coordinates block lookups (`find_matches`), holds blocks via RAII `BlockHolder`, and manages distributed sessions. The `Leader` trait is the core coordination interface.
- **`leader/session/`** — Distributed session protocol: `InitiatorSession` (requester), `ResponderSession` (provider), `ServerSession` (server-side block exposure with optional G3→G2 staging). Sessions track onboarding state: Searching → Holding → Staging → Ready → Complete.
- **`worker/`** — `PhysicalWorker` owns a `TransferManager` and layout handles for actual transfers. `CoordinatedWorker` wraps any `Worker` with the leader's coordination state. The `Worker` and `WorkerTransfers` traits define the execution contract.
- **`worker/group/`** — `SpmdParallelWorkers` broadcasts operations to all workers in parallel (SPMD model) with event aggregation.
- **`worker/velo/`** — RPC layer (`VeloWorkerService`/`VeloWorkerClient`) for remote worker execution via Velo.
- **`offload/`** — Multi-stage async pipeline for tier demotion: PolicyEvaluator → PreconditionAwaiter → Batcher → TransferExecutor. Supports per-container cancellation tokens. **See `src/offload/AGENTS.md` for governance rules before modifying this module.**
- **`object/`** — `ObjectBlockOps` trait for G4 storage. S3 implementation with concurrent uploads/downloads. `ObjectLockManager` for distributed locking via conditional S3 PUTs.
- **`runtime/`** — `KvbmRuntime` bundles tokio, Velo messenger, NixlAgent (RDMA), and EventManager. Built via `KvbmRuntimeBuilder` or quick constructors (`from_env_leader`, `from_env_worker`).
- **`pubsub/`** — Publisher/Subscriber traits with NATS and in-memory stub implementations.
- **`collectives/`** — `CollectiveOps` trait for multi-GPU sync. NCCL implementation and stub for testing. MLA pattern: only rank 0 needs G2/G3; others receive via broadcast.
- **`testing/`** — Feature-gated test utilities: `TestManagerBuilder`, `MessengerPair`, `TestSession`, `EventsPipelineFixture`, `MultiInstancePopulator`, `TestAgent`.

### Documentation

Module docs live in `docs/` and are included via `#[doc = include_str!("../docs/...")]`. When modifying a module, update the corresponding doc file.

### Key Patterns

- **Trait-based abstraction**: `Leader`, `Worker`, `WorkerTransfers`, `ObjectBlockOps`, `CollectiveOps`, `KeyFormatter` — implementations are swappable (real vs. test stubs).
- **RAII resource management**: `BlockHolder` holds blocks during sessions with automatic release on drop. `TransferHandle` tracks offload operations.
- **Builder pattern**: `InstanceLeaderBuilder`, `PhysicalWorkerBuilder`, `KvbmRuntimeBuilder`, `OffloadEngineBuilder`.
- **Execution vs. coordination state**: `PhysicalWorker` owns execution state; `CoordinatedWorker` adds the leader's coordination view. Same API regardless of worker locality.

### Workspace Dependencies

Internal crates: `kvbm-common`, `kvbm-config`, `kvbm-kernels`, `kvbm-logical`, `kvbm-physical`, `velo`, `dynamo-tokens`, `dynamo-memory`.

## Offload Module Governance

The offload module (`src/offload/`) has explicit policies (P1–P6) documented in its README. Before modifying offload code, read `src/offload/AGENTS.md` and the offload docs (`docs/offload.md`, `docs/offload-developer.md`). Off-policy changes require user approval before implementation.
