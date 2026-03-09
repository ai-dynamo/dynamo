# kvbm-engine

`kvbm-engine` provides distributed coordination primitives for KV Block Management (KVBM).
It implements a tiered storage model where KV cache blocks flow between GPU memory, host
DRAM, local disk, and object storage. The crate coordinates leaders (which own block
metadata and make placement decisions) with workers (which execute data transfers via
RDMA, NVMe, or object storage APIs).

## Storage Tier Model

| Tier | Medium | Latency | Capacity | Description |
|------|--------|---------|----------|-------------|
| G1 | GPU HBM | ~ns | Smallest | Active KV cache used by attention kernels |
| G2 | Pinned DRAM | ~us | Medium | Staging area for RDMA transfers and tier promotion |
| G3 | NVMe/SSD | ~ms | Large | Persistent warm-block storage |
| G4 | S3/MinIO | ~100ms | Unlimited | Cold/archival object storage |

## Architecture

```text
                    +-----------------+
                    | InstanceLeader  |
                    |  (find_matches, |
                    |   BlockAccessor)|
                    +--------+--------+
                             |
               +-------------+-------------+
               |                           |
      +--------v--------+        +--------v--------+
      | CoordinatedWorker|       | CoordinatedWorker|
      |   (rank 0)       |       |   (rank 1)       |
      +--------+---------+       +--------+---------+
               |                           |
      +--------v--------+        +--------v--------+
      | PhysicalWorker   |       | PhysicalWorker   |
      | (TransferManager)|       | (TransferManager)|
      +-----------------+        +-----------------+
```

The leader drives workers through the `ParallelWorkers` trait (`SpmdParallelWorkers`
for SPMD execution). For onboarding, the leader creates sessions that progress through
stages: search, hold, prepare (G3->G2), and pull (remote G2->local G2 via RDMA).

## Modules

| Module | Purpose |
|--------|---------|
| `leader` | Block coordination: matching, onboarding sessions, policy-based scanning |
| `worker` | Transfer execution: local, RDMA, and object storage data movement |
| `object` | G4 storage: S3/MinIO client for cold-tier block persistence |
| `offload` | Tier demotion pipeline: batched G2->G3 and G2->G4 offloading |
| `runtime` | Shared infrastructure: `KvbmRuntime`, tokio handle, NIXL agent |
| `pubsub` | Event pub/sub: block-level notifications for cross-instance coordination |
| `collectives` | NCCL collectives for multi-GPU synchronization (feature-gated) |
| `testing` | Test utilities: mock workers, in-memory block managers (feature-gated) |

## Feature Flags

| Flag | Dependencies | Description |
|------|-------------|-------------|
| `default` | `["s3"]` | Default features |
| `s3` | `aws-sdk-s3`, `aws-config`, `rayon`, `tokio-rayon`, `chrono` | S3/MinIO object storage support |
| `collectives` | `nixl-sys`, `nccl` | NIXL + NCCL multi-GPU collectives |
| `nccl` | `cudarc` | NCCL support via cudarc |
| `testing-nccl` | `collectives` | Enable collectives for tests |
| `nats` | `async-nats`, `flume` | NATS-based pub/sub transport |
| `testing` | `kvbm-logical/testing`, `kvbm-physical/testing` | Test utilities and mock infrastructure |
| `nvtx` | `kvbm-config/nvtx` | NVIDIA Tools Extension profiling markers |

## Quick Start

```rust,ignore
use kvbm_engine::{KvbmRuntime, leader::InstanceLeader};

// Build runtime from environment
let runtime = KvbmRuntime::from_env_leader().await?;

// Create a leader instance
let leader = InstanceLeader::new(/* ... */);

// Search for cached blocks
let result = leader.find_matches(&sequence_hashes)?;
```
