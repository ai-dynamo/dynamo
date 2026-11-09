---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Mooncake Store Integration
subtitle: Use Mooncake Store as the KVBM G4 (object storage) tier
---

[Mooncake Store](https://github.com/kvcache-ai/Mooncake) is a distributed KV cache
store that can back the KVBM **G4 tier** — the tier below host memory (G2) and local
SSD (G3). KVBM writes offloaded blocks into Mooncake and onboards them back when a
request reuses that prefix, so a prefix cached by one worker is reusable by any worker
connected to the same Mooncake cluster.

Mooncake is a good fit for G4 when you want lower latency than S3-compatible object
storage and have an RDMA-capable fabric (RoCE or InfiniBand) between the workers and
the Mooncake segment holders.

## Architecture

KVBM talks to Mooncake through two traits:

| Trait | Mooncake implementation | Purpose |
|-------|------------------------|---------|
| `ObjectBlockOps` | `MooncakeObjectBlockClient` | Block-level `has` / `put` / `get` |
| `ObjectLockManager` | `MooncakeLockManager` | Distributed locking for coordinated offloads |

`MooncakeLockManager` is intentionally a no-op: KVBM block keys are content-derived
(`same sequence hash` ⇒ `same block content`), so concurrent writes of the same block
are idempotent and do not need a distributed lock. It grants every lock request and
ignores releases.

Batch operations call Mooncake's native `batch_put_from` / `batch_get_into` /
`batch_is_exist` so a batch of blocks costs one RPC instead of one RPC per block.

## Transport paths

| Protocol | Path | When used |
|----------|------|-----------|
| TCP | Copy | Default. Blocks are serialized into a staging buffer, sent over TCP. |
| RDMA | Zero-copy | `protocol: rdma` **and** `use_zero_copy` **and** a contiguous (fully-contiguous, `Fc`) layout. |
| RDMA | Copy | Fallback for non-contiguous (`Lw`) layouts, or when `use_zero_copy` is off. |

`use_zero_copy` is a builder flag on `MooncakeObjectBlockClient`, not a config field,
and it defaults to `false`:

```rust
let client = MooncakeObjectBlockClient::new(&mc_config)?
    .with_zero_copy(true);
```

> [!WARNING]
> `create_object_client` constructs the Mooncake client **without** calling
> `with_zero_copy(true)`. A client obtained from that factory takes the copy path even
> when `protocol` is `rdma`. To get zero-copy — which is the whole reason to run RDMA —
> build `MooncakeObjectBlockClient` directly and pass it to the offload engine as
> `Arc<dyn ObjectBlockOps>`, or extend the factory.

The zero-copy path requires registering the source/destination memory with the RDMA
device before the first transfer (`register_layout`). Registrations are cached per
layout and released when the owning layout registration is dropped. A layout whose
region is not contiguous takes the copy path regardless of the `use_zero_copy` setting.

## Prerequisites

- A Mooncake Store build with the store component enabled, plus its Rust bindings.
- Mooncake master reachable over gRPC (default port `50051`).
- A metadata service: Mooncake's built-in HTTP metadata server, or etcd.
- For RDMA zero-copy: RDMA-capable NICs (e.g. RoCE `mlx5_*`), matching drivers, and
  the `nixl` environment Mooncake expects.
- The Mooncake source tree checked out where `kvbm-engine` expects it (see below).

## Build

The `mooncake_store` crate is an optional path dependency of `kvbm-engine`:

```toml
# lib/kvbm-engine/Cargo.toml
mooncake_store = { path = "../../../Mooncake/mooncake-store/rust", optional = true }
```

So the Mooncake checkout must sit next to the Dynamo workspace root:

```bash
git clone https://github.com/kvcache-ai/Mooncake ../../Mooncake
```

Build the Mooncake C++ store first (the Rust crate links against it):

```bash
cd ../../Mooncake
cmake -S . -B build-store -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_STORE=ON \
  -DWITH_STORE_RUST=ON
cmake --build build-store -j
```

Then build KVBM with the `mooncake` feature:

```bash
cargo build -p kvbm-engine --features mooncake
```

For CI, or any environment where the Mooncake checkout lives elsewhere, point the
build at it explicitly:

```bash
export MOONCAKE_BUILD_DIR=/path/to/Mooncake/build-store
export MOONCAKE_STORE_LIB_DIR=$MOONCAKE_BUILD_DIR/mooncake-store/src
export LD_LIBRARY_PATH=$MOONCAKE_BUILD_DIR/mooncake-common:$LD_LIBRARY_PATH
```

> [!NOTE]
> `kvbm-engine`'s `default` feature set is `s3`. Enabling Mooncake does not disable it;
> pass `--no-default-features --features mooncake` to get a Mooncake-only build.

## Run a Mooncake cluster

Start the master. The flag combination below serves both the gRPC control plane and
the HTTP metadata service from one process:

```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080 \
  --rpc_port=50051
```

Then start at least one segment holder — a process that contributes memory to the
distributed pool. KVBM clients can hold segments too, but for a pure offload tier it is
cleaner to give the capacity to dedicated holders so that a client's `get` genuinely
traverses the fabric:

```bash
bench_mooncake --role holder --segment-gib 320
```

## Configuration

Mooncake is configured through the `object` section of the KVBM config:

```json
{
  "object": {
    "client": {
      "type": "mooncake",
      "metadata_server": "http://127.0.0.1:8080/metadata",
      "master_server_addr": "127.0.0.1:50051",
      "local_hostname": "node-0",
      "protocol": "rdma",
      "device_name": "mlx5_1,mlx5_2",
      "global_segment_size": 536870912,
      "local_buffer_size": 134217728,
      "namespace": "kvbm-prod",
      "max_concurrent_requests": 16
    }
  }
}
```

The same values can be supplied as environment variables (prefix
`KVBM_OBJECT_CLIENT_`), or by pointing `KVBM_CONFIG_PATH` at a TOML file:

```bash
export KVBM_OBJECT_CLIENT_TYPE=mooncake
export KVBM_OBJECT_CLIENT_METADATA_SERVER=http://127.0.0.1:8080/metadata
export KVBM_OBJECT_CLIENT_MASTER_SERVER_ADDR=127.0.0.1:50051
export KVBM_OBJECT_CLIENT_LOCAL_HOSTNAME=$(hostname)
export KVBM_OBJECT_CLIENT_PROTOCOL=rdma
export KVBM_OBJECT_CLIENT_DEVICE_NAME=mlx5_1,mlx5_2
export KVBM_OBJECT_CLIENT_NAMESPACE=kvbm-prod
```

Precedence, lowest to highest: built-in defaults → `/opt/dynamo/etc/kvbm.toml` →
the file named by `KVBM_CONFIG_PATH` → `KVBM_*` environment variables → JSON passed
programmatically (used by the vLLM connector's `kv_connector_extra_config`).

### Fields

| Field | Default | Description |
|-------|---------|-------------|
| `type` | — | Must be `mooncake`. |
| `metadata_server` | — | `http://host:port/metadata` for the HTTP metadata server, or `etcd://host:2379`. |
| `master_server_addr` | `127.0.0.1:50051` | Master gRPC address. |
| `local_hostname` | `/proc/sys/kernel/hostname`, else `localhost` | Identity used for RDMA addressing. Must be resolvable by peers. |
| `protocol` | `tcp` | `tcp` or `rdma`. |
| `device_name` | `""` (auto-select) | Comma-separated RDMA devices, e.g. `mlx5_1,mlx5_2`. |
| `global_segment_size` | 512 MiB | Memory this client contributes to the pool. Set `0` for a client that should only read/write remote data. |
| `local_buffer_size` | 128 MiB | Staging buffer for transfers. |
| `namespace` | none | Key prefix, for isolating clusters that share one Mooncake deployment. |
| `max_concurrent_requests` | 16 | Concurrency limit for batch operations. |

> [!WARNING]
> `local_hostname` defaults to reading `/proc/sys/kernel/hostname`, which only exists on
> Linux. On macOS (or in a container where that file is unhelpful) the default silently
> becomes `localhost`, which produces confusing RDMA addressing errors. Set it
> explicitly whenever the hostname is not resolvable from the peers.

## Wiring the G4 tier

The object client is attached at construction time. Two entry points exist.

### Single-process: `OffloadEngine`

```rust
use kvbm_engine::offload::{ObjectPipelineConfig, OffloadEngine};
use kvbm_engine::object::{create_lock_manager, create_object_client};
use kvbm_engine::G2;

let config = kvbm_config::KvbmConfig::from_env()?;
let object = config.object.as_ref().expect("object tier not configured");

let object_ops = create_object_client(object, /* rank */ Some(0)).await?;
let lock_manager = create_lock_manager(object, instance_id).await?;

// For RDMA zero-copy, replace the line above with a directly-built client:
//   let object_ops: Arc<dyn ObjectBlockOps> =
//       Arc::new(MooncakeObjectBlockClient::new(mc_config)?.with_zero_copy(true));
// and call register_layout() on each layout before the first transfer.

let mut g4 = ObjectPipelineConfig::<G2>::default();
g4.lock_manager = Some(lock_manager);

let engine = OffloadEngine::builder(leader.clone())
    .with_registry(registry)
    .with_object_ops(object_ops)
    .with_g2_physical_layout(g2_layout)   // source layout for uploads
    .with_g2_to_g4_pipeline(g4)
    .build()?;
```

`with_g2_physical_layout` is required: the G2→G4 executor reads block data straight out
of the worker's pinned host layout, so it must know that layout. `OffloadEngine::builder`
takes the leader as its first argument, and `build()` is synchronous.

### Distributed: leader + workers

In a multi-worker deployment the leader holds no physical layouts, so it cannot upload
by itself. Enable remote G4 instead — the leader routes G2→G4 work to the workers'
`ObjectBlockOps` over RPC, and each worker uploads from its own local G2:

```rust
let engine = OffloadEngine::builder(leader.clone())
    .with_enable_remote_g4(true)
    .build()?;
```

`with_enable_remote_g4` and `with_g2_to_g4_pipeline` are mutually exclusive — use one
or the other. On the leader, attach the client so that prefix-cache search also consults
G4:

```rust
let leader = InstanceLeader::builder()
    .object_client(object_ops)
    // ...
    .build()?;
```

Without `object_client`, the leader never probes G4 and blocks that exist only in
Mooncake will not be reused.

> [!IMPORTANT]
> At the time of writing, the G4 tier is reachable through the `kvbm-engine` Rust API
> only. The client is not yet constructed from `KVBM_OBJECT_CLIENT_*` by the Python
> connector path, and `LeaderLayoutConfig.object` is not yet populated by the leader.
> Deployments that need G4 today must build the offload engine and leader themselves via
> the builders above. The environment variables and JSON config described in this page
> are the forward-compatible surface; they are consumed by `create_object_client` and
> `create_lock_manager`, which the caller invokes.

## Validate the transport

`bench_mooncake` measures put / has / get for KV blocks whose geometry is read from a
HuggingFace `config.json`, so the numbers are per real model rather than synthetic
square buffers. It drives `MooncakeObjectBlockClient` directly.

```bash
cargo build -p kvbm-engine --release --features bench,mooncake
```

Run one holder and one client. The client is given **no segment of its own** so that
every byte actually crosses the transport instead of landing in a local segment:

```bash
MODEL=/models/Qwen3-30B-A3B

mooncake_master --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 --http_metadata_server_port=8080 \
  --rpc_port=50051 &

bench_mooncake --role holder --segment-gib 320 --protocol rdma --device mlx5_1 &

# Copy path (both contiguous and non-contiguous layouts)
bench_mooncake --model-dir $MODEL --protocol rdma --device mlx5_1 --layouts fc,lw --iters 5

# Zero-copy path (requires a contiguous layout)
bench_mooncake --model-dir $MODEL --protocol rdma --device mlx5_1 --layouts fc --zero-copy --iters 5
```

Each row moves `prefix_tokens / page_size` blocks, so it reads as "time to offload or
onboard an N-token prefix for one TP rank".

### Measured throughput

Qwen3-30B-A3B on 1× H20, 4× 400 Gb/s RoCE NICs (`mlx5_1`–`mlx5_4`), 320 GiB holder
segment, concurrency 16, 5 iterations after 1 warm-up. The model shape is 48 layers ×
4 KV heads × 128 head_dim in bf16 = **96 KiB per token** (TP=1).

Contiguous layout, 64-token pages, at two batch sizes:

| Path | 16K-token prefix (1.5 GiB) put / get | 32K-token prefix (3 GiB) put / get |
|------|--------------------------------------|------------------------------------|
| TCP, copy | 1.94 / 1.89 GB/s | 1.93 / 1.89 GB/s |
| TCP, zero-copy | 2.73 / 2.18 GB/s | 2.73 / 2.18 GB/s |
| RDMA, zero-copy, 1 NIC | 43.20 / 44.61 GB/s | 27.04 / 27.64 GB/s |
| RDMA, zero-copy, 4 NICs | 71.28 / 76.74 GB/s | 72.53 / 77.00 GB/s |

The same data as end-to-end latency for one 32K-token prefix on one TP rank:

| Path | put p50 | get p50 |
|------|---------|---------|
| TCP, copy | 1658 ms | 1703 ms |
| TCP, zero-copy | 1179 ms | 1474 ms |
| RDMA, zero-copy, 1 NIC | 119 ms | 116 ms |
| RDMA, zero-copy, 4 NICs | 45 ms | 42 ms |

Across the full sweep (page sizes 16/32/64, prefixes 1K–32K):

| Path | put GB/s | get GB/s |
|------|----------|----------|
| TCP, copy (Fc + Lw) | 1.41 – 2.06 | 1.61 – 2.37 |
| TCP, zero-copy (Fc) | 2.60 – 3.40 | 2.14 – 3.26 |
| RDMA, zero-copy, 1 NIC (Fc) | 25.39 – 43.20 | 27.64 – 44.61 |
| RDMA, zero-copy, 4 NICs (Fc) | 52.03 – 72.53 | 67.41 – 77.00 |

Reading these numbers:

- **RDMA zero-copy is ~20–35× faster than the TCP copy path** on `get`, and ~26× on
  `put`. This is the difference between a 3 GiB prefix taking 1.7 s to offload and
  taking 45 ms.
- A single 400 Gb/s NIC moves 50 GB/s unidirectionally, so 44.6 GB/s is about **89% of
  single-NIC line rate** — the transport is not the bottleneck at one NIC.
- Four NICs give 77 GB/s, i.e. **1.7× one NIC**, not 4×. That is ~38% of the 200 GB/s
  aggregate, so the multi-NIC path is not scaling linearly; expect to tune queue depth
  or concurrency before assuming four NICs buy four times the bandwidth.
- **Single-NIC RDMA degrades at the 3 GiB batch** (27 GB/s vs 43 GB/s at 1.5 GiB) while
  the 4-NIC case does not. Treat the 32K-token single-NIC row as the outlier and prefer
  multiple NICs, or a smaller batch, for that shape.

> [!NOTE]
> All RDMA figures above are the **zero-copy** path only. The copy path did not produce
> measurements over RDMA in this configuration — `MooncakeStore::get()` allocates an
> unregistered `Vec<u8>` for its destination, so the RDMA read fails with
> `AddressNotRegistered` (see the troubleshooting entry below). The TCP rows are the
> copy-path comparison.

## Troubleshooting

### `AddressNotRegistered` on RDMA `get` / `put`

**Symptom:** transfers fail with `Memory region not registered by any active device(s)`
and `error code is AddressNotRegistered`.

**Cause:** the destination buffer was not registered with the RDMA device. This happens
on the copy path, where the staging buffer is an ordinary heap allocation.

**Solution:** use the zero-copy path with a contiguous layout. Alternatively set
`MC_STORE_MEMCPY=1` in the environment to force Mooncake's memcpy fallback, which
accepts unregistered buffers at the cost of an extra copy.

### `Memory region not registered by any active device(s)` on the first transfer

**Cause:** `register_layout` was never called for that layout. The client does not
register implicitly — `put_blocks_with_layout` / `get_blocks_with_layout` assume the
layout's memory was registered beforehand.

**Solution:** call `client.register_layout(&layout)` once per physical layout before the
first zero-copy transfer, and `unregister_layout(&layout)` when the layout is retired.
`bench_mooncake` shows the pattern (`lib/kvbm-engine/bin/bench_mooncake.rs`).

### `local_hostname` resolution failures

**Symptom:** RDMA setup fails, or the master cannot reach the client, with a hostname
the operator did not set.

**Cause:** the default came from `/proc/sys/kernel/hostname` (or fell back to
`localhost` off-Linux) and does not resolve from the peer.

**Solution:** set `local_hostname` (or `KVBM_OBJECT_CLIENT_LOCAL_HOSTNAME`) to an
address or resolvable name that peers can route to.

### Throughput far below the NIC line rate

Check, in order:

1. Is `--zero-copy` on, and is the layout contiguous? Non-contiguous blocks silently
   take the copy path.
2. Is `device_name` set to the RDMA devices, or left empty for auto-selection? Verify the
   holder log reports the expected HCA count, e.g. `Found 4 HCAs`.
3. Is the client contributing its own `global_segment_size`? A large local segment
   absorbs the writes, so the measurement never touches the fabric.
4. Is the holder on the same NUMA node as the NIC? Mooncake allocates NUMA-segmented
   buffers and expects the NIC NUMA node set to be non-empty.

### Benign shutdown warnings

At process exit Mooncake may log:

```text
Failed to restore fork state for memory region at 0x... (1073741824 bytes),
deregister already succeeded: Invalid argument [22]
```

This is expected when a process that registered a large local buffer exits; the
deregistration already succeeded. It does not affect measurements or correctness.

## See Also

- [KVBM Guide](../components/kvbm/kvbm-guide.md) for cache tier configuration and metrics
- [S3 integration](../../lib/kvbm-engine/docs/object.md) for the S3-compatible G4 backend
- [KVBM Design](../design-docs/kvbm-design.md) for the tiered cache architecture
- [Mooncake Store documentation](https://github.com/kvcache-ai/Mooncake) for cluster sizing and deployment
