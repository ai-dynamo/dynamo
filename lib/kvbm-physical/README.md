# kvbm-physical

Physical layout and transfer management for KV cache block storage.

`kvbm-physical` provides the low-level building blocks for mapping KV cache blocks to memory, registering them for RDMA transfers via NIXL, and executing transfers between heterogeneous storage tiers (GPU, host, disk, remote).

## Modules

### `layout` — Block-to-memory mapping

Abstractions for how KV cache blocks are organized in memory.

- **`Layout` trait** — Core abstraction mapping `(block_id, layer_id, outer_id)` to a `MemoryRegion`. Implementations include fully contiguous (single allocation) and layer-separate (one allocation per layer) variants.
- **`KvBlockLayout`** — Describes dimension ordering within a block. Five named formats (`UniversalTP`, `UniversalPP`, `OperationalHND`, `OperationalNHD`, `Custom`) plus `Unknown`. Provides `requires_transform()`, `is_operational()`, and `is_universal()` for kernel selection.
- **`PhysicalLayout`** — Wraps a `Layout` with its physical storage location (`StorageKind`) and NIXL registration metadata (`NixlMetadata`). Constructed via a type-state builder: Config &rarr; Layout type &rarr; Memory allocation &rarr; `build()`.
- **`LayoutConfig`** — Block dimensions: `num_blocks`, `num_layers`, `outer_dim`, `page_size`, `inner_dim`, `dtype_width_bytes`, optional `num_heads`.
- **`KvBlocks`** — Groups block IDs with a shared `PhysicalLayout` and optional `KvBlockLayout` override for cross-format transfers.

### `manager` — Layout registration and transfer orchestration

- **`TransferManager`** — Primary API. Registers layouts, exports/imports RDMA metadata between workers, and executes transfers by handle.
- **`LayoutHandle`** — Compact `u128` encoding `(worker_id, layout_id)`. Identifies a registered layout within a specific worker; not symmetric across workers.
- **`LogicalLayoutDescriptor`** — Bridges a `LayoutHandle` to a `LogicalLayoutHandle` (G1/G2/G3/G4 tier). Enables callers to say "copy from G1 to G2" while `TransferManager` resolves worker-specific physical handles.
- **`SerializedLayout`** — Wire format for RDMA metadata exchange. Packs worker address, NIXL metadata, and layout descriptors into a bincode blob.
- **`WorkerAddress`** — `(worker_id, nixl_agent_name)` pair identifying a worker on the network.

### `transfer` — Transfer configuration and execution

- **`TransferConfig` / builder** — Configures event system, NIXL backends, CUDA device, capabilities, and memory pool before building a `TransferManager`.
- **`TransferOptions`** — Per-transfer configuration: `layer_range`, `nixl_write_notification`, `bounce_buffer`, caller-provided `cuda_stream`, and src/dst `kv_layout` overrides.
- **`TransferCapabilities`** — Feature flags: `allow_gds` (GPU Direct Storage), `allow_gpu_rdma` (device-to-remote without host staging).
- **`TransferPreferences`** — Strategy hints via `NativeVsNixlPolicy` (PreferNative / PreferNixl / Automatic).
- **`TransferCompleteNotification`** — `Either<Ready, EventAwaiter>` implementing `IntoFuture`. Zero-cost for synchronous completions. `aggregate()` composes multiple notifications. `could_yield()` checks if awaiting will suspend.
- **`BounceBuffer`** — Staging area for two-hop transfers (e.g., Device &rarr; Host &rarr; Remote).
- **Checksum utilities** — BLAKE3 block/layer checksums for transfer verification.
- **Fill utilities** — Constant/sequential patterns for testing and initialization.

## Quick Start

```rust,ignore
use kvbm_physical::{TransferManager, TransferOptions};
use kvbm_physical::layout::{LayoutConfig, PhysicalLayout};

// 1. Build the TransferManager (creates NIXL agent, CUDA streams, event system)
let manager = TransferManager::builder()
    .nixl_backend("ucx")
    .cuda_device_id(0)
    .build()?;

// 2. Configure a layout
let config = LayoutConfig::builder()
    .num_blocks(64)
    .num_layers(32)
    .outer_dim(2)
    .page_size(16)
    .inner_dim(128)
    .dtype_width_bytes(2)
    .build()?;

// 3. Build a physical layout (type-state builder: config -> layout type -> memory -> build)
let gpu_layout = PhysicalLayout::builder(manager.nixl_agent().clone())
    .with_config(config.clone())
    .fully_contiguous()
    .allocate_device(0)
    .build()?;

let host_layout = PhysicalLayout::builder(manager.nixl_agent().clone())
    .with_config(config)
    .fully_contiguous()
    .allocate_pinned(Some(0))
    .build()?;

// 4. Register layouts to get handles
let gpu_handle = manager.register_layout(gpu_layout)?;
let host_handle = manager.register_layout(host_layout)?;

// 5. Execute a transfer and await completion
let notification = manager.execute_transfer(
    gpu_handle,
    &[0, 1, 2, 3],        // source block IDs
    host_handle,
    &[0, 1, 2, 3],        // destination block IDs
    TransferOptions::new(),
)?;
notification.await?;
```

## Testing

All functional tests in `kvbm-physical` require a real NIXL installation and a CUDA GPU. They are gated behind two feature flags:

- **`testing-kvbm`** — enables tests that create `NixlAgent` instances (hard-aborts under NIXL stubs)
- **`testing-kvbm`** — enables tests that allocate device memory or launch CUDA kernels

### Running tests

```bash
# Without GPU/NIXL — only the sentinel test runs (confirms skipping)
cargo test -p kvbm-physical

# With GPU + NIXL available
cargo test -p kvbm-physical --features testing-kvbm,testing-kvbm
```

When neither feature is enabled, a single **sentinel test** runs and prints a reminder message. This ensures `cargo test` never silently passes with zero tests.

### What the sentinel test looks like

```
running 1 test
test sentinel::all_functional_tests_skipped___enable_testing_nixl_and_testing_cuda ... ok
```

The `test_version_check_on_deserialization` test in `layout::tests` is the only functional test that runs without feature flags, as it does not require NIXL or CUDA.

## Documentation

- [v1 Migration Guide](docs/v1_migration.md) — Migration from `dynamo-llm::block_manager` to `kvbm-physical`
