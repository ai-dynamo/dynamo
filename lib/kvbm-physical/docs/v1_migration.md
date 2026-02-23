# Migration Guide: block_manager to kvbm-physical

Guide for migrating from `dynamo-llm::block_manager` (v1) to `kvbm-physical`.

## Overview

`kvbm-physical` is a ground-up rewrite of the physical transfer layer from `lib/llm/src/block_manager/`. The core data flow is the same (register layouts, exchange metadata, execute transfers), but `kvbm-physical` adds block format awareness, richer transfer options, and a cleaner separation between logical tiers and physical handles.

Both implementations use the same `vectorized_copy` CUDA kernel. The original embeds it in a `.fatbin` (`lib/llm/src/block_manager/block/transfer/kernels/vectorized_copy.fatbin`) loaded via `cuModuleLoadData`. `kvbm-physical` wraps the same kernel via the `kvbm-kernels` crate with explicit Rust FFI for transparency and testability.

## Type mapping table

| Original (block_manager) | kvbm-physical | Notes |
|--------------------------|---------------|-------|
| `TransportManager` | `TransferManager` | Same role, richer API |
| `LayoutHandle` | `LayoutHandle` | Same concept, same encoding |
| `PhysicalLayout` + builder | `PhysicalLayout` + builder | Same pattern; adds `with_external_device_regions()` |
| `LayoutConfig` | `LayoutConfig` | Same fields + optional `num_heads` |
| `TransferOptions` | `TransferOptions` | Adds `cuda_stream`, `src_kv_layout`, `dst_kv_layout` |
| `TransferCapabilities` | `TransferCapabilities` | Same |
| `TransferPreferences` | `TransferPreferences` | Same |
| `SerializedLayout` | `SerializedLayout` | Same wire format concept |
| `WorkerAddress` | `WorkerAddress` | Same |
| `TransferCompleteNotification` (oneshot) | `TransferCompleteNotification` (`Either`/`EventAwaiter`) | Zero-cost sync path |
| `BounceBufferSpec` (trait object) | `BounceBuffer` (concrete struct) | Simpler, no heap allocation |
| N/A | `LogicalLayoutDescriptor` | **New** — tier bridging |
| N/A | `KvBlockLayout` | **New** — block format awareness |
| N/A | `KvBlocks` | **New** — grouped blocks with layout override |
| `CudaBlockingH2D` / `CudaBlockingD2H` | Removed | Async-only; `.await` for sync behavior |
| `OperationalCopyBackend` | Removed | Replaced by `kvbm_kernels` direct FFI |

## What kvbm-physical adds

### LogicalLayoutDescriptor

Bridges `LayoutHandle` (physical) to `LogicalLayoutHandle` (G1/G2/G3/G4 tier). This is the key new abstraction for multi-worker coordination: callers say "copy from G1 to G2" while `TransferManager` resolves worker-specific handles.

```rust,ignore
// Build descriptor for RDMA exchange
let descriptor = manager.build_logical_descriptor(gpu_handle, LogicalLayoutHandle::G1)?;
```

### KvBlockLayout

Five named block formats plus `Custom` and `Unknown`. Enables type-driven kernel selection for transfers between different dimension orderings.

```rust,ignore
let needs_permute = src_layout.requires_transform(&dst_layout);
```

### kvbm-kernels FFI

The `kvbm_kernels` crate provides `memcpy_batch` using CUDA 12.9+ batch API with automatic fallback to individual copies. This replaces the fatbin-loading approach with direct Rust FFI.

### Stream pooling

4 H2D + 4 D2H streams with round-robin selection, replacing the original 1+1 stream pair. Reduces contention for concurrent transfers.

### Caller-provided CUDA stream

`TransferOptions::cuda_stream` lets the caller pass in a stream. The executor skips event recording; the caller manages synchronization. Useful for layer-wise transfers where all layers must execute on the same stream.

```rust,ignore
let stream = manager.context().acquire_h2d_stream();
let options = TransferOptions::builder()
    .cuda_stream(stream.clone())
    .build()?;
```

### CudaMemPool

Device memory pool for kernel temporary allocations (permute buffers, etc.). Configured via `TransferConfig`:

```rust,ignore
TransferManager::builder()
    .cuda_pool_reserve_size(64 * 1024 * 1024)         // 64 MiB pre-allocated
    .cuda_pool_release_threshold(Some(64 * 1024 * 1024)) // free above this
    .build()?;
```

### TransferCompleteNotification::aggregate()

Compose multiple transfer notifications into one that completes when all are done. Optimizes away the aggregation when all inputs are already complete.

```rust,ignore
let combined = TransferCompleteNotification::aggregate(
    vec![n1, n2, n3],
    manager.context().event_system(),
    &tokio::runtime::Handle::current(),
)?;
combined.await?;
```

### src/dst kv_layout overrides

`TransferOptions` now supports overriding the source and destination block layout interpretation, enabling cross-format transfers without modifying the registered layout.

```rust,ignore
let options = TransferOptions::builder()
    .src_kv_layout(KvBlockLayout::OperationalNHD)
    .dst_kv_layout(KvBlockLayout::UniversalTP)
    .build()?;
```

## What was intentionally removed

### Blocking CUDA strategies

`CudaBlockingH2D` and `CudaBlockingD2H` are removed. All transfers are async. For synchronous behavior, just `.await` immediately:

```rust,ignore
// v1 (blocking)
let result = blocking_h2d_transfer(...);

// kvbm-physical (async, but can be used synchronously)
let notification = manager.execute_transfer(...)?;
notification.await?;
```

### OperationalCopyBackend enum

The `OperationalCopyBackend` enum (which selected between different kernel loading strategies) is removed. `kvbm-physical` uses `kvbm_kernels` direct FFI exclusively, making kernel dispatch transparent.

### Trait object bounce buffer

`BounceBufferSpec` (a trait object requiring heap allocation) is replaced by `BounceBuffer`, a concrete struct wrapping a `LayoutHandle` + block IDs:

```rust,ignore
// v1
struct MyBounce { layout: PhysicalLayout, blocks: Vec<BlockId> }
impl BounceBufferSpec for MyBounce { ... }

// kvbm-physical
let bounce = BounceBuffer::from_handle(host_handle, vec![0, 1, 2, 3]);
```

## Migration steps

### 1. Replace TransportManager with TransferManager

The builder pattern is the same. `TransferManager::builder()` returns the same kind of fluent builder.

```rust,ignore
// v1
let manager = TransportManager::builder()
    .worker_id(0)
    .nixl_backend("ucx")
    .cuda_device_id(0)
    .build()?;

// kvbm-physical
let manager = TransferManager::builder()
    .nixl_backend("ucx")
    .cuda_device_id(0)
    .build()?;
// worker_id is now derived from the event system
```

### 2. Replace TransferOptions

Add new fields as needed. Existing `layer_range` and `nixl_write_notification` work the same way.

```rust,ignore
// v1
let options = TransferOptions::builder()
    .layer_range(0..16)
    .build()?;

// kvbm-physical (same, with optional new fields)
let options = TransferOptions::builder()
    .layer_range(0..16)
    .cuda_stream(stream)        // new: caller-managed stream
    .src_kv_layout(layout)      // new: format override
    .build()?;
```

### 3. Replace BounceBufferSpec with BounceBuffer

```rust,ignore
// v1 — trait object
let spec: Box<dyn BounceBufferSpec> = Box::new(MyBounce::new(layout, blocks));
options.bounce_buffer(spec);

// kvbm-physical — concrete type
let bounce = BounceBuffer::from_handle(host_handle, block_ids);
let options = TransferOptions::builder()
    .bounce_buffer(bounce)
    .build()?;
```

### 4. Replace TransferCompleteNotification await pattern

The notification now implements `IntoFuture` directly instead of wrapping a oneshot channel.

```rust,ignore
// v1
let notification = manager.execute_transfer(...)?;
notification.recv().await??;

// kvbm-physical
let notification = manager.execute_transfer(...)?;
notification.await?;
```

### 5. Add LogicalLayoutDescriptor for multi-worker tier resolution

If you coordinate transfers across multiple workers by tier name (G1, G2, etc.), use `LogicalLayoutDescriptor`:

```rust,ignore
// Build descriptors that include tier information
let g1_desc = manager.build_logical_descriptor(gpu_handle, LogicalLayoutHandle::G1)?;
let g2_desc = manager.build_logical_descriptor(host_handle, LogicalLayoutHandle::G2)?;

// Remote workers can now resolve "copy G1 to G2" to the correct physical handles
```

### 6. Consider KvBlockLayout annotations for cross-format transfers

If your transfers involve blocks stored in different dimension orderings (e.g., operational NHD from the engine vs. universal TP for storage), annotate with `KvBlockLayout`:

```rust,ignore
let options = TransferOptions::builder()
    .src_kv_layout(KvBlockLayout::OperationalNHD)
    .dst_kv_layout(KvBlockLayout::UniversalTP)
    .build()?;
```

This tells the executor to select a permute kernel instead of a direct copy.
