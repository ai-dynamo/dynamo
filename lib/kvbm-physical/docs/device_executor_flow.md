# Device Executor Flow — `transfer/executor/device.rs`

Step-by-step flow for the transfer executor that dispatches through the
`DeviceContextOps` / `DeviceStreamOps` / `DeviceMemPoolOps` trait surface
(defined in [`src/device/traits.rs`](../src/device/traits.rs)).

This document focuses on **what `device.rs` does at runtime**. For the
architecture behind the traits and how XPU/SYCL fits in alongside CUDA,
see [`kvbm_v2_xpu_sycl_enablement.md`](./kvbm_v2_xpu_sycl_enablement.md).

The executor has three functions:

| Function | Purpose |
|---|---|
| `execute_device_transfer` | Top-level dispatch: strategy + stream-pool selection + sync semantics |
| `execute_whole_block_device` | FC→FC fast path: one `batch_copy` of `bytes_per_block` per block |
| `execute_fc_lw_vectorized` | FC↔LW slow path: pool-backed pointer upload + kernel launch |

---

## 1. `execute_device_transfer` — top-level dispatch

Entered from `transfer::executor::execute_direct_transfer` for any
`TransferStrategy` that is not `Memcpy` or NIXL.

```mermaid
flowchart TD
    START([execute_device_transfer]) --> V1[Validate layer count<br/>+ outer dim match]
    V1 --> V2[validate_layout_compatibility]
    V2 --> LR[Resolve layer_range<br/>default = 0..num_layers]
    LR --> WB[whole_block = can_use_whole_block_transfer]

    LR --> DIR{strategy == AsyncD2H<br/>or BlockingD2H?}
    DIR --> IS_D2H[is_d2h = bool]
    WB --> CMS0{caller-provided<br/>device_stream?}
    IS_D2H --> CMS0

    CMS0 -- yes --> USE_CALLER[Use caller stream<br/>caller_manages_sync = true]
    CMS0 -- no + is_d2h --> POOL_D2H[ctx.next_d2h_stream]
    CMS0 -- no + !is_d2h --> POOL_H2D[ctx.next_h2d_stream]

    USE_CALLER --> DISPATCH
    POOL_D2H --> DISPATCH
    POOL_H2D --> DISPATCH

    DISPATCH{whole_block?}
    DISPATCH -- yes --> FN_WB[execute_whole_block_device]
    DISPATCH -- no --> FN_VEC[execute_fc_lw_vectorized]

    FN_WB --> BLK{Blocking strategy?<br/>BlockingH2D or BlockingD2H}
    FN_VEC --> BLK

    BLK -- yes --> SYNC[device_stream.synchronize]
    BLK -- no --> CMS1{caller_manages_sync?}

    SYNC --> DONE_BLK([TransferCompleteNotification::completed])

    CMS1 -- yes --> DONE_CMS([TransferCompleteNotification::completed])
    CMS1 -- no --> EVT[device_stream.record_event]
    EVT --> REG([ctx.register_device_event<br/>→ TransferCompleteNotification])

    style FN_WB fill:#4caf50,stroke:#2e7d32,color:#fff
    style FN_VEC fill:#ff6b35,stroke:#c44520,color:#fff
    style POOL_H2D fill:#90caf9,stroke:#1565c0
    style POOL_D2H fill:#90caf9,stroke:#1565c0
```

### Stream pool selection

`TransferContext` maintains **two** stream pools per device, each
`num_streams` wide, with round-robin acquisition:

| Pool | Acquired via | Used for |
|---|---|---|
| H2D | `ctx.next_h2d_stream()` | Host→device (whole-block batch DMA *and* FC↔LW kernel launches) |
| D2H | `ctx.next_d2h_stream()` | Device→host (whole-block *and* kernel) |

This matches the upstream CUDA design. An earlier iteration maintained
four pools (`copy_{h2d,d2h}` + `compute_{h2d,d2h}`) to target the
Level-Zero BCS/CCS engine split, but that path is gone: both CUDA and
SYCL queues today are engine-class agnostic, so whole-block `batch_copy`
and kernel `vectorized_copy` share the direction pool without losing
concurrency. D2D maps to the H2D pool (`is_d2h = false`); the actual
copy direction is determined by pointer addresses, not the pool label.

On SYCL the pool is backed by round-robin `sycl::queue` slots inside a
single `sycl::context` so USM pointers remain valid across queues —
see `SyclContextCache` in `src/device/sycl/mod.rs`. `KVBM_SYCL_STREAM_POOL_SIZE`
controls the width.

### Sync semantics

1. If the caller provides a `device_stream`, the executor uses it and
   returns `completed()` immediately (the caller owns synchronization).
2. If the strategy is `BlockingH2D` / `BlockingD2H`, the executor calls
   `device_stream.synchronize()` and returns `completed()`.
3. Otherwise (`AsyncH2D` / `AsyncD2H` / `AsyncD2D`), the executor records
   a `DeviceEvent` on the stream and hands it to
   `ctx.register_device_event`, which returns a
   `TransferCompleteNotification` that completes when the event fires.

---

## 2. `execute_whole_block_device` — FC→FC fast path

One pointer pair per block, one DMA per pair, direction auto-detected by
the runtime from the pointer addresses.

```mermaid
flowchart TD
    START([execute_whole_block_device]) --> CHK{num_blocks == 0?}
    CHK -- yes --> DONE([return Ok])
    CHK -- no --> BUILD["Build two host Vec&lt;u64&gt;:<br/>src_ptrs[i] = addr(src_block_ids[i], 0, 0)<br/>dst_ptrs[i] = addr(dst_block_ids[i], 0, 0)"]
    BUILD --> COPY["stream.batch_copy(&src_ptrs, &dst_ptrs, bytes_per_block)"]
    COPY --> LOG[tracing::debug num_blocks + bytes_per_block]
    LOG --> DONE2([return Ok])

    style COPY fill:#4caf50,stroke:#2e7d32,color:#fff
```

`bytes_per_block` comes from `src.layout().config().bytes_per_block()`.
The source and destination layouts are both fully contiguous here, so
one pointer per block is sufficient.

`stream.batch_copy` is the `DeviceStreamOps::batch_copy` method: on CUDA
it calls `kvbm_kernels::memcpy_batch` (using
`cudaMemcpyBatchAsync` when available, falling back to a per-pair
`cudaMemcpyAsync` loop); on SYCL it submits `num_pairs`
`queue.memcpy_raw_async` calls, which the SYCL runtime auto-detects as
H2D / D2H / D2D from the pointer addresses.

---

## 3. `execute_fc_lw_vectorized` — FC↔LW kernel path

One pointer pair per (block, layer, outer) chunk. Pointer arrays are
uploaded to device memory from a `DeviceMemPool`, then a GPU kernel
reads them and performs `total_chunks` parallel copies of `chunk_size`
bytes each.

```mermaid
flowchart TD
    START([execute_fc_lw_vectorized]) --> CALC["Compute:<br/>chunk_size = page × inner × dtype_width<br/>total_chunks = num_blocks × num_layers × outer_dim"]
    CALC --> CHK{total_chunks == 0?}
    CHK -- yes --> DONE([return Ok])
    CHK -- no --> S1["<b>Step 1</b>: build host Vec of u64<br/>src_ptrs, dst_ptrs — one per (block, layer, outer)<br/>with size-match assert per chunk"]

    S1 --> S2["<b>Step 2</b>: pool.alloc_async × 2<br/>src_ptrs_device, dst_ptrs_device<br/>(stream-ordered on CUDA, software pool on SYCL)"]

    S2 --> S3["<b>Step 3</b>: stream.memcpy_htod × 2<br/>upload pointer arrays H→D (async, stream-ordered)"]

    S3 --> S4["<b>Step 4</b>: stream.record_event<br/>upload_event — fence after uploads only"]

    S4 --> S5["<b>Step 5</b>: stream.vectorized_copy<br/>(src_ptrs_device, dst_ptrs_device, chunk_size, total_chunks)<br/>GPU kernel launch"]

    S5 --> S6["<b>Step 6</b>: pool.free_async × 2<br/>stream-ordered free (CUDA) or event-deferred (SYCL)"]

    S6 --> S7["<b>Step 7</b>: upload_event.synchronize<br/>wait H2D only — safe to drop host Vecs<br/>(does NOT wait for the kernel)"]

    S7 --> LOG[tracing::debug total_chunks + chunk_size]
    LOG --> DONE2([return Ok])

    style S1 fill:#e0e0e0,stroke:#757575
    style S2 fill:#64b5f6,stroke:#1565c0,color:#fff
    style S3 fill:#ff8c5a,stroke:#c44520,color:#fff
    style S4 fill:#fff176,stroke:#f9a825
    style S5 fill:#ff6b35,stroke:#c44520,color:#fff
    style S6 fill:#64b5f6,stroke:#1565c0,color:#fff
    style S7 fill:#fff176,stroke:#f9a825
```

### Why record an event before the kernel

Step 4 captures a fence immediately after the two `memcpy_htod` uploads.
Step 7 waits on that event, which blocks only until the H2D copies are
done — *not* until the kernel finishes. This lets the host drop the
source `Vec<u64>` arrays as soon as the uploads complete, while the
kernel keeps running asynchronously. The in-order stream (CUDA) or
in-order `sycl::queue` (SYCL) guarantees the kernel still observes the
uploads.

### Backend specifics

| Step | CUDA | SYCL |
|---|---|---|
| 2. `pool.alloc_async` | `CudaMemPool::alloc_async_raw` using `cuMemAllocFromPoolAsync` on the raw `CUstream` | `SyclMemPoolWrapper::alloc_async` draws from a software free-list over `sycl::malloc_device`; tracks ptr→size in `active_allocs` |
| 3. `stream.memcpy_htod` | `cudarc::driver::result::memcpy_htod_async` | `queue.memcpy_raw_async` |
| 4. `stream.record_event` | `stream.record_event(None)` | `queue.submit_barrier()` wrapped in `SyclEventWrapper` |
| 5. `stream.vectorized_copy` | `kvbm_kernels::vectorized_copy` (`.cu` compiled by nvcc) | `kvbm_kernels::sycl_vectorized_copy` (`.cpp` compiled by `icpx -fsycl`) with the `sycl::queue*` handle |
| 6. `pool.free_async` | `cuMemFreeAsync` stream-ordered | `SyclMemPoolWrapper` records the event and defers the return-to-pool until the event signals (`PendingFree` queue) |
| 7. `event.synchronize` | `cuEventSynchronize` | `sycl::event::wait` |

---

## Relationship to the rest of the executor

```mermaid
flowchart LR
    TM[TransferManager::execute_transfer] --> EXEC[execute_transfer<br/>in executor/mod.rs]
    EXEC -->|Memcpy strategy| MEMCPY[memcpy executor]
    EXEC -->|NixlRead / NixlWrite| NIXL[nixl executor]
    EXEC -->|Async/Blocking H2D/D2H/D2D| DEV[execute_device_transfer<br/>this document]
    DEV --> WB[execute_whole_block_device]
    DEV --> VEC[execute_fc_lw_vectorized]

    style DEV fill:#ff6b35,stroke:#c44520,color:#fff
    style WB fill:#4caf50,stroke:#2e7d32,color:#fff
    style VEC fill:#ff6b35,stroke:#c44520,color:#fff
```

- `memcpy` handles host↔host `Memcpy` (no device involved).
- `nixl` handles any strategy whose name starts with `Nixl*` (RDMA / GDS).
- `device.rs` handles everything device-local:
  `AsyncH2D`, `AsyncD2H`, `AsyncD2D`, `BlockingH2D`, `BlockingD2H`.

Two-hop transfers (e.g. Device → Pinned → Remote) land back in
`execute_direct_transfer` once per hop, so each hop still flows through
this executor independently.

---

## `TransferStrategy` vs. upstream — rename and additions

The `TransferStrategy` enum consumed by this executor differs from
upstream [`ai-dynamo/dynamo`](https://github.com/ai-dynamo/dynamo) `main`
in two distinct ways:

| Change | Upstream main | Branch | Rationale |
|---|---|---|---|
| **Rename** (no behavior change) | `CudaAsyncH2D`, `CudaAsyncD2H`, `CudaAsyncD2D` | `AsyncH2D`, `AsyncD2H`, `AsyncD2D` | These variants are dispatched by the backend-agnostic `device.rs`, which routes them to `CudaStreamWrapper` or `SyclStreamWrapper` through `DeviceStreamOps`. A `Cuda` prefix is misleading on the SYCL path, so the prefix is dropped. Behavior is identical — the rename is a pure find-and-replace. |
| **New variants** | — (panics on `System ↔ Device`) | `BlockingH2D`, `BlockingD2H` | Upstream `strategy.rs` returns `panic!("System to Device transfers are not supported")` for `System ↔ Device(_)`. The branch replaces that panic with the new `Blocking*` strategies, which run an async copy on the stream and then call `device_stream.synchronize()` before returning. Applies on **both** backends — the motivating case (unpinned `System` memory degrading async copies to staged blocking behavior) affects CUDA and SYCL identically, so hiding the variants behind `#[cfg]` would leave the upstream panic in place for CUDA consumers with no benefit. |

Semantics summary:

- `Async*` — enqueues on a stream, records a `DeviceEvent`, returns a
  notification. Caller awaits the notification. Applies to pinned host
  memory or device memory.
- `Blocking*` — enqueues on a stream, calls `device_stream.synchronize()`
  inline, returns `TransferCompleteNotification::completed()`. Applies
  to unpinned (`System`) host memory where the async copy would stage
  internally and gain nothing from the notification path.
- Neither form is backend-specific; both CUDA and SYCL stream wrappers
  implement the same `DeviceStreamOps` contract.

Nothing was removed from the upstream enum. The only removed *behavior*
is the `panic!` on `System ↔ Device`, which is now a supported transfer.
