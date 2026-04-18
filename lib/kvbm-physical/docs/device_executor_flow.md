# Device Executor Flow — `device.rs`

Flowcharts for the three functions in
`lib/kvbm-physical/src/transfer/executor/device.rs`,
with a side-by-side comparison to the original CUDA executor
(`cuda.rs`).

---

## 1. `execute_device_transfer` (top-level dispatch)

```mermaid
flowchart TD
    START([execute_device_transfer]) --> V1[Validate layer/outer dim match]
    V1 --> V2[validate_layout_compatibility]
    V2 --> LR[Resolve layer_range<br/>default = 0..num_layers]
    LR --> WB{can_use_whole_block_transfer?<br/>both FC, full layers}

    WB -- yes --> ENG_WB[Select Copy engine stream<br/>EngineHint::Copy → BCS/DMA]
    WB -- no --> ENG_VEC[Select Compute engine stream<br/>EngineHint::Compute → CCS/kernel]

    ENG_WB --> CS{caller provided<br/>stream?}
    ENG_VEC --> CS

    CS -- yes --> USE_CALLER[Use caller stream]
    CS -- no+WB --> POOL_COPY[ctx.next_copy_h2d/d2h_stream]
    CS -- no+vec --> POOL_COMP[ctx.next_compute_h2d/d2h_stream]

    USE_CALLER --> DISPATCH
    POOL_COPY --> DISPATCH
    POOL_COMP --> DISPATCH

    DISPATCH{whole_block?}
    DISPATCH -- yes --> FN_WB[execute_whole_block_device]
    DISPATCH -- no --> FN_VEC[execute_fc_lw_vectorized]

    FN_WB --> BLK{Blocking strategy?}
    FN_VEC --> BLK

    BLK -- yes --> SYNC[stream.synchronize]
    BLK -- no --> CMS{caller_manages_sync?}

    SYNC --> DONE_BLK([return completed])

    CMS -- yes --> DONE_CMS([return completed])
    CMS -- no --> EVT[stream.record_event]
    EVT --> REG([ctx.register_device_event → notification])

    style FN_WB fill:#4caf50,stroke:#2e7d32,color:#fff
    style FN_VEC fill:#ff6b35,stroke:#c44520,color:#fff
    style ENG_WB fill:#90caf9,stroke:#1565c0
    style ENG_VEC fill:#ffb088,stroke:#c44520
```

---

## 2. `execute_whole_block_device` (FC→FC batch DMA)

```mermaid
flowchart TD
    START([execute_whole_block_device]) --> CHK{num_blocks == 0?}
    CHK -- yes --> DONE([return Ok])
    CHK -- no --> BUILD[Build host Vec of u64 ptrs:<br/>1 src + 1 dst per block<br/>addr = memory_region block,0,0]
    BUILD --> COPY[stream.batch_copy<br/>src_ptrs, dst_ptrs, bytes_per_block]
    COPY --> LOG[tracing::debug]
    LOG --> DONE2([return Ok])

    style COPY fill:#4caf50,stroke:#2e7d32,color:#fff
```

---

## 3. `execute_fc_lw_vectorized` (pool-based GPU kernel)

```mermaid
flowchart TD
    START([execute_fc_lw_vectorized]) --> CALC[Calculate:<br/>chunk_size = page×inner×dtype<br/>total_chunks = blocks×layers×outer]
    CALC --> CHK{total_chunks == 0?}
    CHK -- yes --> DONE([return Ok])
    CHK -- no --> S1["<b>Step 1:</b> Build host Vec&lt;u64&gt;<br/>1 ptr per (block, layer, outer)"]

    S1 --> S2["<b>Step 2:</b> pool.alloc_async × 2<br/>src_ptrs_device, dst_ptrs_device"]

    S2 --> S3["<b>Step 3:</b> stream.memcpy_htod × 2<br/>upload pointer arrays H→D"]

    S3 --> S4["<b>Step 4:</b> stream.record_event<br/>upload_event (H2D fence)"]

    S4 --> S5["<b>Step 5:</b> stream.vectorized_copy<br/>(src_dev, dst_dev, chunk_size, count)<br/>GPU kernel launch"]

    S5 --> S6["<b>Step 6:</b> pool.free_async × 2<br/>stream-ordered free"]

    S6 --> S7["<b>Step 7:</b> upload_event.synchronize<br/>wait H2D only → safe to drop host Vecs"]

    S7 --> LOG[tracing::debug]
    LOG --> DONE2([return Ok])

    style S1 fill:#e0e0e0,stroke:#757575
    style S2 fill:#64b5f6,stroke:#1565c0,color:#fff
    style S3 fill:#ff8c5a,stroke:#c44520,color:#fff
    style S4 fill:#fff176,stroke:#f9a825
    style S5 fill:#ff6b35,stroke:#c44520,color:#fff
    style S6 fill:#64b5f6,stroke:#1565c0,color:#fff
    style S7 fill:#fff176,stroke:#f9a825
```

---

## 4. CUDA `cuda.rs` vs Device `device.rs` — Comparison

### 4.1 `execute_cuda_transfer` vs `execute_device_transfer`

| Aspect | cuda.rs | device.rs | Match? |
|--------|---------|-----------|--------|
| **Validation** | layer count + outer dim + `validate_layout_compatibility` | Same three checks, same order | ✅ Identical |
| **Layer range** | `layer_range.unwrap_or(0..num_layers)` | Same | ✅ |
| **Whole-block check** | `can_use_whole_block_transfer(src, dst, layer_range.as_ref())` | `can_use_whole_block_transfer(src, dst, Some(&layers))` | ✅ Equivalent |
| **Stream selection** | caller → `ctx.next_d2h_streams()` or `next_h2d_streams()` (1 pool, direction only) | caller → engine×direction: `next_copy_h2d`, `next_copy_d2h`, `next_compute_h2d`, `next_compute_d2h` | ✅ Superset — device.rs separates Copy vs Compute engines (CUDA ignores EngineHint, ZE uses it) |
| **caller_manages_sync** | Returns `completed()` if caller provided stream | Same | ✅ |
| **Blocking sync** | Not present — CUDA only has Async strategies | `BlockingH2D` / `BlockingD2H` → `stream.synchronize()` | ✅ Superset — adds blocking support |
| **Event recording** | `stream.record_event(None)` → `ctx.register_cuda_event` | `stream.record_event()` → `ctx.register_device_event` | ✅ Equivalent (backend-agnostic) |
| **Strategy enum** | `CudaAsyncH2D`, `CudaAsyncD2H`, `CudaAsyncD2D` | `AsyncH2D`, `AsyncD2H`, `AsyncD2D`, `BlockingH2D`, `BlockingD2H` | ✅ Superset |

### 4.2 `execute_whole_block_cuda` vs `execute_whole_block_device`

| Aspect | cuda.rs | device.rs | Match? |
|--------|---------|-----------|--------|
| **Empty check** | `num_blocks == 0 → return Ok` | Same | ✅ |
| **Pointer build** | `*const c_void` / `*mut c_void` per block at `(block, 0, 0)` | `u64` per block at `(block, 0, 0)` | ✅ Same addresses, different types |
| **Copy call** | `kvbm_kernels::memcpy_batch(…, BatchedWithFallback, stream)` | `stream.batch_copy(&src, &dst, bytes_per_block)` | ✅ `batch_copy` impl calls memcpy_batch on CUDA, zeCommandListAppendMemoryCopy on ZE |
| **Direction** | `cudaMemcpyDefault` (auto) | Auto-detected per backend | ✅ |
| **Error handling** | Check `cudaError` status | `Result<()>` propagated from trait | ✅ |

### 4.3 `execute_fc_lw_vectorized` (CUDA) vs `execute_fc_lw_vectorized` (device)

| Step | cuda.rs | device.rs | Match? |
|------|---------|-----------|--------|
| **0. Context bind** | `stream.context().bind_to_thread()` | Not in device.rs — done inside `ZeContext` / `CudaContext` impl | ✅ Moved to trait impl |
| **1. Build ptrs** | `Vec<usize>`, triple loop: block×layer×outer | `Vec<u64>`, same triple loop, same order | ✅ Equivalent (`usize` = `u64` on 64-bit) |
| **1b. Size validation** | Not present | Checks `src_region.size() != dst_region.size()` | ✅ Superset — extra safety |
| **2. Alloc pool** | `pool.alloc_async(…, stream)` × 2 | `pool.alloc_async(…, stream)` × 2 | ✅ |
| **3. Upload H2D** | `cuda_result::memcpy_htod_async` × 2 | `stream.memcpy_htod` × 2 | ✅ Equivalent (memcpy_htod wraps same call) |
| **4. Record event** | `stream.record_event(None)` | `stream.record_event()` | ✅ |
| **5. Kernel launch** | `kvbm_kernels::vectorized_copy(src_dev, dst_dev, chunk_size, count, stream)` | `stream.vectorized_copy(src_dev, dst_dev, chunk_size, count)` | ✅ Stream carries backend dispatch |
| **6. Free pool** | `pool.free_async` × 2 | `pool.free_async` × 2 | ✅ |
| **7. Sync uploads** | `pointers_transfered_event.synchronize()` | `upload_event.synchronize()` | ✅ Same semantics |
| **Order of 6↔7** | free_async → then sync (step 6 before step 7) ⚠️ Actually: free first, sync last | Same: free_async before upload_event.synchronize | ✅ Same order |

### 4.4 Verdict

> **`device.rs` is a faithful, backend-agnostic reimplementation of `cuda.rs`.**
>
> All CUDA semantics are preserved 1:1. The differences are strict supersets:
>
> - **EngineHint** — Copy vs Compute stream selection (CUDA ignores it; ZE uses BCS/CCS)
> - **Blocking strategies** — `BlockingH2D` / `BlockingD2H` with `stream.synchronize()`
> - **Size validation** — extra safety check in vectorized path
> - **Type widening** — `usize` → `u64` and `*const c_void` → `u64` (equivalent on 64-bit)
> - **`bind_to_thread`** — moved from call-site into trait impl (cleaner)
>
> No CUDA functionality is lost. The pipeline order (alloc → upload → event →
> kernel → free → sync-upload) is identical.
