# Planner Executor Flow

Documents the planner-driven transfer executor in `kvbm-physical/src/transfer/executor/planner.rs`. Covers entry points, `PlanOutcome` variants, backend dispatch, and the planning pipeline.

For broader KVBM v2 architecture context, see `kvbm_v2_xpu_sycl_enablement.md`.

## Overview

The planner executor (`use_planner = true` path) provides a backend-agnostic entry point for device and NIXL transfers via a single `execute_outcome_device` function that handles all `PlanOutcome` arms across CUDA and SYCL through the device-agnostic `DeviceStream` API.

**Two strategy families are wired through the planner:**
- `TransferStrategy::Async{H2D, D2H, D2D}` — dispatched via `kvbm_kernels::memcpy_batch`
- `TransferStrategy::Nixl{Read, Write, ReadFlipped, WriteFlipped}` — dispatched via NIXL `create_xfer_req` / `post_xfer_req`

**Legacy paths** (other strategies, `use_planner = false`) stay on `execute_direct_transfer`. The planner path uses explicit bail semantics — errors are NOT silently fallen back.

## Entry Points

### execute_planner_device_transfer

Device-family entry point. Validates strategy, plans, resolves stream, then lowers `PlanOutcome` through `execute_outcome_device`.

```rust
pub(crate) fn execute_planner_device_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    device_stream: Option<Arc<DeviceStream>>,
    layer_range: Option<std::ops::Range<usize>>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification>
```

**Bails when:**
- Strategy is not `Async{H2D, D2H, D2D}` — enforced by `validate_device_planner_entry`
- Block-id lists have unequal length — enforced by `validate_planner_block_ids`
- Layout pair requires semantic transform and kernel catalog has no match

### execute_planner_nixl_transfer

NIXL-family entry point. Same validation/planning/lowering stages as device path, then maps `Vec<CopyOp>` to NIXL `XferDescList` instead of `cudaMemcpyAsync`.

```rust
pub(crate) fn execute_planner_nixl_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    bounce_buffer: Option<&BounceBufferInternal>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification>
```

**Bails when:**
- Strategy is not `Nixl{Read, Write, ReadFlipped, WriteFlipped}`
- Block-id lists have unequal length
- Layout pair requires kernel-side transform but no bounce buffer is provided
- Locality check fails (Write requires src local; Read requires dst local)
- `SmallStridedCopy` or `DeviceGraphReplay` outcome produced (internal routing bug)

**Special case:** When the layout pair requires a transform AND a bounce buffer is
supplied, the NIXL path dispatches `dispatch_staged_nixl_transform` which:
1. Pulls raw bytes via NIXL into the local bounce buffer
2. Runs the permute kernel on the bounce buffer data
3. Places the result into the destination layout

## PlanOutcome Enum

Result of `plan_and_lower`. Four variants cover all transfer dispatch paths:

```rust
enum PlanOutcome {
    /// Empty — nothing to transfer
    Empty,
    /// Direct DMA — dispatched via batch_copy
    Direct(Vec<CopyOp>),
    /// Transform kernel — dispatched via permute kernel FFI
    Transform {
        invocation: KernelInvocation,
        block_pairs: Vec<(BlockId, BlockId)>,
        prepared: Arc<PreparedTransferPlan>,
        layer_range: Option<std::ops::Range<usize>>,
    },
    /// Small strided copy — threshold fallback via vectorized_copy
    SmallStridedCopy(Vec<CopyOp>),
    /// Device graph capture/replay — CUDA cuStreamBeginCapture/cuGraphLaunch
    DeviceGraphReplay {
        cache_key: GraphCacheKey,
        ops: Vec<CopyOp>,
    },
}
```

### Candidate Class Telemetry

Each variant maps to a short discriminator string for telemetry:

| Variant | Class Name |
|---------|------------|
| Empty | Empty (never emitted) |
| Direct | DirectDma |
| Transform | TransformKernel |
| SmallStridedCopy | SmallStridedCopy |
| DeviceGraphReplay | DeviceGraphReplay |

## Planning Pipeline

```
execute_planner_device_transfer / execute_planner_nixl_transfer
    |
    v
planner_prelude
    |
    +-- reject_heterogeneous_views_at_entry
    +-- lookup_benchmark_outcome
    +-- lookup_prepared_plan
    |
    v
plan_and_lower
    |
    +-- validate_planner_block_ids (bail if unequal, short-circuit if empty)
    +-- requires_transform? ──YES──> return PlanOutcome::Transform (from prepared plan)
    |         |
    |        NO
    |         v
    +-- physical_to_layout_view → AnnotatedLayout::from_view
    +-- plan_copy(&src_al, &dst_al, &selection, &policy)
    +-- CopyPlan::Direct?
    |      +-- lower_to_candidates
    |      +-- (optionally emit DeviceGraphReplay candidate if enabled)
    |      +-- select_candidate → PlanOutcome::Direct or DeviceGraphReplay
    +-- CopyPlan::Transform(ThresholdFallback)?
    |      +-- lower_to_candidates → select_candidate → PlanOutcome::SmallStridedCopy
    +-- CopyPlan::Transform(Semantic)? → bail (must route through catalog earlier)
    |
    v
PlanOutcome: Empty / Direct / Transform / SmallStridedCopy / DeviceGraphReplay
    |
    v
execute_outcome_device (device path) or NIXL descriptor build (NIXL path)
```

### Step Details

1. **Reject heterogeneous views** — guards against per-axis StorageKind mixing
2. **Benchmark lookup** — optional startup benchmark via BenchmarkKey
3. **Prepared plan lookup** — transform plans cached by (src_handle, dst_handle, strategy, axis_slices)
4. **plan_and_lower** — projects layouts, runs plan_copy, selects candidate

## Backend Dispatch

### Device-Agnostic Paths

Direct, SmallStridedCopy, and Transform arms run on both CUDA and SYCL through device-agnostic helpers:

| Outcome | Helper | Backend API |
|---------|--------|-------------|
| Direct | dispatch_ops_grouped_by_size | DeviceStream::batch_copy |
| SmallStridedCopy | dispatch_small_strided_copy | DeviceStream::vectorized_copy |
| Transform | dispatch_transform_kernel | FFI entrypoints |

### Per-Backend Path

DeviceGraphReplay branches per-backend:

- **CUDA**: dispatch_cuda_graph_replay_planner -> cuStreamBeginCapture / cuGraphInstantiate / cuGraphLaunch
- **SYCL**: dispatch_sycl_graph_replay_planner -> SYCL graph capture/replay

```rust
fn execute_outcome_device(
    outcome: PlanOutcome,
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    device_stream: &Arc<DeviceStream>,
    _ctx: &TransferContext,
) -> Result<()> {
    match outcome {
        PlanOutcome::Empty => unreachable!("handled before calling this fn"),
        PlanOutcome::Direct(ops) => {
            dispatch_ops_grouped_by_size(&ops, device_stream)?;
        }
        PlanOutcome::Transform { invocation, block_pairs, prepared, layer_range } => {
            dispatch_transform_kernel(&invocation, src, dst, &block_pairs, layer_range, device_stream, &prepared)?;
        }
        PlanOutcome::SmallStridedCopy(ops) => {
            dispatch_small_strided_copy(&ops, device_stream)?;
        }
        PlanOutcome::DeviceGraphReplay { cache_key, ops, .. } => {
            match device_stream.backend() {
                DeviceBackend::Cuda => dispatch_cuda_graph_replay_planner(&ops, &cache_key, ctx.graph_cache(), &cuda_stream)?,
                DeviceBackend::Sycl => dispatch_sycl_graph_replay_planner(&ops, device_stream, _ctx)?,
            }
        }
    }
    Ok(())
}
```

## CUDA Graph Replay (PR-7.4.1)

When device_graph_replay is enabled and the scorer selects it, the CUDA path captures N individual cudaMemcpyAsync nodes via MemcpyBatchMode::FallbackOnly:

1. **Cache lookup** — check GraphCache for cache_key
2. **Cache miss** — capture on temporary stream, instantiate, store node handles
3. **Address rebind** — cuGraphExecMemcpyNodeSetParams per (node, op) pair
4. **Launch** — cuGraphLaunch(exec, work_stream)

## NIXL Staged Transform (PR-6.2)

Cross-agent transforms requiring a kernel-side permute use a two-stage executor:

- **Stage 1 (sync)**: NIXL Read pulls src->bounce, or NIXL Write pushes bounce->dst
- **Stage 2 (async)**: Kernel runs bounce->dst locally

Spawns a tokio task that awaits stage 1 notification, then performs stage 2.

## Key Functions

| Function | Purpose |
|----------|---------|
| execute_planner_device_transfer | Device-family entry point |
| execute_planner_nixl_transfer | NIXL-family entry point |
| execute_outcome_device | Dispatches PlanOutcome to backend |
| planner_prelude | Shared validation + lookup pipeline |
| plan_and_lower | Core planning + candidate selection |
| dispatch_ops_grouped_by_size | Batch dispatch grouped by size |
| dispatch_small_strided_copy | Threshold fallback via vectorized_copy |
| dispatch_transform_kernel | Permute kernel dispatch |
| dispatch_cuda_graph_replay_planner | CUDA graph capture/replay |
| dispatch_sycl_graph_replay_planner | SYCL graph capture/replay |
| dispatch_staged_nixl_transform | Two-hop NIXL + kernel transform |

## Transfer Executor Overview

The transfer executor has two code paths:

| Path | Feature Flag | Document |
|------|--------------|----------|
| **Planner-driven** (preferred) | `use_planner = true` | This document |
| **Legacy direct** | `use_planner = false` | [transfer_executor_overview.md](transfer_executor_overview.md) |

The planner path (`use_planner = true`) is the modern preferred path. The legacy path
exists for backward compatibility but will be phased out.

## Related Documentation

- [kvbm_v2_xpu_sycl_enablement.md](kvbm_v2_xpu_sycl_enablement.md) — Full KVBM v2 architecture
- [transfer_executor_overview.md](transfer_executor_overview.md) — Legacy direct executor (for comparison)
- [sycl_pool_and_numa.md](../memory/docs/sycl_pool_and_numa.md) — SYCL memory pool
