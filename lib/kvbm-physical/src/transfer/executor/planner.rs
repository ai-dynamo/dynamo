// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Planner-driven CUDA / NIXL executor (`use_planner = true` path).
//!
//! Wires `transfer::plan::plan_copy` into the existing transfer
//! infrastructure for two strategy families:
//! - [`TransferStrategy::CudaAsync{H2D, D2H, D2D}`] — dispatched via
//!   `kvbm_kernels::memcpy_batch` (PR-5).
//! - [`TransferStrategy::Nixl{Read, Write, ReadFlipped, WriteFlipped}`]
//!   — dispatched via NIXL `create_xfer_req` / `post_xfer_req` (PR-5.6).
//!
//! Other strategies and `use_planner = false` callers stay on the
//! legacy [`super::execute_direct_transfer`] path; this module is only
//! reached when both conditions hold. Errors from the planner path
//! are NOT silently fallen back to the legacy executor — bail
//! semantics are explicit so callers know whether the transfer ran.
//!
//! Pipeline:
//! 1. Reject `KvBlockLayout` pairs that would require a semantic
//!    transform (PR-6 wires the kernel catalog).
//! 2. `physical_to_layout_view` projects each `PhysicalLayout` to a
//!    labelled [`LayoutView`].
//! 3. `AnnotatedLayout::from_view` collapses each view into the
//!    addressable layout the planner expects.
//! 4. `plan_copy` runs with `min_inner_bytes = 0` so any compatible
//!    layout produces [`CopyPlan::Direct`].
//! 5. `lower_to_candidates` + `select_candidate` pick the executable
//!    candidate (PR-5 only emits / accepts `Candidate::DirectDma`).
//! 6. The candidate's `Vec<CopyOp>` is grouped by `size` and dispatched
//!    via `kvbm_kernels::memcpy_batch` (`BatchedWithFallback`). Groups
//!    with distinct sizes get distinct calls; identical-size groups
//!    coalesce into one batch.

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use cudarc::driver::CudaStream;
use cudarc::driver::sys as cu_sys;
use cudarc::runtime::sys::cudaStream_t;
use dynamo_memory::nixl::{XferDescList, XferOp};
use kvbm_kernels::MemcpyBatchMode;

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::layout::KvBlockLayout;
use crate::manager::LayoutHandle;
use crate::transfer::benchmark::{BenchmarkKey, BenchmarkOutcome};
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::graph_cache::{GraphCache, ManagedExecHandle};
use crate::transfer::lower::{
    Candidate, GraphCacheKey, SelectionContext, lower_to_candidates, physical_to_layout_view,
    select_candidate,
};
use crate::transfer::plan::{
    AnnotatedLayout, CopyOp, CopyPlan, CopyPolicy, TransferSelection, TransformReason, plan_copy,
};
use crate::transfer::prepared::{PreparedPlanKey, PreparedTransferPlan};

/// RAII guard that drains any work queued on a CUDA stream before
/// `Drop` returns. Used by [`with_transform_scratch_upload`] to
/// guarantee that async H2D copies feeding the kernel from a
/// prepared-plan scratch lease have fully drained before the lease is
/// returned to its pool — closes the race where a concurrent
/// transfer could re-acquire the leased `Vec` from the pool and
/// overwrite bytes still being read by an in-flight `cudaMemcpyAsync`.
///
/// The guard records a stream event and host-synchronizes on it. If
/// event recording fails (rare; e.g. driver OOM), we fall back to a
/// full-stream `synchronize()` so the drain is best-effort guaranteed
/// even on error paths. Both record failures and synchronize failures
/// are swallowed: this runs from `Drop` and the only sane recovery is
/// to block long enough for CUDA to settle.
struct DrainOnDrop<'a> {
    stream: &'a Arc<CudaStream>,
}

impl<'a> DrainOnDrop<'a> {
    fn new(stream: &'a Arc<CudaStream>) -> Self {
        Self { stream }
    }
}

impl Drop for DrainOnDrop<'_> {
    fn drop(&mut self) {
        match self.stream.record_event(None) {
            Ok(event) => {
                let _ = event.synchronize();
            }
            Err(_) => {
                // record_event failed — fall back to a heavier
                // whole-stream sync so anything queued still drains
                // before the leased Vec returns to the pool.
                let _ = self.stream.synchronize();
            }
        }
    }
}

/// Shared scratch fill + H2D upload for the permute-kernel dispatch
/// path. The single owner of the prepared-plan lease lifetime
/// contract:
/// 1. Acquire a [`TransformScratchLease`] from the plan's pool.
/// 2. Hand the two scratch `Vec<usize>` slots to the caller's `fill`
///    closure — caller populates them via the appropriate emit method
///    on the [`PreparedTransferPlan`].
/// 3. `clone_htod` each slot to device memory.
/// 4. Record + host-synchronize on a stream event (via `DrainOnDrop`)
///    so the H2D copies drain before the lease returns to the pool.
/// 5. Drop the lease; the kernel launch continues async on the stream.
///
/// The `DrainOnDrop` guard fires on every exit path — including `?`
/// early-returns inside `fill` or between `clone_htod` calls — so a
/// concurrent transfer can never re-acquire the leased `Vec` while
/// an H2D against it is still in flight.
fn with_transform_scratch_upload<F>(
    stream: &Arc<CudaStream>,
    prepared: &PreparedTransferPlan,
    fill: F,
) -> Result<(
    cudarc::driver::CudaSlice<usize>,
    cudarc::driver::CudaSlice<usize>,
)>
where
    F: FnOnce(&mut Vec<usize>, &mut Vec<usize>) -> Result<()>,
{
    let mut lease = prepared.acquire_transform_scratch()?;
    let (a_dev, b_dev) = {
        let _drain = DrainOnDrop::new(stream);
        {
            let (a, b) = lease.both_mut();
            fill(a, b)?;
        }
        let a_dev = stream.clone_htod(lease.op_ptrs())?;
        let b_dev = stream.clone_htod(lease.univ_ptrs())?;
        (a_dev, b_dev)
    };
    drop(lease);
    Ok((a_dev, b_dev))
}

/// Dispatch a CudaAsync transfer through the stride-aware planner.
///
/// Returns the same kind of [`TransferCompleteNotification`] the
/// legacy `execute_cuda_transfer` returns, or an `Err` when the
/// transfer cannot be safely handled by the PR-5 planner path.
///
/// Bails (no fallback) when:
/// - the strategy is not one of `CudaAsync{H2D, D2H, D2D}` —
///   enforced by [`validate_cuda_planner_entry`];
/// - the src/dst block-id lists have unequal length —
///   enforced by [`validate_planner_block_ids`];
/// - `src.block_layout()` and `dst.block_layout()` would require a
///   semantic transformation (NHD↔HND, ↔Universal, etc.). The
///   planner-side projection collapses the per-token NHD/HND
///   substructure into a single trailing `Payload` axis, so a
///   raw-copy without going through the kernel catalog would silently
///   transpose-corrupt the data. PR-6.1 wires the kernel catalog
///   and removes this gate from the Cuda* entrypoint.
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_planner_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    layer_range: Option<std::ops::Range<usize>>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    validate_cuda_planner_entry(
        strategy,
        src.layout().block_layout(),
        dst.layout().block_layout(),
    )?;
    // PR-7.3: 4 KiB min_inner_bytes — same-layout copies with a small
    // contiguous tail route to SmallStridedCopy (via vectorized_copy)
    // rather than failing with a no-kernel error.
    let outcome = planner_prelude(
        src,
        dst,
        src_block_ids,
        dst_block_ids,
        strategy,
        CopyPolicy::default(),
        layer_range,
        axis_slices,
        plan_handles,
        "execute_planner_cuda_transfer",
        ctx,
    )?;

    // Acquire a stream (caller-provided or pool-acquired). Direction
    // determines which stream pool we draw from.
    let caller_manages_sync = cuda_stream.is_some();
    let stream = match &outcome {
        PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
        _ => {
            if let Some(s) = cuda_stream {
                s
            } else {
                match strategy {
                    TransferStrategy::CudaAsyncD2H => ctx.next_d2h_streams(),
                    _ => ctx.next_h2d_streams(),
                }
            }
        }
    };

    // PR-7.6: capture telemetry fields from the outcome before dispatch
    // so we can compute bytes/descriptors once without a second projection.
    // `src.layout().config().bytes_per_block()` is a pure integer multiply
    // (no allocation), so it's cheap even when no subscriber is active.
    let tel_class = outcome.candidate_class();
    let tel_descriptors = outcome.descriptor_count();
    let tel_bytes = outcome.coalesced_bytes(src.layout().config().bytes_per_block());
    // `submit_latency_us` brackets from here to after the synchronous
    // dispatch returns. It does NOT include stream-record or event-register
    // time — just the planner-dispatch submit path itself.
    let tel_t0 = std::time::Instant::now();

    match outcome {
        PlanOutcome::Empty => unreachable!("handled above"),
        PlanOutcome::Direct(ops) => {
            // Group by `size` so each `memcpy_batch` call has a
            // uniform `size_per_copy`. Common case is one group;
            // heterogeneous sizes get one batch per size.
            dispatch_ops_grouped_by_size(&ops, stream.as_ref())?;
        }
        PlanOutcome::Transform {
            invocation,
            block_pairs,
            prepared,
            layer_range,
        } => {
            dispatch_transform_kernel(
                &invocation,
                src,
                dst,
                &block_pairs,
                layer_range,
                &stream,
                &prepared,
            )?;
        }
        PlanOutcome::SmallStridedCopy(ops) => {
            dispatch_small_strided_copy(&ops, &stream)?;
        }
        PlanOutcome::CudaGraphReplay { cache_key, ops } => {
            dispatch_cuda_graph_replay_planner(&ops, &cache_key, ctx.graph_cache(), &stream)?;
        }
    }

    let tel_latency_us = tel_t0.elapsed().as_micros() as u64;

    // PR-7.6: structured telemetry event — emitted at DEBUG level.
    //
    // `route` is "cuda" for fast log filtering; `strategy` carries the
    // full `TransferStrategy` Debug repr for callers who need the exact
    // direction (H2D / D2H / D2D).
    //
    // `src_layout_signature` / `dst_layout_signature` are only
    // computed when a DEBUG subscriber has the "kvbm_physical::planner"
    // target enabled — signatures each allocate a small Vec, so we
    // guard with `tracing::enabled!` to avoid noise on the hot path.
    if tracing::enabled!(target: "kvbm_physical::planner", tracing::Level::DEBUG) {
        let src_sig = physical_to_layout_view(src)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        let dst_sig = physical_to_layout_view(dst)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        tracing::debug!(
            target: "kvbm_physical::planner",
            src_layout_signature = %src_sig,
            dst_layout_signature = %dst_sig,
            descriptor_count = tel_descriptors,
            coalesced_bytes = tel_bytes,
            route = "cuda",
            strategy = ?strategy,
            submit_latency_us = tel_latency_us,
            candidate_class = tel_class,
            "planner dispatch"
        );
    }

    if caller_manages_sync {
        return Ok(TransferCompleteNotification::completed());
    }
    let event = stream.record_event(None)?;
    Ok(ctx.register_cuda_event(event))
}

/// Dispatch a NIXL transfer through the stride-aware planner.
///
/// Behaves like [`execute_planner_cuda_transfer`] for the validation,
/// planning, and lowering stages, then maps the lowered
/// [`Vec<CopyOp>`] onto a NIXL `XferDescList` pair instead of
/// `cudaMemcpyAsync`.
///
/// Per-side `MemType` and `device_id` come from each
/// `PhysicalLayout::nixl_metadata` and are applied uniformly to every
/// op (PR-5.6 option (b): a single transfer touches one src + one dst
/// each homogeneous in storage). PR-7+ may carry per-axis storage
/// in `LayoutView` once heterogeneous-storage planning lands.
///
/// Bails (no fallback) when:
/// - the strategy is not one of `Nixl{Read, Write, ReadFlipped, WriteFlipped}` —
///   enforced by [`validate_nixl_planner_entry`];
/// - the src/dst block-id lists have unequal length —
///   enforced by [`validate_planner_block_ids`];
/// - `src.block_layout()` and `dst.block_layout()` would require a
///   kernel-side transform — PR-6.2 wires the Staged executor and
///   removes this gate from the NIXL entrypoint;
/// - locality is wrong for the chosen op (Write requires src local;
///   Read requires dst local — same invariants the legacy executor
///   asserts).
#[allow(clippy::too_many_arguments)]
pub(crate) fn execute_planner_nixl_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    bounce_buffer: Option<&crate::transfer::BounceBufferInternal>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    validate_nixl_planner_entry(
        strategy,
        src.layout().block_layout(),
        dst.layout().block_layout(),
    )?;
    let xfer_op = match strategy {
        TransferStrategy::NixlRead | TransferStrategy::NixlReadFlipped => XferOp::Read,
        TransferStrategy::NixlWrite | TransferStrategy::NixlWriteFlipped => XferOp::Write,
        // unreachable: validate_nixl_planner_entry already rejected
        // non-NIXL strategies above. Kept as a defence-in-depth bail.
        other => bail!("execute_planner_nixl_transfer: strategy {other:?} not a NIXL strategy"),
    };

    // NIXL path: min_inner_bytes = 0. NIXL descriptors are coalesced
    // by plan_copy already, and the SmallStridedCopy (vectorized_copy)
    // path is Cuda-only. Staged NIXL legs also call plan_and_lower
    // with min_inner_bytes = 0 (same same-layout-always-Direct
    // requirement).
    let nixl_policy = CopyPolicy {
        min_inner_bytes: 0,
        coalesce: true,
    };
    let ops = match planner_prelude(
        src,
        dst,
        src_block_ids,
        dst_block_ids,
        strategy,
        nixl_policy,
        None, // layer_range: NIXL paths do not yet support layer-restricted transfers
        axis_slices,
        plan_handles,
        "execute_planner_nixl_transfer",
        ctx,
    )? {
        PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
        PlanOutcome::Direct(ops) => ops,
        PlanOutcome::Transform {
            invocation,
            block_pairs,
            prepared: _,
            layer_range: _, // NIXL transform path runs full-extent
        } => {
            // Staged NIXL transforms run the kernel on the local bounce
            // buffer side; the bounce-side direct copies don't go
            // through the prepared-plan cache yet (future work — they
            // are short-lived per-request layouts, so caching keyed by
            // their handles offers no reuse benefit).
            let bounce = bounce_buffer.ok_or_else(|| {
                anyhow!(
                    "execute_planner_nixl_transfer: cross-agent transform requires \
                 TransferOptions::bounce_buffer to be set (the Staged executor pulls \
                 raw bytes through a local intermediate, runs the kernel, then places). \
                 Pass a registered local-Device PhysicalLayout via TransferOptions::bounce_buffer."
                )
            })?;
            return dispatch_staged_nixl_transform(
                src,
                dst,
                invocation,
                block_pairs,
                bounce,
                strategy,
                xfer_op,
                ctx,
            );
        }
        PlanOutcome::SmallStridedCopy(_) => bail!(
            "execute_planner_nixl_transfer: unexpected SmallStridedCopy outcome — \
             NIXL paths use min_inner_bytes = 0 and must always produce Direct. \
             This is an internal routing bug."
        ),
        PlanOutcome::CudaGraphReplay { .. } => bail!(
            "execute_planner_nixl_transfer: unexpected CudaGraphReplay outcome — \
             graph replay is only valid on Cuda-family routes, not NIXL. \
             This is an internal routing bug."
        ),
    };

    // Worker-side RDMA telemetry: CONTROL clock starts once ops are lowered —
    // it covers metadata + descriptor build + create_xfer_req, i.e. everything
    // up to the NIXL post. (See XferTelemetry / worker_xfer_complete.)
    let t_ctrl0 = std::time::Instant::now();

    let nixl_agent = ctx.nixl_agent();
    let src_metadata = src.nixl_metadata();
    let dst_metadata = dst.nixl_metadata();
    let src_is_local = nixl_agent.name() == src_metadata.agent_name();
    let dst_is_local = nixl_agent.name() == dst_metadata.agent_name();
    match xfer_op {
        XferOp::Write => {
            if !src_is_local {
                bail!(
                    "execute_planner_nixl_transfer: Write (push) requires local src; \
                     src_agent={:?}, local_agent={:?}",
                    src_metadata.agent_name(),
                    nixl_agent.name()
                );
            }
        }
        XferOp::Read => {
            if !dst_is_local {
                bail!(
                    "execute_planner_nixl_transfer: Read (pull) requires local dst; \
                     dst_agent={:?}, local_agent={:?}",
                    dst_metadata.agent_name(),
                    nixl_agent.name()
                );
            }
        }
    }

    let src_mem_type = src_metadata.mem_type();
    let dst_mem_type = dst_metadata.mem_type();
    let src_device_id = src_metadata.device_id();
    let dst_device_id = dst_metadata.device_id();

    // Build XferDescLists. One descriptor per CopyOp on each side —
    // the planner already coalesced contiguous runs, so the
    // descriptor count equals the op count. Filling through the
    // shared `NixlDescPairSink` keeps the lockstep src/dst push
    // contract in one place; the sink also doubles as a counting /
    // benchmark target via `CountingDescSink`.
    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;
    {
        use crate::transfer::prepared::{DescSink, NixlDescPairSink};
        let mut sink = NixlDescPairSink {
            src: &mut src_dl,
            dst: &mut dst_dl,
            src_device_id,
            dst_device_id,
        };
        sink.reserve(ops.len());
        for op in &ops {
            sink.push(op.src_addr, op.dst_addr, op.size);
        }
    }

    // Flipped strategies swap the roles assigned to the descriptor
    // lists at the NIXL layer (the local agent issues the request
    // against the descriptors as if the directionality were inverted).
    if matches!(
        strategy,
        TransferStrategy::NixlReadFlipped | TransferStrategy::NixlWriteFlipped
    ) {
        std::mem::swap(&mut src_dl, &mut dst_dl);
    }

    let remote_agent = match xfer_op {
        XferOp::Write => dst_metadata.agent_name(),
        XferOp::Read => src_metadata.agent_name(),
    };
    // PR-7.6: telemetry for the NIXL direct (non-staged) path.
    // Descriptor count = ops.len() (coalesced); bytes = sum of sizes (the
    // per-rank shard at TP>1, since the local layout is head-sharded).
    let tel_descriptors = ops.len();
    let tel_bytes: usize = ops.iter().map(|o| o.size).sum();

    // CONTROL = create_xfer_req (and the build above, since t_ctrl0); POST =
    // post_xfer_req; COMPLETION is measured from `tel_submitted_at` by the
    // status poller. For an async RDMA read the post returns still_pending=true
    // and completion is polled; a synchronous completion emits inline below.
    let xfer_req = nixl_agent.create_xfer_req(xfer_op, &src_dl, &dst_dl, remote_agent, None)?;
    let tel_ctrl_us = t_ctrl0.elapsed().as_micros() as u64;
    let t_post0 = std::time::Instant::now();
    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
    let tel_post_us = t_post0.elapsed().as_micros() as u64;
    let tel_submitted_at = std::time::Instant::now();

    if tracing::enabled!(target: "kvbm_physical::planner", tracing::Level::DEBUG) {
        let src_sig = physical_to_layout_view(src)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        let dst_sig = physical_to_layout_view(dst)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        tracing::debug!(
            target: "kvbm_physical::planner",
            src_layout_signature = %src_sig,
            dst_layout_signature = %dst_sig,
            descriptor_count = tel_descriptors,
            coalesced_bytes = tel_bytes,
            route = "nixl",
            strategy = ?strategy,
            submit_latency_us = tel_ctrl_us + tel_post_us,
            candidate_class = "DirectDma",
            "planner dispatch"
        );
    }

    // Per-worker RDMA audit telemetry (emitted once on completion). worker_id
    // is the per-process velo id (distinct per TP rank); device_id is the local
    // dst GPU ordinal (rank → GPU → GPU-local NIC for the report).
    let telemetry = crate::transfer::notifications::XferTelemetry {
        worker_id: ctx.worker_id(),
        op: match xfer_op {
            XferOp::Read => "Read",
            XferOp::Write => "Write",
        },
        device_id: dst_device_id,
        num_blocks: dst_block_ids.len(),
        num_descs: tel_descriptors,
        bytes: tel_bytes,
        ctrl_us: tel_ctrl_us,
        post_us: tel_post_us,
        submitted_at: tel_submitted_at,
    };

    if still_pending {
        Ok(ctx.register_nixl_status(xfer_req, Some(telemetry)))
    } else {
        // Synchronous completion never enters the poller — emit inline.
        telemetry.emit_complete(true);
        Ok(TransferCompleteNotification::completed())
    }
}

/// Result of [`plan_and_lower`].
enum PlanOutcome {
    /// The transfer has nothing to do (empty block list, or planner
    /// returned an empty op vec).
    Empty,
    /// Lowered ops to dispatch via `cudaMemcpyBatchAsync` / NIXL
    /// `XferDescList`.
    Direct(Vec<CopyOp>),
    /// PR-6.1: a `KernelInvocation` resolved through the catalog —
    /// dispatch via the matching `kvbm-kernels` FFI entrypoint with
    /// pointer arrays built from the original `PhysicalLayout`s.
    ///
    /// `prepared` is always present — see [`lookup_prepared_plan`].
    /// The dispatch path uses it to reuse pooled scratch arrays for
    /// `op_ptrs` / `univ_ptrs`; when the cache is disabled or the
    /// pair has no handle key, the Arc points to a fresh per-call
    /// plan with the same shape.
    Transform {
        invocation: crate::transfer::kernel_catalog::KernelInvocation,
        block_pairs: Vec<(BlockId, BlockId)>,
        prepared: std::sync::Arc<PreparedTransferPlan>,
        /// Optional contiguous layer subrange the kernel must walk.
        /// `None` = full extent (legacy behaviour); `Some(range)`
        /// restricts emit + kernel invocation to that slice.
        layer_range: Option<std::ops::Range<usize>>,
    },
    /// PR-7.3: threshold-fallback for same-layout copies whose inner
    /// contiguous tail is below `min_inner_bytes`. Each op has the same
    /// `size` (inner_bytes at the cut point). Dispatched via
    /// `kvbm_kernels::vectorized_copy` on the Cuda path.
    SmallStridedCopy(Vec<CopyOp>),
    /// PR-7.4.1: CUDA graph capture/replay. `ops` are the copy
    /// descriptors whose addresses are rebound on each replay.
    CudaGraphReplay {
        cache_key: GraphCacheKey,
        ops: Vec<CopyOp>,
    },
}

impl PlanOutcome {
    /// PR-7.6: short discriminator string used in telemetry events.
    ///
    /// The returned string is the same format as [`Candidate::class_name`].
    /// `Empty` is never emitted (callers short-circuit before the telemetry
    /// event); `StagedTransform` is emitted separately in
    /// [`dispatch_staged_nixl_transform`] which does not produce a
    /// `PlanOutcome`.
    fn candidate_class(&self) -> &'static str {
        match self {
            PlanOutcome::Empty => "Empty", // not emitted; here for completeness
            PlanOutcome::Direct(_) => "DirectDma",
            PlanOutcome::Transform { .. } => "TransformKernel",
            PlanOutcome::SmallStridedCopy(_) => "SmallStridedCopy",
            PlanOutcome::CudaGraphReplay { .. } => "CudaGraphReplay",
        }
    }

    /// Total bytes across all ops, or the kernel-side byte volume for
    /// Transform outcomes.
    ///
    /// For `Direct` / `SmallStridedCopy`: sum of `op.size` over the op vec.
    /// For `Transform`: `block_pairs.len() * bytes_per_block` — the kernel
    ///   moves exactly one block worth of data per pair. `bytes_per_block`
    ///   is taken from the `src` layout (caller passes it in to avoid
    ///   needing another `PhysicalLayout` ref inside the enum).
    fn coalesced_bytes(&self, src_bytes_per_block: usize) -> usize {
        match self {
            PlanOutcome::Empty => 0,
            PlanOutcome::Direct(ops) => ops.iter().map(|o| o.size).sum(),
            PlanOutcome::Transform { block_pairs, .. } => block_pairs.len() * src_bytes_per_block,
            PlanOutcome::SmallStridedCopy(ops) => ops.iter().map(|o| o.size).sum(),
            PlanOutcome::CudaGraphReplay { ops, .. } => ops.iter().map(|o| o.size).sum(),
        }
    }

    /// Number of descriptors / ops in the outcome.
    ///
    /// For `Direct` / `SmallStridedCopy` / `CudaGraphReplay`: op count.
    /// For `Transform`: block-pair count (one kernel call per pair group).
    fn descriptor_count(&self) -> usize {
        match self {
            PlanOutcome::Empty => 0,
            PlanOutcome::Direct(ops) => ops.len(),
            PlanOutcome::Transform { block_pairs, .. } => block_pairs.len(),
            PlanOutcome::SmallStridedCopy(ops) => ops.len(),
            PlanOutcome::CudaGraphReplay { ops, .. } => ops.len(),
        }
    }
}

/// Shared "reject-heterogeneous → benchmark-lookup → prepared-plan-lookup
/// → plan_and_lower" pipeline used by both the CUDA and NIXL planner-path
/// entrypoints.
///
/// Strategy and layout-compatibility checks are NOT done here — each
/// entrypoint enforces its own per-family contract via
/// [`validate_cuda_planner_entry`] / [`validate_nixl_planner_entry`]
/// before calling this. The two entrypoints differ only by:
/// - the `policy` they pass in (Cuda uses default with
///   `min_inner_bytes = 4096`; NIXL uses `{ 0, coalesce: true }`);
/// - the `entry_label` they pass for the heterogeneous-rejection error.
#[allow(clippy::too_many_arguments)]
fn planner_prelude(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    policy: CopyPolicy,
    layer_range: Option<std::ops::Range<usize>>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    entry_label: &'static str,
    ctx: &TransferContext,
) -> Result<PlanOutcome> {
    reject_heterogeneous_views_at_entry(src, dst, entry_label)?;
    let benchmark_outcome = lookup_benchmark_outcome(src, dst, strategy, ctx);
    let prepared_plan = lookup_prepared_plan(src, dst, strategy, &axis_slices, plan_handles, ctx)?;
    plan_and_lower(
        src,
        dst,
        src_block_ids,
        dst_block_ids,
        strategy,
        ctx.capabilities(),
        policy,
        benchmark_outcome,
        prepared_plan,
        layer_range,
        axis_slices,
    )
}

/// PR-7.5: Look up the benchmark cache and build a `BenchmarkKey` from
/// two physical layouts.  Returns `None` if `startup_benchmark` is
/// disabled or if the views can't be projected (safe fallback: scorer
/// uses baseline scores).
fn lookup_benchmark_outcome(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    strategy: TransferStrategy,
    ctx: &TransferContext,
) -> Option<BenchmarkOutcome> {
    if !ctx.capabilities().startup_benchmark {
        return None;
    }
    let src_view = physical_to_layout_view(src).ok()?;
    let dst_view = physical_to_layout_view(dst).ok()?;
    let dtype_w = Some(src.layout().config().dtype_width_bytes as u32);
    let key = BenchmarkKey::new(
        src_view.signature(),
        dst_view.signature(),
        dtype_w,
        strategy,
    );
    ctx.benchmark_cache().lookup(&key)
}

/// Resolve a prepared transfer plan for this pair.
///
/// Returns `Some(Arc<PreparedTransferPlan>)` for layout pairs that go
/// through the kernel catalog (operational↔universal, NHD↔HND);
/// returns `None` for same-layout direct copies, which the planner
/// projects `AnnotatedLayout`s for inline without caching (the
/// projection cost is microseconds against millisecond-scale
/// DMA/RDMA — caching adds complexity without measurable gain).
///
/// When the cache is enabled and `plan_handles` is `Some`, transform
/// plans are shared via the [`PreparedPlanCache`]; otherwise a fresh,
/// uncached plan is built with a private scratch pool.
fn lookup_prepared_plan(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    strategy: TransferStrategy,
    axis_slices: &[kvbm_common::AxisIntersection],
    plan_handles: Option<(LayoutHandle, LayoutHandle)>,
    ctx: &TransferContext,
) -> Result<Option<std::sync::Arc<PreparedTransferPlan>>> {
    let src_kv = src.layout().block_layout();
    let dst_kv = dst.layout().block_layout();
    if !src_kv.requires_transform(&dst_kv) {
        return Ok(None);
    }
    if !axis_slices.is_empty() {
        bail!(
            "lookup_prepared_plan: axis_slices not supported when the layout pair \
             requires a permute-kernel transform (src={:?}, dst={:?})",
            src_kv,
            dst_kv,
        );
    }

    let build = || -> Result<PreparedTransferPlan> {
        let invocation = build_transform_invocation(src, dst)?;
        PreparedTransferPlan::build_transform(invocation, src, dst)
    };

    let plan = if let Some((src_handle, dst_handle)) = plan_handles {
        let key = PreparedPlanKey::new(src_handle, dst_handle, strategy, axis_slices);
        ctx.prepared_plan_cache()
            .get_or_insert_with(ctx.worker_id(), key, build)?
    } else {
        // No handles → no cache key. Build fresh; scratch pool is
        // private to this transfer and drops with the Arc.
        std::sync::Arc::new(build()?)
    };
    Ok(Some(plan))
}

#[allow(clippy::too_many_arguments)]
fn plan_and_lower(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    capabilities: &crate::transfer::TransferCapabilities,
    policy: CopyPolicy,
    benchmark_outcome: Option<BenchmarkOutcome>,
    prepared_plan: Option<std::sync::Arc<PreparedTransferPlan>>,
    layer_range: Option<std::ops::Range<usize>>,
    axis_slices: Vec<kvbm_common::AxisIntersection>,
) -> Result<PlanOutcome> {
    if validate_planner_block_ids(src_block_ids, dst_block_ids)?.is_noop() {
        return Ok(PlanOutcome::Empty);
    }

    let block_pairs: Vec<(BlockId, BlockId)> = src_block_ids
        .iter()
        .zip(dst_block_ids.iter())
        .map(|(&s, &d)| (s, d))
        .collect();

    // Catalog dispatch for layout pairs whose semantics differ
    // (NHD↔HND, operational↔universal, etc.). `plan_copy` would
    // technically still produce a Direct op-set for these via
    // per-coord stride math — each op being a `head_size` byte
    // chunk — but the resulting descriptor count is large
    // (`num_blocks_to_transfer × num_layers × outer_dim × page_size
    // × num_heads`) and the dedicated permute kernel is the
    // intended path. Routing these to the catalog before
    // `plan_copy` runs keeps `plan_copy` focused on same-shape
    // copies and surfaces a no-matching-kernel error precisely
    // when no kernel covers the pair (e.g. NHD↔HND in PR-6.1
    // before PR-6.3 lands a transpose kernel).
    //
    // AB-1d: the transform path builds a whole-block invocation that
    // would silently discard caller-supplied axis_slices — reject
    // explicitly rather than mis-transferring the wrong region.
    // Cross-leader sliced transfers under layout-mismatch + permute
    // is a future PR (combine kernel staged-bounce with a sliced
    // post-kernel leg).
    {
        let src_kv = src.layout().block_layout();
        let dst_kv = dst.layout().block_layout();
        if src_kv.requires_transform(&dst_kv) {
            if !axis_slices.is_empty() {
                bail!(
                    "plan_and_lower: axis_slices not supported when the layout pair \
                     requires a permute-kernel transform (src={:?}, dst={:?}); \
                     cross-leader sliced transfers under layout-mismatch is a future PR",
                    src_kv,
                    dst_kv,
                );
            }
            let prepared = prepared_plan.ok_or_else(|| {
                anyhow!(
                    "plan_and_lower: layout pair requires a permute-kernel transform \
                     (src={:?}, dst={:?}) but no prepared plan was supplied",
                    src_kv,
                    dst_kv,
                )
            })?;
            // Validate layer_range up front so the caller sees a precise
            // error here rather than a generic kernel-launch failure deep
            // in the FFI.
            if let Some(ref r) = layer_range {
                let nl_full = prepared.invocation.num_layers;
                if r.end > nl_full || r.start > r.end {
                    bail!(
                        "plan_and_lower: layer_range {:?} out of bounds for \
                         invocation.num_layers={} (src={:?}, dst={:?})",
                        r,
                        nl_full,
                        src_kv,
                        dst_kv,
                    );
                }
            }
            let invocation = prepared.invocation;
            return Ok(PlanOutcome::Transform {
                invocation,
                block_pairs,
                prepared,
                layer_range,
            });
        }
    }

    // Direct (same-layout / sliced) path: project AnnotatedLayouts
    // inline. The projection is microseconds — too cheap to cache
    // against the millisecond-scale DMA/RDMA the planner is feeding.
    if layer_range.is_some() {
        bail!(
            "plan_and_lower: layer_range is currently only supported on the \
             permute-kernel transform path; same-layout direct copies stay on \
             the legacy executor for layer-restricted transfers."
        );
    }
    if prepared_plan.is_some() {
        bail!(
            "plan_and_lower: prepared plan supplied for a direct same-layout pair; \
             callers must only pass a plan for permute-kernel transform pairs"
        );
    }
    let src_view = physical_to_layout_view(src)?;
    let dst_view = physical_to_layout_view(dst)?;
    let src_al = AnnotatedLayout::from_view(&src_view)?;
    let dst_al = AnnotatedLayout::from_view(&dst_view)?;

    // `block_pairs` was already built above; `plan_copy` consumes
    // a `usize`-typed pair list. AB-1d: thread caller-supplied
    // axis_slices into the selection when present (empty = full extent
    // along every axis, equivalent to TransferSelection::full).
    let plan_block_pairs: Vec<(usize, usize)> = block_pairs.iter().map(|&(s, d)| (s, d)).collect();
    let selection = if axis_slices.is_empty() {
        TransferSelection::full(plan_block_pairs)
    } else {
        TransferSelection {
            block_pairs: plan_block_pairs,
            axis_slices,
        }
    };
    // `policy` is caller-supplied:
    // - Cuda entry uses CopyPolicy::default() (min_inner_bytes = 4096, PR-7.3).
    // - NIXL entry and staged NIXL legs use min_inner_bytes = 0 so same-layout
    //   legs always produce Direct ops; the SmallStridedCopy path is Cuda-only.

    let plan = plan_copy(&src_al, &dst_al, &selection, &policy)?;
    match plan {
        CopyPlan::Direct(ops) if ops.is_empty() => Ok(PlanOutcome::Empty),
        CopyPlan::Direct(_) => {
            let mut candidates = lower_to_candidates(plan)?;
            let total_bytes: usize = match &candidates[..] {
                [Candidate::DirectDma { ops }] => ops.iter().map(|o| o.size).sum(),
                _ => 0,
            };
            // PR-7.4.1: emit a CudaGraphReplay candidate alongside DirectDma on
            // Cuda-family routes when cuda_graph_replay is enabled. The scorer
            // gives CudaGraphReplay a higher score (1050) than DirectDma (1000),
            // so select_candidate picks it when the cap is on. The ops are shared
            // with the DirectDma candidate; rebinding happens per-launch.
            if capabilities.cuda_graph_replay
                && strategy.is_cuda_family()
                && let [Candidate::DirectDma { ops }] = &candidates[..]
            {
                let route_family = match strategy {
                    TransferStrategy::CudaAsyncH2D => 0u8,
                    TransferStrategy::CudaAsyncD2H => 1u8,
                    TransferStrategy::CudaAsyncD2D => 2u8,
                    _ => 3u8,
                };
                // `dtype_width_bytes` from LayoutConfig is always set
                // (defaults to 2 if not explicitly configured). We use it
                // as the dtype discriminant, keeping the cache key
                // consistent regardless of whether `LayoutConfig::dtype`
                // is populated.
                let cache_key = GraphCacheKey {
                    descriptor_count: ops.len(),
                    total_bytes,
                    dtype_width_bytes: Some(src.layout().config().dtype_width_bytes as u32),
                    route_family,
                    candidate_class: 0, // DirectDma-shaped
                };
                let replay_ops = ops.clone();
                candidates.push(Candidate::CudaGraphReplay {
                    cache_key,
                    ops: replay_ops,
                });
            }
            let sel_ctx = SelectionContext {
                strategy,
                descriptor_count: candidates.len(),
                total_bytes,
                dtype: src.layout().config().dtype,
                capabilities,
                benchmark_outcome,
            };
            let chosen = select_candidate(&candidates, &sel_ctx)?;
            match chosen {
                Candidate::DirectDma { ops } => Ok(PlanOutcome::Direct(ops.clone())),
                // PR-7.4.1: CudaGraphReplay selected — emit with ops for capture/rebind.
                Candidate::CudaGraphReplay { cache_key, ops } => Ok(PlanOutcome::CudaGraphReplay {
                    cache_key: cache_key.clone(),
                    ops: ops.clone(),
                }),
                other => bail!(
                    "plan_and_lower: select_candidate returned unexpected variant for \
                     CopyPlan::Direct: {other:?}"
                ),
            }
        }
        // ThresholdFallback: inner_bytes < min_inner_bytes for a same-layout pair.
        // Route through lower_to_candidates -> SmallStridedCopy.
        CopyPlan::Transform {
            reason: TransformReason::ThresholdFallback,
            ref ops,
            ..
        } if ops.is_empty() => Ok(PlanOutcome::Empty),
        CopyPlan::Transform {
            reason: TransformReason::ThresholdFallback,
            ..
        } => {
            let candidates = lower_to_candidates(plan)?;
            let total_bytes: usize = match &candidates[..] {
                [Candidate::SmallStridedCopy { ops }] => ops.iter().map(|o| o.size).sum(),
                _ => 0,
            };
            let sel_ctx = SelectionContext {
                strategy,
                descriptor_count: candidates.len(),
                total_bytes,
                dtype: src.layout().config().dtype,
                capabilities,
                benchmark_outcome,
            };
            let chosen = select_candidate(&candidates, &sel_ctx)?;
            match chosen {
                Candidate::SmallStridedCopy { ops } => {
                    Ok(PlanOutcome::SmallStridedCopy(ops.clone()))
                }
                other => bail!(
                    "plan_and_lower: select_candidate returned unexpected variant for \
                     ThresholdFallback: {other:?}"
                ),
            }
        }
        // Semantic Transform from plan_copy: plan_copy never emits this; this
        // arm is a future-proofing safety net.
        CopyPlan::Transform {
            reason: TransformReason::Semantic,
            ..
        } => bail!(
            "plan_and_lower: plan_copy emitted Transform(Semantic) — semantic transforms \
             must be routed through the kernel catalog before plan_copy is called. \
             This is an internal routing bug."
        ),
        CopyPlan::Staged { .. } => {
            bail!("plan_and_lower: CopyPlan::Staged is reserved (NIXL transforms in PR-6.2)")
        }
    }
}

/// Resolve a kernel for a `CopyPlan::Transform` through the catalog
/// and build the launch parameters. Errors precisely when no kernel
/// covers the (src_kv, dst_kv, dtype) triple.
pub(crate) fn build_transform_invocation(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
) -> Result<crate::transfer::kernel_catalog::KernelInvocation> {
    use crate::transfer::kernel_catalog::{
        KernelInvocation, KernelKind, match_kernel, to_kernel_block_layout,
    };

    let src_kv = src.layout().block_layout();
    let dst_kv = dst.layout().block_layout();
    let cfg = src.layout().config();
    let dst_cfg = dst.layout().config();
    // Per-block-shape gate: ignore `num_blocks` (per-tier capacity)
    // and compare everything else. G1 (GPU HBM) and G2 (pinned host)
    // routinely hold different block counts; rejecting that case
    // makes cross-tier Universal-mode transfers impossible.
    if !cfg.has_same_block_shape(dst_cfg) {
        bail!(
            "build_transform_invocation: src and dst differ on per-block shape — the \
             catalog only dispatches transforms between layouts with identical block \
             geometry (got src.num_layers={}, src.outer_dim={}, src.page_size={}, \
             src.inner_dim={}, src.num_heads={:?}, dst.num_layers={}, dst.outer_dim={}, \
             dst.page_size={}, dst.inner_dim={}, dst.num_heads={:?}). num_blocks is \
             allowed to differ.",
            cfg.num_layers,
            cfg.outer_dim,
            cfg.page_size,
            cfg.inner_dim,
            cfg.num_heads,
            dst_cfg.num_layers,
            dst_cfg.outer_dim,
            dst_cfg.page_size,
            dst_cfg.inner_dim,
            dst_cfg.num_heads,
        );
    }
    // Effective dtype: prefers explicit `cfg.dtype`, falls back to
    // `derive_tensor_dtype_from_width`. Production layouts historically
    // set only `dtype_width_bytes`; the catalog needs a concrete type.
    // For permute kernels (the only transforms today) F16 vs BF16
    // produce identical bytes, so the derived BF16 default is correct
    // wherever the explicit dtype isn't plumbed. FP8 derives to
    // `TensorDataType::FP8` and `match_kernel` rejects it at the
    // catalog (no kernel template) — clean failure rather than silent
    // wrong dispatch.
    let dtype = cfg.effective_dtype().ok_or_else(|| {
        anyhow!(
            "build_transform_invocation: cannot determine dtype — \
             LayoutConfig.dtype is None and dtype_width_bytes={} has no \
             derivation (validated widths are {{1, 2, 4, 8}}; outside that \
             set is a config-validation gap)",
            cfg.dtype_width_bytes,
        )
    })?;
    let nh = cfg.num_heads.ok_or_else(|| {
        anyhow!("build_transform_invocation: cfg.num_heads is required for transform dispatch")
    })?;
    if !cfg.inner_dim.is_multiple_of(nh) {
        bail!(
            "build_transform_invocation: inner_dim ({}) is not divisible by num_heads ({})",
            cfg.inner_dim,
            nh
        );
    }
    let head_dim = cfg.inner_dim / nh;

    let kind = match_kernel(src_kv, dst_kv, dtype).ok_or_else(|| {
        anyhow!(
            "build_transform_invocation: no kernel registered for (src={src_kv:?}, \
             dst={dst_kv:?}, dtype={dtype:?}). Catalog miss — pair has no \
             registered kernel."
        )
    })?;

    // `block_layout` carries the kernel's NHD/HND template parameter:
    // - U↔O: the (single) operational side selects the inner-token
    //   ordering of the block stack.
    // - O↔O (NhdHndTranspose): both sides are operational; the kernel's
    //   `src_layout` flag drives the inner-offset formulas, so the SRC
    //   side wins.
    let operational_kv = match kind {
        KernelKind::UniversalFromBlock => src_kv,
        KernelKind::BlockFromUniversal => dst_kv,
        KernelKind::NhdHndTranspose => src_kv,
    };
    let block_layout = to_kernel_block_layout(operational_kv).ok_or_else(|| {
        anyhow!(
            "build_transform_invocation: operational side {operational_kv:?} \
             has no kernel-side BlockLayout mapping"
        )
    })?;

    Ok(KernelInvocation {
        kind,
        num_layers: cfg.num_layers,
        outer_dim: cfg.outer_dim,
        page_size: cfg.page_size,
        num_heads: nh,
        head_dim,
        dtype,
        block_layout,
    })
}

/// PR-7.7.1: reject any pair where either layout has more than one
/// distinct [`StorageKind`] across its axes.
///
/// Heterogeneous layouts (e.g. some axes on disk, some on device) require
/// composition planning that doesn't exist yet. Failing loudly here converts
/// a latent correctness bug (the planner would treat the heterogeneous view
/// as homogeneous and produce wrong bytes) into a precise, catchable error.
///
/// Composition planning lands in PR-7.7.2+: `Candidate::HeterogeneousComposite`,
/// axis-range sub-plan composition, executor dispatch over disk + device tiers.
///
/// Called after the per-entrypoint strategy guard and before `plan_and_lower`.
/// The projection cost (one `layout_view()` per side) is paid only on this
/// hot path; today's homogeneous-only callers see no regression because
/// `is_heterogeneous()` returns `false` for every existing producer.
fn reject_heterogeneous_views_at_entry(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    entry_label: &str,
) -> Result<()> {
    let src_view = physical_to_layout_view(src)?;
    let dst_view = physical_to_layout_view(dst)?;
    reject_heterogeneous_layout_views(&src_view, &dst_view, entry_label)
}

/// Core heterogeneous-layout guard operating on already-projected
/// [`LayoutView`]s.
///
/// Extracted separately so unit tests can exercise the guard logic with
/// synthetic `LayoutView` fixtures constructed via [`LayoutView::full`],
/// without needing a full [`PhysicalLayout`] stack.
pub(crate) fn reject_heterogeneous_layout_views(
    src_view: &crate::layout::LayoutView,
    dst_view: &crate::layout::LayoutView,
    entry_label: &str,
) -> Result<()> {
    if src_view.is_heterogeneous() {
        bail!(
            "{entry_label}: src layout is heterogeneous (per-axis StorageKind contains \
             more than one distinct kind). Heterogeneous-layout composition is not yet \
             supported by the planner-driven path. Composition planning lands in PR-7.7.2+."
        );
    }
    if dst_view.is_heterogeneous() {
        bail!(
            "{entry_label}: dst layout is heterogeneous (per-axis StorageKind contains \
             more than one distinct kind). Heterogeneous-layout composition is not yet \
             supported by the planner-driven path. Composition planning lands in PR-7.7.2+."
        );
    }
    Ok(())
}

/// Outcome of [`validate_planner_inputs`].
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PlannerInputs {
    /// Inputs are valid and the transfer must proceed.
    Proceed,
    /// Inputs are valid but the transfer is a no-op (empty block list).
    /// Caller short-circuits with a "completed" notification.
    Noop,
}

impl PlannerInputs {
    fn is_noop(&self) -> bool {
        matches!(self, Self::Noop)
    }
}

/// Pure structural validation: block-id list lengths must match, and
/// an empty list short-circuits to a no-op completion. No knowledge
/// of strategy or layouts.
///
/// Extracted so the rejection paths can be tested without a
/// `TransferContext` (which needs a real CUDA stream pool, NIXL agent,
/// and tokio runtime).
pub(crate) fn validate_planner_block_ids(
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
) -> Result<PlannerInputs> {
    if src_block_ids.len() != dst_block_ids.len() {
        bail!(
            "validate_planner_block_ids: src_block_ids ({}) != dst_block_ids ({})",
            src_block_ids.len(),
            dst_block_ids.len()
        );
    }
    if src_block_ids.is_empty() {
        return Ok(PlannerInputs::Noop);
    }
    Ok(PlannerInputs::Proceed)
}

/// Per-entrypoint guard for [`execute_planner_cuda_transfer`].
///
/// Rejects strategies outside the `CudaAsync{H2D,D2H,D2D}` family.
/// Layout-pair compatibility is enforced downstream by the kernel
/// catalog (`build_transform_invocation`): identical layouts go
/// through the Direct path, registered transform pairs through the
/// Transform path, unregistered transform pairs surface a precise
/// no-matching-kernel error.
pub(crate) fn validate_cuda_planner_entry(
    strategy: TransferStrategy,
    _src_block_layout: KvBlockLayout,
    _dst_block_layout: KvBlockLayout,
) -> Result<()> {
    if !matches!(
        strategy,
        TransferStrategy::CudaAsyncH2D
            | TransferStrategy::CudaAsyncD2H
            | TransferStrategy::CudaAsyncD2D
    ) {
        bail!(
            "validate_cuda_planner_entry: strategy {strategy:?} is not a CudaAsync \
             variant — caller routed a non-Cuda strategy into the Cuda planner \
             entrypoint"
        );
    }
    Ok(())
}

/// Per-entrypoint guard for [`execute_planner_nixl_transfer`].
///
/// Rejects strategies outside the `Nixl{Read,Write,ReadFlipped,
/// WriteFlipped}` family. Layout-pair compatibility is enforced
/// downstream — the Staged executor handles `requires_transform=true`
/// pairs via the kernel catalog.
pub(crate) fn validate_nixl_planner_entry(
    strategy: TransferStrategy,
    _src_block_layout: KvBlockLayout,
    _dst_block_layout: KvBlockLayout,
) -> Result<()> {
    if !matches!(
        strategy,
        TransferStrategy::NixlRead
            | TransferStrategy::NixlWrite
            | TransferStrategy::NixlReadFlipped
            | TransferStrategy::NixlWriteFlipped
    ) {
        bail!(
            "validate_nixl_planner_entry: strategy {strategy:?} is not a NIXL \
             variant — caller routed a non-NIXL strategy into the NIXL planner \
             entrypoint"
        );
    }
    Ok(())
}

/// Dispatch a `KernelInvocation` resolved by the catalog: fill the
/// per-side pointer arrays from `(src, dst, block_pairs)` through the
/// prepared plan's scratch pool, push them to device memory, and
/// launch the matching `kvbm-kernels` FFI entrypoint.
///
/// Three kernel kinds share this dispatch:
/// - [`KernelKind::UniversalFromBlock`] / [`KernelKind::BlockFromUniversal`]
///   move data between an operational layout (walked via
///   `Layout::memory_region(block, layer, outer)` — one pointer per
///   `(block, layer, outer)` chunk) and a universal layout (computed
///   as `single_buffer_base + block_id * bytes_per_block` — one
///   pointer per block, FC-only).
/// - [`KernelKind::NhdHndTranspose`] transposes between two
///   operational layouts; both sides are walked the same way.
///
/// `layer_range` restricts the transfer to a contiguous layer slice:
/// emit methods walk only those layers, and the FFI call passes
/// `nl = range.len()`, `nl_full = invocation.num_layers`, `nl_offset =
/// range.start` so the universal-side kernels use the full per-head
/// stride. `None` means full-extent (legacy behaviour).
///
/// The shared scratch + upload + drain path is owned by
/// [`with_transform_scratch_upload`]; this function only picks the
/// per-kind emit method and FFI entrypoint.
pub(crate) fn dispatch_transform_kernel(
    invocation: &crate::transfer::kernel_catalog::KernelInvocation,
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    block_pairs: &[(BlockId, BlockId)],
    layer_range: Option<std::ops::Range<usize>>,
    stream: &Arc<CudaStream>,
    prepared: &PreparedTransferPlan,
) -> Result<()> {
    use cudarc::driver::DevicePtr;

    use crate::transfer::kernel_catalog::KernelKind;

    let stream_raw = stream.cu_stream() as cudaStream_t;
    let nl_full = invocation.num_layers;
    if let Some(ref r) = layer_range {
        if r.end > nl_full || r.start > r.end {
            bail!(
                "dispatch_transform_kernel: layer_range {:?} out of bounds for \
                 invocation.num_layers={}",
                r,
                nl_full,
            );
        }
        if r.is_empty() {
            // Empty layer range — nothing to do.
            return Ok(());
        }
    }
    let (nl, nl_offset) = match &layer_range {
        Some(r) => (r.len(), r.start),
        None => (nl_full, 0),
    };
    let no = invocation.outer_dim;
    let nt = invocation.page_size;
    let nh = invocation.num_heads;
    let hd = invocation.head_dim;

    // Project per-side block-id slices once. The two fill the lease's
    // `op_ptrs` and `univ_ptrs` slots respectively; for op↔op transpose
    // we just reuse those slots as src/dst (the lease doesn't care
    // about the semantic name, only the two capacity-pooled Vecs).
    let a_block_ids: Vec<BlockId> = block_pairs
        .iter()
        .map(|p| match invocation.kind {
            KernelKind::UniversalFromBlock | KernelKind::NhdHndTranspose => p.0,
            KernelKind::BlockFromUniversal => p.1,
        })
        .collect();
    let b_block_ids: Vec<BlockId> = block_pairs
        .iter()
        .map(|p| match invocation.kind {
            KernelKind::UniversalFromBlock | KernelKind::NhdHndTranspose => p.1,
            KernelKind::BlockFromUniversal => p.0,
        })
        .collect();

    let layer_range_ref = layer_range.as_ref();
    let (a_dev, b_dev) = with_transform_scratch_upload(stream, prepared, |a, b| {
        match invocation.kind {
            KernelKind::UniversalFromBlock => prepared.emit_universal_kind_pointers(
                src, // operational side
                &a_block_ids,
                &b_block_ids,
                layer_range_ref,
                a,
                b,
            ),
            KernelKind::BlockFromUniversal => prepared.emit_universal_kind_pointers(
                dst, // operational side
                &a_block_ids,
                &b_block_ids,
                layer_range_ref,
                a,
                b,
            ),
            KernelKind::NhdHndTranspose => prepared.emit_oo_transpose_pointers(
                src,
                dst,
                &a_block_ids,
                &b_block_ids,
                layer_range_ref,
                a,
                b,
            ),
        }
    })?;

    let (a_ptr_raw, _a_guard) = a_dev.device_ptr(stream.as_ref());
    let (b_ptr_raw, _b_guard) = b_dev.device_ptr(stream.as_ref());

    tracing::debug!(
        target: "kvbm_physical::planner",
        kind = ?invocation.kind,
        num_blocks = block_pairs.len(),
        nh = nh,
        nl = nl,
        no = no,
        nt = nt,
        hd = hd,
        dtype = ?invocation.dtype,
        block_layout = ?invocation.block_layout,
        "fused permute kernel dispatch"
    );

    let status = match invocation.kind {
        KernelKind::UniversalFromBlock => {
            // a = operational (src, IN), b = universal (dst, OUT).
            let universal_ptrs = b_ptr_raw as usize as *const *mut c_void;
            let block_ptrs = a_ptr_raw as usize as *const *const c_void;
            unsafe {
                kvbm_kernels::universal_from_block(
                    universal_ptrs,
                    block_ptrs,
                    block_pairs.len(),
                    nh,
                    nl,
                    no,
                    nt,
                    hd,
                    nl_full,
                    nl_offset,
                    invocation.dtype,
                    invocation.block_layout,
                    stream_raw,
                )
            }
        }
        KernelKind::BlockFromUniversal => {
            // a = operational (dst, OUT), b = universal (src, IN).
            let universal_ptrs = b_ptr_raw as usize as *const *const c_void;
            let block_ptrs = a_ptr_raw as usize as *const *mut c_void;
            unsafe {
                kvbm_kernels::block_from_universal(
                    universal_ptrs,
                    block_ptrs,
                    block_pairs.len(),
                    nh,
                    nl,
                    no,
                    nt,
                    hd,
                    nl_full,
                    nl_offset,
                    invocation.dtype,
                    invocation.block_layout,
                    stream_raw,
                )
            }
        }
        KernelKind::NhdHndTranspose => {
            // a = src op, b = dst op. The op↔op kernel doesn't need
            // `nl_full` / `nl_offset` — both sides are operational so
            // the chunk-pointer table already encodes the layer slice,
            // and `nl` here is the slice length.
            unsafe {
                kvbm_kernels::nhd_hnd_transpose(
                    a_ptr_raw as usize as *const *const c_void,
                    b_ptr_raw as usize as *const *mut c_void,
                    block_pairs.len(),
                    nl,
                    no,
                    nt,
                    nh,
                    hd,
                    invocation.dtype,
                    invocation.block_layout,
                    stream_raw,
                )
            }
        }
    };
    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        bail!(
            "dispatch_transform_kernel: kernel launch failed with status={status:?} \
             for kind={:?}, num_blocks_to_transfer={}",
            invocation.kind,
            block_pairs.len(),
        );
    }
    Ok(())
}

/// Owned bundle of planner-internal bits needed by the staged-NIXL
/// executor — both the synchronous stage-1 call site and the spawned
/// stage-2 task.
///
/// The staged executor must spawn a `tokio::task` for stage 2 because
/// it needs to `await` stage 1's notification. Tasks cannot hold
/// `&TransferContext` across an `.await` (not `'static`). This struct
/// clones the small set of bits the two stages need and bundles
/// `register_cuda_event` / `register_nixl_status` /
/// `build_and_post_nixl_leg` methods so both stages call identical
/// code without an `_owned` suffix variant.
///
/// All methods return `Result<TransferCompleteNotification>` — unlike
/// `TransferContext::register_cuda_event` (which panics on alloc
/// failure), these are hot-path helpers that surface errors to their
/// caller for graceful handling.
struct OwnedStagedContext {
    event_system: Arc<velo::EventManager>,
    tx_cuda_event: tokio::sync::mpsc::Sender<
        crate::transfer::notifications::RegisterPollingNotification<
            crate::transfer::notifications::CudaEventChecker,
        >,
    >,
    tx_nixl_status: tokio::sync::mpsc::Sender<
        crate::transfer::notifications::RegisterPollingNotification<
            crate::transfer::notifications::NixlStatusChecker,
        >,
    >,
    raw_agent: dynamo_memory::nixl::Agent,
    nixl_agent: super::super::NixlAgent,
    stream: Arc<CudaStream>,
    /// Copied from `TransferContext` so the staged task can build a
    /// `SelectionContext` for the NIXL leg's `plan_and_lower` call.
    /// `TransferCapabilities` is `Copy` so this is cheap.
    capabilities: crate::transfer::TransferCapabilities,
}

impl OwnedStagedContext {
    /// Snapshot the bits needed for the staged executor from a live
    /// `TransferContext`. The stream is acquired once here so both
    /// stages share the same stream handle.
    fn from_ctx(ctx: &TransferContext) -> Self {
        let nixl_agent = ctx.nixl_agent().clone();
        Self {
            event_system: ctx.event_system().clone(),
            tx_cuda_event: ctx.tx_cuda_event_clone(),
            tx_nixl_status: ctx.tx_nixl_status_clone(),
            raw_agent: nixl_agent.raw_agent().clone(),
            nixl_agent,
            stream: ctx.next_h2d_streams(),
            capabilities: *ctx.capabilities(),
        }
    }

    /// Register a CUDA event for polling completion.
    fn register_cuda_event(
        &self,
        cuda_event: cudarc::driver::CudaEvent,
    ) -> Result<TransferCompleteNotification> {
        let new_event = self.event_system.new_event()?;
        let handle = new_event.into_handle();
        let awaiter = self.event_system.awaiter(handle)?;
        let notification = crate::transfer::notifications::RegisterPollingNotification {
            uuid: uuid::Uuid::new_v4(),
            checker: crate::transfer::notifications::CudaEventChecker::new(cuda_event),
            event_handle: handle,
            telemetry: None,
        };
        self.tx_cuda_event
            .try_send(notification)
            .map_err(|e| anyhow!("staged: failed to enqueue CUDA event notification: {e}"))?;
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    /// Register a NIXL xfer request for polling completion.
    fn register_nixl_status(
        &self,
        xfer_req: dynamo_memory::nixl::XferRequest,
    ) -> Result<TransferCompleteNotification> {
        let new_event = self.event_system.new_event()?;
        let handle = new_event.into_handle();
        let awaiter = self.event_system.awaiter(handle)?;
        let notification = crate::transfer::notifications::RegisterPollingNotification {
            uuid: uuid::Uuid::new_v4(),
            checker: crate::transfer::notifications::NixlStatusChecker::new(
                self.raw_agent.clone(),
                xfer_req,
            ),
            event_handle: handle,
            // Staged NIXL legs (operational↔universal transform) are not part of
            // the uniform remote-search pull path we instrument.
            telemetry: None,
        };
        self.tx_nixl_status
            .try_send(notification)
            .map_err(|e| anyhow!("staged: failed to enqueue NIXL status notification: {e}"))?;
        Ok(TransferCompleteNotification::from_awaiter(awaiter))
    }

    /// Build XferDescLists, create+post a NIXL xfer request, and
    /// register a polling notification.
    ///
    /// Used for both the stage-1 NIXL leg (Read: src→bounce) and the
    /// stage-2 NIXL leg (Write: bounce→dst). The same-KvBlockLayout
    /// constraint on the leg pair means `plan_and_lower` always
    /// returns `Direct`; a `Transform` outcome is an internal bug.
    fn build_and_post_nixl_leg(
        &self,
        src: &PhysicalLayout,
        dst: &PhysicalLayout,
        src_block_ids: &[BlockId],
        dst_block_ids: &[BlockId],
        strategy: TransferStrategy,
        xfer_op: XferOp,
    ) -> Result<TransferCompleteNotification> {
        // Staged NIXL legs always use min_inner_bytes = 0: the staged
        // executor relies on same-KvBlockLayout pairs going Direct so the
        // per-leg plan_and_lower never triggers the SmallStridedCopy path.
        // PR-7.5: no benchmark lookup for staged legs — they always produce
        // DirectDma ops (no competing candidates), so the bonus has no effect.
        let leg_policy = CopyPolicy {
            min_inner_bytes: 0,
            coalesce: true,
        };
        // Staged legs are same-KvBlockLayout (asserted by the Transform
        // arm bailing below), so they always resolve to Direct. The
        // Direct path does not consume a prepared plan — pass None.
        let outcome = plan_and_lower(
            src,
            dst,
            src_block_ids,
            dst_block_ids,
            strategy,
            &self.capabilities,
            leg_policy,
            None,
            None,
            None,       // layer_range: staged NIXL legs run full-extent
            Vec::new(), // axis_slices: staged NIXL legs run full-extent
        )?;
        let ops = match outcome {
            PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
            PlanOutcome::Direct(ops) => ops,
            PlanOutcome::Transform { .. } => bail!(
                "OwnedStagedContext::build_and_post_nixl_leg: unexpected Transform outcome — \
                 staged NIXL leg expects same-KvBlockLayout pair to go Direct"
            ),
            PlanOutcome::SmallStridedCopy(_) => bail!(
                "OwnedStagedContext::build_and_post_nixl_leg: unexpected SmallStridedCopy \
                 outcome — staged NIXL leg uses min_inner_bytes = 0 and must go Direct. \
                 This is an internal routing bug."
            ),
            PlanOutcome::CudaGraphReplay { .. } => bail!(
                "OwnedStagedContext::build_and_post_nixl_leg: unexpected CudaGraphReplay \
                 outcome — staged NIXL legs use min_inner_bytes = 0 and cuda_graph_replay \
                 is disabled on NIXL routes. This is an internal routing bug."
            ),
        };

        let src_metadata = src.nixl_metadata();
        let dst_metadata = dst.nixl_metadata();
        let mut src_dl = XferDescList::new(src_metadata.mem_type())?;
        let mut dst_dl = XferDescList::new(dst_metadata.mem_type())?;
        for op in &ops {
            src_dl.add_desc(op.src_addr, op.size, src_metadata.device_id());
            dst_dl.add_desc(op.dst_addr, op.size, dst_metadata.device_id());
        }
        if matches!(
            strategy,
            TransferStrategy::NixlReadFlipped | TransferStrategy::NixlWriteFlipped
        ) {
            std::mem::swap(&mut src_dl, &mut dst_dl);
        }

        let remote_agent = match xfer_op {
            XferOp::Write => dst_metadata.agent_name(),
            XferOp::Read => src_metadata.agent_name(),
        };
        let xfer_req =
            self.nixl_agent
                .create_xfer_req(xfer_op, &src_dl, &dst_dl, remote_agent, None)?;
        let still_pending = self.nixl_agent.post_xfer_req(&xfer_req, None)?;
        if !still_pending {
            return Ok(TransferCompleteNotification::completed());
        }
        self.register_nixl_status(xfer_req)
    }
}

/// PR-6.2: dispatch a NIXL transfer that requires a kernel-side
/// transform via the Staged executor.
///
/// Cross-agent transforms cannot be done by NIXL alone — NIXL moves
/// raw bytes between agents, but the operational↔universal permute
/// is a CUDA kernel that runs only locally. The Staged executor
/// stitches the two stages together:
///
/// - **NIXL Read (pull)**: NIXL-leg pulls `src → bounce` (raw, same
///   `KvBlockLayout`); kernel-leg runs `bounce → dst` locally.
/// - **NIXL Write (push)**: kernel-leg runs `src → bounce` locally;
///   NIXL-leg pushes `bounce → dst` (raw, same `KvBlockLayout`).
///
/// The intermediate is the caller-supplied
/// [`BounceBufferInternal`]: a registered local `PhysicalLayout`
/// whose `KvBlockLayout` matches the *raw* side of the staged
/// transfer (src for Read, dst for Write).
///
/// Stage 1 is built synchronously and its notification captured.
/// The chain spawns a tokio task that awaits stage 1, then performs
/// stage 2 using an [`OwnedStagedContext`] that holds cloned
/// `NixlAgent`, polling-channel senders, event manager, and CUDA
/// stream. The returned [`TransferCompleteNotification`] resolves
/// when stage 2 completes.
///
/// **Lifecycle.** The spawned chain is fire-and-forget at spawn
/// time — the caller's only handle is the returned notification. If
/// the tokio runtime is dropped before the chain finishes, the outer
/// `velo::Event` is poisoned and the awaiter resolves with an
/// error (not a hang).
#[allow(clippy::too_many_arguments)]
fn dispatch_staged_nixl_transform(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    invocation: crate::transfer::kernel_catalog::KernelInvocation,
    block_pairs: Vec<(BlockId, BlockId)>,
    bounce: &crate::transfer::BounceBufferInternal,
    strategy: TransferStrategy,
    xfer_op: XferOp,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    use crate::transfer::StorageKind;

    let n = block_pairs.len();
    if n == 0 {
        return Ok(TransferCompleteNotification::completed());
    }

    // ──────── Validate bounce contract. ────────
    let bounce_layout = bounce.layout();
    let bounce_kv = bounce_layout.layout().block_layout();
    let bounce_all = bounce.block_ids();
    if bounce_all.len() < n {
        bail!(
            "dispatch_staged_nixl_transform: bounce has {} block ids, need at least {} \
             for this transfer",
            bounce_all.len(),
            n
        );
    }
    let bounce_block_ids: Vec<BlockId> = bounce_all[..n].to_vec();
    if !matches!(bounce_layout.location(), StorageKind::Device(_)) {
        bail!(
            "dispatch_staged_nixl_transform: bounce storage must be Device(_); got {:?} \
             (cross-agent transforms run a CUDA kernel locally)",
            bounce_layout.location()
        );
    }
    let nixl_agent_local = ctx.nixl_agent();
    if bounce_layout.nixl_metadata().agent_name() != nixl_agent_local.name() {
        bail!(
            "dispatch_staged_nixl_transform: bounce agent {:?} != local agent {:?}",
            bounce_layout.nixl_metadata().agent_name(),
            nixl_agent_local.name()
        );
    }
    let src_kv = src.layout().block_layout();
    let dst_kv = dst.layout().block_layout();
    match xfer_op {
        XferOp::Read => {
            if bounce_kv != src_kv {
                bail!(
                    "dispatch_staged_nixl_transform (Read): bounce KvBlockLayout {bounce_kv:?} \
                     must equal src KvBlockLayout {src_kv:?} (the NIXL leg is a raw copy)"
                );
            }
        }
        XferOp::Write => {
            if bounce_kv != dst_kv {
                bail!(
                    "dispatch_staged_nixl_transform (Write): bounce KvBlockLayout {bounce_kv:?} \
                     must equal dst KvBlockLayout {dst_kv:?} (the NIXL leg is a raw copy)"
                );
            }
        }
    }

    // ──────── Locality check (mirrors the Direct path). ────────
    let src_metadata = src.nixl_metadata();
    let dst_metadata = dst.nixl_metadata();
    match xfer_op {
        XferOp::Write => {
            if nixl_agent_local.name() != src_metadata.agent_name() {
                bail!(
                    "dispatch_staged_nixl_transform: Write (push) requires local src; \
                     src_agent={:?}, local_agent={:?}",
                    src_metadata.agent_name(),
                    nixl_agent_local.name()
                );
            }
        }
        XferOp::Read => {
            if nixl_agent_local.name() != dst_metadata.agent_name() {
                bail!(
                    "dispatch_staged_nixl_transform: Read (pull) requires local dst; \
                     dst_agent={:?}, local_agent={:?}",
                    dst_metadata.agent_name(),
                    nixl_agent_local.name()
                );
            }
        }
    }

    // ──────── Block-id partitioning per stage. ────────
    let src_block_ids: Vec<BlockId> = block_pairs.iter().map(|&(s, _)| s).collect();
    let dst_block_ids: Vec<BlockId> = block_pairs.iter().map(|&(_, d)| d).collect();
    let kernel_pairs: Vec<(BlockId, BlockId)> = match xfer_op {
        XferOp::Read => bounce_block_ids
            .iter()
            .zip(dst_block_ids.iter())
            .map(|(&b, &d)| (b, d))
            .collect(),
        XferOp::Write => src_block_ids
            .iter()
            .zip(bounce_block_ids.iter())
            .map(|(&s, &b)| (s, b))
            .collect(),
    };

    // PR-7.6: telemetry timing starts here — brackets the synchronous
    // portion of dispatch_staged_nixl_transform (validation, bounce checks,
    // OwnedStagedContext construction, stage-1 setup, and tokio::spawn).
    // Does NOT include the async stage-2 chain.
    let tel_t0 = std::time::Instant::now();
    // Byte volume = n block-pairs × bytes_per_block on the src side.
    let tel_bytes = n * src.layout().config().bytes_per_block();

    // ──────── Build owned context. ────────
    let staged = OwnedStagedContext::from_ctx(ctx);

    // ──────── Build stage 1 synchronously. ────────
    let stage1_notification = match xfer_op {
        XferOp::Read => staged.build_and_post_nixl_leg(
            src,
            bounce_layout,
            &src_block_ids,
            &bounce_block_ids,
            strategy,
            xfer_op,
        )?,
        XferOp::Write => {
            // Staged transforms run on per-request bounce layouts —
            // no cache reuse possible — but the dispatcher requires a
            // prepared plan. Build one fresh; the scratch pool is
            // private to this one transfer.
            let stage1_prepared = std::sync::Arc::new(PreparedTransferPlan::build_transform(
                invocation,
                src,
                bounce_layout,
            )?);
            dispatch_transform_kernel(
                &invocation,
                src,
                bounce_layout,
                &kernel_pairs,
                None, // staged NIXL transforms run full-extent
                &staged.stream,
                &stage1_prepared,
            )?;
            let cuda_event = staged.stream.record_event(None)?;
            staged.register_cuda_event(cuda_event)?
        }
    };

    // ──────── Outer notification. ────────
    let outer_event = staged.event_system.new_event()?;
    let outer_handle = outer_event.handle();
    let outer_awaiter = staged.event_system.awaiter(outer_handle)?;

    // ──────── Spawn the chain. ────────
    let runtime = ctx.tokio().clone();
    let bounce_owned = bounce_layout.clone();
    let dst_owned = dst.clone();
    let invocation_owned = invocation;

    runtime.spawn(async move {
        if let Err(e) = stage1_notification.await {
            let _ = outer_event.poison(format!("staged stage 1: {e}"));
            return;
        }

        let stage2_result: Result<()> = match xfer_op {
            XferOp::Read => {
                // Stage 2 = local kernel bounce → dst. Build a fresh
                // prepared plan for the bounce→dst pair; see sibling
                // note in the Write branch.
                let prep: Result<TransferCompleteNotification> = (|| {
                    let stage2_prepared =
                        std::sync::Arc::new(PreparedTransferPlan::build_transform(
                            invocation_owned,
                            &bounce_owned,
                            &dst_owned,
                        )?);
                    dispatch_transform_kernel(
                        &invocation_owned,
                        &bounce_owned,
                        &dst_owned,
                        &kernel_pairs,
                        None, // staged NIXL transforms run full-extent
                        &staged.stream,
                        &stage2_prepared,
                    )?;
                    let cuda_event = staged.stream.record_event(None)?;
                    staged.register_cuda_event(cuda_event)
                })();
                match prep {
                    Ok(notif) => notif.await,
                    Err(e) => Err(e),
                }
            }
            XferOp::Write => {
                // Stage 2 = NIXL push bounce → dst.
                let res = staged.build_and_post_nixl_leg(
                    &bounce_owned,
                    &dst_owned,
                    &bounce_block_ids,
                    &dst_block_ids,
                    strategy,
                    xfer_op,
                );
                match res {
                    Ok(notif) => notif.await,
                    Err(e) => Err(e),
                }
            }
        };

        match stage2_result {
            Ok(()) => {
                let _ = outer_event.trigger();
            }
            Err(e) => {
                let _ = outer_event.poison(format!("staged stage 2: {e}"));
            }
        }
    });

    let tel_latency_us = tel_t0.elapsed().as_micros() as u64;

    // PR-7.6: telemetry for the staged NIXL transform path.
    //
    // candidate_class = "StagedTransform" — this path doesn't go through
    // select_candidate; the class name documents the two-hop execution model.
    // `coalesced_bytes` is n * bytes_per_block (src side) — the kernel moves
    // exactly one block's worth of data per pair.
    // `descriptor_count` = block-pair count (one kernel invocation per group).
    if tracing::enabled!(target: "kvbm_physical::planner", tracing::Level::DEBUG) {
        let src_sig = physical_to_layout_view(src)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        let dst_sig = physical_to_layout_view(dst)
            .map(|v| v.signature())
            .map(|s| format!("{s:?}"))
            .unwrap_or_else(|e| format!("<err:{e}>"));
        tracing::debug!(
            target: "kvbm_physical::planner",
            src_layout_signature = %src_sig,
            dst_layout_signature = %dst_sig,
            descriptor_count = n,
            coalesced_bytes = tel_bytes,
            route = "nixl",
            strategy = ?strategy,
            submit_latency_us = tel_latency_us,
            candidate_class = "StagedTransform",
            "planner dispatch"
        );
    }

    Ok(TransferCompleteNotification::from_awaiter(outer_awaiter))
}

/// PR-7.3: dispatch threshold-fallback ops via `kvbm_kernels::vectorized_copy`.
///
/// All ops must share the same `size` — `plan_copy` guarantees this when it
/// emits `CopyPlan::Transform { reason: ThresholdFallback }` because the
/// outer-iteration loop uses a single `inner_bytes` for every descriptor
/// (and does NOT coalesce, which could mix sizes).
///
/// The pointer arrays must be device-accessible. We stage host-side arrays
/// then push them to device via `clone_htod`, mirroring `dispatch_transform_kernel`.
fn dispatch_small_strided_copy(ops: &[CopyOp], stream: &Arc<CudaStream>) -> Result<()> {
    use cudarc::driver::DevicePtr;

    if ops.is_empty() {
        return Ok(());
    }

    // All ops must have the same size — assert in debug builds.
    let copy_size = ops[0].size;
    debug_assert!(
        ops.iter().all(|o| o.size == copy_size),
        "dispatch_small_strided_copy: ops must all have the same size, \
         but got mixed sizes (first={copy_size})"
    );
    if copy_size == 0 {
        return Ok(());
    }

    let stream_raw = stream.cu_stream() as cudaStream_t;

    // Build host-side pointer tables then push to device.
    let mut src_ptrs: Vec<usize> = Vec::with_capacity(ops.len());
    let mut dst_ptrs: Vec<usize> = Vec::with_capacity(ops.len());
    for op in ops {
        src_ptrs.push(op.src_addr);
        dst_ptrs.push(op.dst_addr);
    }

    let src_dev = stream.clone_htod(&src_ptrs)?;
    let dst_dev = stream.clone_htod(&dst_ptrs)?;
    let (src_raw, _src_guard) = src_dev.device_ptr(stream.as_ref());
    let (dst_raw, _dst_guard) = dst_dev.device_ptr(stream.as_ref());

    let status = unsafe {
        kvbm_kernels::vectorized_copy(
            src_raw as usize as *mut *mut c_void,
            dst_raw as usize as *mut *mut c_void,
            copy_size,
            ops.len() as i32,
            stream_raw,
        )
    };
    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        bail!(
            "dispatch_small_strided_copy: vectorized_copy failed with status={status:?}, \
             copy_size={copy_size}, num_ops={}",
            ops.len()
        );
    }
    Ok(())
}

/// Group `ops` by `size` and dispatch each group via
/// `kvbm_kernels::memcpy_batch` in `BatchedWithFallback` mode (try
/// `cudaMemcpyBatchAsync` when the runtime supports it, fall back to
/// individual `cudaMemcpyAsync` otherwise).
fn dispatch_ops_grouped_by_size(ops: &[CopyOp], stream: &CudaStream) -> Result<()> {
    use std::collections::BTreeMap;

    // Stable grouping: map size -> indices into `ops`, in insertion
    // order. BTreeMap keeps deterministic ordering for testability.
    let mut by_size: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, op) in ops.iter().enumerate() {
        by_size.entry(op.size).or_default().push(i);
    }

    let stream_raw = stream.cu_stream() as cudaStream_t;
    for (size, indices) in by_size {
        if size == 0 {
            continue;
        }
        let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(indices.len());
        let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(indices.len());
        for &i in &indices {
            src_ptrs.push(ops[i].src_addr as *const c_void);
            dst_ptrs.push(ops[i].dst_addr as *mut c_void);
        }
        let status = unsafe {
            kvbm_kernels::memcpy_batch(
                src_ptrs.as_ptr(),
                dst_ptrs.as_ptr(),
                size,
                indices.len(),
                MemcpyBatchMode::BatchedWithFallback,
                stream_raw,
            )
        };
        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            return Err(anyhow!(
                "execute_planner_cuda_transfer: memcpy_batch failed with size={size}, \
                 num_copies={}, status={status:?}",
                indices.len()
            ));
        }
    }
    Ok(())
}

/// PR-7.4.1: Dispatch a CUDA graph capture/replay transfer.
///
/// **Path B** (plan §Lesson 16 choice): uses `MemcpyBatchMode::FallbackOnly`
/// during stream capture so CUDA records N individual `cudaMemcpyAsync` nodes
/// (one per op), each independently rebindable via `cuGraphExecMemcpyNodeSetParams`.
///
/// # Algorithm
///
/// 1. **Cache lookup**: check `graph_cache` for `cache_key`.
/// 2. **Cache miss — capture**:
///    a. `cuStreamBeginCapture` on `capture_stream` (a temporary stream, never used for work; capture is per-stream in RELAXED mode).
///    b. Issue `kvbm_kernels::memcpy_batch(FallbackOnly)` on `capture_stream` — N individual `cudaMemcpyAsync` calls captured as N graph nodes.
///    c. `cuStreamEndCapture` → `CUgraph`.
///    d. `cuGraphInstantiate` → `CUgraphExec`.
///    e. `cuGraphGetNodes` → collect node handles in node-list order.
///    f. Filter to MEMCPY-type nodes; verify count == ops.len().
///    g. Store in cache as `ManagedExecHandle`.
/// 3. **Address rebind** (both hit and miss paths):
///    For each (node, op) pair call `cuGraphExecMemcpyNodeSetParams` with a
///    `CUDA_MEMCPY3D` descriptor for the current op's `src_addr` / `dst_addr`.
///    Memory type is `CU_MEMORYTYPE_UNIFIED` (works for both device and
///    pinned-host pointers under CUDA unified addressing).
/// 4. **Launch**: `cuGraphLaunch(exec, work_stream.cu_stream())`.
///
/// # Stream model
///
/// Capture uses an on-the-fly temporary stream distinct from the caller's work
/// stream (CUDA requires the capture stream to be different from any stream
/// currently active in the CUDA context). Launch always uses the caller's
/// `work_stream` — the graph's nodes are not bound to any specific stream at
/// instantiation time.
///
/// # Thread safety
///
/// The `GraphCache` is `Mutex<...>`-guarded. Cache lookup and insertion each
/// hold the lock for the minimal window. Address rebind and launch are done
/// with the exec handle raw pointer acquired from the cache; the lock is NOT
/// held during these CUDA API calls — the caller is responsible for ensuring
/// that the same exec handle is not concurrently launched by two threads
/// (which is a CUDA-level constraint as well).  In kvbm-physical the planner
/// is called synchronously per-transfer, so concurrent replay of the same
/// exec is not possible today.
fn dispatch_cuda_graph_replay_planner(
    ops: &[CopyOp],
    cache_key: &GraphCacheKey,
    graph_cache: &Arc<GraphCache>,
    work_stream: &Arc<CudaStream>,
) -> Result<()> {
    use std::mem::MaybeUninit;

    if ops.is_empty() {
        return Ok(());
    }

    // ── Helper: build a CUDA_MEMCPY3D descriptor for a CopyOp ────────────────
    //
    // Both src and dst are declared as CU_MEMORYTYPE_UNIFIED. Under CUDA
    // unified addressing this covers device-allocated, pinned-host, and
    // managed memory without needing to query the pointer's actual type.
    // The copy goes through the unified address space regardless; CUDA
    // dispatches the optimal DMA path internally.
    let make_memcpy3d = |src_addr: usize, dst_addr: usize, size: usize| -> cu_sys::CUDA_MEMCPY3D {
        let mut p: cu_sys::CUDA_MEMCPY3D = unsafe { MaybeUninit::zeroed().assume_init() };
        p.srcMemoryType = cu_sys::CUmemorytype::CU_MEMORYTYPE_UNIFIED;
        p.srcDevice = src_addr as cu_sys::CUdeviceptr;
        p.dstMemoryType = cu_sys::CUmemorytype::CU_MEMORYTYPE_UNIFIED;
        p.dstDevice = dst_addr as cu_sys::CUdeviceptr;
        p.WidthInBytes = size;
        p.Height = 1;
        p.Depth = 1;
        p
    };

    // ── 1. Cache lookup ───────────────────────────────────────────────────────

    if let Some((exec, nodes)) = graph_cache.get_exec_and_nodes(cache_key) {
        // Cache hit: rebind addresses then launch.
        if nodes.len() != ops.len() {
            bail!(
                "dispatch_cuda_graph_replay_planner: cache hit but node count \
                 mismatch (cached={}, ops={}). Cache key collision or bug.",
                nodes.len(),
                ops.len()
            );
        }
        for (node, op) in nodes.iter().zip(ops.iter()) {
            let params = make_memcpy3d(op.src_addr, op.dst_addr, op.size);
            let result = unsafe {
                cu_sys::cuGraphExecMemcpyNodeSetParams(
                    exec,
                    *node,
                    &params as *const cu_sys::CUDA_MEMCPY3D,
                    std::ptr::null_mut(), // ctx: NULL uses current context
                )
            };
            if result != cu_sys::CUresult::CUDA_SUCCESS {
                bail!(
                    "dispatch_cuda_graph_replay_planner: cuGraphExecMemcpyNodeSetParams \
                     failed with result={result:?}"
                );
            }
        }
        // Launch on the caller's work stream.
        let result = unsafe { cu_sys::cuGraphLaunch(exec, work_stream.cu_stream()) };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphLaunch (cache hit) \
                 failed with result={result:?}"
            );
        }
        return Ok(());
    }

    // ── 2. Cache miss: capture the graph ─────────────────────────────────────

    // Create a temporary stream for capture. We use a brand-new CudaStream
    // via the same CudaContext that backs the work stream. cuStreamBeginCapture
    // with RELAXED mode allows the captured stream to issue operations against
    // any memory visible in the current context — required for D2D transfers
    // since src and dst may be on the same device.
    //
    // SAFETY: `cu_stream` is valid for the lifetime of `work_stream` (shared
    // context); the temporary stream is destroyed at the end of this block.

    // Build a capture stream. We need access to the context to create a new
    // stream.  cudarc doesn't expose `CudaStream::new` with a context handle
    // externally, but we can use `cu_sys::cuStreamCreate` directly.
    let mut capture_stream_raw: cu_sys::CUstream = std::ptr::null_mut();
    {
        let result = unsafe {
            cu_sys::cuStreamCreate(
                &mut capture_stream_raw as *mut cu_sys::CUstream,
                cu_sys::CUstream_flags::CU_STREAM_DEFAULT as u32,
            )
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            bail!(
                "dispatch_cuda_graph_replay_planner: cuStreamCreate failed with \
                 result={result:?}"
            );
        }
    }
    // RAII guard: destroy the capture stream regardless of what follows.
    struct StreamGuard(cu_sys::CUstream);
    impl Drop for StreamGuard {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    let _ = cu_sys::cuStreamDestroy_v2(self.0);
                }
            }
        }
    }
    let _capture_guard = StreamGuard(capture_stream_raw);

    // Begin capture in RELAXED mode so the capture stream can see all
    // device-visible memory (D2D and H2D/D2H pointers).
    {
        let result = unsafe {
            cu_sys::cuStreamBeginCapture_v2(
                capture_stream_raw,
                cu_sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            bail!(
                "dispatch_cuda_graph_replay_planner: cuStreamBeginCapture_v2 \
                 failed with result={result:?}"
            );
        }
    }

    // Issue N individual cudaMemcpyAsync calls on the capture stream via
    // FallbackOnly mode. Each call records a separate MEMCPY graph node,
    // one per op, independently rebindable at replay time.
    {
        let mut src_ptrs: Vec<*const c_void> = Vec::with_capacity(ops.len());
        let mut dst_ptrs: Vec<*mut c_void> = Vec::with_capacity(ops.len());
        // All ops must share the same size for the batch dispatch.
        // The DirectDma path groups by size upstream; CudaGraphReplay
        // is only emitted for same-size groups (descriptor_count encodes
        // the count, total_bytes = count * size_per_op for uniform ops).
        // We dispatch all ops in one memcpy_batch call using the first op's
        // size — mixed sizes within a CudaGraphReplay batch are currently
        // not supported and would be caught by the cache-key collision check.
        let first_size = ops[0].size;
        for op in ops {
            src_ptrs.push(op.src_addr as *const c_void);
            dst_ptrs.push(op.dst_addr as *mut c_void);
        }
        let status = unsafe {
            kvbm_kernels::memcpy_batch(
                src_ptrs.as_ptr(),
                dst_ptrs.as_ptr(),
                first_size,
                ops.len(),
                MemcpyBatchMode::FallbackOnly,
                capture_stream_raw as cudaStream_t,
            )
        };
        if status != cudarc::runtime::sys::cudaError::cudaSuccess {
            // End capture to clean up before bailing.
            let mut _g: cu_sys::CUgraph = std::ptr::null_mut();
            unsafe {
                let _ = cu_sys::cuStreamEndCapture(capture_stream_raw, &mut _g as *mut _);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: memcpy_batch (FallbackOnly) \
                 during capture failed with status={status:?}"
            );
        }
    }

    // End capture → CUgraph.
    let cu_graph = {
        let mut g: cu_sys::CUgraph = std::ptr::null_mut();
        let result = unsafe {
            cu_sys::cuStreamEndCapture(capture_stream_raw, &mut g as *mut cu_sys::CUgraph)
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            bail!(
                "dispatch_cuda_graph_replay_planner: cuStreamEndCapture failed \
                 with result={result:?}"
            );
        }
        g
    };

    // Instantiate the graph → CUgraphExec.
    let cu_graph_exec = {
        let mut exec: cu_sys::CUgraphExec = std::ptr::null_mut();
        let result = unsafe {
            cu_sys::cuGraphInstantiateWithFlags(
                &mut exec as *mut cu_sys::CUgraphExec,
                cu_graph,
                0u64, // no special flags
            )
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphInstantiateWithFlags \
                 failed with result={result:?}"
            );
        }
        exec
    };

    // Collect graph nodes. The node list order corresponds to the capture
    // order — which matches ops insertion order because FallbackOnly issues
    // memcpy calls sequentially.
    let nodes: Vec<cu_sys::CUgraphNode> = {
        // Two-pass: first query count, then collect.
        let mut num_nodes: usize = 0;
        let result = unsafe {
            cu_sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut num_nodes as *mut usize)
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(cu_graph_exec);
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphGetNodes (count query) \
                 failed with result={result:?}"
            );
        }
        if num_nodes != ops.len() {
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(cu_graph_exec);
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: expected {} graph nodes \
                 (one per op), got {num_nodes}. FallbackOnly capture may have \
                 coalesced ops or added extra synchronisation nodes.",
                ops.len()
            );
        }
        let mut node_vec = vec![std::ptr::null_mut::<cu_sys::CUgraphNode_st>(); num_nodes];
        let result = unsafe {
            cu_sys::cuGraphGetNodes(
                cu_graph,
                node_vec.as_mut_ptr(),
                &mut num_nodes as *mut usize,
            )
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(cu_graph_exec);
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphGetNodes (fill) \
                 failed with result={result:?}"
            );
        }
        node_vec
    };

    // Verify all nodes are MEMCPY type.
    for (i, &node) in nodes.iter().enumerate() {
        let mut node_type = cu_sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        let result = unsafe {
            cu_sys::cuGraphNodeGetType(node, &mut node_type as *mut cu_sys::CUgraphNodeType)
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(cu_graph_exec);
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphNodeGetType for node \
                 {i} failed with result={result:?}"
            );
        }
        if node_type != cu_sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY {
            unsafe {
                let _ = cu_sys::cuGraphExecDestroy(cu_graph_exec);
                let _ = cu_sys::cuGraphDestroy(cu_graph);
            }
            bail!(
                "dispatch_cuda_graph_replay_planner: node {i} is not a MEMCPY \
                 node (type={node_type:?}). cudaMemcpyAsync (FallbackOnly) must \
                 have produced a non-memcpy node — unexpected CUDA driver behaviour."
            );
        }
    }

    // Store the exec handle in the cache.
    let handle = ManagedExecHandle {
        exec: cu_graph_exec,
        graph: cu_graph,
        nodes: nodes.clone(),
    };
    graph_cache.insert(cache_key.clone(), handle);

    // ── 3 & 4. Rebind addresses then launch (first-time path) ────────────────

    for (node, op) in nodes.iter().zip(ops.iter()) {
        let params = make_memcpy3d(op.src_addr, op.dst_addr, op.size);
        let result = unsafe {
            cu_sys::cuGraphExecMemcpyNodeSetParams(
                cu_graph_exec,
                *node,
                &params as *const cu_sys::CUDA_MEMCPY3D,
                std::ptr::null_mut(),
            )
        };
        if result != cu_sys::CUresult::CUDA_SUCCESS {
            bail!(
                "dispatch_cuda_graph_replay_planner: cuGraphExecMemcpyNodeSetParams \
                 (first launch) failed with result={result:?}"
            );
        }
    }

    let result = unsafe { cu_sys::cuGraphLaunch(cu_graph_exec, work_stream.cu_stream()) };
    if result != cu_sys::CUresult::CUDA_SUCCESS {
        bail!(
            "dispatch_cuda_graph_replay_planner: cuGraphLaunch (first launch) \
             failed with result={result:?}"
        );
    }

    Ok(())
}

#[cfg(all(test, feature = "testing-kvbm"))]
mod tests {
    use super::*;

    // ──────────── validate_planner_block_ids ────────────

    /// Equal-length non-empty block-id lists are a valid Proceed
    /// case — the structural validator says nothing about strategy
    /// or layouts.
    #[test]
    fn block_ids_validator_passes_equal_length() {
        let r = validate_planner_block_ids(&[0, 1, 2], &[0, 1, 2]);
        assert!(matches!(r, Ok(PlannerInputs::Proceed)));
    }

    /// Empty block lists short-circuit to Noop so the caller can
    /// resolve a "completed" notification without dispatching.
    #[test]
    fn block_ids_validator_returns_noop_on_empty_list() {
        let r = validate_planner_block_ids(&[], &[]);
        assert!(matches!(r, Ok(PlannerInputs::Noop)));
    }

    /// Mismatched block-id list lengths are a structural error.
    #[test]
    fn block_ids_validator_rejects_length_mismatch() {
        let r = validate_planner_block_ids(&[0, 1], &[0]);
        assert!(r.is_err());
    }

    // ──────────── validate_cuda_planner_entry ────────────

    /// Same operational layout + CudaAsync strategy passes.
    #[test]
    fn cuda_entry_passes_same_operational_layout() {
        for s in [
            TransferStrategy::CudaAsyncH2D,
            TransferStrategy::CudaAsyncD2H,
            TransferStrategy::CudaAsyncD2D,
        ] {
            let r = validate_cuda_planner_entry(
                s,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalNHD,
            );
            assert!(r.is_ok(), "strategy {s:?} expected to pass");
        }
    }

    /// PR-6.1: layout-pair compatibility is no longer enforced at the
    /// Cuda entry guard — the kernel catalog dispatches transforms for
    /// pairs it knows about, and surfaces a precise no-matching-kernel
    /// error from `build_transform_invocation` for pairs it doesn't
    /// (e.g. NHD↔HND, which lands in PR-6.3). The entry guard now only
    /// rejects on strategy mismatch.
    #[test]
    fn cuda_entry_accepts_transform_pairs_now_handled_by_catalog() {
        // Operational ↔ Universal — PR-6.1 catalog has both directions.
        assert!(
            validate_cuda_planner_entry(
                TransferStrategy::CudaAsyncD2D,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::Universal,
            )
            .is_ok()
        );
        // NHD ↔ HND — catalog miss (PR-6.3), but the entry guard still
        // accepts; the precise error comes from the catalog at lower
        // time.
        assert!(
            validate_cuda_planner_entry(
                TransferStrategy::CudaAsyncD2D,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalHND,
            )
            .is_ok()
        );
    }

    /// Non-CudaAsync strategies routed into the Cuda entrypoint
    /// are an internal-routing bug; reject explicitly.
    #[test]
    fn cuda_entry_rejects_non_cuda_strategies() {
        for s in [
            TransferStrategy::NixlRead,
            TransferStrategy::NixlWrite,
            TransferStrategy::NixlReadFlipped,
            TransferStrategy::NixlWriteFlipped,
            TransferStrategy::Memcpy,
            TransferStrategy::Invalid,
        ] {
            let r = validate_cuda_planner_entry(
                s,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalNHD,
            );
            assert!(r.is_err(), "strategy {s:?} expected to be rejected");
        }
    }

    // ──────────── validate_nixl_planner_entry ────────────

    /// Same operational layout + every Nixl variant passes.
    #[test]
    fn nixl_entry_passes_same_operational_layout() {
        for s in [
            TransferStrategy::NixlRead,
            TransferStrategy::NixlWrite,
            TransferStrategy::NixlReadFlipped,
            TransferStrategy::NixlWriteFlipped,
        ] {
            let r = validate_nixl_planner_entry(
                s,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalNHD,
            );
            assert!(r.is_ok(), "strategy {s:?} expected to pass");
        }
    }

    /// PR-6.2: layout-pair compatibility is no longer enforced at the
    /// NIXL entry guard — the Staged executor handles
    /// `requires_transform=true` pairs by stitching a local kernel
    /// between the NIXL leg and the placement leg. Pairs the catalog
    /// doesn't cover surface a precise error from
    /// `build_transform_invocation` instead.
    #[test]
    fn nixl_entry_accepts_transform_pairs() {
        // Operational ↔ Universal — PR-6.1 catalog has both directions.
        assert!(
            validate_nixl_planner_entry(
                TransferStrategy::NixlReadFlipped,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::Universal,
            )
            .is_ok()
        );
        // NHD ↔ HND — catalog miss until PR-6.3, but the entry guard
        // accepts; the precise error comes from the catalog at lower
        // time.
        assert!(
            validate_nixl_planner_entry(
                TransferStrategy::NixlReadFlipped,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalHND,
            )
            .is_ok()
        );
    }

    /// Non-NIXL strategies routed into the NIXL entrypoint are an
    /// internal-routing bug; reject explicitly.
    #[test]
    fn nixl_entry_rejects_non_nixl_strategies() {
        for s in [
            TransferStrategy::CudaAsyncH2D,
            TransferStrategy::CudaAsyncD2H,
            TransferStrategy::CudaAsyncD2D,
            TransferStrategy::Memcpy,
            TransferStrategy::Invalid,
        ] {
            let r = validate_nixl_planner_entry(
                s,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::OperationalNHD,
            );
            assert!(r.is_err(), "strategy {s:?} expected to be rejected");
        }
    }

    // ──────────── PR-7.7.1: reject_heterogeneous_layout_views ────────────
    //
    // Tests operate on synthetic `LayoutView::full(...)` fixtures so no
    // `PhysicalLayout` or real CUDA/NIXL infrastructure is needed.
    // The guard logic lives in `reject_heterogeneous_layout_views`; the
    // entrypoint wrappers (`execute_planner_{cuda,nixl}_transfer`) add the
    // `physical_to_layout_view` projection and call it with the result.

    use dynamo_memory::StorageKind;
    use kvbm_common::{KvDim, KvDimLayout, KvDimStrides};

    /// Build a minimal 2-axis homogeneous `LayoutView` with all axes = `kind`.
    fn homogeneous_view(kind: StorageKind) -> crate::layout::LayoutView {
        let layout = KvDimLayout::new(vec![KvDim::Block, KvDim::Page], vec![4, 8]).unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![8 * 64, 64], 2).unwrap();
        let kinds = vec![kind; 2];
        crate::layout::LayoutView::full(layout, strides, vec![0x1000], None, kinds).unwrap()
    }

    /// Build a 2-axis heterogeneous `LayoutView`: axis 0 = Device(0), axis 1 = System.
    fn heterogeneous_view() -> crate::layout::LayoutView {
        let layout = KvDimLayout::new(vec![KvDim::Block, KvDim::Page], vec![4, 8]).unwrap();
        let strides = KvDimStrides::from_byte_strides(vec![8 * 64, 64], 2).unwrap();
        let kinds = vec![StorageKind::Device(0), StorageKind::System];
        crate::layout::LayoutView::full(layout, strides, vec![0x2000], None, kinds).unwrap()
    }

    /// Homogeneous src + homogeneous dst passes — this is the production path.
    #[test]
    fn heterogeneous_guard_accepts_both_homogeneous() {
        let src = homogeneous_view(StorageKind::Device(0));
        let dst = homogeneous_view(StorageKind::Device(0));
        assert!(
            reject_heterogeneous_layout_views(&src, &dst, "test_entry").is_ok(),
            "homogeneous src + dst must pass the guard"
        );
    }

    /// Heterogeneous src is rejected; error message must name the entry label,
    /// "heterogeneous", and "PR-7.7.2" so callers know where the fix lands.
    #[test]
    fn heterogeneous_guard_rejects_heterogeneous_src() {
        let src = heterogeneous_view();
        let dst = homogeneous_view(StorageKind::Device(0));
        let err = reject_heterogeneous_layout_views(&src, &dst, "test_cuda_entry")
            .expect_err("heterogeneous src must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("heterogeneous"),
            "error must mention 'heterogeneous': {msg}"
        );
        assert!(
            msg.contains("PR-7.7.2"),
            "error must mention 'PR-7.7.2' so callers know the future home: {msg}"
        );
        assert!(
            msg.contains("src"),
            "error must identify 'src' as the offending side: {msg}"
        );
    }

    /// Heterogeneous dst is rejected with a matching error.
    #[test]
    fn heterogeneous_guard_rejects_heterogeneous_dst() {
        let src = homogeneous_view(StorageKind::System);
        let dst = heterogeneous_view();
        let err = reject_heterogeneous_layout_views(&src, &dst, "test_nixl_entry")
            .expect_err("heterogeneous dst must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("heterogeneous"),
            "error must mention 'heterogeneous': {msg}"
        );
        assert!(
            msg.contains("PR-7.7.2"),
            "error must mention 'PR-7.7.2': {msg}"
        );
        assert!(
            msg.contains("dst"),
            "error must identify 'dst' as the offending side: {msg}"
        );
    }

    /// Both src and dst heterogeneous: src is checked first, so the error
    /// identifies the src side.
    #[test]
    fn heterogeneous_guard_rejects_both_heterogeneous_reports_src_first() {
        let src = heterogeneous_view();
        let dst = heterogeneous_view();
        let err = reject_heterogeneous_layout_views(&src, &dst, "test_entry")
            .expect_err("both-heterogeneous must be rejected");
        let msg = format!("{err}");
        assert!(
            msg.contains("src"),
            "src-first checking: error must identify 'src': {msg}"
        );
    }

    /// FC-like all-Device view is homogeneous — the common GPU-only production
    /// layout must never trigger the guard.
    #[test]
    fn heterogeneous_guard_accepts_all_device_homogeneous_view() {
        let src = homogeneous_view(StorageKind::Device(0));
        let dst = homogeneous_view(StorageKind::Device(0));
        assert!(reject_heterogeneous_layout_views(&src, &dst, "test_entry").is_ok());
    }

    /// LS-like all-System view is homogeneous — pinned-host layouts used in
    /// system-memory transfers must also pass.
    #[test]
    fn heterogeneous_guard_accepts_all_system_homogeneous_view() {
        let src = homogeneous_view(StorageKind::System);
        let dst = homogeneous_view(StorageKind::System);
        assert!(reject_heterogeneous_layout_views(&src, &dst, "test_entry").is_ok());
    }
}
