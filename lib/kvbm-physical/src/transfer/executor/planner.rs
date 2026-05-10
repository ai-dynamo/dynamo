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
use cudarc::runtime::sys::cudaStream_t;
use dynamo_memory::nixl::{XferDescList, XferOp};
use kvbm_kernels::MemcpyBatchMode;

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::layout::KvBlockLayout;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::lower::{
    Candidate, lower_to_candidates, physical_to_layout_view, select_candidate,
};
use crate::transfer::plan::{
    AnnotatedLayout, CopyOp, CopyPlan, CopyPolicy, TransferSelection, plan_copy,
};

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
pub(crate) fn execute_planner_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    validate_cuda_planner_entry(strategy, src.layout().block_layout(), dst.layout().block_layout())?;

    let outcome = plan_and_lower(src, dst, src_block_ids, dst_block_ids)?;

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

    match outcome {
        PlanOutcome::Empty => unreachable!("handled above"),
        PlanOutcome::Direct(ops) => {
            // Group by `size` so each `memcpy_batch` call has a
            // uniform `size_per_copy`. Common case is one group;
            // heterogeneous sizes get one batch per size.
            dispatch_ops_grouped_by_size(&ops, stream.as_ref())?;
        }
        #[cfg(feature = "permute_kernels")]
        PlanOutcome::Transform {
            invocation,
            block_pairs,
        } => {
            dispatch_transform_kernel(&invocation, src, dst, &block_pairs, &stream)?;
        }
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
pub(crate) fn execute_planner_nixl_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    #[cfg_attr(not(feature = "permute_kernels"), allow(unused_variables))]
    bounce_buffer: Option<&crate::transfer::BounceBufferInternal>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    validate_nixl_planner_entry(strategy, src.layout().block_layout(), dst.layout().block_layout())?;
    let xfer_op = match strategy {
        TransferStrategy::NixlRead | TransferStrategy::NixlReadFlipped => XferOp::Read,
        TransferStrategy::NixlWrite | TransferStrategy::NixlWriteFlipped => XferOp::Write,
        // unreachable: validate_nixl_planner_entry already rejected
        // non-NIXL strategies above. Kept as a defence-in-depth bail.
        other => bail!("execute_planner_nixl_transfer: strategy {other:?} not a NIXL strategy"),
    };

    let ops = match plan_and_lower(src, dst, src_block_ids, dst_block_ids)? {
        PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
        PlanOutcome::Direct(ops) => ops,
        #[cfg(feature = "permute_kernels")]
        PlanOutcome::Transform {
            invocation,
            block_pairs,
        } => {
            let bounce = bounce_buffer.ok_or_else(|| anyhow!(
                "execute_planner_nixl_transfer: cross-agent transform requires \
                 TransferOptions::bounce_buffer to be set (the Staged executor pulls \
                 raw bytes through a local intermediate, runs the kernel, then places). \
                 Pass a registered local-Device PhysicalLayout via TransferOptions::bounce_buffer."
            ))?;
            return dispatch_staged_nixl_transform(
                src, dst, invocation, block_pairs, bounce, strategy, xfer_op, ctx,
            );
        }
    };

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
    // descriptor count equals the op count.
    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;
    for op in &ops {
        src_dl.add_desc(op.src_addr, op.size, src_device_id);
        dst_dl.add_desc(op.dst_addr, op.size, dst_device_id);
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
    let xfer_req = nixl_agent.create_xfer_req(xfer_op, &src_dl, &dst_dl, remote_agent, None)?;
    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;
    if still_pending {
        Ok(ctx.register_nixl_status(xfer_req))
    } else {
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
    #[cfg(feature = "permute_kernels")]
    Transform {
        invocation: crate::transfer::kernel_catalog::KernelInvocation,
        block_pairs: Vec<(BlockId, BlockId)>,
    },
}

/// Shared "validate → project → plan → lower" pipeline used by both
/// the CUDA and NIXL planner-path entrypoints.
///
/// Strategy and layout-compatibility checks are NOT done here — each
/// entrypoint enforces its own per-family contract via
/// [`validate_cuda_planner_entry`] / [`validate_nixl_planner_entry`]
/// before calling this. This stage handles only the structural
/// invariants of the (src, dst, block_ids) triple.
fn plan_and_lower(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
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
    #[cfg(feature = "permute_kernels")]
    {
        let src_kv = src.layout().block_layout();
        let dst_kv = dst.layout().block_layout();
        if src_kv.requires_transform(&dst_kv) {
            let invocation = build_transform_invocation(src, dst)?;
            return Ok(PlanOutcome::Transform {
                invocation,
                block_pairs,
            });
        }
    }

    let src_view = physical_to_layout_view(src)?;
    let dst_view = physical_to_layout_view(dst)?;
    let src_al = AnnotatedLayout::from_view(&src_view)?;
    let dst_al = AnnotatedLayout::from_view(&dst_view)?;

    // `block_pairs` was already built above; `plan_copy` consumes
    // a `usize`-typed pair list.
    let plan_block_pairs: Vec<(usize, usize)> = block_pairs.iter().map(|&(s, d)| (s, d)).collect();
    let selection = TransferSelection::full(plan_block_pairs);
    // PR-6.1 keeps `min_inner_bytes = 0`. Restoring the 4 KiB default
    // would be unsafe — `plan_copy` emits `CopyPlan::Transform` when
    // `inner_bytes < min_inner_bytes` regardless of whether the
    // layouts are semantically identical, and the catalog returns
    // `None` for same-layout pairs (no transform needed). Result:
    // small same-layout copies would become planner errors. PR-7
    // splits "semantic transform required" from "threshold fallback"
    // inside `plan_copy` and registers a small-strided-copy candidate
    // so the threshold can be reinstated.
    let policy = CopyPolicy {
        min_inner_bytes: 0,
        coalesce: true,
    };

    let plan = plan_copy(&src_al, &dst_al, &selection, &policy)?;
    match plan {
        CopyPlan::Direct(ops) if ops.is_empty() => Ok(PlanOutcome::Empty),
        CopyPlan::Direct(_) => {
            let candidates = lower_to_candidates(plan)?;
            let chosen = select_candidate(&candidates)?;
            match chosen {
                Candidate::DirectDma { ops } => Ok(PlanOutcome::Direct(ops.clone())),
                other => bail!(
                    "plan_and_lower: select_candidate returned non-DirectDma for a \
                     CopyPlan::Direct: {other:?}"
                ),
            }
        }
        // Same-KvBlockLayout pairs only reach `plan_copy`; if it
        // somehow returns Transform anyway (e.g. future
        // `min_inner_bytes` policy), surface it as an unhandled
        // case until PR-7's threshold-fallback machinery lands.
        CopyPlan::Transform { .. } => bail!(
            "plan_and_lower: plan_copy emitted Transform for a same-KvBlockLayout \
             pair — PR-6.1's catalog only dispatches when KvBlockLayout differs \
             (handled before plan_copy). PR-7 will introduce threshold-fallback \
             for small-tail same-layout copies."
        ),
        CopyPlan::Staged { .. } => bail!(
            "plan_and_lower: CopyPlan::Staged is reserved (NIXL transforms in PR-6.2)"
        ),
    }
}

/// Resolve a kernel for a `CopyPlan::Transform` through the catalog
/// and build the launch parameters. Errors precisely when no kernel
/// covers the (src_kv, dst_kv, dtype) triple.
#[cfg(feature = "permute_kernels")]
fn build_transform_invocation(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
) -> Result<crate::transfer::kernel_catalog::KernelInvocation> {
    use crate::transfer::kernel_catalog::{KernelInvocation, KernelKind, match_kernel, to_kernel_block_layout};

    let src_kv = src.layout().block_layout();
    let dst_kv = dst.layout().block_layout();
    let cfg = src.layout().config();
    if cfg != dst.layout().config() {
        bail!(
            "build_transform_invocation: src.config != dst.config — the catalog only \
             dispatches transforms between same-shape layouts (got src.num_blocks={}, \
             src.num_layers={}, src.outer_dim={}, src.page_size={}, src.inner_dim={}, \
             dst.num_blocks={}, dst.num_layers={}, dst.outer_dim={}, dst.page_size={}, \
             dst.inner_dim={})",
            cfg.num_blocks, cfg.num_layers, cfg.outer_dim, cfg.page_size, cfg.inner_dim,
            dst.layout().config().num_blocks, dst.layout().config().num_layers,
            dst.layout().config().outer_dim, dst.layout().config().page_size,
            dst.layout().config().inner_dim,
        );
    }
    let dtype = cfg.dtype.ok_or_else(|| {
        anyhow!(
            "build_transform_invocation: cfg.dtype is required for transform dispatch \
             (catalog dispatches on real TensorDataType, not byte width)"
        )
    })?;
    let nh = cfg.num_heads.ok_or_else(|| {
        anyhow!(
            "build_transform_invocation: cfg.num_heads is required for transform dispatch"
        )
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
             dst={dst_kv:?}, dtype={dtype:?}). UniversalPP support is pending."
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
#[cfg_attr(feature = "permute_kernels", allow(unused_variables))]
pub(crate) fn validate_cuda_planner_entry(
    strategy: TransferStrategy,
    src_block_layout: KvBlockLayout,
    dst_block_layout: KvBlockLayout,
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
    // Without the `permute_kernels` feature, the catalog isn't compiled
    // in, so transforms can't be dispatched — keep the conservative
    // guard active in that build.
    #[cfg(not(feature = "permute_kernels"))]
    if src_block_layout.requires_transform(&dst_block_layout) {
        bail!(
            "validate_cuda_planner_entry: src ({src_block_layout:?}) and dst \
             ({dst_block_layout:?}) require a kernel-side transform, but the \
             `permute_kernels` feature is not enabled in kvbm-physical. Drop \
             use_planner=true or build with --features permute_kernels."
        );
    }
    Ok(())
}

/// Per-entrypoint guard for [`execute_planner_nixl_transfer`].
///
/// Rejects strategies outside the `Nixl{Read,Write,ReadFlipped,
/// WriteFlipped}` family. When the `permute_kernels` feature is
/// disabled the layout-pair compatibility check stays active (no
/// catalog → no Staged dispatch); with `permute_kernels` on, the
/// Staged executor handles `requires_transform=true` pairs and the
/// guard collapses to strategy-only.
#[cfg_attr(feature = "permute_kernels", allow(unused_variables))]
pub(crate) fn validate_nixl_planner_entry(
    strategy: TransferStrategy,
    src_block_layout: KvBlockLayout,
    dst_block_layout: KvBlockLayout,
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
    #[cfg(not(feature = "permute_kernels"))]
    if src_block_layout.requires_transform(&dst_block_layout) {
        bail!(
            "validate_nixl_planner_entry: src ({src_block_layout:?}) and dst \
             ({dst_block_layout:?}) require a kernel-side transform, but the \
             `permute_kernels` feature is not enabled in kvbm-physical. Drop \
             use_planner=true or build with --features permute_kernels."
        );
    }
    Ok(())
}

/// Dispatch a `KernelInvocation` resolved by the catalog: build the
/// per-side pointer arrays from `(src, dst, block_pairs)`, push them
/// to device memory, and launch the matching `kvbm-kernels` FFI
/// entrypoint.
///
/// Pointer-array shape per kernel:
/// - **operational side**: one pointer per (block, layer, outer)
///   chunk; total length `num_blocks_to_transfer * num_layers * outer_dim`.
///   Walked via `Layout::memory_region(block, layer, outer)`.
/// - **universal side**: one pointer per block; total length
///   `num_blocks_to_transfer`. Computed as
///   `single_buffer_base + block_id * bytes_per_block` (universal
///   layouts are FC-only, so a single allocation backs all blocks).
#[cfg(feature = "permute_kernels")]
fn dispatch_transform_kernel(
    invocation: &crate::transfer::kernel_catalog::KernelInvocation,
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    block_pairs: &[(BlockId, BlockId)],
    stream: &Arc<CudaStream>,
) -> Result<()> {
    use cudarc::driver::DevicePtr;

    use crate::transfer::kernel_catalog::KernelKind;

    let stream_raw = stream.cu_stream() as cudaStream_t;
    let nl = invocation.num_layers;
    let no = invocation.outer_dim;
    let nt = invocation.page_size;
    let nh = invocation.num_heads;
    let hd = invocation.head_dim;

    // Operational↔operational transpose: both sides are operational
    // block stacks, so the universal-side scaffolding below doesn't
    // apply. Build per-side chunk pointer tables and launch directly.
    if matches!(invocation.kind, KernelKind::NhdHndTranspose) {
        return dispatch_nhd_hnd_transpose_kernel(invocation, src, dst, block_pairs, stream);
    }

    // Determine which side is operational vs universal.
    let (op_layout, op_block_id_of, univ_layout, univ_block_id_of): (
        &PhysicalLayout,
        Box<dyn Fn(&(BlockId, BlockId)) -> BlockId>,
        &PhysicalLayout,
        Box<dyn Fn(&(BlockId, BlockId)) -> BlockId>,
    ) = match invocation.kind {
        KernelKind::UniversalFromBlock => (
            src,
            Box::new(|p: &(BlockId, BlockId)| p.0),
            dst,
            Box::new(|p: &(BlockId, BlockId)| p.1),
        ),
        KernelKind::BlockFromUniversal => (
            dst,
            Box::new(|p: &(BlockId, BlockId)| p.1),
            src,
            Box::new(|p: &(BlockId, BlockId)| p.0),
        ),
        KernelKind::NhdHndTranspose => unreachable!("handled above"),
    };

    // Operational pointer table: nb × nl × no entries, packed as
    // [block_idx][layer*no + outer]. The kernel iterates this table
    // in the same order.
    let mut op_ptrs: Vec<usize> = Vec::with_capacity(block_pairs.len() * nl * no);
    for pair in block_pairs {
        let block_id = op_block_id_of(pair);
        for layer in 0..nl {
            for outer in 0..no {
                let region = op_layout.layout().memory_region(block_id, layer, outer)
                    .map_err(|e| anyhow!(
                        "dispatch_transform_kernel: failed to read operational chunk \
                         (block={block_id}, layer={layer}, outer={outer}): {e:?}"
                    ))?;
                op_ptrs.push(region.addr());
            }
        }
    }

    // Universal pointer table: one base per block. Universal layouts
    // are FC-only (LayerSeparate rejects them at build), so all
    // blocks live in a single allocation and per-block bases are
    // `single_buffer_base + block_id * bytes_per_block`.
    let univ_buffers = univ_layout.layout().memory_regions();
    if univ_buffers.len() != 1 {
        bail!(
            "dispatch_transform_kernel: universal side expects 1 Buffer, got {}",
            univ_buffers.len()
        );
    }
    let univ_base = univ_buffers[0].addr();
    let bytes_per_block = univ_layout.layout().config().bytes_per_block();
    let mut univ_ptrs: Vec<usize> = Vec::with_capacity(block_pairs.len());
    for pair in block_pairs {
        let block_id = univ_block_id_of(pair);
        univ_ptrs.push(univ_base + block_id * bytes_per_block);
    }

    // Push pointer tables to device memory. The kernels expect
    // device-accessible pointer arrays.
    let op_dev = stream.clone_htod(&op_ptrs)?;
    let univ_dev = stream.clone_htod(&univ_ptrs)?;
    let (op_ptr_dev_raw, _op_guard) = op_dev.device_ptr(stream.as_ref());
    let (univ_ptr_dev_raw, _univ_guard) = univ_dev.device_ptr(stream.as_ref());

    let status = match invocation.kind {
        KernelKind::UniversalFromBlock => {
            let universal_ptrs = univ_ptr_dev_raw as usize as *const *mut c_void;
            let block_ptrs = op_ptr_dev_raw as usize as *const *const c_void;
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
                    invocation.dtype,
                    invocation.block_layout,
                    stream_raw,
                )
            }
        }
        KernelKind::BlockFromUniversal => {
            let universal_ptrs = univ_ptr_dev_raw as usize as *const *const c_void;
            let block_ptrs = op_ptr_dev_raw as usize as *const *mut c_void;
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
                    invocation.dtype,
                    invocation.block_layout,
                    stream_raw,
                )
            }
        }
        KernelKind::NhdHndTranspose => unreachable!("handled above"),
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

/// PR-6.3: dispatch the operational↔operational (NHD↔HND) transpose
/// kernel.
///
/// Both sides are operational block stacks, so unlike
/// [`dispatch_transform_kernel`]'s universal-side scaffolding, both
/// pointer tables are built via `Layout::memory_region(b, l, o)` —
/// shaped `[num_blocks_to_transfer × num_layers × outer_dim]` each.
/// The direction is carried on `KernelInvocation::block_layout`
/// (kernel's `src_layout` template parameter).
#[cfg(feature = "permute_kernels")]
fn dispatch_nhd_hnd_transpose_kernel(
    invocation: &crate::transfer::kernel_catalog::KernelInvocation,
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    block_pairs: &[(BlockId, BlockId)],
    stream: &Arc<CudaStream>,
) -> Result<()> {
    use cudarc::driver::DevicePtr;

    let stream_raw = stream.cu_stream() as cudaStream_t;
    let nl = invocation.num_layers;
    let no = invocation.outer_dim;
    let nt = invocation.page_size;
    let nh = invocation.num_heads;
    let hd = invocation.head_dim;

    let build_table = |layout: &PhysicalLayout, block_id_of: fn(&(BlockId, BlockId)) -> BlockId|
        -> Result<Vec<usize>> {
        let mut table: Vec<usize> = Vec::with_capacity(block_pairs.len() * nl * no);
        for pair in block_pairs {
            let block_id = block_id_of(pair);
            for layer in 0..nl {
                for outer in 0..no {
                    let region = layout.layout().memory_region(block_id, layer, outer)
                        .map_err(|e| anyhow!(
                            "dispatch_nhd_hnd_transpose_kernel: failed to read chunk \
                             (block={block_id}, layer={layer}, outer={outer}): {e:?}"
                        ))?;
                    table.push(region.addr());
                }
            }
        }
        Ok(table)
    };
    let src_ptrs = build_table(src, |p| p.0)?;
    let dst_ptrs = build_table(dst, |p| p.1)?;

    let src_dev = stream.clone_htod(&src_ptrs)?;
    let dst_dev = stream.clone_htod(&dst_ptrs)?;
    let (src_ptr_dev_raw, _src_guard) = src_dev.device_ptr(stream.as_ref());
    let (dst_ptr_dev_raw, _dst_guard) = dst_dev.device_ptr(stream.as_ref());

    let status = unsafe {
        kvbm_kernels::nhd_hnd_transpose(
            src_ptr_dev_raw as usize as *const *const c_void,
            dst_ptr_dev_raw as usize as *const *mut c_void,
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
    };
    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        bail!(
            "dispatch_nhd_hnd_transpose_kernel: kernel launch failed with status={status:?}, \
             num_blocks_to_transfer={}, src_layout={:?}",
            block_pairs.len(),
            invocation.block_layout,
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
#[cfg(feature = "permute_kernels")]
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
}

#[cfg(feature = "permute_kernels")]
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
        let outcome = plan_and_lower(src, dst, src_block_ids, dst_block_ids)?;
        let ops = match outcome {
            PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
            PlanOutcome::Direct(ops) => ops,
            PlanOutcome::Transform { .. } => bail!(
                "OwnedStagedContext::build_and_post_nixl_leg: unexpected Transform outcome — \
                 staged NIXL leg expects same-KvBlockLayout pair to go Direct"
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
#[cfg(feature = "permute_kernels")]
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
            dispatch_transform_kernel(
                &invocation,
                src,
                bounce_layout,
                &kernel_pairs,
                &staged.stream,
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
                // Stage 2 = local kernel bounce → dst.
                let prep: Result<TransferCompleteNotification> = (|| {
                    dispatch_transform_kernel(
                        &invocation_owned,
                        &bounce_owned,
                        &dst_owned,
                        &kernel_pairs,
                        &staged.stream,
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

    Ok(TransferCompleteNotification::from_awaiter(outer_awaiter))
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
                KvBlockLayout::UniversalTP,
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
    /// NIXL entry guard when `permute_kernels` is on — the Staged
    /// executor handles `requires_transform=true` pairs by stitching
    /// a local kernel between the NIXL leg and the placement leg.
    /// Pairs the catalog doesn't cover surface a precise error from
    /// `build_transform_invocation` instead.
    #[cfg(feature = "permute_kernels")]
    #[test]
    fn nixl_entry_accepts_transform_pairs_with_permute_kernels() {
        // Operational ↔ Universal — PR-6.1 catalog has both directions.
        assert!(
            validate_nixl_planner_entry(
                TransferStrategy::NixlReadFlipped,
                KvBlockLayout::OperationalNHD,
                KvBlockLayout::UniversalTP,
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
}
