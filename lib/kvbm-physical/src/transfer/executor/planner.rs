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
        PlanOutcome::Transform { .. } => bail!(
            "execute_planner_nixl_transfer: cross-agent transforms are not yet \
             wired (PR-6.2 lands the Staged executor that pulls raw bytes through \
             a local intermediate, runs the kernel, then places). Drop \
             use_planner=true for this transfer."
        ),
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
             dst={dst_kv:?}, dtype={dtype:?}). NHD↔HND lands in PR-6.3; \
             UniversalPP support is also pending."
        )
    })?;

    // The kernel template selects NHD vs HND from the *operational*
    // side of the pair.
    let operational_kv = match kind {
        KernelKind::UniversalFromBlock => src_kv, // operational → universal
        KernelKind::BlockFromUniversal => dst_kv, // universal → operational
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
/// WriteFlipped}` family and layout pairs that would require a
/// kernel-side transform (PR-6.2 drops the layout check once the
/// Staged executor handles cross-agent transforms).
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
    if src_block_layout.requires_transform(&dst_block_layout) {
        bail!(
            "validate_nixl_planner_entry: src ({src_block_layout:?}) and dst \
             ({dst_block_layout:?}) require a kernel-side transform, which is \
             not yet wired in the NIXL planner path (PR-6.2 lands the Staged \
             executor). Drop use_planner=true for this transfer."
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

    /// NHD ↔ HND is a semantic transform — rejected until PR-6.2
    /// wires the Staged executor.
    #[test]
    fn nixl_entry_rejects_nhd_hnd() {
        let r = validate_nixl_planner_entry(
            TransferStrategy::NixlReadFlipped,
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalHND,
        );
        assert!(r.is_err());
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
