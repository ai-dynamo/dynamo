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
use crate::transfer::plan::{AnnotatedLayout, CopyOp, CopyPolicy, TransferSelection, plan_copy};

/// Dispatch a CudaAsync transfer through the stride-aware planner.
///
/// Returns the same kind of [`TransferCompleteNotification`] the
/// legacy `execute_cuda_transfer` returns, or an `Err` when the
/// transfer cannot be safely handled by the PR-5 planner path.
///
/// Bails (no fallback) when:
/// - the strategy is not one of `CudaAsync{H2D, D2H, D2D}`;
/// - the src/dst block-id lists have unequal length;
/// - `src.block_layout()` and `dst.block_layout()` would require a
///   semantic transformation (NHD↔HND, ↔Universal, etc.). The
///   planner-side projection collapses the per-token NHD/HND
///   substructure into a single trailing `Payload` axis, so a
///   raw-copy without going through the kernel catalog would silently
///   transpose-corrupt the data. PR-6 wires the kernel candidates and
///   removes this gate.
pub(crate) fn execute_planner_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    let ops = match plan_and_lower(src, dst, src_block_ids, dst_block_ids, strategy)? {
        PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
        PlanOutcome::Ops(ops) => ops,
    };

    // Acquire a stream (caller-provided or pool-acquired). Direction
    // determines which stream pool we draw from.
    let caller_manages_sync = cuda_stream.is_some();
    let stream = if let Some(s) = cuda_stream {
        s
    } else {
        match strategy {
            TransferStrategy::CudaAsyncD2H => ctx.next_d2h_streams(),
            _ => ctx.next_h2d_streams(),
        }
    };

    // Dispatch ops to CUDA. Group by `size` so each `memcpy_batch`
    // call has a uniform `size_per_copy`. The common case is one
    // group (uniform op size after coalescing); if the planner emits
    // heterogeneous sizes (e.g. partial coalescing across non-
    // contiguous block runs), we issue one batch per size.
    dispatch_ops_grouped_by_size(&ops, stream.as_ref())?;

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
/// - the strategy is not one of `Nixl{Read, Write, ReadFlipped, WriteFlipped}`;
/// - locality is wrong for the chosen op (Write requires src local;
///   Read requires dst local — same invariants the legacy executor
///   asserts);
/// - any condition rejected by [`validate_planner_inputs`] or
///   [`plan_and_lower`].
pub(crate) fn execute_planner_nixl_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    let xfer_op = match strategy {
        TransferStrategy::NixlRead | TransferStrategy::NixlReadFlipped => XferOp::Read,
        TransferStrategy::NixlWrite | TransferStrategy::NixlWriteFlipped => XferOp::Write,
        other => bail!("execute_planner_nixl_transfer: strategy {other:?} not a NIXL strategy"),
    };

    let ops = match plan_and_lower(src, dst, src_block_ids, dst_block_ids, strategy)? {
        PlanOutcome::Empty => return Ok(TransferCompleteNotification::completed()),
        PlanOutcome::Ops(ops) => ops,
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
    /// Lowered ops to dispatch.
    Ops(Vec<CopyOp>),
}

/// Shared "validate → project → plan → lower" pipeline used by both
/// the CUDA and NIXL planner-path entrypoints.
fn plan_and_lower(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
) -> Result<PlanOutcome> {
    let src_block_layout = src.layout().block_layout();
    let dst_block_layout = dst.layout().block_layout();
    if validate_planner_inputs(
        src_block_layout,
        dst_block_layout,
        src_block_ids,
        dst_block_ids,
        strategy,
    )?
    .is_noop()
    {
        return Ok(PlanOutcome::Empty);
    }

    let src_view = physical_to_layout_view(src)?;
    let dst_view = physical_to_layout_view(dst)?;
    let src_al = AnnotatedLayout::from_view(&src_view)?;
    let dst_al = AnnotatedLayout::from_view(&dst_view)?;

    let block_pairs: Vec<(usize, usize)> = src_block_ids
        .iter()
        .zip(dst_block_ids.iter())
        .map(|(&s, &d)| (s, d))
        .collect();
    let selection = TransferSelection::full(block_pairs);
    // PR-5 policy: `min_inner_bytes = 0` — see
    // `execute_planner_cuda_transfer` for rationale.
    let policy = CopyPolicy {
        min_inner_bytes: 0,
        coalesce: true,
    };

    let plan = plan_copy(&src_al, &dst_al, &selection, &policy)?;
    let candidates = lower_to_candidates(plan)?;
    let chosen = select_candidate(&candidates)?;
    let ops = match chosen {
        Candidate::DirectDma { ops } => ops.clone(),
        other => bail!(
            "plan_and_lower: PR-5 only supports DirectDma, got {:?}",
            other
        ),
    };
    if ops.is_empty() {
        return Ok(PlanOutcome::Empty);
    }
    Ok(PlanOutcome::Ops(ops))
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

/// Pure validation gate for `execute_planner_cuda_transfer`. Extracted
/// so the rejection paths can be tested without a `TransferContext`
/// (which needs a real CUDA stream pool, NIXL agent, and tokio
/// runtime).
///
/// Returns `Err` for every condition that bails inside the executor:
/// unsupported strategy, mismatched block-id list lengths, and
/// `requires_transform` layout pairs. Returns `Ok(Noop)` when the
/// transfer has no work to do, `Ok(Proceed)` otherwise.
pub(crate) fn validate_planner_inputs(
    src_block_layout: KvBlockLayout,
    dst_block_layout: KvBlockLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
) -> Result<PlannerInputs> {
    if !matches!(
        strategy,
        TransferStrategy::CudaAsyncH2D
            | TransferStrategy::CudaAsyncD2H
            | TransferStrategy::CudaAsyncD2D
            | TransferStrategy::NixlRead
            | TransferStrategy::NixlWrite
            | TransferStrategy::NixlReadFlipped
            | TransferStrategy::NixlWriteFlipped
    ) {
        bail!(
            "validate_planner_inputs: strategy {strategy:?} not supported in PR-5 \
             (only CudaAsync H2D / D2H / D2D and Nixl Read/Write)"
        );
    }
    if src_block_ids.len() != dst_block_ids.len() {
        bail!(
            "execute_planner_cuda_transfer: src_block_ids ({}) != dst_block_ids ({})",
            src_block_ids.len(),
            dst_block_ids.len()
        );
    }
    if src_block_ids.is_empty() {
        return Ok(PlannerInputs::Noop);
    }
    if src_block_layout.requires_transform(&dst_block_layout) {
        bail!(
            "execute_planner_cuda_transfer: src ({src_block_layout:?}) and dst \
             ({dst_block_layout:?}) require a kernel-side transform, which is not \
             yet wired in the planner path (PR-6 lands the kernel catalog). Drop \
             use_planner=true for this transfer."
        );
    }
    Ok(PlannerInputs::Proceed)
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

    /// Same operational layout: validator passes.
    #[test]
    fn validator_passes_same_operational_layout() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalNHD,
            &[0, 1, 2],
            &[0, 1, 2],
            TransferStrategy::CudaAsyncD2D,
        );
        assert!(matches!(r, Ok(PlannerInputs::Proceed)));
    }

    /// NHD ↔ HND is a semantic transform that PR-5 cannot raw-copy
    /// without corrupting data.
    #[test]
    fn validator_rejects_nhd_hnd() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalHND,
            &[0],
            &[0],
            TransferStrategy::CudaAsyncD2D,
        );
        assert!(r.is_err());
    }

    /// Operational ↔ Universal also requires a kernel transform.
    #[test]
    fn validator_rejects_operational_to_universal() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::UniversalTP,
            &[0],
            &[0],
            TransferStrategy::CudaAsyncD2D,
        );
        assert!(r.is_err());
    }

    /// Empty block lists are a valid no-op (caller short-circuits).
    #[test]
    fn validator_returns_noop_on_empty_block_list() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalNHD,
            &[],
            &[],
            TransferStrategy::CudaAsyncD2D,
        );
        assert!(matches!(r, Ok(PlannerInputs::Noop)));
    }

    /// Mismatched block-id list lengths are rejected.
    #[test]
    fn validator_rejects_block_id_length_mismatch() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalNHD,
            &[0, 1],
            &[0],
            TransferStrategy::CudaAsyncD2D,
        );
        assert!(r.is_err());
    }

    /// CudaAsync and Nixl strategies are accepted; Memcpy / Invalid
    /// are not (PR-5/5.6 wires only those two strategy families).
    #[test]
    fn validator_strategy_acceptance() {
        // Accepted: every CudaAsync direction.
        for s in [
            TransferStrategy::CudaAsyncH2D,
            TransferStrategy::CudaAsyncD2H,
            TransferStrategy::CudaAsyncD2D,
        ] {
            assert!(matches!(
                validate_planner_inputs(
                    KvBlockLayout::OperationalNHD,
                    KvBlockLayout::OperationalNHD,
                    &[0],
                    &[0],
                    s,
                ),
                Ok(PlannerInputs::Proceed)
            ));
        }
        // Accepted: every Nixl variant.
        for s in [
            TransferStrategy::NixlRead,
            TransferStrategy::NixlWrite,
            TransferStrategy::NixlReadFlipped,
            TransferStrategy::NixlWriteFlipped,
        ] {
            assert!(matches!(
                validate_planner_inputs(
                    KvBlockLayout::OperationalNHD,
                    KvBlockLayout::OperationalNHD,
                    &[0],
                    &[0],
                    s,
                ),
                Ok(PlannerInputs::Proceed)
            ));
        }
        // Rejected: Memcpy (CPU host-only path) and Invalid sentinel.
        for s in [TransferStrategy::Memcpy, TransferStrategy::Invalid] {
            assert!(
                validate_planner_inputs(
                    KvBlockLayout::OperationalNHD,
                    KvBlockLayout::OperationalNHD,
                    &[0],
                    &[0],
                    s,
                )
                .is_err()
            );
        }
    }
}
