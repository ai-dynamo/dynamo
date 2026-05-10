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

    let ops = match plan_and_lower(src, dst, src_block_ids, dst_block_ids)? {
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
/// Rejects strategies outside the `CudaAsync{H2D,D2H,D2D}` family and
/// layout pairs that would require a kernel-side transform (PR-6.1
/// drops the layout check once the kernel catalog handles transforms).
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
    if src_block_layout.requires_transform(&dst_block_layout) {
        bail!(
            "validate_cuda_planner_entry: src ({src_block_layout:?}) and dst \
             ({dst_block_layout:?}) require a kernel-side transform, which is not \
             yet wired in the planner path (PR-6.1 lands the kernel catalog). \
             Drop use_planner=true for this transfer."
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

    /// NHD ↔ HND is a semantic transform — rejected until PR-6.1
    /// wires the kernel catalog.
    #[test]
    fn cuda_entry_rejects_nhd_hnd() {
        let r = validate_cuda_planner_entry(
            TransferStrategy::CudaAsyncD2D,
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalHND,
        );
        assert!(r.is_err());
    }

    /// Operational ↔ Universal also requires a kernel transform.
    #[test]
    fn cuda_entry_rejects_operational_to_universal() {
        let r = validate_cuda_planner_entry(
            TransferStrategy::CudaAsyncD2D,
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::UniversalTP,
        );
        assert!(r.is_err());
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
