// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Planner-driven CudaAsync executor (`use_planner = true` path).
//!
//! Wires `transfer::plan::plan_copy` into the existing CUDA copy
//! infrastructure for the
//! [`TransferStrategy::CudaAsyncH2D`] / `CudaAsyncD2H` / `CudaAsyncD2D`
//! strategies. Other strategies and `use_planner = false` callers stay
//! on the legacy [`super::execute_direct_transfer`] path; this module
//! is only reached when both conditions hold. Errors from the planner
//! path are NOT silently fallen back to the legacy executor — bail
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
        return Ok(TransferCompleteNotification::completed());
    }

    // 1. Project to labelled views.
    let src_view = physical_to_layout_view(src)?;
    let dst_view = physical_to_layout_view(dst)?;

    // 2. Project to addressable annotated layouts.
    let src_al = AnnotatedLayout::from_view(&src_view)?;
    let dst_al = AnnotatedLayout::from_view(&dst_view)?;

    // 3. Build the selection (no axis_slices in PR-5 — full-extent
    //    transfers).
    let block_pairs: Vec<(usize, usize)> = src_block_ids
        .iter()
        .zip(dst_block_ids.iter())
        .map(|(&s, &d)| (s, d))
        .collect();
    let selection = TransferSelection::full(block_pairs);
    // PR-5 policy: `min_inner_bytes = 0` so layouts that pass the
    // `requires_transform` gate above always emit `CopyPlan::Direct`
    // even when their inner contiguous tail is small. The default
    // 4 KiB threshold exists for kernel-launch amortisation, and PR-5
    // has no kernel candidate to hand small-tail plans off to. PR-6
    // restores the threshold once `Candidate::TransformKernel` is
    // executable.
    let policy = CopyPolicy {
        min_inner_bytes: 0,
        coalesce: true,
    };

    // 4. Plan the copy.
    let plan = plan_copy(&src_al, &dst_al, &selection, &policy)?;

    // 5. Lower to candidates and pick.
    let candidates = lower_to_candidates(plan)?;
    let chosen = select_candidate(&candidates)?;
    let ops = match chosen {
        Candidate::DirectDma { ops } => ops.clone(),
        other => bail!(
            "execute_planner_cuda_transfer: PR-5 only supports DirectDma, got {:?}",
            other
        ),
    };

    if ops.is_empty() {
        return Ok(TransferCompleteNotification::completed());
    }

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

    // 6. Dispatch ops to CUDA. Group by `size` so each `memcpy_batch`
    //    call has a uniform `size_per_copy`. The common case is one
    //    group (uniform op size after coalescing); if the planner emits
    //    heterogeneous sizes (e.g. partial coalescing across non-
    //    contiguous block runs), we issue one batch per size.
    dispatch_ops_grouped_by_size(&ops, stream.as_ref())?;

    // 7. Synchronisation handoff.
    if caller_manages_sync {
        return Ok(TransferCompleteNotification::completed());
    }
    let event = stream.record_event(None)?;
    Ok(ctx.register_cuda_event(event))
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
    ) {
        bail!(
            "execute_planner_cuda_transfer: strategy {strategy:?} not supported in PR-5 \
             (only CudaAsync H2D / D2H / D2D)"
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

    /// Strategies outside the CudaAsync set are rejected — the
    /// executor's branch wires planner only for those three.
    #[test]
    fn validator_rejects_non_cuda_async_strategy() {
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalNHD,
            &[0],
            &[0],
            TransferStrategy::Memcpy,
        );
        assert!(r.is_err());
        let r = validate_planner_inputs(
            KvBlockLayout::OperationalNHD,
            KvBlockLayout::OperationalNHD,
            &[0],
            &[0],
            TransferStrategy::NixlWrite,
        );
        assert!(r.is_err());
    }
}
