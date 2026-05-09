// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Planner-driven CudaAsync executor (`use_planner = true` path).
//!
//! This is the PR-5 wiring of `transfer::plan::plan_copy` into the
//! existing CUDA copy infrastructure. It runs only for the
//! [`TransferStrategy::CudaAsyncH2D`] / `CudaAsyncD2H` / `CudaAsyncD2D`
//! strategies — other routes fall through to the legacy
//! `execute_direct_transfer` path even when `use_planner = true`.
//!
//! Pipeline:
//! 1. `physical_to_layout_view` projects each `PhysicalLayout` to a
//!    labelled [`LayoutView`].
//! 2. `AnnotatedLayout::from_view` collapses each view into the
//!    addressable layout the planner expects.
//! 3. `plan_copy` produces a [`CopyPlan`].
//! 4. `lower_to_candidates` + `select_candidate` pick the executable
//!    candidate (PR-5 only emits / accepts `Candidate::DirectDma`).
//! 5. The candidate's `Vec<CopyOp>` is grouped by `size` and dispatched
//!    via `kvbm_kernels::memcpy_batch` (FallbackOnly). Groups with
//!    distinct sizes get distinct calls; identical-size groups
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
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::lower::{
    Candidate, lower_to_candidates, physical_to_layout_view, select_candidate,
};
use crate::transfer::plan::{AnnotatedLayout, CopyOp, CopyPolicy, TransferSelection, plan_copy};

/// Dispatch a CudaAsync transfer through the stride-aware planner.
///
/// Triggered when `TransferOptions::use_planner = true` AND the chosen
/// strategy is one of the CudaAsync variants. Other paths return
/// `Ok(None)` so the caller can fall back to the legacy executor.
pub(crate) fn execute_planner_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
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
    let policy = CopyPolicy::default();

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

/// Group `ops` by `size` and dispatch each group via
/// `kvbm_kernels::memcpy_batch` in `FallbackOnly` mode (uniform-size
/// batched memcpy with per-op fallback to individual `cudaMemcpyAsync`
/// when the batch API is unavailable).
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
