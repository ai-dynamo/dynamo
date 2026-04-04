// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Device executor for GPU memory transfers (multi-backend: CUDA/XPU).
//!
//! This executor uses the device abstraction layer to support multiple hardware
//! backends through a unified interface.
//!
//! - **Whole-block (FC→FC)**: builds 1 pointer per block, calls `batch_copy`
//!   with `bytes_per_block`.
//! - **Per-chunk (FC↔LW or partial)**: builds 1 pointer per layer×outer chunk,
//!   calls `vectorized_copy` with `chunk_size` (GPU kernel), which falls back
//!   to `batch_copy` on backends without kernel support.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::device::DeviceStream;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::validate_layout_compatibility;
use crate::transfer::can_use_whole_block_transfer;
use crate::BlockId;
use anyhow::{Result, anyhow};
use std::ops::Range;
use std::sync::Arc;

/// Execute a device transfer between host and device memory.
///
/// This executor handles transfers involving GPU/XPU memory using device abstraction.
/// Supports async and blocking transfers depending on the strategy.
///
/// For eligible transfers (FC→FC full-block), uses whole-block optimization via
/// `batch_copy`. Otherwise uses `vectorized_copy` (GPU kernel with fallback).
pub fn execute_device_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    device_stream: Option<Arc<DeviceStream>>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // Validate layouts
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.num_layers() != dst_layout.num_layers() {
        return Err(anyhow!(
            "Layouts have incompatible layer counts: src={}, dst={}",
            src_layout.num_layers(),
            dst_layout.num_layers()
        ));
    }

    if src_layout.outer_dim() != dst_layout.outer_dim() {
        return Err(anyhow!(
            "Layouts have incompatible outer dimensions: src={}, dst={}",
            src_layout.outer_dim(),
            dst_layout.outer_dim()
        ));
    }

    // Validate layout compatibility (errors if transform would be needed)
    validate_layout_compatibility(src, dst)?;

    // Determine layer range
    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    // Track whether caller provided stream (affects event recording)
    let caller_manages_sync = device_stream.is_some();

    // Get appropriate device stream — use caller-provided or acquire from round-robin pool
    let device_stream = if let Some(s) = device_stream {
        s
    } else {
        match strategy {
            TransferStrategy::AsyncD2H | TransferStrategy::BlockingD2H => ctx.next_d2h_stream(),
            _ => ctx.next_h2d_stream(), // H2D and D2D use h2d_stream
        }
    };

    // Check if whole-block optimization is applicable
    let whole_block = can_use_whole_block_transfer(src, dst, Some(&layers));

    let strategy_name = match strategy {
        TransferStrategy::AsyncH2D | TransferStrategy::BlockingH2D => "H2D",
        TransferStrategy::AsyncD2H | TransferStrategy::BlockingD2H => "D2H",
        TransferStrategy::AsyncD2D => "D2D",
        _ => "Unknown",
    };

    tracing::debug!(
        strategy = strategy_name,
        whole_block,
        num_blocks = src_block_ids.len(),
        "Starting device transfer"
    );

    // Execute the transfer
    if whole_block {
        execute_whole_block_device(src, dst, src_block_ids, dst_block_ids, &device_stream)?;
    } else {
        execute_fc_lw_vectorized(src, dst, src_block_ids, dst_block_ids, layers, &device_stream)?;
    }

    // For blocking strategies, synchronize the stream
    if matches!(strategy, TransferStrategy::BlockingH2D | TransferStrategy::BlockingD2H) {
        device_stream.synchronize()?;
    }

    // If caller provided the stream, they manage synchronization — return completed immediately
    if caller_manages_sync {
        return Ok(TransferCompleteNotification::completed());
    }

    // For async transfers, record an event and register it for completion tracking
    if matches!(
        strategy,
        TransferStrategy::AsyncH2D
            | TransferStrategy::AsyncD2H
            | TransferStrategy::AsyncD2D
    ) {
        let event = device_stream.record_event()?;
        Ok(ctx.register_device_event(event))
    } else {
        // Blocking transfers are already synchronized
        Ok(TransferCompleteNotification::completed())
    }
}

// ======================================================================
// Whole-block transfer (FC→FC)
// ======================================================================

/// Whole-block transfer via `batch_copy` (FC→FC optimization).
///
/// Both source and destination must be fully contiguous. Builds one pointer
/// pair per block and copies `bytes_per_block` per entry.
fn execute_whole_block_device(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    stream: &Arc<DeviceStream>,
) -> Result<()> {
    let bytes_per_block = src.layout().config().bytes_per_block();
    let num_blocks = src_block_ids.len();

    if num_blocks == 0 {
        return Ok(());
    }

    let mut src_ptrs: Vec<u64> = Vec::with_capacity(num_blocks);
    let mut dst_ptrs: Vec<u64> = Vec::with_capacity(num_blocks);

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_region = src.memory_region(src_block_id, 0, 0)?;
        let dst_region = dst.memory_region(dst_block_id, 0, 0)?;
        src_ptrs.push(src_region.addr() as u64);
        dst_ptrs.push(dst_region.addr() as u64);
    }

    stream.batch_copy(&src_ptrs, &dst_ptrs, bytes_per_block)?;

    tracing::debug!(
        num_blocks,
        bytes_per_block,
        "Whole-block transfer completed via batch_copy"
    );

    Ok(())
}

// ======================================================================
// FC↔LW vectorized transfer
// ======================================================================

/// FC↔LW (or partial-layer) transfer via `vectorized_copy`.
///
/// Builds one pointer pair per (block, layer, outer) chunk and dispatches
/// to `vectorized_copy` which uses a GPU kernel (CUDA) or falls back to
/// `batch_copy` (Level-Zero).
fn execute_fc_lw_vectorized(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &Arc<DeviceStream>,
) -> Result<()> {
    let src_layout = src.layout();
    let nl = layers.len();
    let no = src_layout.outer_dim();
    let chunk_size =
        src_layout.page_size() * src_layout.inner_dim() * src_layout.dtype_width_bytes();
    let num_blocks = src_block_ids.len();
    let total_chunks = num_blocks * nl * no;

    if total_chunks == 0 {
        return Ok(());
    }

    let mut src_ptrs: Vec<u64> = Vec::with_capacity(total_chunks);
    let mut dst_ptrs: Vec<u64> = Vec::with_capacity(total_chunks);

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..no {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                src_ptrs.push(src_region.addr() as u64);
                dst_ptrs.push(dst_region.addr() as u64);
            }
        }
    }

    stream.vectorized_copy(&src_ptrs, &dst_ptrs, chunk_size)?;

    tracing::debug!(
        total_chunks,
        chunk_size,
        "Per-chunk transfer completed via vectorized_copy"
    );

    Ok(())
}
