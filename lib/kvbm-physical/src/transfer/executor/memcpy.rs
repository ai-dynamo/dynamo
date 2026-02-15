// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memcpy executor for host-to-host transfers.

use crate::BlockId;
use crate::transfer::PhysicalLayout;
use crate::transfer::TransferContext;
use crate::transfer::context::TransferCompleteNotification;
use crate::transfer::{can_use_whole_block_transfer, validate_layout_compatibility};
use anyhow::Result;
use std::ops::Range;

/// Execute a memcpy transfer between host memory locations.
///
/// This executor handles transfers between System and Pinned memory using
/// standard CPU memcpy operations. The transfer is synchronous and blocking.
///
/// For FC→FC transfers with compatible layouts and full-block transfers,
/// this uses an optimized whole-block copy path (single memcpy per block).
/// Otherwise, falls back to layer-wise copying.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
/// * `_ctx` - Transfer context (unused for memcpy, kept for API consistency)
pub fn execute_memcpy_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    _ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // Validate layouts have compatible structure
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.num_layers() != dst_layout.num_layers() {
        return Err(anyhow::anyhow!(
            "Layouts have incompatible layer counts: src={}, dst={}",
            src_layout.num_layers(),
            dst_layout.num_layers()
        ));
    }

    if src_layout.outer_dim() != dst_layout.outer_dim() {
        return Err(anyhow::anyhow!(
            "Layouts have incompatible outer dimensions: src={}, dst={}",
            src_layout.outer_dim(),
            dst_layout.outer_dim()
        ));
    }

    // Validate layout compatibility (errors if transform would be needed)
    validate_layout_compatibility(src, dst)?;

    let layers = layer_range.clone().unwrap_or(0..src_layout.num_layers());

    // Try whole-block path for FC→FC transfers with compatible layouts
    if can_use_whole_block_transfer(src, dst, layer_range.as_ref()) {
        tracing::debug!(
            num_blocks = src_block_ids.len(),
            bytes_per_block = src_layout.config().bytes_per_block(),
            "Using whole-block memcpy path"
        );
        execute_whole_block_memcpy(src, dst, src_block_ids, dst_block_ids)?;
    } else {
        tracing::debug!(
            num_blocks = src_block_ids.len(),
            layer_range = ?layers,
            src_fc = src_layout.is_fully_contiguous(),
            dst_fc = dst_layout.is_fully_contiguous(),
            "Using layer-wise memcpy path"
        );
        execute_layer_wise_memcpy(src, dst, src_block_ids, dst_block_ids, layers)?;
    }

    // Memcpy is synchronous, so return already-completed notification
    Ok(TransferCompleteNotification::completed())
}

/// Whole-block memcpy for FC→FC with compatible layouts.
///
/// Copies entire blocks in a single memcpy operation per block,
/// leveraging the fully contiguous memory layout.
fn execute_whole_block_memcpy(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
) -> Result<()> {
    let bytes_per_block = src.layout().config().bytes_per_block();

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        // Get block base address (layer=0, outer=0 for FC layout gives base)
        let src_region = src.memory_region(src_block_id, 0, 0)?;
        let dst_region = dst.memory_region(dst_block_id, 0, 0)?;

        unsafe {
            std::ptr::copy_nonoverlapping(
                src_region.addr() as *const u8,
                dst_region.addr() as *mut u8,
                bytes_per_block,
            );
        }
    }
    Ok(())
}

/// Layer-wise memcpy (existing behavior, refactored).
///
/// Copies blocks layer by layer and outer dimension by outer dimension.
/// Used for FC→LW, LW→FC, LW→LW, or partial layer transfers.
fn execute_layer_wise_memcpy(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
) -> Result<()> {
    let src_layout = src.layout();

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..src_layout.outer_dim() {
                // Get source and destination memory regions
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                // Validate sizes match
                if src_region.size() != dst_region.size() {
                    return Err(anyhow::anyhow!(
                        "Memory region size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                // Perform memcpy
                unsafe {
                    let src_ptr = src_region.addr() as *const u8;
                    let dst_ptr = dst_region.addr() as *mut u8;
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, src_region.size());
                }
            }
        }
    }
    Ok(())
}
