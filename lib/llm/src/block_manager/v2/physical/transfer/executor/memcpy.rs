// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memcpy executor for host-to-host transfers.

use crate::block_manager::v2::physical::transfer::PhysicalLayout;
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::Result;
use std::ops::Range;

/// Execute a memcpy transfer between host memory locations.
///
/// This executor handles transfers between System and Pinned memory using
/// standard CPU memcpy operations. The transfer is synchronous and blocking.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `block_pairs` - Pairs of (src_block_id, dst_block_id) to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
pub fn execute_memcpy_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layer_range: Option<Range<usize>>,
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

    // Determine layer range
    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    // Perform synchronous copies
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

    // Memcpy is synchronous, so return already-completed notification
    Ok(TransferCompleteNotification::completed())
}
