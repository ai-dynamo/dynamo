// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Contiguous transfer optimizations for NIXL.
//!
//! This module provides batched transfer building for contiguous memory layouts,
//! dramatically improving performance for large transfers by reducing descriptor count.

use crate::block_manager::v2::physical::layout::PhysicalLayout;
use anyhow::Result;
use nixl_sys::XferDescList;
use std::ops::Range;

/// Build batched contiguous transfer - optimizes contiguous→contiguous transfers.
///
/// Intelligently batches contiguous ranges even when overall block list is fragmented.
/// For example: [0,1,2, 10,11, 50] creates 3 batched descriptors instead of 6 individual ones.
/// This dramatically improves performance for large transfers (e.g., 877 blocks: 12s → <1s).
pub(crate) fn build_batched_contiguous_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()> {
    let src_layout = src.layout();

    if src_block_ids.is_empty() {
        return Ok(());
    }

    // Find contiguous ranges in both source and destination
    // A range is contiguous when both src and dst blocks increment by 1
    let mut ranges: Vec<(usize, usize)> = Vec::new(); // (start_idx, end_idx)
    let mut range_start = 0;

    for i in 1..src_block_ids.len() {
        let src_contiguous = src_block_ids[i] == src_block_ids[i - 1] + 1;
        let dst_contiguous = dst_block_ids[i] == dst_block_ids[i - 1] + 1;

        if !src_contiguous || !dst_contiguous {
            // End of contiguous range
            ranges.push((range_start, i));
            range_start = i;
        }
    }
    // Add final range
    ranges.push((range_start, src_block_ids.len()));

    let total_blocks = src_block_ids.len();
    let mut batched_blocks = 0;

    // Create one descriptor per contiguous range
    for &(start_idx, end_idx) in &ranges {
        let range_len = end_idx - start_idx;

        if range_len == 1 {
            // Single block - use simple descriptor
            let src_block_id = src_block_ids[start_idx];
            let dst_block_id = dst_block_ids[start_idx];

            let first_src = src.memory_region(src_block_id, layers.start, 0)?;
            let first_dst = dst.memory_region(dst_block_id, layers.start, 0)?;
            let total_size = layers.len() * src_layout.outer_dim() * first_src.size();

            src_dl.add_desc(first_src.addr(), total_size, src_device_id);
            dst_dl.add_desc(first_dst.addr(), total_size, dst_device_id);
        } else {
            // Contiguous range - use batched descriptor
            let src_block_id = src_block_ids[start_idx];
            let dst_block_id = dst_block_ids[start_idx];

            let first_src = src.memory_region(src_block_id, layers.start, 0)?;
            let first_dst = dst.memory_region(dst_block_id, layers.start, 0)?;
            let total_size = range_len * layers.len() * src_layout.outer_dim() * first_src.size();

            src_dl.add_desc(first_src.addr(), total_size, src_device_id);
            dst_dl.add_desc(first_dst.addr(), total_size, dst_device_id);

            batched_blocks += range_len;
        }
    }

    if ranges.len() == 1 && batched_blocks == total_blocks {
        // Fully contiguous - single batch
        tracing::info!(
            "⚡ FULLY BATCHED: {} blocks in ONE descriptor ({:.2} MB)",
            total_blocks,
            (total_blocks
                * layers.len()
                * src_layout.outer_dim()
                * src.memory_region(src_block_ids[0], layers.start, 0)?.size()) as f64
                / 1_000_000.0
        );
    } else if batched_blocks > 0 {
        // Partially batched - some ranges
        tracing::info!(
            "⚡ PARTIALLY BATCHED: {} blocks in {} descriptors ({} batched, {} single)",
            total_blocks,
            ranges.len(),
            batched_blocks,
            total_blocks - batched_blocks
        );
    } else {
        // All single blocks
        tracing::debug!(
            "No batching possible: {} individual descriptors",
            total_blocks
        );
    }

    Ok(())
}

/// Build single-descriptor transfer for fully contiguous layouts.
///
/// Optimizes transfers when both source and destination are fully contiguous
/// by creating a single descriptor covering all blocks. Falls back to
/// per-block descriptors if blocks are non-contiguous.
pub(crate) fn build_single_descriptor_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()> {
    let src_layout = src.layout();

    // Check if blocks are contiguous for single-descriptor optimization
    let blocks_are_contiguous = src_block_ids.len() > 1
        && src_block_ids.windows(2).all(|w| w[1] == w[0] + 1)
        && dst_block_ids.windows(2).all(|w| w[1] == w[0] + 1);

    if blocks_are_contiguous {
        // Single descriptor covering all contiguous blocks
        let first_src = src.memory_region(src_block_ids[0], layers.start, 0)?;
        let first_dst = dst.memory_region(dst_block_ids[0], layers.start, 0)?;

        let total_size =
            src_block_ids.len() * layers.len() * src_layout.outer_dim() * first_src.size();

        src_dl.add_desc(first_src.addr(), total_size, src_device_id);
        dst_dl.add_desc(first_dst.addr(), total_size, dst_device_id);

        tracing::trace!(
            "Single descriptor: {} contiguous blocks, {} bytes",
            src_block_ids.len(),
            total_size
        );
    } else {
        // One descriptor per block (non-contiguous)
        for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
            let first_src = src.memory_region(src_block_id, layers.start, 0)?;
            let first_dst = dst.memory_region(dst_block_id, layers.start, 0)?;

            let total_size = layers.len() * src_layout.outer_dim() * first_src.size();

            src_dl.add_desc(first_src.addr(), total_size, src_device_id);
            dst_dl.add_desc(first_dst.addr(), total_size, dst_device_id);
        }
        tracing::trace!("Single-desc mode: {} descriptors", src_block_ids.len());
    }

    Ok(())
}

/// Build per-block transfer with offsets.
///
/// Creates one descriptor per block, preserving byte offsets from the source layout.
/// Used for reading from different offsets within a shared resource (e.g., disk file, RDMA buffer, object storage).
///
/// # Arguments
/// * `get_device_id` - Optional closure to extract per-block device IDs (e.g., object keys).
///                     If None, uses the default device IDs for all blocks.
pub(crate) fn build_per_block_offset_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
    get_device_id: Option<&dyn Fn(&PhysicalLayout, usize, u64) -> u64>,
) -> Result<()> {
    let src_layout = src.layout();

    tracing::debug!("Per-block offset transfer: {} blocks", src_block_ids.len());

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_region = src.memory_region(src_block_id, layers.start, 0)?;
        let dst_region = dst.memory_region(dst_block_id, layers.start, 0)?;
        let total_size = layers.len() * src_layout.outer_dim() * src_region.size();

        // Get device IDs (per-block for object storage, default for others)
        let src_key = get_device_id
            .map(|f| f(src, src_block_id, src_device_id))
            .unwrap_or(src_device_id);
        let dst_key = get_device_id
            .map(|f| f(dst, dst_block_id, dst_device_id))
            .unwrap_or(dst_device_id);

        src_dl.add_desc(src_region.addr(), total_size, src_key);
        dst_dl.add_desc(dst_region.addr(), total_size, dst_key);

        tracing::trace!(
            "  src[{}] (key={}) offset={} → dst[{}] (key={}) offset={}: {} bytes",
            src_block_id, src_key, src_region.addr(),
            dst_block_id, dst_key, dst_region.addr(),
            total_size
        );
    }

    Ok(())
}

/// Build per-block transfer to unique targets.
///
/// Creates one descriptor per block. For object storage writes, each block writes
/// to offset 0 of its unique target. For other backends, preserves offsets.
///
/// # Arguments
/// * `get_device_id` - Optional closure to extract per-block device IDs (e.g., object keys).
///                     If None, uses the default device IDs for all blocks.
/// * `force_zero_offset` - If true, destination offset is always 0 (for object storage writes).
pub(crate) fn build_per_block_unique_target_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
    get_device_id: Option<&dyn Fn(&PhysicalLayout, usize, u64) -> u64>,
    force_zero_offset: bool,
) -> Result<()> {
    let src_layout = src.layout();

    tracing::debug!(
        "Per-block unique target transfer: {} blocks, force_zero_offset={}",
        src_block_ids.len(),
        force_zero_offset
    );

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_region = src.memory_region(src_block_id, layers.start, 0)?;
        let dst_region = dst.memory_region(dst_block_id, layers.start, 0)?;
        let total_size = layers.len() * src_layout.outer_dim() * src_region.size();

        // Get device IDs (per-block for object storage, default for others)
        let src_key = get_device_id
            .map(|f| f(src, src_block_id, src_device_id))
            .unwrap_or(src_device_id);
        let dst_key = get_device_id
            .map(|f| f(dst, dst_block_id, dst_device_id))
            .unwrap_or(dst_device_id);

        // Destination offset: 0 for object storage writes, actual offset otherwise
        let dst_offset = if force_zero_offset { 0 } else { dst_region.addr() };

        src_dl.add_desc(src_region.addr(), total_size, src_key);
        dst_dl.add_desc(dst_offset, total_size, dst_key);

        tracing::trace!(
            "  src[{}] (key={}) → dst[{}] (key={}) offset={}: {} bytes",
            src_block_id, src_key, dst_block_id, dst_key, dst_offset, total_size
        );
    }

    Ok(())
}

/// Build multi-descriptor transfer with per-block device_id support.
///
/// Standard mode: one descriptor per (block, layer, outer) tuple.
/// Used for non-contiguous layouts or complex patterns.
pub(crate) fn build_multi_descriptor_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    get_device_id: &dyn Fn(&PhysicalLayout, usize, u64) -> u64,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()> {
    use anyhow::anyhow;

    let src_layout = src.layout();

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        // Get device_id for this specific block (unique key for ObjectLayout)
        let src_key = get_device_id(src, src_block_id, src_device_id);
        let dst_key = get_device_id(dst, dst_block_id, dst_device_id);

        for layer_id in layers.clone() {
            for outer_id in 0..src_layout.outer_dim() {
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

                src_dl.add_desc(src_region.addr(), src_region.size(), src_key);
                dst_dl.add_desc(dst_region.addr(), dst_region.size(), dst_key);
            }
        }
    }

    let total = src_block_ids.len() * layers.len() * src_layout.outer_dim();
    tracing::trace!("Multi-descriptor mode: {} descriptors", total);
    Ok(())
}

