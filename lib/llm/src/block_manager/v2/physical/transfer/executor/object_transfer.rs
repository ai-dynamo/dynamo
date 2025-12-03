// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage transfer optimizations for NIXL.
//!
//! This module provides optimized transfer building strategies for object storage (Object/MinIO)
//! backends.
//!
//! # Key Constraints
//!
//! **READ (Object → Host):** Supports byte-range offsets via Object GET Range header.
//! Multiple blocks CAN share one Object key with different offsets (TP partitions).
//!
//! **WRITE (Host → Object):** One-to-one mapping ONLY. Object PUT does NOT support
//! partial writes at offsets. Each block MUST map to a unique Object.

use crate::block_manager::v2::physical::layout::PhysicalLayout;
use crate::block_manager::v2::memory::StorageKind;
use anyhow::Result;
use nixl_sys::XferDescList;
use std::collections::HashSet;
use std::ops::Range;

/// Build optimized transfer for ObjectLayout ↔ Contiguous layout.
///
/// # Read Path (Object → Host): Supports byte-range offsets
///
/// Multiple blocks CAN share one Object key with different offsets (TP partitions).
/// Uses Object GET with Range header for partial reads:
/// ```text
/// Object key 99999, offset 0MB       → Block 0 (TP partition 0)
/// Object key 99999, offset 2.3MB     → Block 1 (TP partition 1)
/// Object key 99999, offset 4.6MB     → Block 2 (TP partition 2)
/// Object key 99999, offset 6.9MB     → Block 3 (TP partition 3)
/// ```
///
/// # Write Path (Host → Object): One-to-one mapping ONLY
///
/// **IMPORTANT:** Object PUT does NOT support partial writes at offsets.
/// Each host block MUST map to a separate Object (unique key):
/// ```text
/// Block 0 → Object key 123456  (separate object)
/// Block 1 → Object key 789012  (separate object)
/// Block 2 → Object key 345678  (separate object)
/// ```
///
/// Returns error if multiple blocks share the same Object key (would require offset writes)
///
/// # Arguments
/// * `is_read` - true for Object→Host reads, false for Host→Object writes
/// * `get_device_id` - Closure that extracts Object key for each block
pub(crate) fn build_object_storage_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
    layers: &Range<usize>,
    src_device_id: u64,
    dst_device_id: u64,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
    get_device_id: &dyn Fn(&PhysicalLayout, usize, u64) -> u64,
    is_read: bool,
) -> Result<()> {
    let region_size = |layout: &PhysicalLayout| -> usize {
        let l = layout.layout();
        layers.len() * l.outer_dim() * l.page_size() * l.inner_dim() * l.dtype_width_bytes()
    };

    match is_read {
        true => {
            // READ: Object → Host
            // Each source block may be:
            // 1. A separate Object (different keys) - requires unique key per block
            // 2. A partition within one Object (same key, different offsets) - byte-range GET
            tracing::debug!("Object storage READ: {} blocks from Object", src_block_ids.len());

            src_block_ids
                .iter()
                .zip(dst_block_ids.iter())
                .try_for_each(|(&src_id, &dst_id)| {
                    let src_region = src.memory_region(src_id, layers.start, 0)?;
                    let dst_region = dst.memory_region(dst_id, layers.start, 0)?;
                    let total_size = region_size(src);
                    let src_key = get_device_id(src, src_id, src_device_id);

                    src_dl.add_desc(src_region.addr(), total_size, src_key);
                    dst_dl.add_desc(dst_region.addr(), total_size, dst_device_id);

                    tracing::trace!(
                        "  Object[{src_id}] (key={src_key}) → Host[{dst_id}]: {total_size} bytes"
                    );
                    Ok(())
                })
        }
        false => {
            // WRITE: Host → Object
            // IMPORTANT: Object PUT does NOT support partial writes at offsets.
            // Each block MUST map to a unique Object (one-to-one mapping).

            // Collect all destination Object keys
            let dst_keys: Vec<u64> = dst_block_ids
                .iter()
                .map(|&dst_id| get_device_id(dst, dst_id, dst_device_id))
                .collect();

            // Validate one-to-one mapping: each block must have a unique Object key
            let unique_keys: HashSet<u64> = dst_keys.iter().copied().collect();
            if unique_keys.len() != dst_block_ids.len() {
                return Err(anyhow::anyhow!(
                    "Object storage WRITE requires one-to-one mapping: {} blocks but only {} unique Object keys. \
                     Object PUT does not support partial writes at offsets.",
                    dst_block_ids.len(),
                    unique_keys.len()
                ));
            }

            tracing::debug!(
                "Object storage WRITE: {} blocks → {} Objects",
                src_block_ids.len(),
                unique_keys.len()
            );

            // One descriptor per block (one-to-one mapping)
            src_block_ids
                .iter()
                .zip(dst_block_ids.iter())
                .zip(dst_keys.iter())
                .try_for_each(|((&src_id, &dst_id), &dst_key)| {
                    let src_region = src.memory_region(src_id, layers.start, 0)?;
                    let dst_region = dst.memory_region(dst_id, layers.start, 0)?;
                    let total_size = region_size(src);

                    // Validate destination offset is 0 (Object PUT doesn't support offsets)
                    if dst_region.addr() != 0 {
                        return Err(anyhow::anyhow!(
                            "Object storage WRITE: block {} has non-zero offset {}. \
                             Object PUT does not support partial writes at offsets.",
                            dst_id,
                            dst_region.addr()
                        ));
                    }

                    src_dl.add_desc(src_region.addr(), total_size, src_device_id);
                    dst_dl.add_desc(0, total_size, dst_key); // Always offset 0 for writes

                    tracing::trace!(
                        "  Host[{src_id}] → Object key={dst_key}: {total_size} bytes"
                    );
                    Ok(())
                })
        }
    }
}

/// Extract device_id (Object key) for a specific block.
///
/// ObjectLayout has unique keys per block; other layouts share one device_id.
pub(crate) fn get_object_device_id(
    layout: &PhysicalLayout,
    block_id: usize,
    default_id: u64,
) -> u64 {
    match (layout.location(), layout.layout().is_fully_contiguous()) {
        (StorageKind::Object(_), false) => {
            // ObjectLayout: each block has its own Object key
            layout
                .layout()
                .memory_regions()
                .get(block_id)
                .and_then(|region| match region.storage_kind() {
                    StorageKind::Object(key) => Some(key),
                    _ => None,
                })
                .unwrap_or(default_id)
        }
        _ => default_id,
    }
}
