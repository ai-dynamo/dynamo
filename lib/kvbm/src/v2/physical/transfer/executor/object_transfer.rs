// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Object storage utilities for NIXL transfers.
//!
//! This module provides helper functions for object storage backends,
//! particularly for extracting per-block device IDs (object keys).

use crate::physical::layout::PhysicalLayout;
use dynamo_memory::StorageKind;

/// Extract device_id (object key) for a specific block.
///
/// ObjectLayout has unique keys per block; other layouts share one device_id.
///
/// # Behavior
/// - For `ObjectLayout` (non-contiguous): Returns the unique object key for this block
/// - For `FullyContiguous` with object backing: Returns the shared object key
/// - For other layouts: Returns the default device_id
pub(crate) fn get_object_device_id(
    layout: &PhysicalLayout,
    block_id: usize,
    default_id: u64,
) -> u64 {
    match (layout.location(), layout.layout().is_fully_contiguous()) {
        (StorageKind::Object(_), false) => {
            // ObjectLayout: each block has its own object key
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
