// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::layout::{BlockLayoutConfig, LayoutError};
use crate::block_manager::storage::Storage;

use validator::ValidationError;

/// Validation function for Option<usize> to check if it's Some(power_of_2).
pub fn validate_power_of_2(alignment: usize) -> Result<(), ValidationError> {
    if !alignment.is_power_of_two() {
        // Return validation error if alignment is not a power of 2
        return Err(validator::ValidationError::new(
            "alignment_must_be_power_of_2",
        ));
    }
    // Passes validation if alignment is a power of 2
    Ok(())
}

/// Helper to align a value up to the nearest multiple of alignment.
/// Alignment must be a power of 2.
pub fn align_up(value: usize, alignment: usize) -> usize {
    (value + alignment - 1) & !(alignment - 1)
}

/// Helper to validate that a storage allocation is large enough for a layout.
pub fn validate_storage<S: Storage, C: BlockLayoutConfig>(
    storage: &S,
    config: &C,
) -> Result<usize, LayoutError> {
    let provided_size = storage.size();
    let storage_addr = storage.addr();
    let alignment = config.layout_config().alignment;

    // Calculate base offset needed to align the start of block 0
    let base_offset = if alignment > 1 {
        align_up(storage_addr as usize, alignment) - storage_addr as usize
    } else {
        0
    };

    let total_required_size_with_offset = base_offset + config.layout_data_bytes();

    tracing::debug!(
        provided_size,
        total_required_size_with_offset,
        base_offset,
        required_layout_data_bytes = config.layout_data_bytes(),
        alignment,
        "Validating storage size with base offset and alignment"
    );

    // Validate storage size fits the configuration *with base offset and alignment*
    if provided_size < total_required_size_with_offset {
        tracing::warn!(
            provided_size,
            total_required_size_with_offset,
            "Storage size too small for aligned layout including base offset"
        );
        return Err(LayoutError::InvalidConfig(format!(
            "Storage size {} is less than required size {} (including base offset for alignment)",
            provided_size, total_required_size_with_offset
        )));
    }

    Ok(base_offset)
}

pub fn validate_indices<C: BlockLayoutConfig>(
    config: &C,
    block_idx: usize,
    layer_idx: usize,
    outer_idx: usize,
) -> Result<(), LayoutError> {
    if block_idx >= config.num_blocks() {
        return Err(LayoutError::InvalidBlockIndex(block_idx));
    }

    if layer_idx >= config.num_layers() {
        return Err(LayoutError::InvalidLayerIndex(layer_idx));
    }

    if outer_idx >= config.outer_dim() {
        return Err(LayoutError::InvalidOuterIndex(outer_idx));
    }

    Ok(())
}
