// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::layout::{BlockLayoutConfig, GenericBlockLayout, LayoutError};
use crate::block_manager::storage::Storage;

use validator::ValidationError;

/// Verification result for a memory region
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct RegionVerificationResult {
    /// Block index that was verified
    pub block_idx: usize,
    /// Layer index that was verified
    pub layer_idx: usize,
    /// Outer dimension index that was verified
    pub outer_idx: usize,
    /// Expected memory address for this region
    pub expected_addr: usize,
    /// Actual memory address for this region
    pub actual_addr: usize,
    /// Expected size in bytes for this region
    pub expected_size: usize,
    /// Actual size in bytes for this region
    pub actual_size: usize,
    /// Whether the addresses match
    pub addr_matches: bool,
    /// Whether the sizes match
    pub size_matches: bool,
}

/// Layout verification statistics
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct LayoutVerificationStats {
    /// Total number of memory regions verified
    pub total_regions: usize,
    /// Number of regions with address mismatches
    pub addr_mismatches: usize,
    /// Number of regions with size mismatches
    pub size_mismatches: usize,
    /// Number of regions that passed all verifications
    pub successful_verifications: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct WorkerLayoutVerifier {
    stats: LayoutVerificationStats,
}

#[allow(dead_code)]
impl WorkerLayoutVerifier {
    // Constructor: Start with clean slate
    pub fn new() -> Self {
        Self {
            stats: LayoutVerificationStats::default(),
        }
    }

    pub fn verify_layout_consistency<L: GenericBlockLayout>(
        &mut self,
        layout: &L,
    ) -> Result<Vec<RegionVerificationResult>, LayoutError> {
        // This is the main orchestrator method.
        // It systematically checks every memory region in
        // the layout to ensure consistency.

        self.stats = LayoutVerificationStats::default();
        let mut results = Vec::new();

        // Iterate over all blocks, layers, and outer dimensions
        for block_idx in 0..layout.num_blocks() {
            for layer_idx in 0..layout.num_layers() {
                for outer_idx in 0..layout.outer_dim() {
                    let result =
                        self.verify_memory_region(layout, block_idx, layer_idx, outer_idx)?;
                    self.update_stats(&result);
                    results.push(result);
                }
            }
        }

        Ok(results)
    }

    pub fn verify_memory_region<L: GenericBlockLayout>(
        &mut self,
        layout: &L,
        block_idx: usize,
        layer_idx: usize,
        outer_idx: usize,
    ) -> Result<RegionVerificationResult, LayoutError> {
        let memory_region = layout.memory_region(block_idx, layer_idx, outer_idx)?;

        let config = layout.config();
        let expected_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

        let expected_addr = memory_region.addr();

        Ok(RegionVerificationResult {
            block_idx,
            layer_idx,
            outer_idx,
            expected_addr,
            actual_addr: memory_region.addr(),
            expected_size,
            actual_size: memory_region.size(),
            addr_matches: expected_addr == memory_region.addr(),
            size_matches: expected_size == memory_region.size(),
        })
    }

    fn update_stats(&mut self, result: &RegionVerificationResult) {
        self.stats.total_regions += 1;
        if !result.addr_matches {
            self.stats.addr_mismatches += 1;
        }
        if !result.size_matches {
            self.stats.size_mismatches += 1;
        }
        if result.addr_matches && result.size_matches {
            self.stats.successful_verifications += 1;
        }
    }

    pub fn has_critical_mismatches(&self) -> bool {
        self.stats.size_mismatches > 0
    }
}

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

#[cfg(test)]
mod worker_verification_tests {
    use super::*;
    use crate::block_manager::LayoutConfig;
    use crate::block_manager::layout::{FullyContiguous, LayerSeparate};
    use crate::block_manager::storage::tests::{NullDeviceAllocator, NullDeviceStorage};

    // Test constants (same as layout.rs tests)
    const NUM_BLOCKS: usize = 7;
    const NUM_LAYERS: usize = 5;
    const OUTER_DIM: usize = 2;
    const PAGE_SIZE: usize = 4;
    const INNER_DIM: usize = 13;
    const DTYPE_WIDTH_BYTES: usize = 4;

    fn create_test_config() -> LayoutConfig {
        LayoutConfig {
            num_blocks: NUM_BLOCKS,
            num_layers: NUM_LAYERS,
            outer_dim: OUTER_DIM,
            page_size: PAGE_SIZE,
            inner_dim: INNER_DIM,
            alignment: 1,
            dtype_width_bytes: DTYPE_WIDTH_BYTES,
        }
    }

    fn create_fully_contiguous_layout() -> Result<FullyContiguous<NullDeviceStorage>, LayoutError> {
        let config = create_test_config();
        FullyContiguous::allocate(config, &NullDeviceAllocator)
    }

    fn create_layer_separate_layout() -> Result<LayerSeparate<NullDeviceStorage>, LayoutError> {
        let config = create_test_config();
        LayerSeparate::allocate(config, &NullDeviceAllocator, true) // outer_contiguous = true
    }

    #[test]
    fn test_verify_initialisation() {
        let verifier = WorkerLayoutVerifier::new();

        assert_eq!(verifier.stats.total_regions, 0);
        assert_eq!(verifier.stats.addr_mismatches, 0);
        assert_eq!(verifier.stats.size_mismatches, 0);
        assert_eq!(verifier.stats.successful_verifications, 0);

        assert!(!verifier.has_critical_mismatches());
    }

    #[test]
    fn test_layer_separate_verification() {
        let layout = create_layer_separate_layout().expect("Failed to create LayerSeparate layout");
        let mut verifier = WorkerLayoutVerifier::new();
        let results = verifier
            .verify_layout_consistency(&layout)
            .expect("Failed to verify layout");

        assert_eq!(results.len(), NUM_BLOCKS * NUM_LAYERS * OUTER_DIM);
        assert!(
            !verifier.has_critical_mismatches(),
            "Expected no critical mismatches but got: total={}, size_mismatches={}, successful={}",
            verifier.stats.total_regions,
            verifier.stats.size_mismatches,
            verifier.stats.successful_verifications
        );
    }

    #[test]
    fn test_fully_contiguous_verification() {
        let layout =
            create_fully_contiguous_layout().expect("Failed to create FullyContiguous layout");
        let mut verifier = WorkerLayoutVerifier::new();
        let results = verifier
            .verify_layout_consistency(&layout)
            .expect("Failed to verify layout");
        assert_eq!(results.len(), NUM_BLOCKS * NUM_LAYERS * OUTER_DIM);

        assert!(
            !verifier.has_critical_mismatches(),
            "Expected no critical mismatches but got: total={}, size_mismatches={}, successful={}",
            verifier.stats.total_regions,
            verifier.stats.size_mismatches,
            verifier.stats.successful_verifications
        );
    }
}
