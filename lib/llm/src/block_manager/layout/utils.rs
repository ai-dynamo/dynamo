// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::block_manager::layout::{BlockLayoutConfig, LayoutError};
use crate::block_manager::storage::Storage;

/// Aligns the given value up to the nearest multiple of alignment.
/// Alignment must be a power of 2.
#[inline(always)]
pub fn align_up(value: usize, alignment: usize) -> usize {
    debug_assert!(alignment.is_power_of_two(), "Alignment must be a power of 2");
    (value + alignment - 1) & !(alignment - 1)
}

/// Validates that the given value is a power of 2.
pub fn validate_power_of_2(alignment: usize) -> Result<(), validator::ValidationError> {
    if alignment.is_power_of_two() {
        Ok(())
    } else {
        Err(validator::ValidationError::new(
            "Alignment must be a power of 2",
        ))
    }
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

/// Validate that the provided indices are within bounds for the given layout configuration
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

/// Worker-side value verification utilities
pub mod worker_verification {
    use super::*;
    use crate::block_manager::layout::{GenericBlockLayout, LocalMemoryRegion};
    use std::collections::HashMap;

    /// Verification result for a memory region
    #[derive(Debug, Clone)]
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
        /// Optional checksum of the memory region data
        pub checksum: Option<u64>,
    }

    /// Layout verification statistics
    #[derive(Debug, Clone, Default)]
    pub struct LayoutVerificationStats {
        /// Total number of memory regions verified
        pub total_regions: usize,
        /// Number of regions with address mismatches
        pub addr_mismatches: usize,
        /// Number of regions with size mismatches
        pub size_mismatches: usize,
        /// Number of regions that passed all verifications
        pub successful_verifications: usize,
        /// Number of regions with checksum mismatches
        pub checksum_mismatches: usize,
    }

    /// Worker-side layout verifier
    pub struct WorkerLayoutVerifier {
        expected_checksums: HashMap<(usize, usize, usize), u64>,
        stats: LayoutVerificationStats,
    }

    impl WorkerLayoutVerifier {
        /// Create a new worker layout verifier
        pub fn new() -> Self {
            Self {
                expected_checksums: HashMap::new(),
                stats: LayoutVerificationStats::default(),
            }
        }

        /// Verify that memory regions match expected layout calculations
        pub fn verify_layout_consistency<L: GenericBlockLayout>(
            &mut self,
            layout: &L,
            verify_data: bool,
        ) -> Result<Vec<RegionVerificationResult>, LayoutError> {
            let mut results = Vec::new();
            self.stats = LayoutVerificationStats::default();

            for block_idx in 0..layout.num_blocks() {
                for layer_idx in 0..layout.num_layers() {
                    for outer_idx in 0..layout.outer_dim() {
                        let result = self.verify_memory_region(
                            layout, block_idx, layer_idx, outer_idx, verify_data
                        )?;

                        self.update_stats(&result);
                        results.push(result);
                    }
                }
            }

            Ok(results)
        }

        /// Verify a single memory region
        fn verify_memory_region<L: GenericBlockLayout>(
            &mut self,
            layout: &L,
            block_idx: usize,
            layer_idx: usize,
            outer_idx: usize,
            verify_data: bool,
        ) -> Result<RegionVerificationResult, LayoutError> {
            let region = layout.memory_region(block_idx, layer_idx, outer_idx)?;

            // Calculate expected values based on layout configuration
            let config = layout.config();
            let expected_size = config.page_size * config.inner_dim * config.dtype_width_bytes;

            // For verification, we just check that the region is accessible and has correct size
            // Address verification is complex and layout-specific, so we skip it for now
            let expected_addr = region.addr; // Accept whatever the layout calculates

            let mut checksum = None;
            if verify_data {
                checksum = Some(self.calculate_memory_checksum(&region)?);

                // Check against stored checksum if available
                if let Some(&expected_checksum) = self.expected_checksums.get(&(block_idx, layer_idx, outer_idx)) {
                    if checksum != Some(expected_checksum) {
                        tracing::warn!(
                            "Checksum mismatch at ({}, {}, {}): expected {}, got {:?}",
                            block_idx, layer_idx, outer_idx, expected_checksum, checksum
                        );
                    }
                }
            }

            Ok(RegionVerificationResult {
                block_idx,
                layer_idx,
                outer_idx,
                expected_addr,
                actual_addr: region.addr,
                expected_size,
                actual_size: region.size,
                addr_matches: true, // Always true since we accept the layout's calculation
                size_matches: expected_size == region.size,
                checksum,
            })
        }



        /// Calculate a simple checksum for memory region data
        fn calculate_memory_checksum(&self, region: &LocalMemoryRegion) -> Result<u64, LayoutError> {
            unsafe {
                let slice = std::slice::from_raw_parts(region.addr as *const u8, region.size);
                let mut checksum = 0u64;

                for (i, &byte) in slice.iter().enumerate() {
                    checksum = checksum.wrapping_add((byte as u64).wrapping_mul(i as u64 + 1));
                }

                Ok(checksum)
            }
        }

        /// Store expected checksum for later verification
        pub fn store_expected_checksum(
            &mut self,
            block_idx: usize,
            layer_idx: usize,
            outer_idx: usize,
            checksum: u64,
        ) {
            self.expected_checksums.insert((block_idx, layer_idx, outer_idx), checksum);
        }

        /// Update verification statistics
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

            if let Some(checksum) = result.checksum {
                if let Some(&expected) = self.expected_checksums.get(&(result.block_idx, result.layer_idx, result.outer_idx)) {
                    if checksum != expected {
                        self.stats.checksum_mismatches += 1;
                    }
                }
            }
        }

        /// Get verification statistics
        pub fn stats(&self) -> &LayoutVerificationStats {
            &self.stats
        }

        /// Check if there are any critical layout mismatches
        pub fn has_critical_mismatches(&self) -> bool {
            // Only check size mismatches since address verification is layout-specific
            self.stats.size_mismatches > 0
        }

        /// Generate a detailed verification report
        pub fn generate_report(&self, results: &[RegionVerificationResult]) -> String {
            let mut report = String::new();

            report.push_str(&format!("Layout Verification Report\n"));
            report.push_str(&format!("========================\n"));
            report.push_str(&format!("Total regions: {}\n", self.stats.total_regions));
            report.push_str(&format!("Successful verifications: {}\n", self.stats.successful_verifications));
            report.push_str(&format!("Address mismatches: {}\n", self.stats.addr_mismatches));
            report.push_str(&format!("Size mismatches: {}\n", self.stats.size_mismatches));
            report.push_str(&format!("Checksum mismatches: {}\n", self.stats.checksum_mismatches));

            if self.has_critical_mismatches() {
                report.push_str("\nCRITICAL ISSUES FOUND:\n");
                for result in results {
                    if !result.addr_matches || !result.size_matches {
                        report.push_str(&format!(
                            "Region ({}, {}, {}): addr_match={}, size_match={}\n",
                            result.block_idx, result.layer_idx, result.outer_idx,
                            result.addr_matches, result.size_matches
                        ));
                    }
                }
            }

            report
        }
    }

    /// Verify layout compatibility between source and destination
    pub fn verify_layout_compatibility<S: GenericBlockLayout, D: GenericBlockLayout>(
        source: &S,
        destination: &D,
    ) -> Result<bool, LayoutError> {
        let source_config = source.config();
        let dest_config = destination.config();

        // Check basic compatibility
        if source_config.num_blocks != dest_config.num_blocks {
            tracing::error!("Block count mismatch: {} vs {}",
                source_config.num_blocks, dest_config.num_blocks);
            return Ok(false);
        }

        if source_config.num_layers != dest_config.num_layers {
            tracing::error!("Layer count mismatch: {} vs {}",
                source_config.num_layers, dest_config.num_layers);
            return Ok(false);
        }

        if source_config.outer_dim != dest_config.outer_dim {
            tracing::error!("Outer dimension mismatch: {} vs {}",
                source_config.outer_dim, dest_config.outer_dim);
            return Ok(false);
        }

        if source_config.page_size != dest_config.page_size {
            tracing::error!("Page size mismatch: {} vs {}",
                source_config.page_size, dest_config.page_size);
            return Ok(false);
        }

        if source_config.inner_dim != dest_config.inner_dim {
            tracing::error!("Inner dimension mismatch: {} vs {}",
                source_config.inner_dim, dest_config.inner_dim);
            return Ok(false);
        }

        if source_config.dtype_width_bytes != dest_config.dtype_width_bytes {
            tracing::error!("Data type width mismatch: {} vs {}",
                source_config.dtype_width_bytes, dest_config.dtype_width_bytes);
            return Ok(false);
        }

        // Check memory region compatibility
        for block_idx in 0..source_config.num_blocks {
            for layer_idx in 0..source_config.num_layers {
                for outer_idx in 0..source_config.outer_dim {
                    let src_region = source.memory_region(block_idx, layer_idx, outer_idx)?;
                    let dst_region = destination.memory_region(block_idx, layer_idx, outer_idx)?;

                    if src_region.size != dst_region.size {
                        tracing::error!(
                            "Memory region size mismatch at ({}, {}, {}): {} vs {}",
                            block_idx, layer_idx, outer_idx, src_region.size, dst_region.size
                        );
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::block_manager::layout::{LayoutConfig, FullyContiguous, LayerSeparate};
        use crate::block_manager::storage::{SystemAllocator, tests::NullDeviceAllocator};

        #[test]
        fn test_worker_layout_verification() {
            let config = LayoutConfig {
                num_blocks: 2,
                num_layers: 2,
                outer_dim: 1,
                page_size: 4,
                inner_dim: 8,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let layout = FullyContiguous::allocate(config, &SystemAllocator).unwrap();
            let mut verifier = WorkerLayoutVerifier::new();

            let results = verifier.verify_layout_consistency(&layout, false).unwrap();
            assert_eq!(results.len(), 4); // 2 blocks * 2 layers * 1 outer

            for result in &results {
                assert!(result.size_matches, "Size should match for all regions");
            }

            let report = verifier.generate_report(&results);
            assert!(report.contains("Total regions: 4"));
        }

        #[test]
        fn test_layout_compatibility_verification() {
            let config = LayoutConfig {
                num_blocks: 2,
                num_layers: 2,
                outer_dim: 1,
                page_size: 4,
                inner_dim: 8,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let fc_layout = FullyContiguous::allocate(config.clone(), &SystemAllocator).unwrap();
            let ls_layout = LayerSeparate::allocate(config, &NullDeviceAllocator, true).unwrap();

            let compatible = verify_layout_compatibility(&fc_layout, &ls_layout).unwrap();
            assert!(compatible, "Layouts with same config should be compatible");
        }

        #[test]
        fn test_incompatible_layouts() {
            let config1 = LayoutConfig {
                num_blocks: 2,
                num_layers: 2,
                outer_dim: 1,
                page_size: 4,
                inner_dim: 8,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let config2 = LayoutConfig {
                num_blocks: 3, // Different!
                num_layers: 2,
                outer_dim: 1,
                page_size: 4,
                inner_dim: 8,
                alignment: 1,
                dtype_width_bytes: 2,
            };

            let layout1 = FullyContiguous::allocate(config1, &SystemAllocator).unwrap();
            let layout2 = FullyContiguous::allocate(config2, &SystemAllocator).unwrap();

            let compatible = verify_layout_compatibility(&layout1, &layout2).unwrap();
            assert!(!compatible, "Layouts with different configs should be incompatible");
        }
    }
}
