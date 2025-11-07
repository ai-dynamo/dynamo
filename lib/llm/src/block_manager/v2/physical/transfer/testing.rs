// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round-trip testing infrastructure for transfer verification.
//!
//! This module provides utilities for testing data integrity across transfers
//! by comparing checksums after round-trip operations:
//! 1. Source blocks (host) → Intermediate (device/disk/remote)
//! 2. Intermediate → Destination blocks (host, different IDs)
//! 3. Verify checksums match between source and destination

use super::{
    BlockChecksum, FillPattern, PhysicalLayout, StorageKind, compute_block_checksums,
    fill_blocks, transfer_blocks,
};
use super::context::TransferContext;
use anyhow::{Result, anyhow};
use std::collections::HashMap;

/// Result of a round-trip test.
#[derive(Debug)]
pub struct RoundTripTestResult {
    /// Source block checksums (keyed by source block ID)
    pub source_checksums: HashMap<usize, BlockChecksum>,

    /// Destination block checksums (keyed by destination block ID)
    pub dest_checksums: HashMap<usize, BlockChecksum>,

    /// Block ID mapping used (src_id, dst_id)
    pub block_mapping: Vec<(usize, usize)>,

    /// Whether all checksums matched
    pub success: bool,

    /// Mismatched blocks (if any)
    pub mismatches: Vec<(usize, usize)>, // (src_id, dst_id) pairs that didn't match
}

impl RoundTripTestResult {
    /// Check if the round-trip test passed.
    pub fn is_success(&self) -> bool {
        self.success
    }

    /// Get the number of blocks tested.
    pub fn num_blocks(&self) -> usize {
        self.block_mapping.len()
    }

    /// Get a detailed report of the test results.
    pub fn report(&self) -> String {
        if self.success {
            format!(
                "Round-trip test PASSED: {}/{} blocks verified successfully",
                self.num_blocks(),
                self.num_blocks()
            )
        } else {
            format!(
                "Round-trip test FAILED: {}/{} blocks mismatched\nMismatches: {:?}",
                self.mismatches.len(),
                self.num_blocks(),
                self.mismatches
            )
        }
    }
}

/// Builder for round-trip tests.
///
/// This allows configuring a test that transfers data from source blocks
/// to intermediate storage and back to different destination blocks,
/// verifying data integrity via checksums.
pub struct RoundTripTest {
    /// Source physical layout (must be local)
    source: PhysicalLayout,

    /// Intermediate physical layout (can be remote/device/disk)
    intermediate: PhysicalLayout,

    /// Destination physical layout (must be local)
    destination: PhysicalLayout,

    /// Block mapping: (src_id, intermediate_id, dst_id)
    block_mapping: Vec<(usize, usize, usize)>,

    /// Fill pattern for source blocks
    fill_pattern: FillPattern,
}

impl RoundTripTest {
    /// Create a new round-trip test.
    ///
    /// # Arguments
    /// * `source` - Source physical layout (must be local)
    /// * `intermediate` - Intermediate physical layout
    /// * `destination` - Destination physical layout (must be local)
    pub fn new(
        source: PhysicalLayout,
        intermediate: PhysicalLayout,
        destination: PhysicalLayout,
    ) -> Result<Self> {
        if source.is_remote() {
            return Err(anyhow!("Source layout must be local"));
        }
        if destination.is_remote() {
            return Err(anyhow!("Destination layout must be local"));
        }

        Ok(Self {
            source,
            intermediate,
            destination,
            block_mapping: Vec::new(),
            fill_pattern: FillPattern::Sequential,
        })
    }

    /// Set the fill pattern for source blocks.
    pub fn with_fill_pattern(mut self, pattern: FillPattern) -> Self {
        self.fill_pattern = pattern;
        self
    }

    /// Add a block mapping for the round-trip test.
    ///
    /// # Arguments
    /// * `src_id` - Source block ID
    /// * `intermediate_id` - Intermediate block ID
    /// * `dst_id` - Destination block ID
    pub fn add_block_mapping(
        mut self,
        src_id: usize,
        intermediate_id: usize,
        dst_id: usize,
    ) -> Self {
        self.block_mapping.push((src_id, intermediate_id, dst_id));
        self
    }

    /// Add multiple block mappings at once.
    ///
    /// This is a convenience method for adding several mappings.
    pub fn with_block_mappings(mut self, mappings: &[(usize, usize, usize)]) -> Self {
        self.block_mapping.extend_from_slice(mappings);
        self
    }

    /// Run the round-trip test.
    ///
    /// # Workflow
    /// 1. Fill source blocks with the specified pattern
    /// 2. Compute source checksums
    /// 3. Transfer source → intermediate
    /// 4. Transfer intermediate → destination
    /// 5. Compute destination checksums
    /// 6. Compare checksums
    ///
    /// # Arguments
    /// * `ctx` - Transfer context with CUDA stream and NIXL agent
    pub async fn run(self, ctx: &TransferContext) -> Result<RoundTripTestResult> {
        if self.block_mapping.is_empty() {
            return Err(anyhow!("No block mappings specified"));
        }

        // Step 1: Fill source blocks
        let src_ids: Vec<usize> = self.block_mapping.iter().map(|(src, _, _)| *src).collect();
        fill_blocks(&self.source, &src_ids, self.fill_pattern)?;

        // Step 2: Compute source checksums
        let source_checksums = compute_block_checksums(&self.source, &src_ids)?;

        // Step 3: Transfer source → intermediate
        let src_ids_intermediate: Vec<usize> =
            self.block_mapping.iter().map(|(src, _, _)| *src).collect();
        let inter_ids_from_src: Vec<usize> = self
            .block_mapping
            .iter()
            .map(|(_, inter, _)| *inter)
            .collect();
        let notification = transfer_blocks(
            &self.source,
            &self.intermediate,
            &src_ids_intermediate,
            &inter_ids_from_src,
            ctx,
        )?;
        notification.await?;

        // Step 4: Transfer intermediate → destination
        let inter_ids_to_dst: Vec<usize> = self
            .block_mapping
            .iter()
            .map(|(_, inter, _)| *inter)
            .collect();
        let dst_ids_from_inter: Vec<usize> =
            self.block_mapping.iter().map(|(_, _, dst)| *dst).collect();
        let notification = transfer_blocks(
            &self.intermediate,
            &self.destination,
            &inter_ids_to_dst,
            &dst_ids_from_inter,
            ctx,
        )?;
        notification.await?;

        // Step 5: Compute destination checksums
        let dst_ids: Vec<usize> = self.block_mapping.iter().map(|(_, _, dst)| *dst).collect();
        let dest_checksums = compute_block_checksums(&self.destination, &dst_ids)?;

        // Step 6: Compare checksums
        let mut mismatches = Vec::new();
        for (src_id, _, dst_id) in &self.block_mapping {
            let src_checksum = &source_checksums[src_id];
            let dst_checksum = &dest_checksums[dst_id];

            if src_checksum != dst_checksum {
                mismatches.push((*src_id, *dst_id));
            }
        }

        let success = mismatches.is_empty();
        let block_mapping: Vec<(usize, usize)> = self
            .block_mapping
            .iter()
            .map(|(src, _, dst)| (*src, *dst))
            .collect();

        Ok(RoundTripTestResult {
            source_checksums,
            dest_checksums,
            block_mapping,
            success,
            mismatches,
        })
    }
}

#[cfg(test, features = "testing-cuda")]
mod tests {
    use super::*;
    use crate::block_manager::v2::layout::{
        FullyContiguousLayout, Layout, LayoutConfig, MemoryRegion, OwnedMemoryRegion,
    };
    use std::sync::Arc;

    // Helper to create a minimal transfer context for testing
    // In real tests with CUDA/NIXL, this would be properly constructed
    fn create_test_context() -> TransferContext {
        // For now, we'll skip these tests if CUDA is not available
        // In the future, we can mock TransferContext or use conditional compilation
        todo!("Create test context - requires CUDA/NIXL setup")
    }

    #[tokio::test]
    async fn test_round_trip_host_to_host() {
        // Create three layouts: source, intermediate, destination
        let (src_layout, _src_mem) = create_test_layout(4);
        let (inter_layout, _inter_mem) = create_test_layout(4);
        let (dst_layout, _dst_mem) = create_test_layout(4);

        let source = PhysicalLayout::new_local(src_layout, StorageKind::System);
        let intermediate = PhysicalLayout::new_local(inter_layout, StorageKind::Pinned);
        let destination = PhysicalLayout::new_local(dst_layout, StorageKind::System);

        // Build round-trip test with different block IDs
        // Source: blocks [0, 1, 2, 3]
        // Intermediate: blocks [0, 1, 2, 3]
        // Destination: blocks [0, 1, 2, 3] (different memory than source)
        let test = RoundTripTest::new(source, intermediate, destination)
            .unwrap()
            .with_fill_pattern(FillPattern::Sequential)
            .add_block_mapping(0, 0, 0)
            .add_block_mapping(1, 1, 1)
            .add_block_mapping(2, 2, 2)
            .add_block_mapping(3, 3, 3);

        // Create a transfer context (requires actual CUDA/NIXL setup)
        let ctx = create_test_context();

        // Run the test
        let result = test.run(&ctx).await.unwrap();

        assert!(result.is_success(), "{}", result.report());
        assert_eq!(result.num_blocks(), 4);
    }

    #[tokio::test]
    async fn test_round_trip_different_block_ids() {
        // Create layouts with enough blocks
        let (src_layout, _src_mem) = create_test_layout(8);
        let (inter_layout, _inter_mem) = create_test_layout(8);
        let (dst_layout, _dst_mem) = create_test_layout(8);

        let source = PhysicalLayout::new_local(src_layout, StorageKind::System);
        let intermediate = PhysicalLayout::new_local(inter_layout, StorageKind::Pinned);
        let destination = PhysicalLayout::new_local(dst_layout, StorageKind::System);

        // Test with non-overlapping block IDs
        // Source: blocks [0, 1, 2, 3]
        // Intermediate: blocks [2, 3, 4, 5]
        // Destination: blocks [4, 5, 6, 7]
        let test = RoundTripTest::new(source, intermediate, destination)
            .unwrap()
            .with_fill_pattern(FillPattern::BlockBased)
            .with_block_mappings(&[(0, 2, 4), (1, 3, 5), (2, 4, 6), (3, 5, 7)]);

        let ctx = create_test_context();
        let result = test.run(&ctx).await.unwrap();

        assert!(result.is_success(), "{}", result.report());
        assert_eq!(result.num_blocks(), 4);
    }

    #[test]
    fn test_round_trip_builder() {
        let (src_layout, _) = create_test_layout(4);
        let (inter_layout, _) = create_test_layout(4);
        let (dst_layout, _) = create_test_layout(4);

        let source = PhysicalLayout::new_local(src_layout, StorageKind::System);
        let intermediate = PhysicalLayout::new_local(inter_layout, StorageKind::Pinned);
        let destination = PhysicalLayout::new_local(dst_layout, StorageKind::System);

        let test = RoundTripTest::new(source, intermediate, destination)
            .unwrap()
            .with_fill_pattern(FillPattern::Constant(42))
            .add_block_mapping(0, 0, 1)
            .add_block_mapping(1, 1, 2);

        assert_eq!(test.block_mapping.len(), 2);
    }

    #[test]
    fn test_round_trip_requires_local_source() {
        let (src_layout, _) = create_test_layout(1);
        let (inter_layout, _) = create_test_layout(1);
        let (dst_layout, _) = create_test_layout(1);

        let source =
            PhysicalLayout::new_remote(src_layout, StorageKind::System, "remote".to_string());
        let intermediate = PhysicalLayout::new_local(inter_layout, StorageKind::Pinned);
        let destination = PhysicalLayout::new_local(dst_layout, StorageKind::System);

        let result = RoundTripTest::new(source, intermediate, destination);
        assert!(result.is_err());
    }

    #[test]
    fn test_round_trip_requires_local_destination() {
        let (src_layout, _) = create_test_layout(1);
        let (inter_layout, _) = create_test_layout(1);
        let (dst_layout, _) = create_test_layout(1);

        let source = PhysicalLayout::new_local(src_layout, StorageKind::System);
        let intermediate = PhysicalLayout::new_local(inter_layout, StorageKind::Pinned);
        let destination =
            PhysicalLayout::new_remote(dst_layout, StorageKind::System, "remote".to_string());

        let result = RoundTripTest::new(source, intermediate, destination);
        assert!(result.is_err());
    }
}
