// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block checksum computation for verification.
//!
//! This module provides utilities to compute checksums of blocks for
//! round-trip test verification.

use crate::block_manager::v2::memory::StorageKind;

use super::PhysicalLayout;

use aligned_vec::{AVec, avec};
use anyhow::{Result, anyhow};
use blake3::Hasher;

use std::{
    collections::HashMap,
    fs::File,
    io::{Read, Seek},
    mem::ManuallyDrop,
    ops::Range,
    os::fd::FromRawFd,
};

use cudarc::runtime::sys::{cudaMemcpy, cudaMemcpyKind};

pub type BlockChecksum = String;

/// Compute checksums for a list of blocks.
///
/// # Arguments
/// * `layout` - The physical layout containing the blocks
/// * `block_ids` - List of block IDs to checksum
///
/// # Returns
/// A map from block ID to its checksum
///
/// # Errors
/// Returns an error if:
/// - Layout is remote (cannot checksum remote memory directly)
/// - Block IDs are out of range
pub fn compute_block_checksums(
    layout: &PhysicalLayout,
    block_ids: &[usize],
) -> Result<HashMap<usize, BlockChecksum>> {
    let mut checksums = HashMap::new();

    for &block_id in block_ids {
        let checksum = compute_single_block_checksum(layout, block_id, None)?;
        checksums.insert(block_id, checksum);
    }

    Ok(checksums)
}

/// Compute checksums for specific layers in blocks.
///
/// # Arguments
/// * `layout` - The physical layout containing the blocks
/// * `block_ids` - List of block IDs to checksum
/// * `layer_range` - Range of layers to include in checksum
///
/// # Returns
/// A map from block ID to its checksum (for the specified layers only)
pub fn compute_layer_checksums(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    layer_range: Range<usize>,
) -> Result<HashMap<usize, BlockChecksum>> {
    let config = layout.layout().config();
    if layer_range.end > config.num_layers {
        return Err(anyhow!(
            "Layer range {:?} exceeds num_layers {}",
            layer_range,
            config.num_layers
        ));
    }

    let mut checksums = HashMap::new();

    for &block_id in block_ids {
        let checksum = compute_single_block_checksum(layout, block_id, Some(layer_range.clone()))?;
        checksums.insert(block_id, checksum);
    }

    Ok(checksums)
}

/// Compute checksum for a single block.
fn compute_single_block_checksum(
    layout: &PhysicalLayout,
    block_id: usize,
    layer_range: Option<Range<usize>>,
) -> Result<String> {
    let config = layout.layout().config();

    if block_id >= config.num_blocks {
        return Err(anyhow!("Block ID {} out of range", block_id));
    }

    let num_layers = config.num_layers;
    let outer_dim = config.outer_dim;

    let layers = layer_range.unwrap_or(0..num_layers);

    // validate layer range
    if layers.end > config.num_layers {
        return Err(anyhow!(
            "Layer range {:?} exceeds num_layers {}",
            layers,
            config.num_layers
        ));
    }

    let mut hasher = Hasher::new();

    // Iterate over all layers and outer dimensions
    for layer_id in layers {
        for outer_id in 0..outer_dim {
            let region = layout.memory_region(block_id, layer_id, outer_id)?;

            match layout.location() {
                StorageKind::System | StorageKind::Pinned => {
                    let slice = unsafe {
                        std::slice::from_raw_parts(region.addr() as *const u8, region.size())
                    };
                    hasher.update(slice);
                }
                StorageKind::Device(_) => {
                    let system_region: Vec<u8> = vec![0; region.size()];
                    unsafe {
                        cudaMemcpy(
                            system_region.as_ptr() as *mut std::ffi::c_void,
                            region.addr() as *const std::ffi::c_void,
                            region.size(),
                            cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        );
                    }
                    hasher.update(system_region.as_slice());
                }
                StorageKind::Disk(fd) => {
                    let mut system_region: AVec<u8, _> = avec![[4096]| 0; region.size()];

                    let mut file = ManuallyDrop::new(unsafe { File::from_raw_fd(fd as i32) });
                    file.seek(std::io::SeekFrom::Start(region.addr() as u64))?;
                    file.read_exact(&mut system_region)?;
                    hasher.update(system_region.as_slice());
                }
            }
        }
    }

    Ok(hasher.finalize().to_string())
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::block_manager::v2::physical::transfer::{FillPattern, fill_blocks};

    #[test]
    fn test_checksum_constant_pattern() {
        let physical = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        fill_blocks(&physical, &[0, 1], FillPattern::Constant(42)).unwrap();

        let checksums = compute_block_checksums(&physical, &[0, 1]).unwrap();

        // Both blocks should have the same checksum values (same pattern)
        assert_eq!(checksums[&0], checksums[&1]);

        let memory_region = physical.memory_region(0, 0, 0).unwrap();
        let slice = unsafe {
            std::slice::from_raw_parts(memory_region.addr() as *const u8, memory_region.size())
        };
        assert!(slice.iter().all(|&b| b == 42));

        let mut hasher = Hasher::new();
        hasher.update(slice);
        let checksum_mr_slice = hasher.finalize().to_string();

        let vec = vec![42; memory_region.size()];
        let mut hasher = Hasher::new();
        hasher.update(&vec);
        let checksum_vec = hasher.finalize().to_string();

        assert_eq!(checksum_mr_slice, checksum_vec);
    }

    // #[test]
    // fn test_checksum_different_patterns() {
    //     let (layout, _memory) = create_test_layout(2);
    //     let physical = PhysicalLayout::new_local(layout, StorageLocation::System);

    //     // Fill blocks with different patterns
    //     fill_blocks(&physical, &[0], FillPattern::Constant(42)).unwrap();
    //     fill_blocks(&physical, &[1], FillPattern::Constant(100)).unwrap();

    //     let checksums = compute_block_checksums(&physical, &[0, 1]).unwrap();

    //     // Blocks should have different checksums
    //     assert_ne!(checksums[&0], checksums[&1]);
    // }

    // #[test]
    // fn test_checksum_matches() {
    //     let (layout1, _memory1) = create_test_layout(1);
    //     let (layout2, _memory2) = create_test_layout(1);

    //     let physical1 = PhysicalLayout::new_local(layout1, StorageLocation::System);
    //     let physical2 = PhysicalLayout::new_local(layout2, StorageLocation::System);

    //     // Fill both with same pattern
    //     fill_blocks(&physical1, &[0], FillPattern::Sequential).unwrap();
    //     fill_blocks(&physical2, &[0], FillPattern::Sequential).unwrap();

    //     let checksum1 = compute_block_checksums(&physical1, &[0]).unwrap();
    //     let checksum2 = compute_block_checksums(&physical2, &[0]).unwrap();

    //     // Checksums should match (ignoring block_id)
    //     assert!(checksum1[&0].matches(&checksum2[&0]));
    // }

    // #[test]
    // fn test_layer_checksums() {
    //     let (layout, _memory) = create_test_layout(1);
    //     let physical = PhysicalLayout::new_local(layout, StorageLocation::System);

    //     // Fill entire block
    //     fill_blocks(&physical, &[0], FillPattern::Sequential).unwrap();

    //     // Compute checksums for different layer ranges
    //     let full_checksum = compute_block_checksums(&physical, &[0]).unwrap();
    //     let layer0_checksum = compute_layer_checksums(&physical, &[0], 0..1).unwrap();
    //     let layer1_checksum = compute_layer_checksums(&physical, &[0], 1..2).unwrap();

    //     // Layer checksums should be different from full checksum
    //     assert_ne!(full_checksum[&0].byte_count, layer0_checksum[&0].byte_count);
    //     assert_ne!(full_checksum[&0].byte_count, layer1_checksum[&0].byte_count);

    //     // Layer 0 and Layer 1 should have same byte count (same size)
    //     assert_eq!(
    //         layer0_checksum[&0].byte_count,
    //         layer1_checksum[&0].byte_count
    //     );
    // }

    // #[test]
    // fn test_checksum_remote_layout_fails() {
    //     let (layout, _memory) = create_test_layout(1);
    //     let physical =
    //         PhysicalLayout::new_remote(layout, StorageLocation::System, "remote".to_string());

    //     let result = compute_block_checksums(&physical, &[0]);
    //     assert!(result.is_err());
    //     assert!(result.unwrap_err().to_string().contains("remote"));
    // }
}
