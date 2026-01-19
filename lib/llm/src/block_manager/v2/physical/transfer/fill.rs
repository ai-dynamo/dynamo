// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Block filling operations for testing.
//!
//! This module provides utilities to populate blocks with specific patterns
//! for verification in round-trip tests.

use super::PhysicalLayout;

use crate::block_manager::v2::memory::StorageKind;
use aligned_vec::{AVec, avec};
use anyhow::{Result, anyhow};
use cudarc::runtime::sys::{cudaMemcpy, cudaMemcpyKind};

use std::{
    fs::File,
    io::{Seek, Write},
    mem::ManuallyDrop,
    ops::Range,
    os::fd::FromRawFd,
};

/// Fill strategy for block memory.
#[derive(Debug, Clone, Copy)]
pub enum FillPattern {
    /// Fill with a constant byte value
    Constant(u8),

    /// Fill with a sequential pattern: block_id + layer_id + offset % 256
    Sequential,
}

/// Fill blocks in a physical layout with a specific pattern.
///
/// This operation directly writes to memory and should only be used on
/// local layouts. Remote layouts cannot be filled directly.
///
/// # Arguments
/// * `layout` - The physical layout containing the blocks
/// * `block_ids` - List of block IDs to fill
/// * `pattern` - Fill pattern to use
///
/// # Errors
/// Returns an error if:
/// - Layout is remote (cannot fill remote memory directly)
/// - Block IDs are out of range
/// - Memory access fails
pub fn fill_blocks(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    pattern: FillPattern,
) -> Result<()> {
    // Can only fill local layouts
    let config = layout.layout().config();
    let num_layers = config.num_layers;
    let outer_dim = config.outer_dim;

    for &block_id in block_ids {
        if block_id >= config.num_blocks {
            return Err(anyhow!("Block ID {} out of range", block_id));
        }

        // Fill all layers and outer dimensions for this block
        for layer_id in 0..num_layers {
            for outer_id in 0..outer_dim {
                let region = layout.memory_region(block_id, layer_id, outer_id)?;

                match layout.location() {
                    StorageKind::System | StorageKind::Pinned => {
                        fill_memory_region(
                            region.addr(),
                            region.size(),
                            block_id,
                            layer_id,
                            pattern,
                        )?;
                    }
                    StorageKind::Device(_) => {
                        let system_region: Vec<u8> = vec![0; region.size()];
                        fill_memory_region(
                            system_region.as_ptr() as usize,
                            system_region.len(),
                            block_id,
                            layer_id,
                            pattern,
                        )?;
                        unsafe {
                            cudaMemcpy(
                                region.addr() as *mut std::ffi::c_void,
                                system_region.as_ptr() as *const std::ffi::c_void,
                                region.size(),
                                cudaMemcpyKind::cudaMemcpyHostToDevice,
                            );
                        }
                    }
                    StorageKind::Disk(fd) => {
                        let system_region: AVec<u8, _> = avec![[4096]| 0; region.size()];
                        fill_memory_region(
                            system_region.as_ptr() as usize,
                            system_region.len(),
                            block_id,
                            layer_id,
                            pattern,
                        )?;

                        let mut file = ManuallyDrop::new(unsafe { File::from_raw_fd(fd as i32) });

                        file.seek(std::io::SeekFrom::Start(region.addr() as u64))?;
                        file.write_all(&system_region)?;
                        file.sync_all()?;
                        file.flush()?;
                    }
                }
            }
        }
    }

    Ok(())
}

/// Fill a subset of layers in blocks with a specific pattern.
///
/// # Arguments
/// * `layout` - The physical layout containing the blocks
/// * `block_ids` - List of block IDs to fill
/// * `layer_range` - Range of layers to fill
/// * `pattern` - Fill pattern to use
pub fn fill_layers(
    layout: &PhysicalLayout,
    block_ids: &[usize],
    layer_range: Range<usize>,
    pattern: FillPattern,
) -> Result<()> {
    let config = layout.layout().config();
    let num_layers = config.num_layers;
    let outer_dim = config.outer_dim;

    if layer_range.end > num_layers {
        return Err(anyhow!(
            "Layer range {:?} exceeds num_layers {}",
            layer_range,
            num_layers
        ));
    }

    for &block_id in block_ids {
        if block_id >= config.num_blocks {
            return Err(anyhow!("Block ID {} out of range", block_id));
        }

        // Fill specified layers and all outer dimensions
        for layer_id in layer_range.clone() {
            for outer_id in 0..outer_dim {
                let region = layout.memory_region(block_id, layer_id, outer_id)?;
                fill_memory_region(region.addr(), region.size(), block_id, layer_id, pattern)?;
            }
        }
    }

    Ok(())
}

/// Fill a memory region with the specified pattern.
///
/// # Safety
/// This function performs unsafe memory writes. The caller must ensure:
/// - The memory region is valid and accessible
/// - No other references exist to this memory
fn fill_memory_region(
    addr: usize,
    size: usize,
    block_id: usize,
    layer_id: usize,
    pattern: FillPattern,
) -> Result<()> {
    unsafe {
        let ptr = addr as *mut u8;
        match pattern {
            FillPattern::Constant(value) => {
                std::ptr::write_bytes(ptr, value, size);
            }
            FillPattern::Sequential => {
                for offset in 0..size {
                    let value = ((block_id + layer_id + offset) % 256) as u8;
                    ptr.add(offset).write(value);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::tests::*;
    use super::*;
    use crate::block_manager::v2::memory::actions::Slice;

    #[test]
    fn test_fill_blocks_constant() {
        let physical = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        fill_blocks(&physical, &[0, 1], FillPattern::Constant(42)).unwrap();

        // Verify all bytes are set to 42
        assert!(
            physical
                .memory_region(0, 0, 0)
                .unwrap()
                .as_slice()
                .unwrap()
                .iter()
                .all(|&b| b == 42)
        );
    }

    #[test]
    fn test_fill_blocks_sequential() {
        let physical = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        fill_blocks(&physical, &[0, 1], FillPattern::Sequential).unwrap();

        let mr = physical.memory_region(0, 0, 0).unwrap();
        let mr_slice = mr.as_slice().unwrap();

        // Verify pattern is applied (spot check a few bytes)
        let first_byte = mr_slice[0];
        let second_byte = mr_slice[1];
        assert_eq!(first_byte, 0);
        assert_eq!(second_byte, first_byte.wrapping_add(1));

        let mr = physical.memory_region(1, 1, 0).unwrap();
        let mr_slice = mr.as_slice().unwrap();

        let first_byte = mr_slice[0];
        let second_byte = mr_slice[1];
        assert_eq!(first_byte, 2);
        assert_eq!(second_byte, first_byte.wrapping_add(1));
    }

    #[test]
    fn test_fill_layers() {
        let physical = builder(2)
            .fully_contiguous()
            .allocate_system()
            .build()
            .unwrap();

        // Fill only layer 0
        fill_layers(&physical, &[0], 0..1, FillPattern::Constant(0)).unwrap();
        fill_layers(&physical, &[0], 1..2, FillPattern::Constant(1)).unwrap();
        fill_layers(&physical, &[1], 0..1, FillPattern::Constant(100)).unwrap();
        fill_layers(&physical, &[1], 1..2, FillPattern::Constant(101)).unwrap();

        let mr_00 = physical.memory_region(0, 0, 0).unwrap().as_slice().unwrap()[0];
        let mr_01 = physical.memory_region(0, 1, 0).unwrap().as_slice().unwrap()[0];
        let mr_10 = physical.memory_region(1, 0, 0).unwrap().as_slice().unwrap()[0];
        let mr_11 = physical.memory_region(1, 1, 0).unwrap().as_slice().unwrap()[0];
        assert_eq!(mr_00, 0);
        assert_eq!(mr_01, 1);
        assert_eq!(mr_10, 100);
        assert_eq!(mr_11, 101);
    }
}
