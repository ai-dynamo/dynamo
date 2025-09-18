// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::TransferError;
use crate::block_manager::storage::{DeviceStorage, PinnedStorage};
use anyhow::Result;
use cudarc::driver::result as cuda_result;
use std::ops::Range;

type CudaMemcpyFnPtr = unsafe fn(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError>;

fn cuda_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<CudaMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::CudaAsyncH2D => Ok(cuda_memcpy_h2d),
        TransferStrategy::CudaAsyncD2H => Ok(cuda_memcpy_d2h),
        TransferStrategy::CudaAsyncD2D => Ok(cuda_memcpy_d2d),
        _ => Err(TransferError::ExecutionError(
            "Unsupported copy strategy for CUDA memcpy async".into(),
        )),
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        let expected_strategy =
            expected_strategy::<Source::StorageType, Destination::StorageType>();
        assert_eq!(strategy, expected_strategy);
    }

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());

        unsafe {
            memcpy_fn(
                src_view.as_ptr(),
                dst_view.as_mut_ptr(),
                src_view.size(),
                stream,
            )?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            stream,
            strategy,
        )?;
    }
    Ok(())
}

/// Copy a range of layers from a source to a destination using CUDA memcpy
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    stream: &CudaStream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        let expected_strategy =
            expected_strategy::<Source::StorageType, Destination::StorageType>();
        assert_eq!(strategy, expected_strategy);
    }

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            unsafe {
                memcpy_fn(
                    src_view.as_ptr(),
                    dst_view.as_mut_ptr(),
                    src_view.size(),
                    stream,
                )?;
            }
        }
    }
    Ok(())
}

/// Helper function to perform the appropriate CUDA memcpy based on storage types
// Allow dead code because it's used in debug assertions
#[allow(dead_code)]
fn expected_strategy<Source: Storage, Dest: Storage>() -> TransferStrategy {
    match (
        std::any::TypeId::of::<Source>(),
        std::any::TypeId::of::<Dest>(),
    ) {
        (src, dst)
            if src == std::any::TypeId::of::<PinnedStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            TransferStrategy::CudaAsyncH2D
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<PinnedStorage>() =>
        {
            TransferStrategy::CudaAsyncD2H
        }
        (src, dst)
            if src == std::any::TypeId::of::<DeviceStorage>()
                && dst == std::any::TypeId::of::<DeviceStorage>() =>
        {
            TransferStrategy::CudaAsyncD2D
        }
        _ => TransferStrategy::Invalid,
    }
}

/// H2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        let src_slice = std::slice::from_raw_parts(src_ptr, size);
        cuda_result::memcpy_htod_async(dst_ptr as u64, src_slice, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?
    };
    Ok(())
}

/// D2H Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
        cuda_result::memcpy_dtoh_async(dst_slice, src_ptr as u64, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
    }
    Ok(())
}

/// D2D Implementation
#[inline(always)]
unsafe fn cuda_memcpy_d2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination device pointer is null");
    debug_assert!(
        (src_ptr as usize + size <= dst_ptr as usize)
            || (dst_ptr as usize + size <= src_ptr as usize),
        "Source and destination device memory regions must not overlap for D2D copy"
    );

    unsafe {
        cuda_result::memcpy_dtod_async(dst_ptr as u64, src_ptr as u64, size, stream.cu_stream())
            .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?
    };
    Ok(())
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::storage::{
        DeviceAllocator, PinnedAllocator, StorageAllocator, StorageMemset,
    };

    #[test]
    fn test_memset_and_transfer() {
        // Create allocators
        let device_allocator = DeviceAllocator::default();
        let pinned_allocator = PinnedAllocator::default();

        let ctx = device_allocator.ctx().clone();

        // Create CUDA stream
        let stream = ctx.new_stream().unwrap();

        // Allocate host and device memory
        let mut host = pinned_allocator.allocate(1024).unwrap();
        let mut device = device_allocator.allocate(1024).unwrap();

        // Set a pattern in host memory
        StorageMemset::memset(&mut host, 42, 0, 1024).unwrap();

        // Verify host memory was set correctly
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }

        // Copy host to device
        unsafe {
            cuda_memcpy_h2d(host.as_ptr(), device.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure H2D copy is complete
        stream.synchronize().unwrap();

        // Clear host memory
        StorageMemset::memset(&mut host, 0, 0, 1024).unwrap();

        // Verify host memory was cleared
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 0));
        }

        // Copy back from device to host
        unsafe {
            cuda_memcpy_d2h(device.as_ptr(), host.as_mut_ptr(), 1024, stream.as_ref()).unwrap();
        }

        // Synchronize to ensure D2H copy is complete before verifying
        stream.synchronize().unwrap();

        // Verify the original pattern was restored
        unsafe {
            let ptr = host.as_ptr();
            let slice = std::slice::from_raw_parts(ptr, 1024);
            assert!(slice.iter().all(|&x| x == 42));
        }
    }

    // ============================================================================
    // CUDA TRANSFER TESTS FOR LAYOUT COMPATIBILITY
    // ============================================================================

    mod layout_transfer_tests {
        use super::*;
        use crate::block_manager::layout::{FullyContiguous, LayerSeparate, LayoutConfig, LayoutType, GenericBlockLayout};
        use crate::block_manager::storage::{DeviceStorage, PinnedStorage, SystemStorage};

        const TEST_NUM_BLOCKS: usize = 4;
        const TEST_NUM_LAYERS: usize = 3;
        const TEST_OUTER_DIM: usize = 2;
        const TEST_PAGE_SIZE: usize = 8;
        const TEST_INNER_DIM: usize = 16;
        const TEST_DTYPE_WIDTH_BYTES: usize = 2;

        fn create_test_config() -> LayoutConfig {
            LayoutConfig {
                num_blocks: TEST_NUM_BLOCKS,
                num_layers: TEST_NUM_LAYERS,
                outer_dim: TEST_OUTER_DIM,
                page_size: TEST_PAGE_SIZE,
                inner_dim: TEST_INNER_DIM,
                alignment: 256, // GPU-friendly alignment
                dtype_width_bytes: TEST_DTYPE_WIDTH_BYTES,
            }
        }

        /// Test H2D transfers between FullyContiguous host and LayerSeparate device layouts
        #[test]
        fn test_h2d_fc_host_to_ls_device() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create FullyContiguous host layout
            let host_layout = FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();

            // Create LayerSeparate device layout
            let device_layout = LayerSeparate::allocate(config, &device_allocator, true).unwrap();

            // Test data transfer for each memory region
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        let device_region = device_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();

                        // Verify regions have same size
                        assert_eq!(host_region.size(), device_region.size(),
                            "Region size mismatch at ({}, {}, {})", block_idx, layer_idx, outer_idx);

                        // Create test pattern
                        let pattern = ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Fill host memory with pattern
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8, host_region.size()
                            );
                            host_slice.fill(pattern);
                        }

                        // Transfer H2D
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                    }
                }
            }

            stream.synchronize().unwrap();

            // Verify transfers by copying back and checking patterns
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        let device_region = device_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();

                        let expected_pattern = ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Create temporary verification buffer
                        let mut verify_buffer = pinned_allocator.allocate(host_region.size()).unwrap();

                        // Copy back from device
                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                verify_buffer.as_mut_ptr(),
                                host_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Verify pattern
                        unsafe {
                            let verify_slice = std::slice::from_raw_parts(
                                verify_buffer.as_ptr(), host_region.size()
                            );
                            assert!(verify_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx, layer_idx, outer_idx, expected_pattern,
                                &verify_slice[0..std::cmp::min(8, verify_slice.len())]);
                        }
                    }
                }
            }
        }

        /// Test D2H transfers from LayerSeparate device to FullyContiguous host
        #[test]
        fn test_d2h_ls_device_to_fc_host() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create LayerSeparate device layout (block contiguous)
            let device_layout = LayerSeparate::allocate(config.clone(), &device_allocator, false).unwrap();

            // Create FullyContiguous host layout
            let host_layout = FullyContiguous::allocate(config, &pinned_allocator).unwrap();

            // Initialize device memory with patterns using a temporary host buffer
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let device_region = device_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        let pattern = ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8) | 0x80;

                        // Create temp buffer with pattern
                        let mut temp_buffer = pinned_allocator.allocate(device_region.size()).unwrap();
                        unsafe {
                            let temp_slice = std::slice::from_raw_parts_mut(
                                temp_buffer.as_mut_ptr(), device_region.size()
                            );
                            temp_slice.fill(pattern);
                        }

                        // Copy pattern to device
                        unsafe {
                            cuda_memcpy_h2d(
                                temp_buffer.as_ptr(),
                                device_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Clear host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8, host_region.size()
                            );
                            host_slice.fill(0);
                        }
                    }
                }
            }

            // Transfer D2H
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let device_region = device_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        let host_region = host_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();

                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Verify patterns in host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        let expected_pattern = ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8) | 0x80;

                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8, host_region.size()
                            );
                            assert!(host_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx, layer_idx, outer_idx, expected_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]);
                        }
                    }
                }
            }
        }

        /// Test bidirectional transfers with layout compatibility verification
        #[test]
        fn test_bidirectional_layout_transfers() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            let config = create_test_config();

            // Create both layout types
            let host_fc = FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();
            let device_ls_outer = LayerSeparate::allocate(config.clone(), &device_allocator, true).unwrap();
            let device_ls_block = LayerSeparate::allocate(config, &device_allocator, false).unwrap();

            // Test round-trip: Host FC -> Device LS (outer) -> Device LS (block) -> Host FC
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let original_pattern = ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8) | 0x40;

                        // Step 1: Initialize host FC with pattern
                        let host_region = host_fc.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8, host_region.size()
                            );
                            host_slice.fill(original_pattern);
                        }

                        // Step 2: Transfer to device LS outer
                        let device_outer_region = device_ls_outer.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_outer_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }

                        // Step 3: Transfer between device layouts (D2D)
                        let device_block_region = device_ls_block.memory_region(block_idx, layer_idx, outer_idx).unwrap();
                        unsafe {
                            cuda_memcpy_d2d(
                                device_outer_region.addr() as *const u8,
                                device_block_region.addr() as *mut u8,
                                device_outer_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }

                        stream.synchronize().unwrap();

                        // Step 4: Clear host and transfer back
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8, host_region.size()
                            );
                            host_slice.fill(0);
                        }

                        unsafe {
                            cuda_memcpy_d2h(
                                device_block_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_block_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Step 5: Verify pattern survived the round trip
                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8, host_region.size()
                            );
                            assert!(host_slice.iter().all(|&x| x == original_pattern),
                                "Round-trip pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx, layer_idx, outer_idx, original_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]);
                        }
                    }
                }
            }
        }

        /// Test transfer performance and alignment impact
        #[test]
        fn test_layout_transfer_alignment_performance() {
            let device_allocator = DeviceAllocator::default();
            let pinned_allocator = PinnedAllocator::default();
            let ctx = device_allocator.ctx().clone();
            let stream = ctx.new_stream().unwrap();

            // Test different alignments
            for alignment in [1, 64, 256, 512] {
                let config = LayoutConfig {
                    num_blocks: 2,
                    num_layers: 2,
                    outer_dim: 1,
                    page_size: 1024,
                    inner_dim: 256,
                    alignment,
                    dtype_width_bytes: 4,
                };

                let host_layout = FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();
                let device_layout = FullyContiguous::allocate(config, &device_allocator).unwrap();

                // Measure transfer time (basic timing)
                let start = std::time::Instant::now();

                for block_idx in 0..2 {
                    for layer_idx in 0..2 {
                        let host_region = host_layout.memory_region(block_idx, layer_idx, 0).unwrap();
                        let device_region = device_layout.memory_region(block_idx, layer_idx, 0).unwrap();

                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref()
                            ).unwrap();
                        }
                    }
                }
                stream.synchronize().unwrap();

                let duration = start.elapsed();

                // Verify alignment was applied correctly
                let region = host_layout.memory_region(0, 0, 0).unwrap();
                if alignment > 1 {
                    assert_eq!(region.addr() % alignment, 0,
                        "Memory not aligned to {} bytes", alignment);
                }

                println!("Transfer with alignment {} took {:?}", alignment, duration);
            }
        }
    }
}
