// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    let src_data = sources.block_data(private::PrivateToken);
    let dst_data = destinations.block_data_mut(private::PrivateToken);
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
    let src_data = sources.block_data(private::PrivateToken);
    let dst_data = destinations.block_data_mut(private::PrivateToken);
    let memcpy_fn = cuda_memcpy_fn_ptr(&strategy)?;

    #[cfg(debug_assertions)]
    {
        let expected_strategy =
            expected_strategy::<Source::StorageType, Destination::StorageType>();
        assert_eq!(strategy, expected_strategy);
    }

    for layer_idx in layer_range {
        let src_view = src_data.layer_view(layer_idx)?;
        let mut dst_view = dst_data.layer_view_mut(layer_idx)?;

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

    let src_slice = std::slice::from_raw_parts(src_ptr, size);
    cuda_result::memcpy_htod_async(dst_ptr as u64, src_slice, stream.cu_stream())
        .map_err(|e| TransferError::ExecutionError(format!("CUDA H2D memcpy failed: {}", e)))?;
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

    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, size);
    cuda_result::memcpy_dtoh_async(dst_slice, src_ptr as u64, stream.cu_stream())
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2H memcpy failed: {}", e)))?;
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

    cuda_result::memcpy_dtod_async(dst_ptr as u64, src_ptr as u64, size, stream.cu_stream())
        .map_err(|e| TransferError::ExecutionError(format!("CUDA D2D memcpy failed: {}", e)))?;
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
}
