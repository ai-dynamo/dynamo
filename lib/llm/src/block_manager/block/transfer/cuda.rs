// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::TransferError;
use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};

use anyhow::Result;
use cudarc::driver::CudaStream;
use cudarc::driver::result as cuda_result;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Simple pinned memory allocation
pub fn allocate_pinned_memory(size: usize) -> Result<u64, TransferError> {
    // 16-byte alignment for vectorized operations
    let aligned_size = (size + 15) & !15;

    if aligned_size == 0 {
        return Err(TransferError::ExecutionError(
            "Invalid allocation size".to_string(),
        ));
    }

    unsafe {
        let result = cuda_result::malloc_host(aligned_size, 0);
        match result {
            Ok(ptr) => {
                let ptr_value = ptr as u64;
                tracing::debug!(
                    "Allocated pinned memory: {}KB, ptr=0x{:x}",
                    aligned_size / 1024,
                    ptr_value
                );
                Ok(ptr_value)
            }
            Err(e) => {
                tracing::error!("Pinned memory allocation failed: {}", e);
                Err(TransferError::ExecutionError(format!(
                    "Pinned memory allocation failed: {}",
                    e
                )))
            }
        }
    }
}

// Global storage for kernel function - store as usize to avoid Send/Sync issues
static COPY_KERNEL_MODULE: Mutex<Option<usize>> = Mutex::new(None);
static COPY_KERNEL_FUNCTION: Mutex<Option<usize>> = Mutex::new(None);

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

/// Collect K/V cache addresses from source and destination blocks
fn collect_kv_addresses<Source, Destination>(
    sources: &[Source],
    destinations: &[Destination],
    num_layers: usize,
    num_outer_dims: usize,
) -> Result<(Vec<u64>, Vec<u64>), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if sources.is_empty() {
        return Err(TransferError::ExecutionError(
            "No source blocks provided".to_string(),
        ));
    }

    let total_address_pairs = sources.len() * num_layers * num_outer_dims;
    let mut src_addresses = Vec::with_capacity(total_address_pairs);
    let mut dst_addresses = Vec::with_capacity(total_address_pairs);

    let src_block_data: Vec<_> = sources.iter().map(|block| block.block_data()).collect();
    let dst_block_data: Vec<_> = destinations
        .iter()
        .map(|block| block.block_data())
        .collect();

    for (src_data, dst_data) in src_block_data.iter().zip(dst_block_data.iter()) {
        for layer_idx in 0..num_layers {
            for outer_idx in 0..num_outer_dims {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let dst_view = dst_data.layer_view(layer_idx, outer_idx)?;

                unsafe {
                    src_addresses.push(src_view.as_ptr() as u64);
                    dst_addresses.push(dst_view.as_ptr() as u64);
                }
            }
        }
    }

    Ok((src_addresses, dst_addresses))
}

/// Launch CUDA kernel directly with pinned buffer pointers (no address copying)
unsafe fn launch_copy_kernel_direct(
    src_pinned_ptr: u64,
    dst_pinned_ptr: u64,
    address_count: usize,
    layer_size: usize,
    stream: &CudaStream,
) -> Result<(), TransferError> {
    // Get kernel function
    let kernel = get_copy_kernel()?;

    tracing::debug!(
        "LAUNCHING KERNEL: {} pairs, src=0x{:x}, dst=0x{:x}",
        address_count,
        src_pinned_ptr,
        dst_pinned_ptr
    );

    let threads_per_block = 256u32;
    let max_blocks = 1024u32;
    let blocks_needed = std::cmp::min(max_blocks, address_count as u32);

    let grid_dim = (blocks_needed, 1, 1);
    let block_dim = (threads_per_block, 1, 1);

    // cuLaunchKernel expects pointers to parameter values
    let src_ptr_param = src_pinned_ptr;
    let dst_ptr_param = dst_pinned_ptr;
    let size_param = layer_size;
    let num_pairs_param = address_count as i32;

    let params = [
        &src_ptr_param as *const _ as *mut std::ffi::c_void,
        &dst_ptr_param as *const _ as *mut std::ffi::c_void,
        &size_param as *const _ as *mut std::ffi::c_void,
        &num_pairs_param as *const _ as *mut std::ffi::c_void,
    ];

    let result = unsafe {
        cudarc::driver::sys::cuLaunchKernel(
            kernel,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            0, // shared memory
            stream.cu_stream(),
            params.as_ptr() as *mut *mut std::ffi::c_void,
            std::ptr::null_mut(), // extra
        )
    };

    if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
        tracing::error!("Kernel launch failed: {:?}", result);
        return Err(TransferError::ExecutionError(format!(
            "CUDA kernel launch failed: {:?}",
            result
        )));
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CachedBlockDimensions {
    num_layers: usize,
    num_outer_dims: usize,
    layer_size: usize,
}

static BLOCK_DIMENSIONS_CACHE: OnceLock<CachedBlockDimensions> = OnceLock::new();

fn get_cached_block_dimensions<T: BlockDataProvider>(
    block: &T,
) -> Result<CachedBlockDimensions, TransferError> {
    Ok(*BLOCK_DIMENSIONS_CACHE
        .get_or_init(|| calculate_block_dimensions_from_layout(block).unwrap()))
}

fn calculate_block_dimensions_from_layout<T: BlockDataProvider>(
    block: &T,
) -> Result<CachedBlockDimensions, TransferError> {
    let block_data = block.block_data();

    // Get dimensions directly from layout (pre-computed values)
    let num_layers = block_data.num_layers();
    let num_outer_dims = block_data.num_outer_dims();
    let layer_size = block_data.layer_view(0, 0).map(|v| v.size()).unwrap_or(0);

    Ok(CachedBlockDimensions {
        num_layers,
        num_outer_dims,
        layer_size,
    })
}

pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &CudaStream,
    ctx: &crate::block_manager::block::transfer::TransferContext,
) -> Result<Option<(Vec<u64>, usize)>, TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let _context_guard = stream.context().bind_to_thread();
    // Get cached dimensions (calculated once per program lifetime!)
    let dims = get_cached_block_dimensions(&sources[0])?;

    // Use cached dimensions
    let (src_addresses, dst_addresses) =
        collect_kv_addresses(sources, destinations, dims.num_layers, dims.num_outer_dims)?;

    tracing::debug!(
        "Using vectorized_copy for {} blocks [{}L×{}O×{}B], {} address pairs",
        sources.len(),
        dims.num_layers,
        dims.num_outer_dims,
        dims.layer_size,
        src_addresses.len()
    );

    // Use pool-based approach with TransferResources
    let resources = crate::block_manager::block::transfer::context::TransferResources::acquire_for_kernel_launch(
        ctx,
        src_addresses.len()
    )?;

    // Copy addresses to pinned buffers
    resources.copy_addresses_to_buffers(&src_addresses, &dst_addresses)?;

    tracing::debug!(
        " Using pooled pinned buffers: src=0x{:x}, dst=0x{:x} ({} address pairs)",
        resources.src_ptr(),
        resources.dst_ptr(),
        src_addresses.len()
    );

    // Launch kernel with pooled resources (addresses already copied)
    unsafe {
        launch_copy_kernel_direct(
            resources.src_ptr(),
            resources.dst_ptr(),
            src_addresses.len(),
            dims.layer_size,
            stream,
        )?;
    }

    tracing::debug!("vectorized_copy completed - resources will be returned to pool automatically");
    Ok(None) // No manual cleanup needed - TransferResources handles it via Drop
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

// Load the vectorized_copy module from FATBIN
fn get_copy_kernel_module() -> Result<cudarc::driver::sys::CUmodule, TransferError> {
    let mut module_guard = COPY_KERNEL_MODULE.lock().unwrap();

    if let Some(module_ptr) = *module_guard {
        return Ok(module_ptr as cudarc::driver::sys::CUmodule);
    }

    // Load the module on first access
    let module = match load_embedded_fatbin() {
        Ok(module) => {
            tracing::debug!("Successfully loaded embedded FATBIN module");
            module
        }
        Err(embedded_err) => {
            tracing::debug!("Embedded FATBIN loading failed: {:?}", embedded_err);
            match load_runtime_fatbin() {
                Ok(module) => {
                    tracing::debug!("Successfully loaded runtime FATBIN module");
                    module
                }
                Err(runtime_err) => {
                    tracing::error!("  Both FATBIN loading methods failed:");
                    tracing::error!("  Embedded error: {:?}", embedded_err);
                    tracing::error!("  Runtime error: {:?}", runtime_err);
                    return Err(TransferError::ExecutionError(
                        "No vectorized_copy FATBIN found (tried embedded and runtime paths)"
                            .to_string(),
                    ));
                }
            }
        }
    };

    let module_ptr = module as usize;
    *module_guard = Some(module_ptr);
    Ok(module as cudarc::driver::sys::CUmodule)
}

// Get the vectorized_copy function
fn get_copy_kernel() -> Result<cudarc::driver::sys::CUfunction, TransferError> {
    let mut func_guard = COPY_KERNEL_FUNCTION.lock().unwrap();

    if let Some(func_ptr) = *func_guard {
        return Ok(func_ptr as cudarc::driver::sys::CUfunction);
    }

    // Load the function on first access
    let module = get_copy_kernel_module()?;
    let func = unsafe {
        let mut func = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("vectorised_copy").unwrap();
        let result =
            cudarc::driver::sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            func
        } else {
            return Err(TransferError::ExecutionError(format!(
                "Failed to get kernel function: {:?}",
                result
            )));
        }
    };

    let func_ptr = func as usize;
    *func_guard = Some(func_ptr);
    Ok(func as cudarc::driver::sys::CUfunction)
}

// Try to load embedded FATBIN (compile-time) - only compiled when FATBIN is available
#[cfg(have_vec_copy_fatbin)]
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // FATBIN was copied to OUT_DIR by build.rs and embedded here
    const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/vectorized_copy.fatbin"));
    tracing::debug!("Loading embedded FATBIN ({} bytes)", FATBIN.len());
    unsafe {
        let mut module = std::ptr::null_mut();
        let result = cudarc::driver::sys::cuModuleLoadData(
            &mut module,
            FATBIN.as_ptr() as *const std::ffi::c_void,
        );
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            tracing::debug!("Embedded FATBIN module loaded successfully: {:p}", module);
            return Ok(module);
        } else {
            tracing::error!(
                "Embedded FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                result
            );
        }
    }

    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
}

// Fallback implementation when FATBIN is not available at compile time
#[cfg(not(have_vec_copy_fatbin))]
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    tracing::debug!("No embedded FATBIN available (not compiled with have_vec_copy_fatbin)");
    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
}

// Try to load FATBIN from filesystem (runtime)
fn load_runtime_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // 1. Check runtime environment variable first
    if let Ok(runtime_path) = std::env::var("DYNAMO_FATBIN_PATH")
        && let Ok(fatbin_data) = std::fs::read(&runtime_path)
    {
        tracing::debug!("Loading FATBIN from runtime env var: {}", runtime_path);
        unsafe {
            let mut module = std::ptr::null_mut();
            let result = cudarc::driver::sys::cuModuleLoadData(
                &mut module,
                fatbin_data.as_ptr() as *const std::ffi::c_void,
            );
            if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                tracing::debug!("Runtime FATBIN module loaded successfully: {:p}", module);
                return Ok(module);
            } else {
                tracing::error!(
                    "Runtime FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                    result
                );
            }
        }
    }

    // 2. Check standard runtime locations
    let runtime_paths = ["./src/block_manager/block/transfer/kernels/vectorized_copy.fatbin"];

    for path in &runtime_paths {
        if let Ok(fatbin_data) = std::fs::read(path) {
            tracing::debug!("Loading FATBIN from runtime path: {}", path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(
                    &mut module,
                    fatbin_data.as_ptr() as *const std::ffi::c_void,
                );
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    tracing::debug!(
                        "Runtime path FATBIN module loaded successfully: {:p}",
                        module
                    );
                    return Ok(module);
                } else {
                    tracing::error!(
                        "Runtime path FATBIN cuModuleLoadData failed with CUDA error: {:?}",
                        result
                    );
                }
            }
        } else {
            tracing::debug!("Could not read FATBIN file: {}", path);
        }
    }

    Err(cudarc::driver::DriverError(
        cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND,
    ))
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
