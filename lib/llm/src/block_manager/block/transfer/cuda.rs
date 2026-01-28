// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::TransferError;
use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};

use anyhow::Result;
use cudarc::driver::CudaStream;
use cudarc::driver::result as cuda_result;
use cudarc::driver::sys::{CUevent_flags, CUresult, cuMemcpyHtoDAsync_v2};
use dynamo_runtime::config::environment_names::cuda as env_cuda;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::OnceLock;

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
        tracing::error!(
            "Kernel launch failed: {:?} - kernel params: {} pairs, layer_size={}, src=0x{:x}, dst=0x{:x}",
            result,
            address_count,
            layer_size,
            src_pinned_ptr,
            dst_pinned_ptr
        );
        return Err(TransferError::ExecutionError(format!(
            "CUDA kernel launch failed: {:?} (address_count={}, layer_size={})",
            result, address_count, layer_size
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
) -> Result<(), TransferError>
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

    let size = src_addresses.len() * std::mem::size_of::<u64>();

    let pool = ctx.cuda_mem_pool().ok_or_else(|| {
        TransferError::ExecutionError(
            "TransferContext was not instantiated with a CudaPool; please report this error"
                .to_string(),
        )
    })?;

    // Allocate DEVICE memory from pool (stream-ordered)
    let src_buffer = pool.alloc_async(size, stream).map_err(|e| {
        TransferError::ExecutionError(format!("CUDA pool allocation failed: {}", e))
    })?;
    let dst_buffer = pool.alloc_async(size, stream).map_err(|e| {
        TransferError::ExecutionError(format!("CUDA pool allocation failed: {}", e))
    })?;

    // Copy address buffers from host to device using stream-ordered H2D memcpy
    let result_src = unsafe {
        cuMemcpyHtoDAsync_v2(
            src_buffer,
            src_addresses.as_ptr() as *const std::ffi::c_void,
            size,
            stream.cu_stream(),
        )
    };
    if result_src != CUresult::CUDA_SUCCESS {
        return Err(TransferError::ExecutionError(format!(
            "H2D memcpy for src buffer failed: {:?}",
            result_src
        )));
    }

    let result_dst = unsafe {
        cuMemcpyHtoDAsync_v2(
            dst_buffer,
            dst_addresses.as_ptr() as *const std::ffi::c_void,
            size,
            stream.cu_stream(),
        )
    };
    if result_dst != CUresult::CUDA_SUCCESS {
        return Err(TransferError::ExecutionError(format!(
            "H2D memcpy for dst buffer failed: {:?}",
            result_dst
        )));
    }

    // Record event and synchronize to ensure H2D completes before host vectors drop
    // This is critical: the async H2D memcpy is still reading from src_addresses/dst_addresses
    // host memory when it returns. We must wait for completion before those vectors are dropped.
    let h2d_event = stream
        .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))
        .map_err(|e| TransferError::ExecutionError(format!("Failed to record H2D event: {}", e)))?;

    // Launch kernel (reads from device buffers)
    unsafe {
        launch_copy_kernel_direct(
            src_buffer,
            dst_buffer,
            src_addresses.len(),
            dims.layer_size,
            stream,
        )?;
    }

    // Free buffers immediately (stream-ordered - CUDA ensures kernel completes first)
    pool.free_async(src_buffer, stream)
        .map_err(|e| TransferError::ExecutionError(format!("Failed to free src buffer: {}", e)))?;
    pool.free_async(dst_buffer, stream)
        .map_err(|e| TransferError::ExecutionError(format!("Failed to free dst buffer: {}", e)))?;

    // By synchronizing here, we enqueue all the work to the stream, then wait.
    // There is cpu overheads associated with each of those calls.
    // We might as well amortize the transfer of the pointers with those launch overheads.
    h2d_event
        .synchronize()
        .map_err(|e| TransferError::ExecutionError(format!("Failed to sync H2D event: {}", e)))?;

    Ok(())
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
    if let Ok(runtime_path) = std::env::var(env_cuda::DYNAMO_FATBIN_PATH)
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

    // ============================================================================
    // CUDA TRANSFER TESTS FOR LAYOUT COMPATIBILITY
    // ============================================================================

    mod layout_transfer_tests {
        use super::*;
        use crate::block_manager::layout::{
            FullyContiguous, GenericBlockLayout, LayerSeparate, LayoutConfig,
        };

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
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        // Verify regions have same size
                        assert_eq!(
                            host_region.size(),
                            device_region.size(),
                            "Region size mismatch at ({}, {}, {})",
                            block_idx,
                            layer_idx,
                            outer_idx
                        );

                        // Create test pattern
                        let pattern =
                            ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Fill host memory with pattern
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(pattern);
                        }

                        // Transfer H2D
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }

            stream.synchronize().unwrap();

            // Verify transfers by copying back and checking patterns
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        let expected_pattern =
                            ((block_idx as u8) << 4) | ((layer_idx as u8) << 2) | (outer_idx as u8);

                        // Create temporary verification buffer
                        let mut verify_buffer =
                            pinned_allocator.allocate(host_region.size()).unwrap();

                        // Copy back from device
                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                verify_buffer.as_mut_ptr(),
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Verify pattern
                        unsafe {
                            let verify_slice = std::slice::from_raw_parts(
                                verify_buffer.as_ptr(),
                                host_region.size(),
                            );
                            assert!(
                                verify_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                expected_pattern,
                                &verify_slice[0..std::cmp::min(8, verify_slice.len())]
                            );
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
            let device_layout =
                LayerSeparate::allocate(config.clone(), &device_allocator, false).unwrap();

            // Create FullyContiguous host layout
            let host_layout = FullyContiguous::allocate(config, &pinned_allocator).unwrap();

            // Initialize device memory with patterns using a temporary host buffer
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x80;

                        // Create temp buffer with pattern
                        let mut temp_buffer =
                            pinned_allocator.allocate(device_region.size()).unwrap();
                        unsafe {
                            let temp_slice = std::slice::from_raw_parts_mut(
                                temp_buffer.as_mut_ptr(),
                                device_region.size(),
                            );
                            temp_slice.fill(pattern);
                        }

                        // Copy pattern to device
                        unsafe {
                            cuda_memcpy_h2d(
                                temp_buffer.as_ptr(),
                                device_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Clear host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
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
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();

                        unsafe {
                            cuda_memcpy_d2h(
                                device_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
            }
            stream.synchronize().unwrap();

            // Verify patterns in host layout
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let host_region = host_layout
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        let expected_pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x80;

                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8,
                                host_region.size(),
                            );
                            assert!(
                                host_slice.iter().all(|&x| x == expected_pattern),
                                "Pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                expected_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]
                            );
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
            let device_ls_outer =
                LayerSeparate::allocate(config.clone(), &device_allocator, true).unwrap();
            let device_ls_block =
                LayerSeparate::allocate(config, &device_allocator, false).unwrap();

            // Test round-trip: Host FC -> Device LS (outer) -> Device LS (block) -> Host FC
            for block_idx in 0..TEST_NUM_BLOCKS {
                for layer_idx in 0..TEST_NUM_LAYERS {
                    for outer_idx in 0..TEST_OUTER_DIM {
                        let original_pattern = ((block_idx as u8) << 4)
                            | ((layer_idx as u8) << 2)
                            | (outer_idx as u8)
                            | 0x40;

                        // Step 1: Initialize host FC with pattern
                        let host_region = host_fc
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(original_pattern);
                        }

                        // Step 2: Transfer to device LS outer
                        let device_outer_region = device_ls_outer
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_outer_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }

                        // Step 3: Transfer between device layouts (D2D)
                        let device_block_region = device_ls_block
                            .memory_region(block_idx, layer_idx, outer_idx)
                            .unwrap();
                        unsafe {
                            cuda_memcpy_d2d(
                                device_outer_region.addr() as *const u8,
                                device_block_region.addr() as *mut u8,
                                device_outer_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }

                        stream.synchronize().unwrap();

                        // Step 4: Clear host and transfer back
                        unsafe {
                            let host_slice = std::slice::from_raw_parts_mut(
                                host_region.addr() as *mut u8,
                                host_region.size(),
                            );
                            host_slice.fill(0);
                        }

                        unsafe {
                            cuda_memcpy_d2h(
                                device_block_region.addr() as *const u8,
                                host_region.addr() as *mut u8,
                                device_block_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                        stream.synchronize().unwrap();

                        // Step 5: Verify pattern survived the round trip
                        unsafe {
                            let host_slice = std::slice::from_raw_parts(
                                host_region.addr() as *const u8,
                                host_region.size(),
                            );
                            assert!(
                                host_slice.iter().all(|&x| x == original_pattern),
                                "Round-trip pattern mismatch at ({}, {}, {}) - expected {}, got {:?}",
                                block_idx,
                                layer_idx,
                                outer_idx,
                                original_pattern,
                                &host_slice[0..std::cmp::min(8, host_slice.len())]
                            );
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

                let host_layout =
                    FullyContiguous::allocate(config.clone(), &pinned_allocator).unwrap();
                let device_layout = FullyContiguous::allocate(config, &device_allocator).unwrap();

                // Measure transfer time (basic timing)
                let start = std::time::Instant::now();

                for block_idx in 0..2 {
                    for layer_idx in 0..2 {
                        let host_region =
                            host_layout.memory_region(block_idx, layer_idx, 0).unwrap();
                        let device_region = device_layout
                            .memory_region(block_idx, layer_idx, 0)
                            .unwrap();

                        unsafe {
                            cuda_memcpy_h2d(
                                host_region.addr() as *const u8,
                                device_region.addr() as *mut u8,
                                host_region.size(),
                                stream.as_ref(),
                            )
                            .unwrap();
                        }
                    }
                }
                stream.synchronize().unwrap();

                let duration = start.elapsed();

                // Verify alignment was applied correctly
                let region = host_layout.memory_region(0, 0, 0).unwrap();
                if alignment > 1 {
                    assert_eq!(
                        region.addr() % alignment,
                        0,
                        "Memory not aligned to {} bytes",
                        alignment
                    );
                }

                println!("Transfer with alignment {} took {:?}", alignment, duration);
            }
        }
    }
}
