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

mod context;
mod cuda;
mod memcpy;
mod nixl;
mod strategy;

use super::*;

use crate::block_manager::storage::{
    nixl::{NixlRegisterableStorage, NixlStorage},
    DeviceStorage, DiskStorage, PinnedStorage, SystemStorage, StorageType,
};

use cudarc::driver::CudaStream;

use nixl_sys::NixlDescriptor;
use nixl_sys::XferOp::{Read, Write};
use std::ops::Range;
use tokio::sync::oneshot;


// Removed unused imports for zero-copy scatter kernel implementation
use cudarc::driver::result as cuda_result;
use std::sync::{Mutex, OnceLock};
use std::collections::VecDeque;

// Pre-allocated device buffers for kernel pointer arrays
const MAX_PREALLOCATED_BLOCKS: usize = 1024;
const TRANSFER_BUFFER_POOL_SIZE: usize = 8;  // Number of concurrent buffers

struct TransferBuffer {
    id: usize,
    src_ptrs: u64,  // Device pointer to u64 array
    dst_ptrs: u64,  // Device pointer to u64 array
    capacity: usize,
}

struct TransferBufferPool {
    available: Mutex<VecDeque<TransferBuffer>>,
    total_buffers: usize,
}

impl TransferBufferPool {
    fn acquire(&self) -> Option<TransferBuffer> {
        let mut available = self.available.lock().unwrap();
        available.pop_front()
    }

    fn release(&self, buffer: TransferBuffer) {
        let mut available = self.available.lock().unwrap();
        available.push_back(buffer);
    }

    fn len(&self) -> usize {
        let available = self.available.lock().unwrap();
        available.len()
    }
}

// Global storage for transfer buffer pool and kernel function
// Note: Store CUDA pointers as usize to avoid Send/Sync issues with raw pointers
static TRANSFER_BUFFER_POOL: OnceLock<Result<TransferBufferPool, String>> = OnceLock::new();
static COPY_KERNEL_MODULE: Mutex<Option<usize>> = Mutex::new(None);
static COPY_KERNEL_FUNCTION: Mutex<Option<usize>> = Mutex::new(None);

// Load the copy_kernel_KV module from FATBIN
fn get_copy_kernel_module() -> Result<cudarc::driver::sys::CUmodule, TransferError> {
    let mut module_guard = COPY_KERNEL_MODULE.lock().unwrap();

    if let Some(module_ptr) = *module_guard {
        return Ok(module_ptr as cudarc::driver::sys::CUmodule);
    }

    // Load the module on first access
    let module = if let Ok(module) = load_embedded_fatbin() {
        module
    } else if let Ok(module) = load_runtime_fatbin() {
        module
    } else {
        return Err(TransferError::ExecutionError("No copy_kernel_KV FATBIN found (tried embedded and runtime paths)".to_string()));
    };

    let module_ptr = module as usize;
    *module_guard = Some(module_ptr);
    Ok(module as cudarc::driver::sys::CUmodule)
}

// Get the copy_kernel_KV function
fn get_copy_kernel() -> Result<cudarc::driver::sys::CUfunction, TransferError> {
    let mut func_guard = COPY_KERNEL_FUNCTION.lock().unwrap();

    if let Some(func_ptr) = *func_guard {
        return Ok(func_ptr as cudarc::driver::sys::CUfunction);
    }

    // Load the function on first access
    let module = get_copy_kernel_module()?;
    let func = unsafe {
        let mut func = std::ptr::null_mut();
        let func_name = std::ffi::CString::new("copy_kernel_KV").unwrap();
        let result = cudarc::driver::sys::cuModuleGetFunction(&mut func, module, func_name.as_ptr());
        if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
            func
        } else {
            return Err(TransferError::ExecutionError(format!("Failed to get kernel function: {:?}", result)));
        }
    };

    let func_ptr = func as usize;
    *func_guard = Some(func_ptr);
    Ok(func as cudarc::driver::sys::CUfunction)
}

// Get or initialize pre-allocated transfer buffer pool
fn get_transfer_buffer_pool() -> Result<&'static TransferBufferPool, TransferError> {
    let result = TRANSFER_BUFFER_POOL.get_or_init(|| {
        let ptr_array_size = MAX_PREALLOCATED_BLOCKS * std::mem::size_of::<u64>();
        let mut buffers = VecDeque::with_capacity(TRANSFER_BUFFER_POOL_SIZE);

        for i in 0..TRANSFER_BUFFER_POOL_SIZE {
            let (src_ptrs, dst_ptrs) = unsafe {
                let src_ptrs = match cuda_result::malloc_sync(ptr_array_size) {
                    Ok(ptr) => ptr,
                    Err(e) => return Err(format!("Failed to allocate transfer src buffer {}: {}", i, e)),
                };
                let dst_ptrs = match cuda_result::malloc_sync(ptr_array_size) {
                    Ok(ptr) => ptr,
                    Err(e) => return Err(format!("Failed to allocate transfer dst buffer {}: {}", i, e)),
                };
                (src_ptrs, dst_ptrs)
            };

            buffers.push_back(TransferBuffer {
                id: i,
                src_ptrs,
                dst_ptrs,
                capacity: MAX_PREALLOCATED_BLOCKS,
            });
        }

        println!("📦 Allocated {} transfer buffers, {} blocks per buffer ({} KB each)",
                 TRANSFER_BUFFER_POOL_SIZE, MAX_PREALLOCATED_BLOCKS, ptr_array_size / 1024);

        Ok(TransferBufferPool {
            available: Mutex::new(buffers),
            total_buffers: TRANSFER_BUFFER_POOL_SIZE,
        })
    });

    match result {
        Ok(pool) => Ok(pool),
        Err(e) => Err(TransferError::ExecutionError(e.clone())),
    }
}

// RAII wrapper for automatic buffer release
struct TransferBufferGuard {
    buffer: Option<TransferBuffer>,
    pool: &'static TransferBufferPool,
}

impl TransferBufferGuard {
    fn new(buffer: TransferBuffer, pool: &'static TransferBufferPool) -> Self {
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    fn get(&self) -> &TransferBuffer {
        self.buffer.as_ref().unwrap()
    }
}

impl Drop for TransferBufferGuard {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

// Try to load embedded FATBIN (compile-time)
fn load_embedded_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // Check if FATBIN was embedded at compile time
    if option_env!("DYNAMO_FATBIN_AVAILABLE").is_some() {
        // FATBIN was copied to OUT_DIR by build.rs and embedded here
        const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/copy_kernel_kv.fatbin"));
        println!("📦 Loading embedded FATBIN ({} bytes)", FATBIN.len());
        unsafe {
            let mut module = std::ptr::null_mut();
            let result = cudarc::driver::sys::cuModuleLoadData(&mut module, FATBIN.as_ptr() as *const std::ffi::c_void);
            if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Ok(module);
            }
        }
    }

    Err(cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))
}

// Try to load FATBIN from filesystem (runtime)
fn load_runtime_fatbin() -> Result<cudarc::driver::sys::CUmodule, cudarc::driver::DriverError> {
    // 1. Check runtime environment variable first
    if let Ok(runtime_path) = std::env::var("DYNAMO_FATBIN_PATH") {
        if let Ok(fatbin_data) = std::fs::read(&runtime_path) {
            println!("📁 Loading FATBIN from runtime env var: {}", runtime_path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(&mut module, fatbin_data.as_ptr() as *const std::ffi::c_void);
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Ok(module);
                }
            }
        }
    }

    // 2. Check standard runtime locations (priority order)
    let runtime_paths = [
        "./src/block_manager/block/transfer/kernels/copy_kernel_kv.fatbin",  // Primary: Next to transfer module
        "./kernels/copy_kernel_kv.fatbin",                                   // Working directory kernels
        "./copy_kernel_kv.fatbin",                                           // Current directory
        "/usr/local/lib/dynamo/kernels/copy_kernel_kv.fatbin",               // System install
        "/opt/dynamo/kernels/copy_kernel_kv.fatbin",                         // Alternative system
    ];

    for path in &runtime_paths {
        if let Ok(fatbin_data) = std::fs::read(path) {
            println!("📁 Loading FATBIN from runtime path: {}", path);
            unsafe {
                let mut module = std::ptr::null_mut();
                let result = cudarc::driver::sys::cuModuleLoadData(&mut module, fatbin_data.as_ptr() as *const std::ffi::c_void);
                if result == cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                    return Ok(module);
                }
            }
        }
    }

    Err(cudarc::driver::DriverError(cudarc::driver::sys::cudaError_enum::CUDA_ERROR_FILE_NOT_FOUND))
}

pub use crate::block_manager::storage::{CudaAccessible, Local, Remote};
pub use async_trait::async_trait;
pub use context::TransferContext;

/// A block that can be the target of a write
pub trait Writable {}

/// A block that can be the source of a read
pub trait Readable {}

pub trait Mutable: Readable + Writable {}

pub trait Immutable: Readable {}

#[derive(Debug)]
pub enum BlockTarget {
    Source,
    Destination,
}

#[derive(Debug, thiserror::Error)]
pub enum TransferError {
    #[error("Builder configuration error: {0}")]
    BuilderError(String),
    #[error("Transfer execution failed: {0}")]
    ExecutionError(String),
    #[error("Incompatible block types provided: {0}")]
    IncompatibleTypes(String),
    #[error("Mismatched source/destination counts: {0} sources, {1} destinations")]
    CountMismatch(usize, usize),
    #[error("Block operation failed: {0}")]
    BlockError(#[from] BlockError),
    // TODO: Add NIXL specific errors
    #[error("No blocks provided")]
    NoBlocksProvided,

    #[error("Mismatched {0:?} block set index: {1} != {2}")]
    MismatchedBlockSetIndex(BlockTarget, usize, usize),

    #[error("Mismatched {0:?} worker ID: {1} != {2}")]
    MismatchedWorkerID(BlockTarget, usize, usize),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NixlTransfer {
    Read,
    Write,
}

impl NixlTransfer {
    pub fn as_xfer_op(&self) -> nixl_sys::XferOp {
        match self {
            NixlTransfer::Read => Read,
            NixlTransfer::Write => Write,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    Memcpy,
    CudaAsyncH2D,
    CudaAsyncD2H,
    CudaAsyncD2D,
    CudaBlockingH2D,
    CudaBlockingD2H,
    Nixl(NixlTransfer),
    Invalid,
}

/// Trait for determining the transfer strategy for writing from a local
/// source to a target destination which could be local or remote
pub trait WriteToStrategy<Target> {
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

/// Trait for determining the transfer strategy for reading from a
/// `Source` which could be local or remote into `Self` which must
/// be both local and writable.
pub trait ReadFromStrategy<Source> {
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Invalid
    }
}

impl<RB: ReadableBlock, WB: WritableBlock> WriteToStrategy<WB> for RB
where
    <RB as StorageTypeProvider>::StorageType:
        Local + WriteToStrategy<<WB as StorageTypeProvider>::StorageType>,
{
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        <<RB as StorageTypeProvider>::StorageType as WriteToStrategy<
            <WB as StorageTypeProvider>::StorageType,
        >>::write_to_strategy()
    }
}

impl<WB: WritableBlock, RB: ReadableBlock> ReadFromStrategy<RB> for WB
where
    <RB as StorageTypeProvider>::StorageType: Remote,
    <WB as StorageTypeProvider>::StorageType: NixlRegisterableStorage,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

pub fn handle_local_transfer<RB, WB>(
    sources: &[RB],
    targets: &mut [WB],
    ctx: Arc<TransferContext>,
) -> Result<oneshot::Receiver<()>, TransferError>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    WB: WritableBlock,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    let (tx, rx) = oneshot::channel();

    match RB::write_to_strategy() {
        TransferStrategy::Memcpy => {
            for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                // TODO: Unlike all other transfer strategies, this is fully blocking.
                // We probably want some sort of thread pool to handle these.
                memcpy::copy_block(src, dst)?;
            }

            tx.send(()).unwrap();
            Ok(rx)
        }
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {

            if RB::write_to_strategy() == TransferStrategy::CudaAsyncH2D {
                // Verify H2D: Host -> Device transfer
                if sources.len() > 0 {
                    if let (Ok(first_src), Ok(first_dst)) = (
                        sources[0].block_data().layer_view(0, 0),
                        targets[0].block_data().layer_view(0, 0)
                    ) {
                        unsafe {
                            // Get storage types and block IDs
                            let src_storage = sources[0].block_data().storage_type();
                            let dst_storage = targets[0].block_data().storage_type();
                            let src_block_id = sources[0].block_data().block_id();
                            let dst_block_id = targets[0].block_data().block_id();

                            // Verify H2D direction: source should be Pinned/System, dest should be Device
                            let direction_correct = matches!(src_storage, StorageType::Pinned | StorageType::System) &&
                                                  matches!(dst_storage, StorageType::Device(_));

                            print!("H2D: {} blocks | ", sources.len());
                            print!("src[{}]={:?}@{:p}", src_block_id, src_storage, first_src.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_src) = sources[sources.len()-1].block_data().layer_view(0, 0) {
                                    let last_src_id = sources[sources.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_src_id, last_src.as_ptr());
                                }
                            }
                            print!(" → dst[{}]={:?}@{:p}", dst_block_id, dst_storage, first_dst.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_dst) = targets[targets.len()-1].block_data().layer_view(0, 0) {
                                    let last_dst_id = targets[targets.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_dst_id, last_dst.as_ptr());
                                }
                            }

                            if direction_correct {
                                println!(" ✅");
                            } else {
                                println!(" ❌ WRONG_DIRECTION");
                            }
                        }
                    }
                }

                // Use custom copy_kernel_KV for multiple H2D blocks only
                if sources.len() > 1 {
                    // Check if blocks are fully contiguous (required for our simple kernel)
                    let src_data = sources[0].block_data();
                    let is_fully_contiguous = src_data.is_fully_contiguous();

                    if is_fully_contiguous {
                        tracing::info!("Using copy_kernel_KV (FATBIN) for {} H2D blocks [FullyContiguous]", sources.len());

                        let block_size = src_data.block_view()?.size();
                        let mut src_ptrs = Vec::with_capacity(sources.len());
                        let mut dst_ptrs = Vec::with_capacity(sources.len());

                        for (src, dst) in sources.iter().zip(targets.iter()) {
                            let src_view = src.block_data().block_view()?;
                            let dst_view = dst.block_data().block_view()?;

                            unsafe {
                                src_ptrs.push(src_view.as_ptr() as u64);
                                dst_ptrs.push(dst_view.as_ptr() as u64);
                            }
                        }

                        tracing::info!("Copy size: {} bytes, {} pairs [FullyContiguous blocks]", block_size, sources.len());

                        // For now, skip pooled buffers to avoid CUDA context issues during initialization
                        // TODO: Implement proper CUDA context-aware buffer pool initialization
                        let use_pooled = false; // sources.len() <= MAX_PREALLOCATED_BLOCKS;
                        let buffer_guard = None::<TransferBufferGuard>;

                    if let Some(ref guard) = buffer_guard {
                        // Use pooled buffer - fast path
                        let buffer = guard.get();

                        unsafe {
                            // Bind CUDA context to current thread for pooled operations
                            let _context_guard = ctx.stream().context().bind_to_thread();

                            // Copy pointer arrays to pooled device memory
                            cuda_result::memcpy_htod_async(buffer.src_ptrs, &src_ptrs, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy source pointers: {}", e)))?;
                            cuda_result::memcpy_htod_async(buffer.dst_ptrs, &dst_ptrs, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy dest pointers: {}", e)))?;

                            // Launch kernel with pooled buffers
                            let kernel = get_copy_kernel()?;

                            // Optimal grid sizing for grid-stride kernel
                            let threads_per_block = 256u32;
                            let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                            let blocks_needed = std::cmp::min(max_blocks, sources.len() as u32);

                            let grid_dim = (blocks_needed, 1, 1);
                            let block_dim = (threads_per_block, 1, 1);

                            // cuLaunchKernel expects pointers to parameter values
                            let src_ptr_param = buffer.src_ptrs;
                            let dst_ptr_param = buffer.dst_ptrs;
                            let size_param = block_size;
                            let num_pairs_param = sources.len() as i32; // Match kernel's int num_pairs

                            let params = [
                                &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                                &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                            ];

                            let result = cudarc::driver::sys::cuLaunchKernel(
                                kernel,
                                grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                                block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                                0, // shared memory
                                ctx.stream().cu_stream(),
                                params.as_ptr() as *mut *mut std::ffi::c_void,
                                std::ptr::null_mut(), // extra
                            );
                            if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                                return Err(TransferError::ExecutionError(format!("Kernel launch failed: {:?}", result)));
                            }
                        }

                        let pool = get_transfer_buffer_pool()?;
                        tracing::info!("copy_kernel_KV launched with pooled buffer {} (available: {}/{})",
                                 buffer.id, pool.len(), pool.total_buffers);
                    } else {
                        // Fall back to malloc/memcpy/free approach using pinned host memory
                        // Note: This path has potential memory leak due to async completion
                        if !use_pooled {
                            tracing::info!("Block count {} exceeds buffer capacity {}, using malloc fallback",
                                     sources.len(), MAX_PREALLOCATED_BLOCKS);
                        } else {
                            tracing::info!("No pooled buffers available, using malloc fallback");
                        }

                        // Collect all source and destination addresses for each layer's K/V components
                        let mut src_addresses = Vec::new();
                        let mut dst_addresses = Vec::new();

                        let num_layers = sources[0].block_data().num_layers();
                        for (src_block, dst_block) in sources.iter().zip(targets.iter()) {
                            for layer_idx in 0..num_layers {
                                // K cache addresses (outer_dim = 0)
                                if let (Ok(src_k), Ok(dst_k)) = (
                                    src_block.block_data().layer_view(layer_idx, 0),
                                    dst_block.block_data().layer_view(layer_idx, 0)
                                ) {
                                    unsafe {
                                        src_addresses.push(src_k.as_ptr() as u64);
                                        dst_addresses.push(dst_k.as_ptr() as u64);
                                    }
                                }
                                // V cache addresses (outer_dim = 1)
                                if let (Ok(src_v), Ok(dst_v)) = (
                                    src_block.block_data().layer_view(layer_idx, 1),
                                    dst_block.block_data().layer_view(layer_idx, 1)
                                ) {
                                    unsafe {
                                        src_addresses.push(src_v.as_ptr() as u64);
                                        dst_addresses.push(dst_v.as_ptr() as u64);
                                    }
                                }
                            }
                        }

                        let layer_size = sources[0].block_data().layer_view(0, 0)?.size();
                        tracing::info!("Copy size: {} bytes, {} transfers [FullyContiguous fallback: {} src/dst pairs]",
                                 layer_size, src_addresses.len(), src_addresses.len());

                        let ptr_array_size = src_addresses.len() * std::mem::size_of::<u64>();
                        let mut d_src_ptrs = 0u64;
                        let mut d_dst_ptrs = 0u64;

                        unsafe {
                            // Bind CUDA context to current thread for allocations
                            let _context_guard = ctx.stream().context().bind_to_thread();

                            // Allocate device memory for pointer arrays
                            d_src_ptrs = cuda_result::malloc_sync(ptr_array_size)
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate device source pointers: {}", e)))?;
                            d_dst_ptrs = cuda_result::malloc_sync(ptr_array_size)
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate device dest pointers: {}", e)))?;

                            // Copy pointer arrays to device (direct from Vec)
                            cuda_result::memcpy_htod_async(d_src_ptrs, &src_addresses, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy source pointers: {}", e)))?;
                            cuda_result::memcpy_htod_async(d_dst_ptrs, &dst_addresses, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy dest pointers: {}", e)))?;

                            // Launch kernel using fallback malloc approach
                            let kernel = get_copy_kernel()?;

                            // Optimal grid sizing for grid-stride kernel (not one block per transfer!)
                            let threads_per_block = 256u32;
                            let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                            let blocks_needed = std::cmp::min(max_blocks, src_addresses.len() as u32);

                            let grid_dim = (blocks_needed, 1, 1);
                            let block_dim = (threads_per_block, 1, 1);

                            tracing::info!("Grid-stride kernel: {} pairs, {} blocks × {} threads (grid-stride handles overflow)",
                                     src_addresses.len(), blocks_needed, threads_per_block);

                            // cuLaunchKernel expects pointers to parameter values
                            let src_ptr_param = d_src_ptrs;
                            let dst_ptr_param = d_dst_ptrs;
                            let size_param = layer_size;  // Size per layer component (32KB)
                            let num_pairs_param = src_addresses.len() as i32; // Match kernel's int num_pairs

                            let params = [
                                &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                                &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                            ];

                            let result = cudarc::driver::sys::cuLaunchKernel(
                                kernel,
                                grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                                block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                                0, // shared memory
                                ctx.stream().cu_stream(),
                                params.as_ptr() as *mut *mut std::ffi::c_void,
                                std::ptr::null_mut(), // extra
                            );
                            if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                                return Err(TransferError::ExecutionError(format!("FullyContiguous kernel launch failed: {:?}", result)));
                            }

                            // Note: Memory cleanup deferred - will be cleaned up when stream completes
                            // TODO: Implement proper async cleanup or use pooled buffers to avoid this issue
                            // let _ = cuda_result::free_sync(d_src_ptrs);
                            // let _ = cuda_result::free_sync(d_dst_ptrs);
                        }

                        tracing::info!("copy_kernel_KV launched successfully (FullyContiguous malloc fallback)");
                    }
                    } else {
                        // LayerSeparate layout detected - collect all layer addresses for scatter kernel
                        let num_layers = src_data.num_layers();
                        let num_outer_dims = src_data.num_outer_dims();
                        let layer_size = src_data.layer_view(0, 0)?.size();
                        let total_size = layer_size * num_layers * num_outer_dims;

                        tracing::info!("Using copy_kernel_KV (FATBIN) for {} LayerSeparate blocks [{}L×{}O×{}B]",
                                 sources.len(), num_layers, num_outer_dims, layer_size);

                        // Collect all source and destination addresses for each layer's K/V components
                        let mut src_addresses = Vec::new();
                        let mut dst_addresses = Vec::new();

                        for (src_block, dst_block) in sources.iter().zip(targets.iter()) {
                            for layer_idx in 0..num_layers {
                                // K cache addresses (outer_dim = 0)
                                if let (Ok(src_k), Ok(dst_k)) = (
                                    src_block.block_data().layer_view(layer_idx, 0),
                                    dst_block.block_data().layer_view(layer_idx, 0)
                                ) {
                                    unsafe {
                                        src_addresses.push(src_k.as_ptr() as u64);
                                        dst_addresses.push(dst_k.as_ptr() as u64);
                                    }
                                }
                                // V cache addresses (outer_dim = 1)
                                if let (Ok(src_v), Ok(dst_v)) = (
                                    src_block.block_data().layer_view(layer_idx, 1),
                                    dst_block.block_data().layer_view(layer_idx, 1)
                                ) {
                                    unsafe {
                                        src_addresses.push(src_v.as_ptr() as u64);
                                        dst_addresses.push(dst_v.as_ptr() as u64);
                                    }
                                }
                            }
                        }

                        tracing::info!("Copy size: {} bytes, {} transfers [LayerSeparate: {} src/dst pairs]",
                                 layer_size, src_addresses.len(), src_addresses.len());

                        // Launch kernel with collected LayerSeparate addresses
                        let ptr_array_size = src_addresses.len() * std::mem::size_of::<u64>();
                        let mut d_src_ptrs = 0u64;
                        let mut d_dst_ptrs = 0u64;

                        unsafe {
                            // Bind CUDA context to current thread for allocations
                            let _context_guard = ctx.stream().context().bind_to_thread();

                            // Allocate device memory for pointer arrays
                            d_src_ptrs = cuda_result::malloc_sync(ptr_array_size)
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate LayerSeparate source pointers: {}", e)))?;
                            d_dst_ptrs = cuda_result::malloc_sync(ptr_array_size)
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to allocate LayerSeparate dest pointers: {}", e)))?;

                            // Copy pointer arrays to device
                            cuda_result::memcpy_htod_async(d_src_ptrs, &src_addresses, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy LayerSeparate source pointers: {}", e)))?;
                            cuda_result::memcpy_htod_async(d_dst_ptrs, &dst_addresses, ctx.stream().cu_stream())
                                .map_err(|e| TransferError::ExecutionError(format!("Failed to copy LayerSeparate dest pointers: {}", e)))?;

                            // Launch kernel for LayerSeparate transfers
                            let kernel = get_copy_kernel()?;

                            // Optimal grid sizing for grid-stride kernel (not one block per transfer!)
                            let threads_per_block = 256u32;
                            let max_blocks = 1024u32; // Reasonable limit, let grid-stride handle the rest
                            let blocks_needed = std::cmp::min(max_blocks, src_addresses.len() as u32);

                            let grid_dim = (blocks_needed, 1, 1);
                            let block_dim = (threads_per_block, 1, 1);

                            tracing::info!(" LayerSeparate grid-stride: {} pairs, {} blocks × {} threads",
                                     src_addresses.len(), blocks_needed, threads_per_block);

                            // cuLaunchKernel expects pointers to parameter values
                            let src_ptr_param = d_src_ptrs;
                            let dst_ptr_param = d_dst_ptrs;
                            let size_param = layer_size;  // Size per layer component (32KB)
                            let num_pairs_param = src_addresses.len() as i32; // Match kernel's int num_pairs

                            let params = [
                                &src_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &dst_ptr_param as *const _ as *mut std::ffi::c_void,    // Pointer to device address
                                &size_param as *const _ as *mut std::ffi::c_void,       // Pointer to size value
                                &num_pairs_param as *const _ as *mut std::ffi::c_void,  // Pointer to num_pairs (int)
                            ];

                            let result = cudarc::driver::sys::cuLaunchKernel(
                                kernel,
                                grid_dim.0, grid_dim.1, grid_dim.2,  // grid dimensions
                                block_dim.0, block_dim.1, block_dim.2,  // block dimensions
                                0, // shared memory
                                ctx.stream().cu_stream(),
                                params.as_ptr() as *mut *mut std::ffi::c_void,
                                std::ptr::null_mut(), // extra
                            );
                            if result != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                                return Err(TransferError::ExecutionError(format!("LayerSeparate kernel launch failed: {:?}", result)));
                            }

                            // Note: Memory cleanup deferred - will be cleaned up when stream completes
                            // TODO: Implement proper async cleanup or use pooled buffers to avoid this issue
                            // let _ = cuda_result::free_sync(d_src_ptrs);
                            // let _ = cuda_result::free_sync(d_dst_ptrs);
                        }

                        tracing::info!("copy_kernel_KV launched successfully for LayerSeparate layout");
                    }
                } else {
                    // Fall back to individual copy for single H2D blocks
                    for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                        cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                    }
                }
            } else if RB::write_to_strategy() == TransferStrategy::CudaAsyncD2H {
                // Verify D2H: Device -> Host transfer
                if sources.len() > 0 {
                    if let (Ok(first_src), Ok(first_dst)) = (
                        sources[0].block_data().layer_view(0, 0),
                        targets[0].block_data().layer_view(0, 0)
                    ) {
                        unsafe {
                            // Get storage types and block IDs
                            let src_storage = sources[0].block_data().storage_type();
                            let dst_storage = targets[0].block_data().storage_type();
                            let src_block_id = sources[0].block_data().block_id();
                            let dst_block_id = targets[0].block_data().block_id();

                            // Verify D2H direction: source should be Device, dest should be Pinned/System
                            let direction_correct = matches!(src_storage, StorageType::Device(_)) &&
                                                  matches!(dst_storage, StorageType::Pinned | StorageType::System);

                            print!("D2H: {} blocks | ", sources.len());
                            print!("src[{}]={:?}@{:p}", src_block_id, src_storage, first_src.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_src) = sources[sources.len()-1].block_data().layer_view(0, 0) {
                                    let last_src_id = sources[sources.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_src_id, last_src.as_ptr());
                                }
                            }
                            print!(" → dst[{}]={:?}@{:p}", dst_block_id, dst_storage, first_dst.as_ptr());
                            if sources.len() > 1 {
                                if let Ok(last_dst) = targets[targets.len()-1].block_data().layer_view(0, 0) {
                                    let last_dst_id = targets[targets.len()-1].block_data().block_id();
                                    print!("..{}@{:p}", last_dst_id, last_dst.as_ptr());
                                }
                            }

                            if direction_correct {
                                tracing::info!(" CORRECT DIRECTION");
                            } else {
                                tracing::info!(" WRONG DIRECTION");
                            }
                        }
                    }
                }

                // D2H transfers use standard block copy
                tracing::info!("ziqif copy_blocks");
                cuda::copy_blocks(sources, targets, ctx.stream().as_ref(), RB::write_to_strategy())?;
                // for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                //     cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                // }
            } else {
                // For all other cases (D2D, etc.), use standard block copy
                tracing::info!("ziqif copy_blocks");
                cuda::copy_blocks(sources, targets, ctx.stream().as_ref(), RB::write_to_strategy())?;
                // for (src, dst) in sources.iter().zip(targets.iter_mut()) {
                //     cuda::copy_block(src, dst, ctx.stream().as_ref(), RB::write_to_strategy())?;
                // }
            }

            // todo: acquire an cuda event, recored on the stream, drop the stream, await on the event.
            ctx.cuda_event(tx)?;
            Ok(rx)
        }
        TransferStrategy::Nixl(transfer_type) => {
            let transfer_fut = nixl::write_blocks_to(sources, targets, &ctx, transfer_type)?;

            ctx.async_rt_handle().spawn(async move {
                transfer_fut.await;
                tx.send(()).unwrap();
            });
            Ok(rx)
        }
        _ => Err(TransferError::IncompatibleTypes(format!(
            "Unsupported copy strategy: {:?}",
            RB::write_to_strategy()
        ))),
    }
}

pub trait WriteTo<Target> {
    fn write_to(
        &self,
        dst: &mut Vec<Target>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>;
}

impl<RB, WB, L: LocalityProvider> WriteTo<WB> for Vec<RB>
where
    RB: ReadableBlock + WriteToStrategy<WB> + Local,
    <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
    <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
    RB: BlockDataProvider<Locality = L>,
    WB: WritableBlock + BlockDataProviderMut<Locality = L>,
{
    fn write_to(
        &self,
        dst: &mut Vec<WB>,
        ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError> {
        L::handle_transfer(self, dst, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
    }
}






