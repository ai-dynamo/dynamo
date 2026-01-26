// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA executor for GPU memory transfers.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::v2::physical::transfer::context::TransferCompleteNotification;
use crate::v2::physical::transfer::{can_use_whole_block_transfer, validate_layout_compatibility};
use anyhow::{Result, anyhow};
use cudarc::driver::{CudaStream, result as cuda_result};
use cudarc::runtime::sys::cudaStream_t;
use dynamo_memory::CudaMemPool;
use std::ffi::c_void;
use std::ops::Range;
use std::sync::Arc;

// #[cfg(test)]
// mod cuda_kernel_tests;

/// Execute a CUDA transfer between host and device memory.
///
/// This executor handles transfers involving GPU memory using CUDA APIs.
/// Supports async and blocking transfers depending on the strategy.
///
/// # Arguments
/// * `src` - Source physical layout
/// * `dst` - Destination physical layout
/// * `src_block_ids` - Source block IDs to transfer
/// * `dst_block_ids` - Destination block IDs to transfer
/// * `layer_range` - Optional range of layers to transfer (None = all layers)
/// * `strategy` - CUDA transfer strategy (H2D, D2H, D2D, async or blocking)
/// * `cuda_stream` - Optional caller-provided stream. If provided, use this stream
///   and skip event recording (caller manages sync). Returns completed() immediately.
/// * `ctx` - Transfer context with CUDA stream
pub fn execute_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
    cuda_stream: Option<Arc<CudaStream>>,
    ctx: &TransferContext,
) -> Result<TransferCompleteNotification> {
    // Validate layouts
    let src_layout = src.layout();
    let dst_layout = dst.layout();

    if src_layout.num_layers() != dst_layout.num_layers() {
        return Err(anyhow!(
            "Layouts have incompatible layer counts: src={}, dst={}",
            src_layout.num_layers(),
            dst_layout.num_layers()
        ));
    }

    if src_layout.outer_dim() != dst_layout.outer_dim() {
        return Err(anyhow!(
            "Layouts have incompatible outer dimensions: src={}, dst={}",
            src_layout.outer_dim(),
            dst_layout.outer_dim()
        ));
    }

    // Validate layout compatibility (errors if transform would be needed)
    validate_layout_compatibility(src, dst)?;

    // Determine layer range
    let layers = layer_range.clone().unwrap_or(0..src_layout.num_layers());

    // Check if we can use optimized whole-block transfer
    let use_whole_block = can_use_whole_block_transfer(src, dst, layer_range.as_ref());

    // Track whether caller provided stream (affects event recording)
    let caller_manages_sync = cuda_stream.is_some();

    // Get appropriate CUDA stream - use caller-provided or acquire from pool
    let stream = if let Some(s) = cuda_stream {
        s
    } else {
        match strategy {
            TransferStrategy::CudaAsyncD2H => ctx.next_d2h_streams(),
            _ => ctx.next_h2d_streams(), // H2D and D2D use h2d_stream
        }
    };

    // Perform CUDA transfers based on strategy
    // Determine direction name for logging
    let strategy_name = match strategy {
        TransferStrategy::CudaAsyncH2D => "H2D",
        TransferStrategy::CudaAsyncD2H => "D2H",
        TransferStrategy::CudaAsyncD2D => "D2D",
        _ => "Unknown",
    };

    match strategy {
        TransferStrategy::CudaAsyncH2D
        | TransferStrategy::CudaAsyncD2H
        | TransferStrategy::CudaAsyncD2D => {
            if use_whole_block {
                // FC→FC: Use unified whole-block path with batched memcpy
                // Direction auto-detected by cudaMemcpyDefault
                tracing::debug!(
                    strategy = strategy_name,
                    num_blocks = src_block_ids.len(),
                    bytes_per_block = src_layout.config().bytes_per_block(),
                    "Using whole-block transfer (auto direction)"
                );
                execute_whole_block_cuda(src, dst, src_block_ids, dst_block_ids, stream.as_ref())?;
            } else {
                // FC↔LW: Use vectorized_copy kernel directly
                tracing::debug!(
                    strategy = strategy_name,
                    num_blocks = src_block_ids.len(),
                    num_layers = layers.len(),
                    "Using vectorized_copy for FC↔LW transfer"
                );
                execute_fc_lw_vectorized(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers.clone(),
                    stream.as_ref(),
                    ctx.cuda_pool(),
                )?;
            }
        }
        _ => {
            return Err(anyhow!("Invalid CUDA transfer strategy: {:?}", strategy));
        }
    }

    // If caller provided the stream, they manage synchronization - return completed immediately
    if caller_manages_sync {
        return Ok(TransferCompleteNotification::completed());
    }

    // For async transfers, record an event and register it for completion tracking
    if matches!(
        strategy,
        TransferStrategy::CudaAsyncH2D
            | TransferStrategy::CudaAsyncD2H
            | TransferStrategy::CudaAsyncD2D
    ) {
        let event = stream.record_event(None)?;
        Ok(ctx.register_cuda_event(event))
    } else {
        // Blocking transfers are already synchronized
        Ok(TransferCompleteNotification::completed())
    }
}

// ============================================================================
// Whole-Block Transfer Functions (FC→FC optimization)
// ============================================================================

/// Unified whole-block transfer using batched memcpy.
///
/// NO device pointer allocation needed. Direction is auto-detected by CUDA
/// from pointer types using cudaMemcpyDefault.
///
/// Uses cudaMemcpyBatchAsync when available (CUDA 12.9+), falling back to
/// individual cudaMemcpyAsync calls on older CUDA versions.
fn execute_whole_block_cuda(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    stream: &cudarc::driver::CudaStream,
) -> Result<()> {
    let bytes_per_block = src.layout().config().bytes_per_block();
    let num_blocks = src_block_ids.len();

    if num_blocks == 0 {
        return Ok(());
    }

    // Build host pointer arrays
    let mut src_ptrs: Vec<*const std::ffi::c_void> = Vec::with_capacity(num_blocks);
    let mut dst_ptrs: Vec<*mut std::ffi::c_void> = Vec::with_capacity(num_blocks);

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        let src_region = src.memory_region(src_block_id, 0, 0)?;
        let dst_region = dst.memory_region(dst_block_id, 0, 0)?;
        src_ptrs.push(src_region.addr() as *const std::ffi::c_void);
        dst_ptrs.push(dst_region.addr() as *mut std::ffi::c_void);
    }

    // Use batched memcpy - handles CUDA 12.9+ batch API with automatic fallback
    let status = unsafe {
        dynamo_kvbm_kernels::memcpy_batch(
            src_ptrs.as_ptr(),
            dst_ptrs.as_ptr(),
            bytes_per_block,
            num_blocks,
            stream.cu_stream() as cudarc::runtime::sys::cudaStream_t,
        )
    };

    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        return Err(anyhow!("memcpy_batch failed: {:?}", status));
    }

    tracing::debug!(
        num_blocks,
        bytes_per_block,
        batch_available = dynamo_kvbm_kernels::is_memcpy_batch_available(),
        "Whole-block transfer completed"
    );

    Ok(())
}

// ============================================================================
// FC↔LW Transfer using vectorized_copy kernel
// ============================================================================

/// Execute FC↔LW transfer using vectorized_copy kernel.
///
/// This function builds flat (src, dst) pointer arrays for all chunks across all blocks,
/// uploads them to device memory, and calls the vectorized_copy kernel directly.
///
/// Benefits over the old operational_copy approach:
/// - Simpler: One kernel, no backend selection logic
/// - Faster: 16-byte (int4) loads when aligned (vs 8-byte in operational_copy_vectorized)
/// - All offset math on host: Kernel just copies bytes
/// - Handles any alignment: Falls back gracefully to 8/4/1-byte copies
fn execute_fc_lw_vectorized(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &CudaStream,
    pool: &CudaMemPool,
) -> Result<()> {
    // Bind CUDA context to current thread before any CUDA operations.
    stream.context().bind_to_thread()?;

    let src_layout = src.layout();
    let nl = layers.len();
    let no = src_layout.outer_dim();
    let chunk_size =
        src_layout.page_size() * src_layout.inner_dim() * src_layout.dtype_width_bytes();
    let num_blocks = src_block_ids.len();
    let total_chunks = num_blocks * nl * no;

    if total_chunks == 0 {
        return Ok(());
    }

    // Build flat pointer arrays on host
    let mut src_ptrs: Vec<usize> = Vec::with_capacity(total_chunks);
    let mut dst_ptrs: Vec<usize> = Vec::with_capacity(total_chunks);

    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..no {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;
                src_ptrs.push(src_region.addr());
                dst_ptrs.push(dst_region.addr());
            }
        }
    }

    // Allocate device memory for pointer arrays
    let src_ptrs_device = pool.alloc_async(total_chunks * std::mem::size_of::<usize>(), stream)?;
    let dst_ptrs_device = pool.alloc_async(total_chunks * std::mem::size_of::<usize>(), stream)?;

    // Upload pointer arrays to device
    unsafe {
        cuda_result::memcpy_htod_async(
            src_ptrs_device,
            std::slice::from_raw_parts(
                src_ptrs.as_ptr() as *const u8,
                total_chunks * std::mem::size_of::<usize>(),
            ),
            stream.cu_stream(),
        )?;
        cuda_result::memcpy_htod_async(
            dst_ptrs_device,
            std::slice::from_raw_parts(
                dst_ptrs.as_ptr() as *const u8,
                total_chunks * std::mem::size_of::<usize>(),
            ),
            stream.cu_stream(),
        )?;
    }

    let pointers_transfered_event = stream.record_event(None)?;

    // Call vectorized_copy kernel
    let status = unsafe {
        dynamo_kvbm_kernels::vectorized_copy(
            src_ptrs_device as *mut *mut c_void,
            dst_ptrs_device as *mut *mut c_void,
            chunk_size,
            total_chunks as i32,
            stream.cu_stream() as cudaStream_t,
        )
    };

    // Free device allocations back to the pool
    pool.free_async(src_ptrs_device, stream)?;
    pool.free_async(dst_ptrs_device, stream)?;

    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        return Err(anyhow!("vectorized_copy failed: {:?}", status));
    }

    tracing::debug!(
        total_chunks,
        chunk_size,
        "FC↔LW vectorized_copy transfer completed"
    );

    pointers_transfered_event.synchronize()?;

    Ok(())
}
