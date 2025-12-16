// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA executor for GPU memory transfers.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::BlockId;
use crate::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::{Result, anyhow};
use cudarc::driver::result as cuda_result;
use dynamo_kvbm_kernels::{OperationalCopyBackend, OperationalCopyDirection, TensorDataType};
use std::ops::Range;

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
/// * `ctx` - Transfer context with CUDA stream
pub fn execute_cuda_transfer(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layer_range: Option<Range<usize>>,
    strategy: TransferStrategy,
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

    // Determine layer range
    let layers = layer_range.unwrap_or(0..src_layout.num_layers());

    // Get appropriate CUDA stream based on transfer direction
    let stream = match strategy {
        TransferStrategy::CudaAsyncD2H | TransferStrategy::CudaBlockingD2H => {
            ctx.next_d2h_streams()
        }
        _ => ctx.next_h2d_streams(), // H2D and D2D use h2d_stream
    };

    // Perform CUDA transfers based on strategy
    match strategy {
        TransferStrategy::CudaAsyncH2D => {
            let backend = select_backend_for_layouts(src, dst);
            tracing::debug!(
                strategy = "CudaAsyncH2D",
                ?backend,
                num_blocks = src_block_ids.len(),
                "Attempting kernel-based transfer"
            );
            if let Err(e) = try_execute_operational_kernel(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers.clone(),
                stream.as_ref(),
                backend,
            ) {
                // Fallback to memcpy-based path
                tracing::debug!(
                    strategy = "CudaAsyncH2D",
                    error = %e,
                    "Kernel-based transfer failed, falling back to memcpy"
                );
                execute_h2d(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            } else {
                tracing::debug!(
                    strategy = "CudaAsyncH2D",
                    ?backend,
                    "Kernel-based transfer succeeded"
                );
            }
        }
        TransferStrategy::CudaAsyncD2H => {
            let backend = select_backend_for_layouts(src, dst);
            tracing::debug!(
                strategy = "CudaAsyncD2H",
                ?backend,
                num_blocks = src_block_ids.len(),
                "Attempting kernel-based transfer"
            );
            if let Err(e) = try_execute_operational_kernel(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers.clone(),
                stream.as_ref(),
                backend,
            ) {
                // Fallback to memcpy-based path
                tracing::debug!(
                    strategy = "CudaAsyncD2H",
                    error = %e,
                    "Kernel-based transfer failed, falling back to memcpy"
                );
                execute_d2h(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            } else {
                tracing::debug!(
                    strategy = "CudaAsyncD2H",
                    ?backend,
                    "Kernel-based transfer succeeded"
                );
            }
        }
        TransferStrategy::CudaAsyncD2D => {
            let backend = select_backend_for_layouts(src, dst);
            tracing::debug!(
                strategy = "CudaAsyncD2D",
                ?backend,
                num_blocks = src_block_ids.len(),
                "Attempting kernel-based transfer"
            );
            if let Err(e) = try_execute_operational_kernel(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers.clone(),
                stream.as_ref(),
                backend,
            ) {
                // Fallback to memcpy-based path
                tracing::debug!(
                    strategy = "CudaAsyncD2D",
                    error = %e,
                    "Kernel-based transfer failed, falling back to memcpy"
                );
                execute_d2d(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            } else {
                tracing::debug!(
                    strategy = "CudaAsyncD2D",
                    ?backend,
                    "Kernel-based transfer succeeded"
                );
            }
        }
        TransferStrategy::CudaBlockingH2D => {
            tracing::debug!(
                strategy = "CudaBlockingH2D",
                num_blocks = src_block_ids.len(),
                "Executing blocking memcpy transfer"
            );
            execute_h2d(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers,
                stream.as_ref(),
            )?;
            // Synchronize immediately for blocking transfer
            stream.synchronize()?;
        }
        TransferStrategy::CudaBlockingD2H => {
            tracing::debug!(
                strategy = "CudaBlockingD2H",
                num_blocks = src_block_ids.len(),
                "Executing blocking memcpy transfer"
            );
            execute_d2h(
                src,
                dst,
                src_block_ids,
                dst_block_ids,
                layers,
                stream.as_ref(),
            )?;
            // Synchronize immediately for blocking transfer
            stream.synchronize()?;
        }
        _ => {
            return Err(anyhow!("Invalid CUDA transfer strategy: {:?}", strategy));
        }
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

/// Execute host-to-device transfer.
fn execute_h2d(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &cudarc::driver::CudaStream,
) -> Result<()> {
    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..src.layout().outer_dim() {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                unsafe {
                    let src_ptr = src_region.addr() as *const u8;
                    let dst_ptr = dst_region.addr() as u64;
                    let src_slice = std::slice::from_raw_parts(src_ptr, src_region.size());
                    cuda_result::memcpy_htod_async(dst_ptr, src_slice, stream.cu_stream())?;
                }
            }
        }
    }
    Ok(())
}

/// Execute device-to-host transfer.
fn execute_d2h(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &cudarc::driver::CudaStream,
) -> Result<()> {
    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..src.layout().outer_dim() {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                unsafe {
                    let src_ptr = src_region.addr() as u64;
                    let dst_ptr = dst_region.addr() as *mut u8;
                    let dst_slice = std::slice::from_raw_parts_mut(dst_ptr, dst_region.size());
                    cuda_result::memcpy_dtoh_async(dst_slice, src_ptr, stream.cu_stream())?;
                }
            }
        }
    }
    Ok(())
}

/// Execute device-to-device transfer.
fn execute_d2d(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &cudarc::driver::CudaStream,
) -> Result<()> {
    for (&src_block_id, &dst_block_id) in src_block_ids.iter().zip(dst_block_ids.iter()) {
        for layer_id in layers.clone() {
            for outer_id in 0..src.layout().outer_dim() {
                let src_region = src.memory_region(src_block_id, layer_id, outer_id)?;
                let dst_region = dst.memory_region(dst_block_id, layer_id, outer_id)?;

                if src_region.size() != dst_region.size() {
                    return Err(anyhow!(
                        "Size mismatch at block=({},{}), layer={}, outer={}: src={}, dst={}",
                        src_block_id,
                        dst_block_id,
                        layer_id,
                        outer_id,
                        src_region.size(),
                        dst_region.size()
                    ));
                }

                unsafe {
                    let src_ptr = src_region.addr() as u64;
                    let dst_ptr = dst_region.addr() as u64;
                    cuda_result::memcpy_dtod_async(
                        dst_ptr,
                        src_ptr,
                        src_region.size(),
                        stream.cu_stream(),
                    )?;
                }
            }
        }
    }
    Ok(())
}

/// Check if layouts are compatible for kernel-based operational copy.
///
/// Compatible when at least one side is fully contiguous (Operational).
/// The kernel works with any combination of FC and LayerWise layouts.
fn are_layouts_kernel_compatible(src: &PhysicalLayout, dst: &PhysicalLayout) -> bool {
    let src_fc = src.layout().is_fully_contiguous();
    let dst_fc = dst.layout().is_fully_contiguous();

    // At least one must be FC (operational) - the other can be either FC or LW
    src_fc || dst_fc
}

/// Map dtype width in bytes to TensorDataType enum.
fn map_dtype(dtype_width_bytes: usize) -> Option<TensorDataType> {
    match dtype_width_bytes {
        2 => Some(TensorDataType::F16), // Default to F16 for 2-byte types
        4 => Some(TensorDataType::F32),
        8 => Some(TensorDataType::F64),
        _ => None,
    }
}

/// Select the optimal backend for operational copy based on layout contiguity.
///
/// Decision logic:
/// 1. If CUDA 12.9+ is available, use Auto (which prioritizes cudaMemcpyBatchAsync)
/// 2. If both layouts are fully contiguous, use cudaMemcpyAsync
/// 3. If one or both layouts are not fully contiguous (mixed), use the vectorized kernel
fn select_backend_for_layouts(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
) -> OperationalCopyBackend {
    // If batch copy available (CUDA 12.9+), let Auto handle it
    if dynamo_kvbm_kernels::is_memcpy_batch_available() {
        return OperationalCopyBackend::Auto;
    }

    let src_fc = src.layout().is_fully_contiguous();
    let dst_fc = dst.layout().is_fully_contiguous();

    if src_fc && dst_fc {
        // Both contiguous: cudaMemcpyAsync is optimal
        OperationalCopyBackend::MemcpyAsync
    } else {
        // Mixed contiguity: use vectorized kernel
        OperationalCopyBackend::KernelOnly
    }
}

/// Execute device-to-device transfer using operational copy kernels.
///
/// This uses the kvbm-kernels operational_copy for FC↔LW or FC↔FC transfers.
/// Falls back to memcpy-based path on incompatibility or errors.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn try_execute_operational_kernel(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    src_block_ids: &[BlockId],
    dst_block_ids: &[BlockId],
    layers: Range<usize>,
    stream: &cudarc::driver::CudaStream,
    backend: OperationalCopyBackend,
) -> Result<()> {
    // Bind CUDA context to current thread before any CUDA operations.
    // This is necessary because this function may be called from any tokio worker thread,
    // but malloc_sync() requires the context to be current on the calling thread.
    stream.context().bind_to_thread()?;

    // Check compatibility
    if !are_layouts_kernel_compatible(src, dst) {
        return Err(anyhow!("Layouts not compatible for kernel-based copy"));
    }

    let src_layout = src.layout();
    let _dst_layout = dst.layout();

    // Map dtype
    let dtype = map_dtype(src_layout.dtype_width_bytes()).ok_or_else(|| {
        anyhow!(
            "Unsupported dtype width: {}",
            src_layout.dtype_width_bytes()
        )
    })?;

    // Determine which side is operational (FC)
    let src_is_operational = src_layout.is_fully_contiguous();
    let direction = if src_is_operational {
        OperationalCopyDirection::OperationalToBlock
    } else {
        OperationalCopyDirection::BlockToOperational
    };

    let (operational_layout, block_layout) = if src_is_operational {
        (src, dst)
    } else {
        (dst, src)
    };

    // Compute kernel parameters
    let nl = layers.len();
    let no = src_layout.outer_dim();
    let inner = src_layout.page_size() * src_layout.inner_dim();
    let elem_size = src_layout.dtype_width_bytes();
    let num_blocks = src_block_ids.len();

    // Build pointer tables
    // For block layout: flat array of pointers [block][layer][outer]
    // For operational layout: array of base pointers per block
    let mut block_ptrs_host: Vec<usize> = Vec::with_capacity(num_blocks * nl * no);
    let mut operational_ptrs_host: Vec<usize> = Vec::with_capacity(num_blocks);

    let (block_src_ids, block_dst_ids) = if src_is_operational {
        (dst_block_ids, src_block_ids)
    } else {
        (src_block_ids, dst_block_ids)
    };

    for (&block_src_id, &block_dst_id) in block_src_ids.iter().zip(block_dst_ids.iter()) {
        // Collect block pointers for this block across all layers and outer dims
        for layer_id in layers.clone() {
            for outer_id in 0..no {
                let region = block_layout.memory_region(block_src_id, layer_id, outer_id)?;
                block_ptrs_host.push(region.addr());
            }
        }

        // Collect operational base pointer for this block
        // For FC layout, compute base for the first layer in range
        let op_base = operational_layout.memory_region(block_dst_id, layers.start, 0)?;
        operational_ptrs_host.push(op_base.addr());
    }

    // Allocate device memory for pointer tables
    let block_ptrs_device_raw =
        unsafe { cuda_result::malloc_sync(block_ptrs_host.len() * std::mem::size_of::<usize>())? };
    let op_ptrs_device_raw = unsafe {
        cuda_result::malloc_sync(operational_ptrs_host.len() * std::mem::size_of::<usize>())?
    };

    // Copy pointer tables to device
    unsafe {
        let block_ptrs_slice = std::slice::from_raw_parts(
            block_ptrs_host.as_ptr() as *const u8,
            block_ptrs_host.len() * std::mem::size_of::<usize>(),
        );
        cuda_result::memcpy_htod_async(
            block_ptrs_device_raw,
            block_ptrs_slice,
            stream.cu_stream(),
        )?;

        let op_ptrs_slice = std::slice::from_raw_parts(
            operational_ptrs_host.as_ptr() as *const u8,
            operational_ptrs_host.len() * std::mem::size_of::<usize>(),
        );
        cuda_result::memcpy_htod_async(op_ptrs_device_raw, op_ptrs_slice, stream.cu_stream())?;
    }

    // Launch kernel
    let status = unsafe {
        dynamo_kvbm_kernels::operational_copy(
            block_ptrs_host.as_ptr() as *const *const std::ffi::c_void,
            block_ptrs_device_raw as usize as *const *const std::ffi::c_void,
            operational_ptrs_host.as_ptr() as *const *mut std::ffi::c_void,
            op_ptrs_device_raw as usize as *const *const std::ffi::c_void,
            num_blocks,
            nl,
            no,
            inner,
            elem_size,
            dtype,
            direction,
            backend,
            stream.cu_stream() as cudarc::runtime::sys::cudaStream_t,
        )
    };

    // Free device allocations
    unsafe {
        let _ = cuda_result::free_async(block_ptrs_device_raw, stream.cu_stream());
        let _ = cuda_result::free_async(op_ptrs_device_raw, stream.cu_stream());
    }

    if status != cudarc::runtime::sys::cudaError::cudaSuccess {
        return Err(anyhow!(
            "Kernel launch failed with CUDA error: {:?}",
            status
        ));
    }

    Ok(())
}
