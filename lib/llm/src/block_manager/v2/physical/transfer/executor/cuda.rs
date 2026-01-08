// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA executor for GPU memory transfers.

use super::TransferContext;
use super::{PhysicalLayout, TransferStrategy};
use crate::block_manager::v2::kernels::OperationalCopyBackend;
use crate::block_manager::v2::physical::transfer::context::TransferCompleteNotification;
use anyhow::{Result, anyhow};
use cudarc::driver::result as cuda_result;
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
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
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
        TransferStrategy::CudaAsyncD2H | TransferStrategy::CudaBlockingD2H => ctx.d2h_stream(),
        _ => ctx.h2d_stream(), // H2D and D2D use h2d_stream
    };

    // Perform CUDA transfers based on strategy
    match strategy {
        TransferStrategy::CudaAsyncH2D => {
            let backend = ctx.operational_backend();
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
                tracing::debug!("Kernel-based H2D failed ({}), falling back to memcpy", e);
                execute_h2d(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            }
        }
        TransferStrategy::CudaAsyncD2H => {
            let backend = ctx.operational_backend();
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
                tracing::debug!("Kernel-based D2H failed ({}), falling back to memcpy", e);
                execute_d2h(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            }
        }
        TransferStrategy::CudaAsyncD2D => {
            // Try kernel-based path first, fall back to memcpy on error
            let backend = ctx.operational_backend();
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
                tracing::debug!("Kernel-based D2D failed ({}), falling back to memcpy", e);
                execute_d2d(
                    src,
                    dst,
                    src_block_ids,
                    dst_block_ids,
                    layers,
                    stream.as_ref(),
                )?;
            }
        }
        TransferStrategy::CudaBlockingH2D => {
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
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
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
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
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
    src_block_ids: &[usize],
    dst_block_ids: &[usize],
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

/// TODO: For now, we've stubbed this out just so we can merge.
/// For now, we'll always just fall back to memcpy.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn try_execute_operational_kernel(
    _src: &PhysicalLayout,
    _dst: &PhysicalLayout,
    _src_block_ids: &[usize],
    _dst_block_ids: &[usize],
    _layers: Range<usize>,
    _stream: &cudarc::driver::CudaStream,
    _backend: OperationalCopyBackend,
) -> Result<()> {
    anyhow::bail!("Not implemented.");
}
