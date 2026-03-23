// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::block::{BlockDataProvider, BlockDataProviderMut};
use crate::block_manager::storage::ze::ZeCommandQueue;
use std::ffi::c_void;
use std::ops::Range;

type ZeMemcpyFnPtr =
    fn(src_ptr: *const u8, dst_ptr: *mut u8, size: usize, queue: &ZeCommandQueue)
        -> Result<(), TransferError>;

#[derive(Debug, Default)]
pub struct ZeMemPool;

fn ze_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<ZeMemcpyFnPtr, TransferError> {
    match strategy {
        TransferStrategy::AsyncH2D | TransferStrategy::BlockingH2D => Ok(ze_memcpy_h2d),
        TransferStrategy::AsyncD2H | TransferStrategy::BlockingD2H => Ok(ze_memcpy_d2h),
        _ => Err(TransferError::ExecutionError(format!(
            "Unsupported ZE copy strategy: {:?}",
            strategy
        ))),
    }
}

/// Copy a block using ZE memcpy primitives.
pub fn copy_block<'a, Source, Destination>(
    sources: &'a Source,
    destinations: &'a mut Destination,
    queue: &ZeCommandQueue,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = ze_memcpy_fn_ptr(&strategy)?;

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());
        unsafe {
            tracing::debug!(
                "ZE copy_block contiguous: strategy={:?}, src=0x{:x}, dst=0x{:x}, size={}",
                strategy,
                src_view.as_ptr() as usize,
                dst_view.as_mut_ptr() as usize,
                src_view.size()
            );
            memcpy_fn(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size(), queue)?;
        }
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(
            0..src_data.num_layers(),
            sources,
            destinations,
            queue,
            strategy,
        )?;
    }

    Ok(())
}

/// Temporary ZE implementation for API parity with CUDA custom kernel path.
///
/// This currently falls back to batched ZE memcpys with a single command list.
pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    queue: &ZeCommandQueue,
    _ctx: &crate::block_manager::block::transfer::TransferContext,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if sources.len() != destinations.len() {
        return Err(TransferError::CountMismatch(
            sources.len(),
            destinations.len(),
        ));
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!(
            "ZE custom batch command list creation failed: {:?}",
            e
        ))
    })?;

    for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
        let src_data = src.block_data();
        let dst_data = dst.block_data_mut();

        if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
            let src_view = src_data.block_view()?;
            let mut dst_view = dst_data.block_view_mut()?;

            debug_assert_eq!(src_view.size(), dst_view.size());

            if src_view.size() == 0 {
                continue;
            }

            unsafe {
                tracing::debug!(
                    "ZE copy_blocks_with_customized_kernel contiguous: src=0x{:x}, dst=0x{:x}, size={}",
                    src_view.as_ptr() as usize,
                    dst_view.as_mut_ptr() as usize,
                    src_view.size()
                );
                list.append_memcpy(
                    dst_view.as_mut_ptr() as *mut c_void,
                    src_view.as_ptr() as *const c_void,
                    src_view.size(),
                )
                .map_err(|e| {
                    TransferError::ExecutionError(format!(
                        "ZE custom batch append_memcpy (contiguous) failed: {:?}",
                        e
                    ))
                })?;
            }
        } else {
            assert_eq!(src_data.num_layers(), dst_data.num_layers());
            assert_eq!(src_data.num_outer_dims(), dst_data.num_outer_dims());

            for layer_idx in 0..src_data.num_layers() {
                for outer_idx in 0..src_data.num_outer_dims() {
                    let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                    let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                    debug_assert_eq!(src_view.size(), dst_view.size());

                    if src_view.size() == 0 {
                        continue;
                    }

                    unsafe {
                        tracing::debug!(
                            "ZE copy_blocks_with_customized_kernel layered: layer={}, outer={}, src=0x{:x}, dst=0x{:x}, size={}",
                            layer_idx,
                            outer_idx,
                            src_view.as_ptr() as usize,
                            dst_view.as_mut_ptr() as usize,
                            src_view.size()
                        );
                        list.append_memcpy(
                            dst_view.as_mut_ptr() as *mut c_void,
                            src_view.as_ptr() as *const c_void,
                            src_view.size(),
                        )
                        .map_err(|e| {
                            TransferError::ExecutionError(format!(
                                "ZE custom batch append_memcpy (layered) failed: {:?}",
                                e
                            ))
                        })?;
                    }
                }
            }
        }
    }

    list.close().map_err(|e| {
        TransferError::ExecutionError(format!("ZE custom batch list close failed: {:?}", e))
    })?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE custom batch queue submit failed: {:?}", e))
    })?;

    Ok(())
}

/// Copy a range of layers from source to destination using ZE memcpy.
pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    sources: &'a Source,
    destinations: &'a mut Destination,
    queue: &ZeCommandQueue,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = sources.block_data();
    let dst_data = destinations.block_data_mut();
    let memcpy_fn = ze_memcpy_fn_ptr(&strategy)?;

    assert_eq!(src_data.num_outer_dims(), dst_data.num_outer_dims());

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

            debug_assert_eq!(src_view.size(), dst_view.size());
            unsafe {
                tracing::debug!(
                    "ZE copy_layers: strategy={:?}, layer={}, outer={}, src=0x{:x}, dst=0x{:x}, size={}",
                    strategy,
                    layer_idx,
                    outer_idx,
                    src_view.as_ptr() as usize,
                    dst_view.as_mut_ptr() as usize,
                    src_view.size()
                );
                memcpy_fn(src_view.as_ptr(), dst_view.as_mut_ptr(), src_view.size(), queue)?;
            }
        }
    }

    Ok(())
}

#[inline(always)]
fn ze_memcpy_d2h(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    queue: &ZeCommandQueue,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source ZE device pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination host pointer is null");

    if size == 0 {
        return Ok(());
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!("ZE D2H command list creation failed: {:?}", e))
    })?;
    list.append_memcpy(dst_ptr as *mut c_void, src_ptr as *const c_void, size)
        .map_err(|e| TransferError::ExecutionError(format!("ZE D2H append_memcpy failed: {:?}", e)))?;
    list.close()
        .map_err(|e| TransferError::ExecutionError(format!("ZE D2H list close failed: {:?}", e)))?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE D2H queue submit failed: {:?}", e))
    })?;
    Ok(())
}

#[inline(always)]
fn ze_memcpy_h2d(
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
    queue: &ZeCommandQueue,
) -> Result<(), TransferError> {
    debug_assert!(!src_ptr.is_null(), "Source host pointer is null");
    debug_assert!(!dst_ptr.is_null(), "Destination ZE device pointer is null");

    if size == 0 {
        return Ok(());
    }

    let mut list = queue.create_command_list().map_err(|e| {
        TransferError::ExecutionError(format!("ZE H2D command list creation failed: {:?}", e))
    })?;
    list.append_memcpy(dst_ptr as *mut c_void, src_ptr as *const c_void, size)
        .map_err(|e| TransferError::ExecutionError(format!("ZE H2D append_memcpy failed: {:?}", e)))?;
    list.close()
        .map_err(|e| TransferError::ExecutionError(format!("ZE H2D list close failed: {:?}", e)))?;
    queue.execute_nonblocking(&mut list).map_err(|e| {
        TransferError::ExecutionError(format!("ZE H2D queue submit failed: {:?}", e))
    })?;
    Ok(())
}
