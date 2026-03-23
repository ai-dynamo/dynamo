// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! HPU transfer backend scaffolding for phase 4.

use super::*;

use crate::block_manager::storage::hpu::{
    copy_device_to_device_raw, copy_device_to_host_raw, copy_host_to_device_raw,
};
use std::ops::Range;
use synapse::Stream;

type SynapseMemcpyFn = fn(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), TransferError>;

fn synapse_memcpy_fn_ptr(strategy: &TransferStrategy) -> Result<SynapseMemcpyFn, TransferError> {
    match strategy {
        TransferStrategy::AsyncH2D | TransferStrategy::BlockingH2D => Ok(synapse_memcpy_h2d),
        TransferStrategy::AsyncD2H | TransferStrategy::BlockingD2H => Ok(synapse_memcpy_d2h),
        TransferStrategy::AsyncD2D => Ok(synapse_memcpy_d2d),
        other => Err(TransferError::ExecutionError(format!(
            "Unsupported Synapse copy strategy: {:?}",
            other
        ))),
    }
}

fn synapse_memcpy_h2d(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), TransferError> {
    copy_host_to_device_raw(stream, src_ptr, dst_ptr, size)
        .map_err(|e| TransferError::ExecutionError(e.to_string()))
}

fn synapse_memcpy_d2h(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), TransferError> {
    copy_device_to_host_raw(stream, src_ptr, dst_ptr, size)
        .map_err(|e| TransferError::ExecutionError(e.to_string()))
}

fn synapse_memcpy_d2d(
    stream: &Stream,
    src_ptr: *const u8,
    dst_ptr: *mut u8,
    size: usize,
) -> Result<(), TransferError> {
    copy_device_to_device_raw(stream, src_ptr, dst_ptr, size)
        .map_err(|e| TransferError::ExecutionError(e.to_string()))
}

pub fn copy_block<'a, Source, Destination>(
    src: &'a Source,
    dst: &'a mut Destination,
    stream: &Stream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();
    let memcpy_fn = synapse_memcpy_fn_ptr(&strategy)?;

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_view = src_data.block_view()?;
        let mut dst_view = dst_data.block_view_mut()?;

        debug_assert_eq!(src_view.size(), dst_view.size());
        let src_ptr = unsafe { src_view.as_ptr() };
        let dst_ptr = unsafe { dst_view.as_mut_ptr() };
        memcpy_fn(stream, src_ptr, dst_ptr, src_view.size())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        copy_layers(0..src_data.num_layers(), src, dst, stream, strategy)
    }
}

pub fn copy_blocks_with_customized_kernel<'a, Source, Destination>(
    sources: &'a [Source],
    destinations: &'a mut [Destination],
    stream: &Stream,
    _ctx: &TransferContext,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    if sources.len() != destinations.len() {
        return Err(TransferError::CountMismatch(sources.len(), destinations.len()));
    }

    for (src, dst) in sources.iter().zip(destinations.iter_mut()) {
        copy_block(src, dst, stream, TransferStrategy::AsyncD2D)?;
    }
    Ok(())
}

pub fn copy_layers<'a, Source, Destination>(
    layer_range: Range<usize>,
    src: &'a Source,
    dst: &'a mut Destination,
    stream: &Stream,
    strategy: TransferStrategy,
) -> Result<(), TransferError>
where
    Source: BlockDataProvider,
    Destination: BlockDataProviderMut,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();
    let memcpy_fn = synapse_memcpy_fn_ptr(&strategy)?;

    assert_eq!(src_data.num_outer_dims(), dst_data.num_outer_dims());

    for layer_idx in layer_range {
        for outer_idx in 0..src_data.num_outer_dims() {
            let src_view = src_data.layer_view(layer_idx, outer_idx)?;
            let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;
            debug_assert_eq!(src_view.size(), dst_view.size());
            let src_ptr = unsafe { src_view.as_ptr() };
            let dst_ptr = unsafe { dst_view.as_mut_ptr() };
            memcpy_fn(stream, src_ptr, dst_ptr, src_view.size())?;
        }
    }

    Ok(())
}
