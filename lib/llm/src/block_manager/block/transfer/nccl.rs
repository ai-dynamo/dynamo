// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NCCL collective broadcast operations for block data.
//!
//! This module provides functions for broadcasting block data across multiple
//! GPUs using NCCL collective operations.

use super::*;

use std::ffi::c_void;
use std::ops::Range;

use anyhow::{Context, Result};
use cudarc::driver::sys::CUstream;
use cudarc::nccl::sys::{
    ncclBcast, ncclComm_t, ncclDataType_t, ncclGroupEnd, ncclGroupStart, ncclResult_t,
};

/// Check NCCL result and convert to anyhow::Result
fn check_nccl_result(result: ncclResult_t) -> Result<()> {
    if result == ncclResult_t::ncclSuccess {
        Ok(())
    } else {
        anyhow::bail!("NCCL error: {:?}", result)
    }
}

/// RAII guard for NCCL group operations.
///
/// Calls `ncclGroupStart` on creation and `ncclGroupEnd` on drop.
/// Use this to batch multiple NCCL operations efficiently.
///
/// # Example
/// ```ignore
/// let group = unsafe { NcclGroup::new()? };
/// unsafe { bcast_block(&block1, root, comm, stream)?; }
/// unsafe { bcast_block(&block2, root, comm, stream)?; }
/// drop(group); // Submits all queued operations
/// ```
///
/// # Safety
/// Creating an `NcclGroup` is unsafe because:
/// - All ranks must create and drop the group collectively
/// - NCCL operations between creation and drop must be valid
pub struct NcclGroup {
    _private: (), // Prevent construction outside of new()
}

impl NcclGroup {
    /// Start a new NCCL group.
    ///
    /// # Safety
    /// - All ranks must call this collectively
    /// - The group must be dropped (ending the group) before any synchronization
    pub unsafe fn new() -> Result<Self> {
        let result = unsafe { ncclGroupStart() };
        check_nccl_result(result).context("ncclGroupStart failed")?;
        Ok(Self { _private: () })
    }
}

impl Drop for NcclGroup {
    fn drop(&mut self) {
        // Safety: If we successfully created the group, we must end it.
        // Panicking here is acceptable as failing to end a group is unrecoverable.
        let result = unsafe { ncclGroupEnd() };
        if result != ncclResult_t::ncclSuccess {
            // Log error but don't panic in drop to avoid double-panic
            tracing::error!("ncclGroupEnd failed in NcclGroup drop: {:?}", result);
        }
    }
}

/// Broadcast a block to all ranks.
///
/// If the block is fully contiguous, uses a single NCCL broadcast call.
/// Otherwise, falls back to layer-by-layer broadcast via [`bcast_layer`].
///
/// This function should be called from within an [`NcclGroup`] scope for
/// efficient batching of multiple broadcasts.
///
/// # Safety
/// - `comm` must be a valid NCCL communicator
/// - `stream` must be a valid CUDA stream
/// - All ranks must call this collectively with matching parameters
/// - The block's memory must be valid GPU memory accessible by the NCCL communicator
/// - Should be called within an [`NcclGroup`] scope
///
/// # Arguments
/// * `block` - The block to broadcast (source on root, destination on other ranks)
/// * `root` - The rank that owns the source data
/// * `comm` - The NCCL communicator
/// * `stream` - The CUDA stream to use for the operation
pub unsafe fn bcast_block<B>(block: &B, root: i32, comm: ncclComm_t, stream: CUstream) -> Result<()>
where
    B: BlockDataProvider,
{
    let data = block.block_data();

    if data.is_fully_contiguous() {
        let view = data.block_view().context("Failed to get block view")?;
        let ptr = unsafe { view.as_ptr() } as usize;
        let size = view.size();

        let result = unsafe {
            ncclBcast(
                ptr as *mut c_void,
                size,
                ncclDataType_t::ncclChar,
                root,
                comm,
                stream.cast(),
            )
        };
        check_nccl_result(result).context("ncclBcast failed")
    } else {
        // Fall back to layer-by-layer broadcast for non-contiguous blocks
        unsafe { bcast_layer(block, None, root, comm, stream) }
    }
}

/// Broadcast block layers to all ranks.
///
/// Iterates over layer views and broadcasts each one. Use this when only a
/// subset of layers should be broadcast, or when the block layout is not
/// fully contiguous.
///
/// This function should be called from within an [`NcclGroup`] scope for
/// efficient batching of multiple broadcasts.
///
/// # Safety
/// - `comm` must be a valid NCCL communicator
/// - `stream` must be a valid CUDA stream
/// - All ranks must call this collectively with matching parameters
/// - The block's memory must be valid GPU memory accessible by the NCCL communicator
/// - Should be called within an [`NcclGroup`] scope
///
/// # Arguments
/// * `block` - The block containing layers to broadcast
/// * `layer_range` - Optional range of layers to broadcast. If None, broadcasts all layers.
/// * `root` - The rank that owns the source data
/// * `comm` - The NCCL communicator
/// * `stream` - The CUDA stream to use for the operation
pub unsafe fn bcast_layer<B>(
    block: &B,
    layer_range: Option<Range<usize>>,
    root: i32,
    comm: ncclComm_t,
    stream: CUstream,
) -> Result<()>
where
    B: BlockDataProvider,
{
    let data = block.block_data();
    let layer_range = layer_range.unwrap_or(0..data.num_layers());

    for layer_idx in layer_range {
        for outer_idx in 0..data.num_outer_dims() {
            let view = data
                .layer_view(layer_idx, outer_idx)
                .context("Failed to get layer view")?;
            let ptr = unsafe { view.as_ptr() } as usize;
            let size = view.size();

            let result = unsafe {
                ncclBcast(
                    ptr as *mut c_void,
                    size,
                    ncclDataType_t::ncclChar,
                    root,
                    comm,
                    stream.cast(),
                )
            };
            check_nccl_result(result).context("ncclBcast failed in layer loop")?;
        }
    }

    Ok(())
}
