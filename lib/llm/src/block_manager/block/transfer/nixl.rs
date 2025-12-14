// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use super::remote::{RemoteBlockDescriptor, RemoteKey, RemoteStorageKind, RemoteTransferDirection};
use super::context::RemoteTransferContext;

use anyhow::Result;
use nixl_sys::{Agent as NixlAgent, MemoryRegion, MemType, NixlDescriptor, XferDescList, XferOp, XferRequest, XferStatus};
use std::future::Future;
use tokio_util::sync::CancellationToken;

use crate::block_manager::storage::ObjectStorage;

fn append_xfer_request<Source, Destination>(
    src: &Source,
    dst: &mut Destination,
    src_dl: &mut XferDescList,
    dst_dl: &mut XferDescList,
) -> Result<()>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    let src_data = src.block_data();
    let dst_data = dst.block_data_mut();

    if src_data.is_fully_contiguous() && dst_data.is_fully_contiguous() {
        let src_desc = src_data.block_view()?.as_nixl_descriptor();
        let dst_desc = dst_data.block_view_mut()?.as_nixl_descriptor_mut();

        unsafe {
            src_dl.add_desc(
                src_desc.as_ptr() as usize,
                src_desc.size(),
                src_desc.device_id(),
            );

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            );
        }

        Ok(())
    } else {
        assert_eq!(src_data.num_layers(), dst_data.num_layers());
        for layer_idx in 0..src_data.num_layers() {
            for outer_idx in 0..src_data.num_outer_dims() {
                let src_view = src_data.layer_view(layer_idx, outer_idx)?;
                let mut dst_view = dst_data.layer_view_mut(layer_idx, outer_idx)?;

                debug_assert_eq!(src_view.size(), dst_view.size());

                let src_desc = src_view.as_nixl_descriptor();
                let dst_desc = dst_view.as_nixl_descriptor_mut();

                unsafe {
                    src_dl.add_desc(
                        src_desc.as_ptr() as usize,
                        src_desc.size(),
                        src_desc.device_id(),
                    );

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Copy a block from a source to a destination using CUDA memcpy
pub fn write_blocks_to<Source, Destination>(
    src: &[Source],
    dst: &mut [Destination],
    ctx: &Arc<TransferContext>,
    transfer_type: NixlTransfer,
) -> Result<Box<dyn Future<Output = ()> + Send + Sync + Unpin>>
where
    Source: BlockDataProvider,
    Source::StorageType: NixlDescriptor,
    Destination: BlockDataProviderMut,
    Destination::StorageType: NixlDescriptor,
{
    if src.is_empty() || dst.is_empty() {
        return Ok(Box::new(std::future::ready(())));
    }
    assert_eq!(src.len(), dst.len());

    let nixl_agent_arc = ctx.as_ref().nixl_agent();
    let nixl_agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .expect("NIXL agent not found");

    let src_mem_type = src
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();
    let dst_mem_type = dst
        .first()
        .unwrap()
        .block_data()
        .storage_type()
        .nixl_mem_type();

    let mut src_dl = XferDescList::new(src_mem_type)?;
    let mut dst_dl = XferDescList::new(dst_mem_type)?;

    for (src, dst) in src.iter().zip(dst.iter_mut()) {
        append_xfer_request(src, dst, &mut src_dl, &mut dst_dl)?;
    }

    let xfer_req = nixl_agent.create_xfer_req(
        transfer_type.as_xfer_op(),
        &src_dl,
        &dst_dl,
        &nixl_agent.name(),
        None,
    )?;

    let still_pending = nixl_agent.post_xfer_req(&xfer_req, None)?;

    if still_pending {
        Ok(Box::new(Box::pin(async move {
            let nixl_agent = nixl_agent_arc
                .as_ref()
                .as_ref()
                .expect("NIXL agent not found");

            loop {
                match nixl_agent.get_xfer_status(&xfer_req) {
                    Ok(XferStatus::Success) => break, // Transfer is complete.
                    Ok(XferStatus::InProgress) => {
                        tokio::time::sleep(std::time::Duration::from_millis(5)).await
                    } // Transfer is still in progress.
                    Err(e) => {
                        tracing::error!("Error getting transfer status: {}", e);
                        break;
                    }
                }
            }
        })))
    } else {
        Ok(Box::new(std::future::ready(())))
    }
}


/// Execute a remote storage transfer (object storage or disk).
///
/// This function handles the NIXL-level execution of remote transfers.
/// It supports both object storage and remote disk.
///
/// # Arguments
///
/// * `direction` - Whether this is an onboard (read) or offload (write)
/// * `kind` - Type of remote storage (Object or Disk)
/// * `descriptors` - Remote block descriptors with keys and sizes
/// * `local_blocks` - Local host blocks (source for offload, destination for onboard)
/// * `ctx` - Remote transfer context
/// * `cancel_token` - Cancellation token for cooperative cancellation
///
/// # Returns
///
/// `Ok(())` on success, or `TransferError` on failure/cancellation.
pub async fn execute_remote_transfer<LB>(
    direction: RemoteTransferDirection,
    kind: RemoteStorageKind,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    if descriptors.is_empty() || local_blocks.is_empty() {
        return Ok(());
    }

    if descriptors.len() != local_blocks.len() {
        return Err(TransferError::CountMismatch(
            descriptors.len(),
            local_blocks.len(),
        ));
    }

    // Check for early cancellation
    if cancel_token.is_cancelled() {
        return Err(TransferError::Cancelled);
    }

    let nixl_agent_arc = ctx.nixl_agent();
    let agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .ok_or_else(|| TransferError::ExecutionError("NIXL agent not available".to_string()))?;

    let num_blocks = descriptors.len();

    // Get block size from first local block
    let first_block = &local_blocks[0];
    let block_size = first_block.block_data().block_view()?.size();

    tracing::debug!(
        "Remote transfer: {} blocks, direction={:?}, kind={:?}, block_size={}",
        num_blocks,
        direction,
        kind,
        block_size
    );

    match kind {
        RemoteStorageKind::Object => {
            execute_object_transfer(
                agent,
                direction,
                descriptors,
                local_blocks,
                block_size,
                ctx,
                cancel_token,
            )
            .await
        }
        RemoteStorageKind::Disk => {
            execute_disk_transfer(
                agent,
                direction,
                descriptors,
                local_blocks,
                block_size,
                ctx,
                cancel_token,
            )
            .await
        }
    }
}

/// Execute object storage transfer.
async fn execute_object_transfer<LB>(
    agent: &NixlAgent,
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    let num_blocks = descriptors.len();
    let _default_bucket = ctx.default_bucket().unwrap_or("default");

    // Use a scope block to ensure all non-Send types are dropped before await
    let (xfer_req, still_pending) = {
        // Register ALL object storage regions with NIXL
        let mut obj_storages = Vec::with_capacity(num_blocks);
        let mut _registration_handles = Vec::with_capacity(num_blocks);

        // TODO: Add support for string-based object keys via metadata in nixl-sys Rust bindings.
        // For now, we pass the sequence hash (u64) directly as device_id.

        for desc in descriptors.iter() {
            let bucket = match desc.key() {
                RemoteKey::Object(obj_key) => obj_key.bucket.as_str(),
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Object key for object storage transfer".to_string(),
                    ));
                }
            };

            // Use sequence hash directly as device_id - NIXL uses this as the object key
            let object_key = desc.sequence_hash().ok_or_else(|| {
                TransferError::ExecutionError(format!(
                    "Descriptor missing sequence_hash: {:?}",
                    desc.key()
                ))
            })?;

            let obj_storage = ObjectStorage::new(bucket, object_key, block_size).map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create ObjectStorage: {:?}", e))
            })?;

            let handle = agent.register_memory(&obj_storage, None).map_err(|e| {
                TransferError::ExecutionError(format!(
                    "Failed to register object storage: {:?}",
                    e
                ))
            })?;

            obj_storages.push(obj_storage);
            _registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        let mut src_dl = XferDescList::new(MemType::Dram)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e)))?;
        let mut dst_dl = XferDescList::new(MemType::Object)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e)))?;

        for (block, desc) in local_blocks.iter().zip(descriptors.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            src_dl.add_desc(addr, block_size, 0); // device_id=0 for host
            dst_dl.add_desc(0, block_size, desc.sequence_hash().unwrap());
        }

        // Determine the transfer operation
        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

        // Create transfer request
        let agent_name = agent.name();
        let xfer_req = agent
            .create_xfer_req(xfer_op, &src_dl, &dst_dl, &agent_name, None)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e)))?;

        let still_pending = agent
            .post_xfer_req(&xfer_req, None)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e)))?;

        (xfer_req, still_pending)
    };

    // Wait for completion with cancellation support
    if still_pending {
        poll_transfer_completion(agent, &xfer_req, cancel_token).await?;
    }

    tracing::debug!(
        "Object transfer complete: {} blocks, direction={:?}",
        num_blocks,
        direction
    );

    Ok(())
}

/// Execute disk storage transfer.
async fn execute_disk_transfer<LB>(
    agent: &NixlAgent,
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    _ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{

    let num_blocks = descriptors.len();

    // Use a scope block to ensure all non-Send types are dropped before await
    let (xfer_req, still_pending) = {
        // Build transfer descriptor lists for disk
        let mut src_dl = XferDescList::new(MemType::Dram)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e)))?;
        let mut dst_dl = XferDescList::new(MemType::File)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e)))?;

        for (block, desc) in local_blocks.iter().zip(descriptors.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            src_dl.add_desc(addr, block_size, 0);

            // Use key hash as device_id for disk
            let disk_key = desc.key().nixl_device_id();
            dst_dl.add_desc(0, block_size, disk_key);
        }

        // Determine the transfer operation
        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

        // Create transfer request
        let agent_name = agent.name();
        let xfer_req = agent
            .create_xfer_req(xfer_op, &src_dl, &dst_dl, &agent_name, None)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e)))?;

        let still_pending = agent
            .post_xfer_req(&xfer_req, None)
            .map_err(|e| TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e)))?;

        (xfer_req, still_pending)
    };

    // Wait for completion with cancellation support
    if still_pending {
        poll_transfer_completion(agent, &xfer_req, cancel_token).await?;
    }

    tracing::debug!(
        "Disk transfer complete: {} blocks, direction={:?}",
        num_blocks,
        direction
    );

    Ok(())
}

/// Poll for transfer completion with cancellation support.
async fn poll_transfer_completion(
    agent: &NixlAgent,
    xfer_req: &XferRequest,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError> {
    let poll_interval = tokio::time::Duration::from_micros(100);

    loop {
        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(TransferError::Cancelled);
            }
            _ = tokio::time::sleep(poll_interval) => {
                let status = agent.get_xfer_status(xfer_req).map_err(|e| {
                    TransferError::ExecutionError(format!("Failed to get transfer status: {:?}", e))
                })?;

                match status {
                    XferStatus::Success => return Ok(()),
                    XferStatus::InProgress => continue,
                    // Handle other status values if they exist
                    #[allow(unreachable_patterns)]
                    other => {
                        return Err(TransferError::ExecutionError(format!(
                            "Transfer failed with status: {:?}",
                            other
                        )));
                    }
                }
            }
        }
    }
}
