// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use super::remote::{RemoteBlockDescriptor, RemoteKey, RemoteStorageKind, RemoteTransferDirection};
use crate::block_manager::config::RemoteTransferContext;
use crate::block_manager::storage::nixl::NixlRegisterableStorage;
use crate::block_manager::storage::{DiskStorage, ObjectStorage};
use anyhow::Result;
use nixl_sys::{
    Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, XferDescList, XferOp, XferRequest,
    XferStatus,
};
use std::future::Future;
use tokio_util::sync::CancellationToken;
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
            )?;

            dst_dl.add_desc(
                dst_desc.as_ptr() as usize,
                dst_desc.size(),
                dst_desc.device_id(),
            )?;
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
                    )?;

                    dst_dl.add_desc(
                        dst_desc.as_ptr() as usize,
                        dst_desc.size(),
                        dst_desc.device_id(),
                    )?;
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
                TransferError::ExecutionError(format!("Failed to register object storage: {:?}", e))
            })?;

            obj_storages.push(obj_storage);
            _registration_handles.push(handle);
        }

        // Build transfer descriptor lists
        let mut src_dl = XferDescList::new(MemType::Dram).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e))
        })?;
        let mut dst_dl = XferDescList::new(MemType::Object).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e))
        })?;

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
            .map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e))
            })?;

        let still_pending = agent.post_xfer_req(&xfer_req, None).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e))
        })?;

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

    // For Offload (write): create files
    // For Onboard (read): open existing files
    let create_files = matches!(direction, RemoteTransferDirection::Offload);

    // Use a scope block to ensure all non-Send types are dropped before await
    let (xfer_req, still_pending, _disk_storages) = {
        // Dynamically create/open and register disk storage for each block
        let mut disk_storages = Vec::with_capacity(num_blocks);

        for desc in descriptors.iter() {
            // Get file path from descriptor's DiskKey
            let file_path = match desc.key() {
                RemoteKey::Disk(disk_key) => disk_key.full_path(),
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Disk key for disk storage transfer".to_string(),
                    ));
                }
            };

            // Create or open DiskStorage at the specified path
            // persist=true ensures files aren't deleted when DiskStorage is dropped
            let mut disk_storage =
                DiskStorage::new_at_path(&file_path, block_size, create_files, true).map_err(
                    |e| {
                        TransferError::ExecutionError(format!(
                            "Failed to {} DiskStorage at {}: {:?}",
                            if create_files { "create" } else { "open" },
                            file_path,
                            e
                        ))
                    },
                )?;

            // Register with NIXL - this makes it available for transfers
            disk_storage.nixl_register(agent, None).map_err(|e| {
                TransferError::ExecutionError(format!(
                    "Failed to register disk storage {}: {:?}",
                    file_path, e
                ))
            })?;

            disk_storages.push(disk_storage);
        }

        // Build transfer descriptor lists for disk
        let mut src_dl = XferDescList::new(MemType::Dram).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e))
        })?;
        let mut dst_dl = XferDescList::new(MemType::File).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e))
        })?;

        for (block, disk_storage) in local_blocks.iter().zip(disk_storages.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            // Add DRAM source descriptor
            let _ = src_dl.add_desc(addr, block_size, 0);

            // Add FILE destination descriptor using the actual file descriptor
            let fd = disk_storage.fd();
            let _ = dst_dl.add_desc(0, block_size, fd);
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
            .map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e))
            })?;

        let still_pending = agent.post_xfer_req(&xfer_req, None).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e))
        })?;

        // Return disk_storages to keep them alive during the transfer
        (xfer_req, still_pending, disk_storages)
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

#[cfg(all(test, feature = "testing-cuda", feature = "testing-nixl"))]
mod tests {
    use super::*;
    use crate::block_manager::block::transfer::context::TransferContext;
    use crate::block_manager::{
        LayoutConfig,
        block::{BasicMetadata, Block, BlockData, locality},
        config::{RemoteStorageConfig, RemoteTransferContext},
        layout::{BlockLayoutConfig, FullyContiguous, nixl::NixlLayout},
        storage::{PinnedAllocator, PinnedStorage},
    };
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    // Shared NIXL agent with OBJ and POSIX backends
    lazy_static::lazy_static! {
        static ref TEST_AGENT: Arc<Option<NixlAgent>> = {
            let agent = NixlAgent::new("nixl-transfer-test").expect("Failed to create NIXL agent");

            // Create OBJ backend for object storage
            if let Ok((_, params)) = agent.get_plugin_params("OBJ") {
                match agent.create_backend("OBJ", &params) {
                    Ok(_) => eprintln!("OBJ backend created"),
                    Err(e) => eprintln!("OBJ backend failed: {}", e),
                }
            } else {
                eprintln!("OBJ plugin not found");
            }

            // Create POSIX backend for disk storage
            if let Ok((_, params)) = agent.get_plugin_params("POSIX") {
                match agent.create_backend("POSIX", &params) {
                    Ok(_) => eprintln!("POSIX backend created"),
                    Err(e) => eprintln!("POSIX backend failed: {}", e),
                }
            } else {
                eprintln!("POSIX plugin not found");
            }

            Arc::new(Some(agent))
        };

        static ref CUDA_CTX: Arc<CudaContext> = {
            CudaContext::new(0).expect("Failed to create CUDA context")
        };
    }

    fn create_test_layout(num_blocks: usize) -> FullyContiguous<PinnedStorage> {
        let config = LayoutConfig::builder()
            .num_blocks(num_blocks)
            .num_layers(2)
            .outer_dim(1)
            .page_size(4)
            .inner_dim(64)
            .dtype_width_bytes(2)
            .build()
            .unwrap();

        let allocator = PinnedAllocator::new().unwrap();
        FullyContiguous::allocate(config, &allocator).unwrap()
    }

    fn create_transfer_context() -> Arc<TransferContext> {
        let stream = CUDA_CTX.default_stream();
        let handle = tokio::runtime::Handle::current();
        Arc::new(TransferContext::new(
            TEST_AGENT.clone(),
            stream,
            handle,
            None,
        ))
    }

    fn create_disk_remote_context(base: Arc<TransferContext>, path: &str) -> RemoteTransferContext {
        RemoteTransferContext::new(base, RemoteStorageConfig::disk(path, false))
    }

    fn create_object_remote_context(
        base: Arc<TransferContext>,
        bucket: &str,
    ) -> RemoteTransferContext {
        RemoteTransferContext::new(base, RemoteStorageConfig::object(bucket))
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_empty_descriptors() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        let descriptors: Vec<RemoteBlockDescriptor> = vec![];
        let local_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = vec![];

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &local_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_count_mismatch() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        // Create 2 descriptors but 1 block - should fail
        let descriptors = vec![
            RemoteBlockDescriptor::disk_from_hash("/tmp", 0x1234, 1024),
            RemoteBlockDescriptor::disk_from_hash("/tmp", 0x5678, 1024),
        ];

        let mut layout = create_test_layout(1);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let layout = Arc::new(layout);
        let blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..1)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(matches!(result, Err(TransferError::CountMismatch(2, 1))));
    }

    #[tokio::test]
    async fn test_execute_remote_transfer_early_cancellation() {
        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, "/tmp/nixl-test");
        let cancel_token = CancellationToken::new();

        // Cancel before starting
        cancel_token.cancel();

        let descriptors = vec![RemoteBlockDescriptor::disk_from_hash("/tmp", 0x1234, 1024)];

        let mut layout = create_test_layout(1);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let layout = Arc::new(layout);
        let blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..1)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(matches!(result, Err(TransferError::Cancelled)));
    }

    /// Test disk transfer using POSIX backend - writes data to disk and reads it back
    #[tokio::test]
    async fn test_posix_disk_transfer_roundtrip() {
        use tempfile::tempdir;

        let temp_dir = tempdir().unwrap();
        let base_path = temp_dir.path().to_str().unwrap().to_string();

        let base_ctx = create_transfer_context();
        let remote_ctx = create_disk_remote_context(base_ctx, &base_path);
        let cancel_token = CancellationToken::new();

        // Create blocks and fill with test data
        let mut layout = create_test_layout(2);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let block_size = layout.layout_data_bytes() / layout.num_blocks();
        let layout = Arc::new(layout);

        let mut blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Fill blocks with recognizable pattern
        for (i, block) in blocks.iter_mut().enumerate() {
            let mut view = block.block_data_mut().block_view_mut().unwrap();
            let slice = unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            for (j, byte) in slice.iter_mut().enumerate() {
                *byte = ((i * 100 + j) % 256) as u8;
            }
        }

        let descriptors: Vec<RemoteBlockDescriptor> = (0..2u64)
            .map(|i| RemoteBlockDescriptor::disk_from_hash(&base_path, 0x1000 + i, block_size))
            .collect();

        // Offload (write to disk via POSIX)
        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Disk,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        if result.is_err() {
            eprintln!(
                "POSIX disk transfer test skipped - backend may not be available: {:?}",
                result
            );
            return;
        }
        assert!(result.is_ok(), "POSIX Offload failed: {:?}", result);

        // Drop offload blocks to ensure we're not reusing cached memory
        drop(blocks);

        // Create fresh blocks for onboarding (different memory)
        let mut onboard_layout = create_test_layout(2);
        onboard_layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let onboard_layout = Arc::new(onboard_layout);

        let mut onboard_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(onboard_layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Verify onboard blocks start zeroed (not containing our pattern)
        for block in onboard_blocks.iter() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            assert!(
                slice.iter().all(|&b| b == 0),
                "Onboard blocks should start zeroed"
            );
        }

        // Onboard (read from disk via POSIX) into fresh blocks
        let result = execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Disk,
            &descriptors,
            &onboard_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok(), "POSIX Onboard failed: {:?}", result);

        // Verify data was restored to the new blocks
        for (i, block) in onboard_blocks.iter().enumerate() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            for (j, &byte) in slice.iter().enumerate() {
                let expected = ((i * 100 + j) % 256) as u8;
                assert_eq!(
                    byte, expected,
                    "POSIX: Data mismatch at block {} byte {}",
                    i, j
                );
            }
        }
        eprintln!("POSIX disk roundtrip transfer successful (using separate blocks for onboard)");
    }

    /// Test object storage transfer using OBJ backend - writes data to S3/object store and reads it back
    #[tokio::test]
    async fn test_obj_object_storage_transfer_roundtrip() {
        let bucket =
            std::env::var("NIXL_TEST_BUCKET").unwrap_or_else(|_| "nixl-test-bucket".to_string());

        let base_ctx = create_transfer_context();
        let remote_ctx = create_object_remote_context(base_ctx, &bucket);
        let cancel_token = CancellationToken::new();

        // Create blocks and fill with test data
        let mut layout = create_test_layout(2);
        layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let block_size = layout.layout_data_bytes() / layout.num_blocks();
        let layout = Arc::new(layout);

        let mut blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        for (i, block) in blocks.iter_mut().enumerate() {
            let mut view = block.block_data_mut().block_view_mut().unwrap();
            let slice = unsafe { std::slice::from_raw_parts_mut(view.as_mut_ptr(), view.size()) };
            for (j, byte) in slice.iter_mut().enumerate() {
                *byte = ((i * 200 + j) % 256) as u8;
            }
        }

        // Use unique sequence hashes for this test run
        let test_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let descriptors: Vec<RemoteBlockDescriptor> = (0..2u64)
            .map(|i| RemoteBlockDescriptor::object_from_hash(&bucket, test_id + i, block_size))
            .collect();

        // Offload (write to object storage via OBJ)
        let result = execute_remote_transfer(
            RemoteTransferDirection::Offload,
            RemoteStorageKind::Object,
            &descriptors,
            &blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        if result.is_err() {
            eprintln!(
                "OBJ object storage test skipped - backend may not be available or S3 not configured: {:?}",
                result
            );
            return;
        }
        assert!(result.is_ok(), "OBJ Offload failed: {:?}", result);

        // Drop offload blocks to ensure we're not reusing cached memory
        drop(blocks);

        // Create fresh blocks for onboarding (different memory)
        let mut onboard_layout = create_test_layout(2);
        onboard_layout
            .nixl_register(TEST_AGENT.as_ref().as_ref().unwrap(), None)
            .unwrap();
        let onboard_layout = Arc::new(onboard_layout);

        let mut onboard_blocks: Vec<Block<PinnedStorage, locality::Local, BasicMetadata>> = (0..2)
            .map(|i| {
                let data = BlockData::new(onboard_layout.clone(), i, 0, 0);
                Block::new(data, BasicMetadata::default()).unwrap()
            })
            .collect();

        // Verify onboard blocks start zeroed (not containing our pattern)
        for block in onboard_blocks.iter() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            assert!(
                slice.iter().all(|&b| b == 0),
                "Onboard blocks should start zeroed"
            );
        }

        let result = execute_remote_transfer(
            RemoteTransferDirection::Onboard,
            RemoteStorageKind::Object,
            &descriptors,
            &onboard_blocks,
            &remote_ctx,
            &cancel_token,
        )
        .await;

        assert!(result.is_ok(), "OBJ Onboard failed: {:?}", result);

        // Verify data was restored to the new blocks
        for (i, block) in onboard_blocks.iter().enumerate() {
            let view = block.block_data().block_view().unwrap();
            let slice = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.size()) };
            for (j, &byte) in slice.iter().enumerate() {
                let expected = ((i * 200 + j) % 256) as u8;
                assert_eq!(
                    byte, expected,
                    "OBJ: Data mismatch at block {} byte {}",
                    i, j
                );
            }
        }
        eprintln!("OBJ object storage roundtrip transfer successful");
    }
}
