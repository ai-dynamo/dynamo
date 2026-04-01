// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL OBJ-backend transfer implementation (object / S3 storage).

use super::*;
use super::remote::{RemoteBlockDescriptor, RemoteKey, RemoteTransferDirection};
use crate::block_manager::config::RemoteTransferContext;
use crate::block_manager::storage::ObjectStorage;
use nixl_sys::{MemType, NixlDescriptor, XferDescList, XferOp};
use tokio_util::sync::CancellationToken;

/// Execute an object-storage (S3 / NIXL OBJ) transfer.
///
/// `pub(crate)` — called exclusively from [`super::nixl::execute_remote_transfer`].
pub(crate) async fn execute_object_transfer<LB>(
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
    let nixl_agent_arc = ctx.nixl_agent();
    let agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .ok_or_else(|| TransferError::ExecutionError("NIXL agent not available".to_string()))?;

    let num_blocks = descriptors.len();
    let worker_id = ctx.worker_id() as usize;

    // Resolve bucket template once — `{worker_id}` is substituted if present.
    // Falls back to the per-descriptor bucket when no template is configured.
    let resolved_bucket_from_ctx = ctx.config().resolve_bucket(worker_id);

    // Use a scope block to ensure all non-Send types are dropped before await
    let (xfer_req, still_pending) = {
        // Register ALL object storage regions with NIXL, collecting hashes for reuse below.
        let mut obj_storages = Vec::with_capacity(num_blocks);
        let mut _registration_handles = Vec::with_capacity(num_blocks);
        let mut sequence_hashes = Vec::with_capacity(num_blocks);

        for desc in descriptors.iter() {
            // Prefer the ctx-level resolved bucket (supports {worker_id} templates);
            // fall back to the per-descriptor bucket for backwards compatibility.
            let desc_bucket = match desc.key() {
                RemoteKey::Object(obj_key) => obj_key.bucket.as_str(),
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Object key for object storage transfer".to_string(),
                    ));
                }
            };
            let bucket = resolved_bucket_from_ctx.as_deref().unwrap_or(desc_bucket);

            let sequence_hash = desc.sequence_hash().ok_or_else(|| {
                TransferError::ExecutionError(format!(
                    "Descriptor missing sequence_hash: {:?}",
                    desc.key()
                ))
            })?;
            sequence_hashes.push(sequence_hash);

            let obj_storage = ObjectStorage::new(bucket, sequence_hash, block_size).map_err(|e| {
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

        for (block, &seq_hash) in local_blocks.iter().zip(sequence_hashes.iter()) {
            let block_view = block.block_data().block_view()?;
            let addr = unsafe { block_view.as_ptr() as usize };

            src_dl.add_desc(addr, block_size, 0);
            dst_dl.add_desc(0, block_size, seq_hash);
        }

        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

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

    if still_pending {
        use tracing::Instrument;
        let nixl_span = tracing::info_span!(
            "nixl_io",
            otel.name = match direction {
                RemoteTransferDirection::Onboard => "kvbm.nixl_read",
                RemoteTransferDirection::Offload => "kvbm.nixl_write",
            },
            description = "NIXL object store I/O (post + completion wait)",
            num_blocks,
            direction = ?direction,
        );
        super::nixl::poll_transfer_completion(agent, &xfer_req, cancel_token)
            .instrument(nixl_span)
            .await?;
    }

    tracing::debug!(
        "Object transfer complete: {} blocks, direction={:?}, worker={}",
        num_blocks,
        direction,
        worker_id,
    );

    Ok(())
}
