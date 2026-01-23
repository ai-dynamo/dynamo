// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::utils::RemoteTransferRequest;
use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    BasicMetadata, Storage,
    block::{
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{
            TransferContext, WriteTo, WriteToStrategy, checksum::compute_checksum,
            remote::RemoteTransferPipeline,
        },
    },
    config::RemoteTransferContext,
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    distributed::vllm::g4_checksum_enabled,
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};
use tokio_util::sync::CancellationToken;

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        Self {
            max_batch_size: MAX_TRANSFER_BATCH_SIZE,
        }
    }

    pub async fn execute_batched_transfer(
        &self,
        handler: &BlockTransferHandler,
        request: BlockTransferRequest,
    ) -> Result<()> {
        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        let batches = blocks.chunks(self.max_batch_size);

        let batch_futures: Vec<_> = batches
            .map(|batch| {
                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                    sequence_hashes: None,
                };
                handler.execute_transfer_direct(batch_request)
            })
            .collect();

        // Execute all batches concurrently
        tracing::debug!("Executing {} batches concurrently", batch_futures.len());

        match try_join_all(batch_futures).await {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Batched connector transfer failed: {}", e);
                Err(e)
            }
        }
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
/// Also handles remote storage transfers (G4 object storage, remote disk) when configured.
#[derive(Clone)]
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    remote_context: Option<Arc<RemoteTransferContext>>,
    cancel_token: CancellationToken,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        remote_context: Option<Arc<RemoteTransferContext>>,
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            remote_context,
            cancel_token,
        })
    }

    fn get_local_data<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        blocks.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| {
                    let block_data = b.block_data() as &dyn Any;

                    block_data
                        .downcast_ref::<LocalBlockData<S>>()
                        .unwrap()
                        .clone()
                })
                .collect()
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_list: &Option<LocalBlockDataList<Source>>,
        target_pool_list: &Option<LocalBlockDataList<Target>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage + NixlDescriptor,
        Target: Storage + NixlDescriptor,
        // Check that the source block is readable, local, and writable to the target block.
        LocalBlockData<Source>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<LocalBlockData<Target>>,
        // Check that the target block is writable.
        LocalBlockData<Target>: WritableBlock<StorageType = Target>,
        LocalBlockData<Source>: BlockDataProvider<Locality = locality::Local>,
        LocalBlockData<Target>: BlockDataProviderMut<Locality = locality::Local>,
    {
        let Some(source_pool_list) = source_pool_list else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_list) = target_pool_list else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources: Vec<LocalBlockData<Source>> = source_idxs
            .map(|idx| source_pool_list[idx].clone())
            .collect();
        let mut targets: Vec<LocalBlockData<Target>> = target_idxs
            .map(|idx| target_pool_list[idx].clone())
            .collect();

        // Perform the transfer, and return the notifying channel.
        match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        }
    }

    /// Execute transfer with batching to prevent resource exhaustion
    pub async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    /// Execute transfer directly without batching (used by the batcher)
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        tracing::debug!("request: {request:#?}");

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Device, Disk) => self.begin_transfer(&self.device, &self.disk, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        Ok(())
    }

    /// Execute a remote transfer (G4 object storage or remote disk).
    ///
    /// Handles both onboard (remote -> host -> device) and offload (device -> host -> remote)
    /// transfers, with optional checksum logging when G4 validation is enabled.
    pub async fn execute_remote_transfer(&self, request: RemoteTransferRequest) -> Result<()> {
        let remote_ctx = self
            .remote_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Remote transfer context not configured"))?;

        let host_blocks = self
            .host
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Host blocks required for remote transfers"))?;

        let pipeline = request.pipeline.to_pipeline();
        let num_blocks = pipeline.num_blocks();
        let direction = pipeline.direction();
        let is_onboard = direction.is_onboard();

        tracing::debug!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks,
            direction = ?direction,
            "executing remote transfer"
        );

        // Get bounce buffer block IDs (host blocks used as staging)
        let bounce_ids = pipeline
            .bounce_block_ids()
            .ok_or_else(|| anyhow::anyhow!("Remote transfer requires bounce buffer block IDs"))?;

        // Get the host blocks for this transfer
        let bounce_blocks: Vec<LocalBlockData<PinnedStorage>> = bounce_ids
            .iter()
            .map(|&idx| host_blocks[idx].clone())
            .collect();

        // For offload: compute checksums before writing to remote
        if !is_onboard && g4_checksum_enabled() {
            self.log_checksums_for_offload(&bounce_blocks, &pipeline, &request.request_id);
        }

        // Execute the remote <-> host transfer
        pipeline
            .execute(&bounce_blocks, remote_ctx.as_ref(), &self.cancel_token)
            .await
            .map_err(|e| anyhow::anyhow!("Remote transfer failed: {}", e))?;

        // For onboard: compute checksums after reading from remote
        if is_onboard && g4_checksum_enabled() {
            self.log_checksums_for_onboard(&bounce_blocks, &pipeline, &request.request_id);
        }

        // If this is a full pipeline with device blocks, execute host <-> device transfer
        if let Some(device_ids) = pipeline.device_block_ids() {
            if !device_ids.is_empty() {
                let block_pairs: Vec<(usize, usize)> = if is_onboard {
                    // Onboard: host -> device
                    bounce_ids
                        .iter()
                        .copied()
                        .zip(device_ids.iter().copied())
                        .collect()
                } else {
                    // Offload: device -> host (already done before remote transfer)
                    // This case is typically handled separately
                    vec![]
                };

                if !block_pairs.is_empty() {
                    let local_request = BlockTransferRequest {
                        from_pool: if is_onboard { Host } else { Device },
                        to_pool: if is_onboard { Device } else { Host },
                        blocks: block_pairs,
                        connector_req: None,
                        sequence_hashes: None,
                    };
                    self.execute_transfer(local_request).await?;
                }
            }
        }

        tracing::debug!(
            target: "kvbm-g4",
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks,
            "remote transfer completed"
        );

        Ok(())
    }

    /// Log checksums for blocks being offloaded to remote storage.
    fn log_checksums_for_offload(
        &self,
        bounce_blocks: &[LocalBlockData<PinnedStorage>],
        pipeline: &RemoteTransferPipeline,
        request_id: &str,
    ) {
        use crate::block_manager::block::data::BlockDataViews;

        let descriptors = pipeline.descriptors();
        for (block, desc) in bounce_blocks.iter().zip(descriptors.iter()) {
            let remote_key = format!("{}/{}", desc.key().location(), desc.key().key_str());
            let seq_hash = desc.sequence_hash().unwrap_or(0);

            // Get block view and compute checksum from raw pointer
            match block.local_block_view() {
                Ok(view) => {
                    let checksum = unsafe {
                        let ptr = view.as_ptr();
                        let size = view.size();
                        let slice = std::slice::from_raw_parts(ptr, size);
                        compute_checksum(slice)
                    };

                    tracing::info!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        remote_key,
                        checksum,
                        phase = "pre-offload",
                        "checksum computed before remote write"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        error = %e,
                        "failed to get block view for checksum"
                    );
                }
            }
        }
    }

    /// Log checksums for blocks being onboarded from remote storage.
    fn log_checksums_for_onboard(
        &self,
        bounce_blocks: &[LocalBlockData<PinnedStorage>],
        pipeline: &RemoteTransferPipeline,
        request_id: &str,
    ) {
        use crate::block_manager::block::data::BlockDataViews;

        let descriptors = pipeline.descriptors();
        for (block, desc) in bounce_blocks.iter().zip(descriptors.iter()) {
            let remote_key = format!("{}/{}", desc.key().location(), desc.key().key_str());
            let seq_hash = desc.sequence_hash().unwrap_or(0);

            // Get block view and compute checksum from raw pointer
            match block.local_block_view() {
                Ok(view) => {
                    let checksum = unsafe {
                        let ptr = view.as_ptr();
                        let size = view.size();
                        let slice = std::slice::from_raw_parts(ptr, size);
                        compute_checksum(slice)
                    };

                    tracing::info!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        remote_key,
                        checksum,
                        phase = "post-onboard",
                        "checksum computed after remote read"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "kvbm-g4",
                        request_id,
                        block_idx = seq_hash,
                        error = %e,
                        "failed to get block view for checksum"
                    );
                }
            }
        }
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        // Try to parse as RemoteTransferRequest first, then fall back to BlockTransferRequest
        let result = if let Ok(remote_request) =
            serde_json::from_slice::<RemoteTransferRequest>(&message.data[0])
        {
            // Handle remote transfer (G4 object storage)
            let operation_id = remote_request.operation_id;

            tracing::debug!(
                target: "kvbm-g4",
                request_id = %remote_request.request_id,
                operation_id = %operation_id,
                num_blocks = remote_request.num_blocks(),
                is_onboard = remote_request.is_onboard(),
                "handling remote transfer request"
            );

            if let Some(connector_req) = remote_request.connector_req.clone() {
                let client = self
                    .scheduler_client
                    .as_ref()
                    .expect("scheduler client is required")
                    .clone();

                let handle = client.schedule_transfer(connector_req).await?;
                assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

                match self.execute_remote_transfer(remote_request).await {
                    Ok(_) => {
                        handle.mark_complete(Ok(())).await;
                        Ok(())
                    }
                    Err(e) => {
                        handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await;
                        Err(e)
                    }
                }
            } else {
                self.execute_remote_transfer(remote_request).await
            }
        } else {
            // Handle local block transfer
            let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

            if let Some(req) = request.connector_req.take() {
                let operation_id = req.uuid;

                tracing::debug!(
                    request_id = %req.request_id,
                    operation_id = %operation_id,
                    "scheduling transfer"
                );

                let client = self
                    .scheduler_client
                    .as_ref()
                    .expect("scheduler client is required")
                    .clone();

                let handle = client.schedule_transfer(req).await?;

                // we don't support cancellation yet
                assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

                match self.execute_transfer(request).await {
                    Ok(_) => {
                        handle.mark_complete(Ok(())).await;
                        Ok(())
                    }
                    Err(e) => {
                        handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await;
                        Err(e)
                    }
                }
            } else {
                self.execute_transfer(request).await
            }
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
