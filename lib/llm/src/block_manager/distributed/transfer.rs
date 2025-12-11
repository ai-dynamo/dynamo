// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use super::transfer_object::ObjectTransferHandler;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    Storage,
    block::{
        BasicMetadata, Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock,
        WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    config::ObjectStorageConfig,
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    v2::logical::distributed::traits::DistributedRegistry,
    v2::logical::external_registry::SequenceHashRegistry,
    v2::physical::{
        layout::{LayoutConfig, PhysicalLayout},
        manager::TransportManager,
        transfer::{LayoutHandle, NixlAgent},
        transfer::options::TransferOptions,
    },
};

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

    pub async fn execute_batched_transfer<T: BlockTransferDirectHandler>(
        &self,
        handler: &T,
        request: BlockTransferRequest,
    ) -> Result<()> {
        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        // When batching, we need to slice sequence_hashes to match each batch
        // sequence_hashes[i] corresponds to blocks[i]
        let hashes = request.sequence_hashes.as_ref();

        let batch_futures: Vec<_> = blocks
            .chunks(self.max_batch_size)
            .enumerate()
            .map(|(batch_idx, batch)| {
                let start_idx = batch_idx * self.max_batch_size;
                let end_idx = start_idx + batch.len();

                // Slice hashes to match this batch (if present)
                let batch_hashes = hashes.map(|h| {
                    h.get(start_idx..end_idx)
                        .map(|slice| slice.to_vec())
                        .unwrap_or_default()
                }).filter(|h| !h.is_empty());

                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                    sequence_hashes: batch_hashes,
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

#[async_trait]
pub trait BlockTransferHandler: Send + Sync {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()>;

    fn scheduler_client(&self) -> Option<TransferSchedulerClient>;

    /// Enable downcasting to concrete handler types (e.g., for G4 integration).
    fn as_any(&self) -> &dyn Any;
}

#[async_trait]
pub trait BlockTransferDirectHandler {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()>;
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
#[derive(Clone)]
pub struct BlockTransferHandlerV1 {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    // add worker-connector scheduler client here
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV1 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV1 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
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
}

impl BlockTransferHandlerV1 {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        // add worker-connector scheduler client here
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
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
}

#[derive(Clone)]
pub struct BlockTransferHandlerV2 {
    device_handle: Option<LayoutHandle>,
    host_handle: Option<LayoutHandle>,
    disk_handle: Option<LayoutHandle>,
    transport_manager: TransportManager,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,

    // G4 resources (for creating ObjectTransferHandler)
    layout_config: Option<LayoutConfig>,
    nixl_agent: Option<NixlAgent>,

    /// Optional G4 object handler for write-through offloads.
    /// Set via `set_object_handler()` after initialization.
    /// Uses OnceLock for interior mutability (can set after Arc wrapping).
    object_handler: std::sync::OnceLock<Arc<ObjectTransferHandler>>,
}

impl BlockTransferHandlerV2 {
    pub fn new(
        device_layout: Option<PhysicalLayout>,
        host_layout: Option<PhysicalLayout>,
        disk_layout: Option<PhysicalLayout>,
        transport_manager: TransportManager,
        scheduler_client: Option<TransferSchedulerClient>,
        layout_config: Option<LayoutConfig>,
        nixl_agent: Option<NixlAgent>,
    ) -> Result<Self> {
        Ok(Self {
            device_handle: device_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            host_handle: host_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            disk_handle: disk_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            transport_manager,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            layout_config,
            nixl_agent,
            object_handler: std::sync::OnceLock::new(),
        })
    }

    /// Set the object handler for G4 write-through.
    ///
    /// Must be called after handler creation if G4 is enabled.
    /// Uses OnceLock so this can be called on &self (after Arc wrapping).
    pub fn set_object_handler(&self, handler: Arc<ObjectTransferHandler>) {
        let _ = self.object_handler.set(handler);
    }

    /// Get a reference to the object handler (if set).
    pub fn object_handler(&self) -> Option<&Arc<ObjectTransferHandler>> {
        self.object_handler.get()
    }

    /// Get the device layout handle.
    pub fn device_handle(&self) -> Option<LayoutHandle> {
        self.device_handle
    }

    /// Get the host layout handle (used for bounce buffers in G4 transfers).
    pub fn host_handle(&self) -> Option<LayoutHandle> {
        self.host_handle
    }

    /// Get a reference to the transport manager.
    pub fn transport_manager(&self) -> &TransportManager {
        &self.transport_manager
    }

    /// Get a reference to the layout config (if available).
    pub fn layout_config(&self) -> Option<&LayoutConfig> {
        self.layout_config.as_ref()
    }

    /// Get a reference to the NIXL agent (if available).
    pub fn nixl_agent(&self) -> Option<&NixlAgent> {
        self.nixl_agent.as_ref()
    }

    /// Create an ObjectTransferHandler using this handler's resources.
    ///
    /// Returns None if:
    /// - Host layout is not configured (required for bounce buffers)
    /// - Layout config or agent is not available (V1 mode)
    pub fn create_object_handler(
        &self,
        config: ObjectStorageConfig,
        local_registry: SequenceHashRegistry,
        distributed_registry: Option<std::sync::Arc<dyn DistributedRegistry>>,
    ) -> Option<ObjectTransferHandler> {
        let host_handle = self.host_handle?;
        let layout_config = self.layout_config.clone()?;
        let agent = self.nixl_agent.clone()?;

        ObjectTransferHandler::new(
            self.transport_manager.clone(),
            local_registry,
            distributed_registry,
            config,
            host_handle,
            layout_config,
            agent,
        )
        .ok()
    }
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV2 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV2 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        let is_device_to_host = matches!(
            (request.from_pool(), request.to_pool()),
            (Device, Host)
        );

        let (src, dst) = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => (self.device_handle.as_ref(), self.host_handle.as_ref()),
            (Device, Disk) => (self.device_handle.as_ref(), self.disk_handle.as_ref()),
            (Host, Device) => (self.host_handle.as_ref(), self.device_handle.as_ref()),
            (Host, Disk) => (self.host_handle.as_ref(), self.disk_handle.as_ref()),
            (Disk, Device) => (self.disk_handle.as_ref(), self.device_handle.as_ref()),
            _ => return Err(anyhow::anyhow!("Invalid transfer type.")),
        };

        if let (Some(src), Some(dst)) = (src, dst) {
            let src_block_ids = request
                .blocks()
                .iter()
                .map(|(from, _)| *from)
                .collect::<Vec<_>>();
            let dst_block_ids = request
                .blocks()
                .iter()
                .map(|(_, to)| *to)
                .collect::<Vec<_>>();

            self.transport_manager
                .execute_transfer(
                    *src,
                    &src_block_ids,
                    *dst,
                    &dst_block_ids,
                    TransferOptions::default(),
                )?
                .await?;

            // G4 write-through: After D→H transfer completes, also offload to object storage
            if is_device_to_host {
                if let (Some(obj_handler), Some(hashes)) =
                    (self.object_handler.get(), &request.sequence_hashes)
                {
                    if !hashes.is_empty() {
                        let host_handle = self.host_handle
                            .ok_or_else(|| anyhow::anyhow!("Host handle required for G4 write-through"))?;

                        match obj_handler.offload(host_handle, &dst_block_ids, hashes).await {
                            Ok(count) => {
                                tracing::debug!(
                                    "G4 write-through: offloaded {} of {} blocks to object storage",
                                    count,
                                    dst_block_ids.len()
                                );
                            }
                            Err(e) => {
                                // G4 offload failure is non-fatal - log and continue
                                tracing::warn!(
                                    "G4 write-through failed (non-fatal): {}",
                                    e
                                );
                            }
                        }
                    }
                }
            }
        } else {
            return Err(anyhow::anyhow!("Invalid transfer type."));
        }

        Ok(())
    }
}

#[async_trait]
impl<T: ?Sized + BlockTransferHandler> Handler for T {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
            let operation_id = req.uuid;

            tracing::debug!(
                request_id = %req.request_id,
                operation_id = %operation_id,
                "scheduling transfer"
            );

            let client = self
                .scheduler_client()
                .expect("scheduler client is required");

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
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
