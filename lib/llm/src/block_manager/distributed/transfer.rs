// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A named group of (device, host, disk) block pools.
/// The primary KV cache is group 0; additional caches (e.g. indexer k cache for DSA) are group 1+.
/// All groups share the same `num_blocks` and `page_size` but may differ in `num_layers`,
/// `outer_dim`, `inner_dim`, and `dtype`.
#[derive(Clone)]
pub struct CacheGroup {
    pub name: String,
    pub device: Option<LocalBlockDataList<DeviceStorage>>,
    pub host: Option<LocalBlockDataList<PinnedStorage>>,
    pub disk: Option<LocalBlockDataList<DiskStorage>>,
}

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
///
/// Holds one or more [`CacheGroup`]s. The primary KV cache is always group 0.
/// Additional caches (e.g. DSA indexer k cache) are group 1+.
/// Transfer operations iterate all groups for the same block index mapping,
/// guaranteeing lockstep movement.
#[derive(Clone)]
pub struct BlockTransferHandler {
    cache_groups: Vec<CacheGroup>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    // add worker-connector scheduler client here
}

impl BlockTransferHandler {
    /// Backwards-compatible constructor that creates a single primary cache group.
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        // add worker-connector scheduler client here
    ) -> Result<Self> {
        let primary = CacheGroup {
            name: "primary".to_string(),
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
        };
        Self::new_multi(vec![primary], context, scheduler_client)
    }

    /// Create a handler with multiple cache groups. Group 0 is the primary KV cache.
    pub fn new_multi(
        cache_groups: Vec<CacheGroup>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
    ) -> Result<Self> {
        assert!(!cache_groups.is_empty(), "At least one cache group required");

        Ok(Self {
            cache_groups,
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
        })
    }

    /// Returns a reference to the primary (first) cache group.
    pub fn primary(&self) -> &CacheGroup {
        &self.cache_groups[0]
    }

    /// Returns the number of cache groups.
    pub fn num_cache_groups(&self) -> usize {
        self.cache_groups.len()
    }

    pub fn get_local_data<S: Storage>(
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
    /// Iterates all cache groups for the same block index mapping to ensure lockstep.
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?} across {} cache group(s)",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool(),
            self.cache_groups.len()
        );

        tracing::debug!("request: {request:#?}");

        // Fire begin_transfer for all cache groups, then await all notifications
        let mut notifies = Vec::with_capacity(self.cache_groups.len());

        for group in &self.cache_groups {
            let notify = match (request.from_pool(), request.to_pool()) {
                (Device, Host) => {
                    self.begin_transfer(&group.device, &group.host, request.clone())
                        .await
                }
                (Device, Disk) => {
                    self.begin_transfer(&group.device, &group.disk, request.clone())
                        .await
                }
                (Host, Device) => {
                    self.begin_transfer(&group.host, &group.device, request.clone())
                        .await
                }
                (Host, Disk) => {
                    self.begin_transfer(&group.host, &group.disk, request.clone())
                        .await
                }
                (Disk, Device) => {
                    self.begin_transfer(&group.disk, &group.device, request.clone())
                        .await
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid transfer type."));
                }
            }?;
            notifies.push(notify);
        }

        for notify in notifies {
            notify.await?;
        }
        Ok(())
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

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
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
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
