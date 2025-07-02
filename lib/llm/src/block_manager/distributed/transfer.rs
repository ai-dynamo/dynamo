// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    block::{
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
        Block, BlockDataProvider, ReadableBlock, WritableBlock,
    },
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    BasicMetadata, BlockMetadata, Storage,
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};
use tokio::sync::Mutex;

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A manager for a pool of blocks.
/// This performs two functions:
/// - It provides a way to get blocks from the pool.
/// - It returns blocks to the pool after their transfer is complete.
// TODO: This seems like a bit of an ugly workaround. Surely there's a better way to do this.
struct BlockTransferPoolManager<S: Storage> {
    blocks: Arc<Mutex<LocalBlockDataList<S>>>,
}

impl<S: Storage> BlockTransferPoolManager<S> {
    fn new<M: BlockMetadata>(blocks: Vec<LocalBlock<S, M>>) -> Result<Self> {
        let blocks = blocks
            .into_iter()
            .map(|b| {
                let block_data = b.block_data() as &dyn Any;

                block_data
                    .downcast_ref::<LocalBlockData<S>>()
                    .unwrap()
                    .clone()
            })
            .collect();
        let blocks = Arc::new(Mutex::new(blocks));

        Ok(Self { blocks })
    }

    /// Get a set of blocks from the pool.
    async fn get_blocks(&self, block_idxs: impl Iterator<Item = usize>) -> Vec<LocalBlockData<S>> {
        let blocks_handle = self.blocks.lock().await;

        block_idxs
            .map(|idx| {
                // This shouldn't ever fail. If it does, it indicates a logic error on the leader.
                // TODO: This seems a bit fragile.
                blocks_handle[idx].clone()
            })
            .collect()
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
pub struct BlockTransferHandler {
    device: Option<BlockTransferPoolManager<DeviceStorage>>,
    host: Option<BlockTransferPoolManager<PinnedStorage>>,
    disk: Option<BlockTransferPoolManager<DiskStorage>>,
    context: Arc<TransferContext>,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
    ) -> Result<Self> {
        Ok(Self {
            device: device_blocks.map(|blocks| BlockTransferPoolManager::new(blocks).unwrap()),
            host: host_blocks.map(|blocks| BlockTransferPoolManager::new(blocks).unwrap()),
            disk: disk_blocks.map(|blocks| BlockTransferPoolManager::new(blocks).unwrap()),
            context,
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_manager: &Option<BlockTransferPoolManager<Source>>,
        target_pool_manager: &Option<BlockTransferPoolManager<Target>>,
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
    {
        let Some(source_pool_manager) = source_pool_manager else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_manager) = target_pool_manager else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources = source_pool_manager.get_blocks(source_idxs).await;
        let mut targets = target_pool_manager.get_blocks(target_idxs).await;

        // Perform the transfer, and return the notifying channel.
        let channel = match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        };

        channel
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

        let request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        message.ack().await?;

        Ok(())
    }
}
