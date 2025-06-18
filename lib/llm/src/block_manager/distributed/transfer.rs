// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    block::{
        transfer::{TransferContext, WriteTo, WriteToStrategy},
        Block, BlockIdentifier, ReadableBlock, WritableBlock,
    },
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    BasicMetadata, BlockMetadata, Storage,
};

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

type BlockList<S, M> = Vec<Option<Block<S, M>>>;

/// A list of blocks that are being transferred.
/// Ensures that blocks are returned to the pool after their transfer is complete.
struct TransferList<S: Storage, M: BlockMetadata> {
    blocks: Arc<Mutex<BlockList<S, M>>>,
    list: Option<Vec<Block<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> TransferList<S, M> {
    fn new(blocks: Arc<Mutex<BlockList<S, M>>>, list: Vec<Block<S, M>>) -> Self {
        Self {
            blocks,
            list: Some(list),
        }
    }

    fn get(&self) -> &Vec<Block<S, M>> {
        self.list.as_ref().unwrap()
    }

    fn get_mut(&mut self) -> &mut Vec<Block<S, M>> {
        self.list.as_mut().unwrap()
    }

    async fn return_blocks(mut self) -> Result<()> {
        let list = self.list.take().unwrap();
        let mut blocks_handle = self.blocks.lock().await;
        for block in list {
            let id = block.block_id();
            if blocks_handle[id].is_some() {
                return Err(anyhow::anyhow!("Block already returned"));
            }
            blocks_handle[id] = Some(block);
        }

        Ok(())
    }
}

impl<S: Storage, M: BlockMetadata> Drop for TransferList<S, M> {
    fn drop(&mut self) {
        if self.list.is_some() {
            panic!("TransferList not returned!");
        }
    }
}

/// A manager for a pool of blocks.
/// This performs two functions:
/// - It provides a way to get blocks from the pool.
/// - It returns blocks to the pool after their transfer is complete.
// TODO: This seems like a bit of an ugly workaround. Surely there's a better way to do this.
struct BlockTransferPoolManager<S: Storage, M: BlockMetadata> {
    blocks: Arc<Mutex<BlockList<S, M>>>,
}

impl<S: Storage, M: BlockMetadata> BlockTransferPoolManager<S, M> {
    fn new(blocks: Vec<Block<S, M>>) -> Result<Self> {
        let blocks = blocks.into_iter().map(Some).collect();
        let blocks = Arc::new(Mutex::new(blocks));

        Ok(Self { blocks })
    }

    /// Get a set of blocks from the pool.
    async fn get_blocks(&self, block_idxs: impl Iterator<Item = usize>) -> TransferList<S, M> {
        let mut blocks_handle = self.blocks.lock().await;

        let mut list = Vec::new();
        for idx in block_idxs {
            // This shouldn't ever fail. If it does, it indicates a logic error on the leader.
            // TODO: This seems a bit fragile.
            list.push(blocks_handle[idx].take().unwrap());
        }
        TransferList::new(self.blocks.clone(), list)
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
pub struct BlockTransferHandler {
    device: Option<BlockTransferPoolManager<DeviceStorage, BasicMetadata>>,
    host: Option<BlockTransferPoolManager<PinnedStorage, BasicMetadata>>,
    disk: Option<BlockTransferPoolManager<DiskStorage, BasicMetadata>>,
    context: Arc<TransferContext>,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<Block<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<Block<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<Block<DiskStorage, BasicMetadata>>>,
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
    async fn begin_transfer<Source, Target, Metadata>(
        &self,
        source_pool_manager: &Option<BlockTransferPoolManager<Source, Metadata>>,
        target_pool_manager: &Option<BlockTransferPoolManager<Target, Metadata>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage,
        Target: Storage,
        Metadata: BlockMetadata,
        // Check that the source block is readable, local, and writable to the target block.
        Block<Source, Metadata>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<Block<Target, Metadata>>,
        // Check that the target block is writable.
        Block<Target, Metadata>: WritableBlock<StorageType = Target>,
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
        let channel = match sources
            .get()
            .write_to(targets.get_mut(), true, self.context.clone())
        {
            Ok(Some(channel)) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
            Ok(None) => {
                panic!("Failed to write blocks. No channel returned. This should never happen.")
            }
        };

        sources.return_blocks().await?;
        targets.return_blocks().await?;

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
