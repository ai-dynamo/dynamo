// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    block::{
        transfer::{TransferContext, WriteTo, WriteToStrategy},
        Block, BlockIdentifier, MutableBlock, ReadableBlock, WritableBlock,
    },
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    BasicMetadata, BlockMetadata, Storage,
};

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

type BlockList<S, M> = Vec<Option<Block<S, M>>>;

/// A manager for a pool of blocks.
/// This performs two functions:
/// - It provides a way to get blocks from the pool.
/// - It returns blocks to the pool after their transfer is complete.
// TODO: This seems like a bit of an ugly workaround. Surely there's a better way to do this.
struct BlockTransferPoolManager<S: Storage, M: BlockMetadata> {
    blocks: Arc<Mutex<BlockList<S, M>>>,
    return_sender: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
}

impl<S: Storage, M: BlockMetadata> BlockTransferPoolManager<S, M> {
    fn new(blocks: Vec<Block<S, M>>, cancel_token: CancellationToken) -> Result<Self> {
        // Create our return channel.
        let (return_tx, return_rx) = tokio::sync::mpsc::unbounded_channel();

        let blocks = blocks.into_iter().map(Some).collect();
        let blocks = Arc::new(Mutex::new(blocks));

        // Kick off the task that returns blocks to the pool.
        let blocks_clone = blocks.clone();
        CriticalTaskExecutionHandle::new(
            |cancel_token| Self::return_worker(blocks_clone, return_rx, cancel_token),
            cancel_token,
            "Block transfer pool manager return worker",
        )?
        .detach();

        Ok(Self {
            blocks,
            return_sender: return_tx,
        })
    }

    /// A task that returns blocks to the pool.
    async fn return_worker(
        blocks: Arc<Mutex<BlockList<S, M>>>,
        mut receiver: tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(block) = receiver.recv() => {
                    let mut blocks_handle = blocks.lock().await;
                    let id = block.block_id();
                    blocks_handle[id] = Some(block);
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }

    /// Get a set of blocks from the pool.
    async fn get_blocks(&self, block_idxs: impl Iterator<Item = usize>) -> Vec<MutableBlock<S, M>> {
        let mut blocks_handle = self.blocks.lock().await;

        let mut blocks = Vec::new();
        for idx in block_idxs {
            // This shouldn't ever fail. If it does, it indicates a logic error on the leader.
            // TODO: This seems a bit fragile.
            blocks.push(MutableBlock::new(
                blocks_handle[idx].take().unwrap(),
                self.return_sender.clone(),
            ));
        }
        blocks
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
        cancel_token: CancellationToken,
    ) -> Result<Self> {
        Ok(Self {
            device: device_blocks
                .map(|blocks| BlockTransferPoolManager::new(blocks, cancel_token.clone()).unwrap()),
            host: host_blocks
                .map(|blocks| BlockTransferPoolManager::new(blocks, cancel_token.clone()).unwrap()),
            disk: disk_blocks
                .map(|blocks| BlockTransferPoolManager::new(blocks, cancel_token.clone()).unwrap()),
            context,
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target, Metadata>(
        &self,
        source_pool_manager: &Option<BlockTransferPoolManager<Source, Metadata>>,
        target_pool_manager: &Option<BlockTransferPoolManager<Target, Metadata>>,
        request: BlockTransferRequest,
    ) -> anyhow::Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage,
        Target: Storage,
        Metadata: BlockMetadata,
        // Check that the source block is readable, local, and writable to the target block.
        MutableBlock<Source, Metadata>: ReadableBlock<StorageType = Source>
            + Local
            + WriteToStrategy<MutableBlock<Target, Metadata>>,
        // Check that the target block is writable.
        MutableBlock<Target, Metadata>: WritableBlock<StorageType = Target>,
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

        // For now, write_to is only implemented for Vec<Arc<ReadableBlock>>.
        // Because of this, we need to wrap our source blocks in Arcs.
        // TODO: This shouldn't be necessary.
        let sources = sources.into_iter().map(Arc::new).collect::<Vec<_>>();

        // Perform the transfer, and return the notifying channel.
        let channel = sources
            .write_to(&mut targets, true, self.context.clone())?
            .unwrap();

        Ok(channel)
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
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
