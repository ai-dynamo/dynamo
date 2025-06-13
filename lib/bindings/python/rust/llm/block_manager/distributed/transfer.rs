// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use utils::*;
use zmq::*;

use BlockTransferPool::*;

use llm_rs::block_manager::{
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
use tokio_util::sync::CancellationToken;

struct BlockTransferPoolManager<S: Storage, M: BlockMetadata> {
    blocks: Arc<Mutex<Vec<Option<Block<S, M>>>>>,
    return_sender: tokio::sync::mpsc::UnboundedSender<Block<S, M>>,
}

impl<S: Storage, M: BlockMetadata> BlockTransferPoolManager<S, M> {
    fn new(blocks: Vec<Block<S, M>>, cancel_token: CancellationToken) -> Result<Self> {
        let (return_tx, return_rx) = tokio::sync::mpsc::unbounded_channel();

        let blocks = blocks.into_iter().map(|b| Some(b)).collect();
        let blocks = Arc::new(Mutex::new(blocks));

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

    async fn return_worker(
        blocks: Arc<Mutex<Vec<Option<Block<S, M>>>>>,
        mut receiver: tokio::sync::mpsc::UnboundedReceiver<Block<S, M>>,
        cancel_token: CancellationToken,
    ) -> Result<()> {
        loop {
            tokio::select! {
                Some(block) = receiver.recv() => {
                    let mut blocks_handle = blocks.lock().await;
                    let id = block.block_id();
                    blocks_handle[id as usize] = Some(block);
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }

    async fn get_blocks(&self, block_idxs: impl Iterator<Item = usize>) -> Vec<MutableBlock<S, M>> {
        let mut blocks_handle = self.blocks.lock().await;

        let mut blocks = Vec::new();
        for idx in block_idxs {
            // This shouldn't ever fail. If it does, it indicates a logic error on the leader.
            blocks.push(MutableBlock::new(
                blocks_handle[idx].take().unwrap(),
                self.return_sender.clone(),
            ));
        }
        blocks
    }
}

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

    async fn begin_transfer<Source: Storage, Target: Storage, Metadata: BlockMetadata>(
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

        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        let sources = source_pool_manager.get_blocks(source_idxs).await;
        let mut targets = target_pool_manager.get_blocks(target_idxs).await;

        let sources = sources.into_iter().map(|b| Arc::new(b)).collect::<Vec<_>>();

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
