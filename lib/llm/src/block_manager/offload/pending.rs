// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Transfer Managers
//!
//! Transfer managers are responsible for multiple things:
//! - Before the transfer:
//!     - Rate-limiting the number of transfers that can be initiated concurrently. This is implemented through bounded channels.
//!         - Due to the nature of the [`super::OffloadManager`], we only apply this rate-limiting to offloads.
//! - During the transfer:
//!     - Initiating the transfer
//!     - Holding strong references to blocks being transfered.
//! - After the transfer:
//!     - Dropping these references once the transfer is complete.
//!     - Registering the blocks with the target pool.
//!     - Returning the registered blocks to the caller.
//!
//! This is implemented through the [`TransferManager`] trait, which takes a single [`PendingTransfer`]
//! and initiates the transfer.
//!
//! Since CUDA and NIXL transfers use completely different semantics, we implement two separate transfer managers.
//!
//! ## Workflow
//! 1. A transfer request is made by calling [`TransferManager::begin_transfer`]
//! 2. [`TransferManager::begin_transfer`] performs the transfer, and enqueues relevant data into a bounded channel.
//! 3. A worker thread (consuming this bounded channel and enforcing rate limiting) awaits the incoming transfers.
//! 4. After a transfer is complete, the worker thread registers the blocks with the target pool, and returns the registered blocks to the caller.

use std::pin::Pin;
use std::sync::Arc;
use std::thread::spawn;
use tokio::sync::mpsc;

use crate::block_manager::block::{
    transfer::{WriteTo, WriteToStrategy},
    BlockError, BlockExt, BlockMetadata, BlockState, MutableBlock, ReadableBlock, WritableBlock,
};
use crate::block_manager::pool::BlockPoolError;
use crate::block_manager::state::TransferContext;
use crate::block_manager::storage::{Local, Storage};
use crate::block_manager::BlockPool;

use anyhow::Result;
use async_trait::async_trait;
use cudarc::driver::{sys::CUevent_flags, CudaEvent};
use futures::{future::join_all, stream::FuturesUnordered, StreamExt};

use super::BlockResult;

/// Manage a set of pending transfers.
pub struct PendingTransfer<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    /// The block being copied from.
    sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
    /// The block being copied to.
    targets: Vec<MutableBlock<Target, Metadata>>,
    /// The oneshot sender that optionally returns the registered blocks once the transfer is complete.
    completion_indicator: Option<oneshot::Sender<BlockResult<Target, Metadata>>>,
    /// The target pool that will receive the registered block.
    target_registration_pool: Arc<Option<BlockPool<Target, Metadata>>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingTransfer<Source, Target, Metadata>
{
    pub fn new(
        sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
        targets: Vec<MutableBlock<Target, Metadata>>,
        completion_indicator: Option<oneshot::Sender<BlockResult<Target, Metadata>>>,
        target_registration_pool: Arc<Option<BlockPool<Target, Metadata>>>,
    ) -> Self {
        Self {
            sources,
            targets,
            completion_indicator,
            target_registration_pool,
        }
    }

    fn handle_complete(self) -> Result<()> {
        let Self {
            targets,
            target_registration_pool,
            completion_indicator,
            ..
        } = self;

        if let Some(target_registration_pool) = target_registration_pool.as_ref() {
            let blocks = target_registration_pool.register_blocks_blocking(targets)?;

            if let Some(completion_indicator) = completion_indicator {
                completion_indicator.send(Ok(blocks))?;
            }
        }

        Ok(())
    }
}

fn transfer_metadata<Source: Storage, Target: Storage, Metadata: BlockMetadata>(
    source: &Arc<MutableBlock<Source, Metadata>>,
    target: &mut MutableBlock<Target, Metadata>,
) -> Result<()> {
    // Only registered blocks can be transferred. There are upstream checks for this, so this shouldn't ever fail.
    if let BlockState::Registered(reg_handle) = source.state() {
        // Bring the block back to the 'Reset' state.
        target.reset();
        // Transfer metadata.
        target.update_metadata(source.metadata().clone());
        // Copy tokens
        target.apply_token_block(reg_handle.token_block().clone())?;
    } else {
        Err(BlockPoolError::BlockError(BlockError::InvalidState(
            "Block is not registered.".to_string(),
        )))?;
    }

    Ok(())
}

#[async_trait]
pub trait TransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata>:
    Send + Sync
{
    /// Begin a transfer. Blocks if the pending queue is full.
    async fn begin_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()>;
}

pub struct CudaTransferManager<Source: Storage, Target: Storage, Metadata: BlockMetadata> {
    pending_transfer_q: mpsc::Sender<(PendingTransfer<Source, Target, Metadata>, CudaEvent)>,
    transfer_ctx: Arc<TransferContext>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    CudaTransferManager<Source, Target, Metadata>
{
    pub fn new(transfer_ctx: Arc<TransferContext>, max_depth: usize) -> Self {
        let (tx, mut rx) =
            mpsc::channel::<(PendingTransfer<Source, Target, Metadata>, CudaEvent)>(max_depth);

        spawn(move || {
            while let Some((pending_transfer, event)) = rx.blocking_recv() {
                // Wait for the event.
                event.synchronize()?;
                // Only finalize the transfer after the event is signaled.
                pending_transfer.handle_complete()?;
            }
            Ok::<(), anyhow::Error>(())
        });

        Self {
            pending_transfer_q: tx,
            transfer_ctx,
        }
    }
}

#[async_trait]
impl<Source, Target, Metadata> TransferManager<Source, Target, Metadata>
    for CudaTransferManager<Source, Target, Metadata>
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
    async fn begin_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        for (source, target) in pending_transfer
            .sources
            .iter()
            .zip(pending_transfer.targets.iter_mut())
        {
            transfer_metadata(source, target)?;
            source.write_to(target, None, self.transfer_ctx.clone())?;
        }

        // Use a cuda event to record the completion of the transfers.
        let event = self
            .transfer_ctx
            .stream()
            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

        // Send the pending transfer and event to the worker thread.
        // If the queue is full, we block the worker until space becomes available.
        self.pending_transfer_q
            .send((pending_transfer, event))
            .await?;

        Ok(())
    }
}

pub struct DiskTransferManager {
    futures_tx: mpsc::Sender<Pin<Box<dyn std::future::Future<Output = ()> + Send + Sync>>>,
    transfer_ctx: Arc<TransferContext>,
}

impl DiskTransferManager {
    pub fn new(transfer_ctx: Arc<TransferContext>, max_size: usize) -> Self {
        let (futures_tx, mut futures_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            // Keep track of our pending transfers.
            // Consume the futures as they complete, while also receiving new ones.

            let mut pending_transfers = FuturesUnordered::new();
            loop {
                tokio::select! {
                    Some(future) = futures_rx.recv() => {
                        // If we're at max size, block the worker thread on the next() call until we have capacity.
                        while pending_transfers.len() >= max_size {
                            pending_transfers.next().await;
                        }
                        // Once we have capacity, push the new future onto the queue.
                        pending_transfers.push(future);
                    }
                    Some(_) = pending_transfers.next(), if !pending_transfers.is_empty() => {
                        // A transfer completed, just continue to process more
                    }
                    else => {
                        // Both branches are pending, wait for one to become ready
                        tokio::task::yield_now().await;
                    }
                }
            }
        });

        Self {
            futures_tx,
            transfer_ctx,
        }
    }
}

#[async_trait]
impl<Source, Target, Metadata> TransferManager<Source, Target, Metadata> for DiskTransferManager
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
    async fn begin_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        let futures = pending_transfer
            .sources
            .iter()
            .zip(pending_transfer.targets.iter_mut())
            .map(|(source, target)| {
                transfer_metadata(source, target).unwrap();
                // Initiate the transfer, and get a future indicating completion.
                source
                    .nixl_write_to(target, None, self.transfer_ctx.clone())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let completion_future = async move {
            let _ = join_all(futures).await;
            pending_transfer.handle_complete().unwrap();
        };

        // Futures_(tx/rx) has a capacity of 1. If the queue worker has received another future and is awaiting next() due to a full `FuturesUnordered`,
        // this call will block until the worker has processed the prior future.
        self.futures_tx.send(Box::pin(completion_future)).await?;

        Ok(())
    }
}
