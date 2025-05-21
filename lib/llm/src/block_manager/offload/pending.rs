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
//! 1. A transfer request is made by calling [`TransferManager::enqueue_transfer`]
//! 2. [`TransferManager::enqueue_transfer`] performs the transfer, and enqueues relevant data into a bounded channel.
//! 3. A worker thread (consuming this bounded channel and enforcing rate limiting) awaits the incoming transfers.
//! 4. After a transfer is complete, the worker thread registers the blocks with the target pool, and returns the registered blocks to the caller.

use std::marker::PhantomData;
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
use futures::{stream::FuturesUnordered, StreamExt};

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
    target_pool: Arc<BlockPool<Target, Metadata>>,
}

impl<Source: Storage, Target: Storage, Metadata: BlockMetadata>
    PendingTransfer<Source, Target, Metadata>
{
    pub fn new(
        sources: Vec<Arc<MutableBlock<Source, Metadata>>>,
        targets: Vec<MutableBlock<Target, Metadata>>,
        completion_indicator: Option<oneshot::Sender<BlockResult<Target, Metadata>>>,
        target_pool: Arc<BlockPool<Target, Metadata>>,
    ) -> Self {
        assert_eq!(sources.len(), targets.len());
        Self {
            sources,
            targets,
            completion_indicator,
            target_pool,
        }
    }

    fn handle_complete(self) -> Result<()> {
        let Self {
            sources,
            mut targets,
            target_pool,
            completion_indicator,
            ..
        } = self;

        for (source, target) in sources.iter().zip(targets.iter_mut()) {
            transfer_metadata(source, target)?;
        }

        let blocks = target_pool.register_blocks_blocking(targets)?;

        if let Some(completion_indicator) = completion_indicator {
            completion_indicator.send(Ok(blocks))?;
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
    async fn enqueue_transfer(
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
    pub fn new(transfer_ctx: Arc<TransferContext>, max_concurrent_transfers: usize) -> Self {
        let (tx, mut rx) = mpsc::channel::<(PendingTransfer<Source, Target, Metadata>, CudaEvent)>(
            max_concurrent_transfers,
        );

        spawn(move || {
            while let Some((pending_transfer, event)) = rx.blocking_recv() {
                // Wait for the event.
                event.synchronize()?;
                // Only finalize the transfer after the event is signaled.
                match pending_transfer.handle_complete() {
                    Ok(_) => {}
                    Err(e) => {
                        // The only case where this can fail is if the progress engine is shutdown.
                        // This is not a problem, so we can just ignore it.
                        tracing::warn!("Error handling transfer completion: {:?}", e);
                    }
                }
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
    async fn enqueue_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        pending_transfer.sources.write_to(
            &mut pending_transfer.targets,
            None,
            self.transfer_ctx.clone(),
        )?;

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
    pub fn new(transfer_ctx: Arc<TransferContext>, max_concurrent_transfers: usize) -> Self {
        let (futures_tx, mut futures_rx) = mpsc::channel(1);

        tokio::spawn(async move {
            // Keep track of our pending transfers.
            // Consume the futures as they complete, while also receiving new ones.

            let mut pending_transfers = FuturesUnordered::new();
            loop {
                tokio::select! {
                    Some(future) = futures_rx.recv() => {
                        // If we're at max size, block the worker thread on the next() call until we have capacity.
                        while pending_transfers.len() >= max_concurrent_transfers {
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
    async fn enqueue_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        let future = pending_transfer.sources.nixl_write_to(
            &mut pending_transfer.targets,
            None,
            self.transfer_ctx.clone(),
        )?;

        let completion_future = async move {
            let _ = future.await;
            match pending_transfer.handle_complete() {
                Ok(_) => {}
                Err(e) => {
                    // The only case where this can fail is if the progress engine is being shutdown.
                    // This is not a problem, so we can just ignore it.
                    tracing::warn!("Error handling transfer completion: {:?}", e);
                }
            }
        };

        // Futures_(tx/rx) has a capacity of 1. If the queue worker has received another future and is awaiting next() due to a full `FuturesUnordered`,
        // this call will block until the worker has processed the prior future.
        self.futures_tx.send(Box::pin(completion_future)).await?;

        Ok(())
    }
}

/// A transfer manager that enforces a max batch size for transfers.
pub struct TransferBatcher<Source, Target, Metadata, Manager>
where
    Source: Storage,
    Target: Storage,
    Metadata: BlockMetadata,
    Manager: TransferManager<Source, Target, Metadata>,
{
    transfer_manager: Manager,
    max_transfer_batch_size: usize,
    _phantom: PhantomData<(Source, Target, Metadata)>,
}

impl<Source, Target, Metadata, Manager> TransferBatcher<Source, Target, Metadata, Manager>
where
    Source: Storage,
    Target: Storage,
    Metadata: BlockMetadata,
    Manager: TransferManager<Source, Target, Metadata>,
{
    pub fn new(transfer_manager: Manager, max_transfer_batch_size: usize) -> Self {
        Self {
            transfer_manager,
            max_transfer_batch_size,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Source, Target, Metadata, Manager> TransferManager<Source, Target, Metadata>
    for TransferBatcher<Source, Target, Metadata, Manager>
where
    Source: Storage,
    Target: Storage,
    Metadata: BlockMetadata,
    Manager: TransferManager<Source, Target, Metadata>,
{
    async fn enqueue_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Metadata>,
    ) -> Result<()> {
        // If it's smaller than the max batch size, just enqueue it.
        if pending_transfer.sources.len() < self.max_transfer_batch_size {
            return self
                .transfer_manager
                .enqueue_transfer(pending_transfer)
                .await;
        }

        // Otherwise, we need to split the transfer into multiple smaller transfers.

        let PendingTransfer {
            mut sources,
            mut targets,
            completion_indicator,
            target_pool,
        } = pending_transfer;

        let mut indicators = Vec::new();

        while !sources.is_empty() {
            let sources = sources
                .drain(..std::cmp::min(self.max_transfer_batch_size, sources.len()))
                .collect();
            let targets = targets
                .drain(..std::cmp::min(self.max_transfer_batch_size, targets.len()))
                .collect();

            // If we have a completion indicator, we need to create a new one for each sub-transfer.
            let indicator = if completion_indicator.is_some() {
                let (batch_tx, batch_rx) = oneshot::channel();
                indicators.push(batch_rx);
                Some(batch_tx)
            } else {
                None
            };

            let request = PendingTransfer::new(sources, targets, indicator, target_pool.clone());
            // Enqueue our reduced transfer. This may block if the queue is full.
            self.transfer_manager.enqueue_transfer(request).await?;
        }

        if let Some(completion_indicator) = completion_indicator {
            tokio::spawn(async move {
                let mut results = Vec::new();

                for indicator in indicators.into_iter() {
                    // Await each sub-transfer, and append the results to our final results.
                    let result = match indicator.await.unwrap() {
                        Ok(result) => result,
                        Err(e) => {
                            tracing::error!("Error receiving transfer results: {:?}", e);
                            completion_indicator.send(Err(e)).unwrap();
                            return;
                        }
                    };
                    results.extend(result);
                }

                // Send the final results to the top-level completion indicator.
                completion_indicator.send(Ok(results)).unwrap();
            });
        }

        Ok(())
    }
}
