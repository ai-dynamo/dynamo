// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

use nixl_sys::NixlDescriptor;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

use crate::block_manager::block::{
    BlockDataProvider, BlockDataProviderMut, BlockError, BlockMetadata, BlockState, ImmutableBlock,
    MutableBlock, ReadableBlock, WritableBlock,
    locality::LocalityProvider,
    transfer::{TransferContext, WriteTo, WriteToStrategy},
};
use crate::block_manager::pool::{BlockPool, BlockPoolError};
use crate::block_manager::storage::{Local, Storage};

use anyhow::Result;
use async_trait::async_trait;
use futures::{StreamExt, stream::FuturesUnordered};

use super::BlockResult;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

/// Manage a set of pending transfers.
pub struct PendingTransfer<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
> {
    /// The block being copied from.
    sources: Vec<ImmutableBlock<Source, Locality, Metadata>>,
    /// The block being copied to.
    targets: Vec<MutableBlock<Target, Locality, Metadata>>,
    /// The oneshot sender that optionally returns the registered blocks once the transfer is complete.
    completion_indicator: Option<oneshot::Sender<BlockResult<Target, Locality, Metadata>>>,
    /// The target pool that will receive the registered block.
    target_pool: Arc<dyn BlockPool<Target, Locality, Metadata>>,
}

impl<Source: Storage, Target: Storage, Locality: LocalityProvider, Metadata: BlockMetadata>
    PendingTransfer<Source, Target, Locality, Metadata>
{
    pub fn new(
        sources: Vec<ImmutableBlock<Source, Locality, Metadata>>,
        targets: Vec<MutableBlock<Target, Locality, Metadata>>,
        completion_indicator: Option<oneshot::Sender<BlockResult<Target, Locality, Metadata>>>,
        target_pool: Arc<dyn BlockPool<Target, Locality, Metadata>>,
    ) -> Self {
        assert_eq!(sources.len(), targets.len());
        Self {
            sources,
            targets,
            completion_indicator,
            target_pool,
        }
    }

    async fn handle_complete(self) -> Result<()> {
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

        let blocks = target_pool.register_blocks(targets).await?;

        tracing::debug!("Transfer complete. Registered {} blocks.", blocks.len());

        if let Some(completion_indicator) = completion_indicator {
            completion_indicator
                .send(Ok(blocks))
                .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
        }

        Ok(())
    }
}

fn transfer_metadata<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
>(
    source: &ImmutableBlock<Source, Locality, Metadata>,
    target: &mut MutableBlock<Target, Locality, Metadata>,
) -> Result<()> {
    // Only registered blocks can be transferred. There are upstream checks for this, so this shouldn't ever fail.
    if let BlockState::Registered(reg_handle, _) = source.state() {
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
pub trait TransferManager<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
>: Send + Sync
{
    /// Begin a transfer. Blocks if the pending queue is full.
    async fn enqueue_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Locality, Metadata>,
    ) -> Result<()>;
}

struct TransferCompletionManager<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
> {
    num_blocks_transferred: usize,
    _phantom: PhantomData<(Source, Target, Locality, Metadata)>,
}

impl<Source: Storage, Target: Storage, Locality: LocalityProvider, Metadata: BlockMetadata>
    TransferCompletionManager<Source, Target, Locality, Metadata>
{
    pub fn new() -> Self {
        Self {
            num_blocks_transferred: 0,
            _phantom: PhantomData,
        }
    }

    pub async fn handle_complete(
        &mut self,
        pending_transfer: PendingTransfer<Source, Target, Locality, Metadata>,
    ) -> Result<()> {
        self.num_blocks_transferred += pending_transfer.sources.len();

        match pending_transfer.handle_complete().await {
            Ok(_) => {}
            Err(e) => {
                // The only case where this can fail is if the progress engine is being shutdown.
                // This is not a problem, so we can just ignore it.
                tracing::warn!("Error handling transfer completion: {:?}", e);
            }
        }

        Ok(())
    }
}

type TransferFuture<Source, Target, Locality, Metadata> = Pin<
    Box<
        dyn std::future::Future<Output = PendingTransfer<Source, Target, Locality, Metadata>>
            + Send
            + Sync,
    >,
>;

pub struct LocalTransferManager<
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
> {
    futures_tx: mpsc::Sender<TransferFuture<Source, Target, Locality, Metadata>>,
    transfer_ctx: Arc<TransferContext>,
}

impl<Source: Storage, Target: Storage, Locality: LocalityProvider, Metadata: BlockMetadata>
    LocalTransferManager<Source, Target, Locality, Metadata>
{
    pub fn new(
        transfer_ctx: Arc<TransferContext>,
        max_concurrent_transfers: usize,
        runtime: &Handle,
        cancellation_token: CancellationToken,
    ) -> Result<Self> {
        let (futures_tx, mut futures_rx) = mpsc::channel(1);

        let mut completion_manager = TransferCompletionManager::new();

        CriticalTaskExecutionHandle::new_with_runtime(
            move |cancel_token| async move {
                let mut pending_transfers: FuturesUnordered<TransferFuture<Source, Target, Locality, Metadata>> = FuturesUnordered::new();
                loop {
                    tokio::select! {

                        _ = cancel_token.cancelled() => {
                            return Ok(());
                        }

                        Some(future) = futures_rx.recv() => {
                            // If we're at max size, block the worker thread on the next() call until we have capacity.
                            while pending_transfers.len() >= max_concurrent_transfers {
                                if let Some(pending_transfer) = pending_transfers.next().await {
                                    completion_manager.handle_complete(pending_transfer).await?;
                                } else {
                                    break;
                                }
                            }

                            pending_transfers.push(future);
                        }
                        Some(pending_transfer) = pending_transfers.next(), if !pending_transfers.is_empty() => {
                            completion_manager.handle_complete(pending_transfer).await?;
                        }
                    }
                }
            },
            cancellation_token.clone(),
            "Local Transfer Manager",
            runtime,
        )?
        .detach();

        Ok(Self {
            futures_tx,
            transfer_ctx,
        })
    }
}

#[async_trait]
impl<Source, Target, Locality, Metadata> TransferManager<Source, Target, Locality, Metadata>
    for LocalTransferManager<Source, Target, Locality, Metadata>
where
    Source: Storage + NixlDescriptor,
    Target: Storage + NixlDescriptor,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
    // Check that the source block is readable, local, and writable to the target block.
    ImmutableBlock<Source, Locality, Metadata>: ReadableBlock<StorageType = Source>
        + Local
        + WriteToStrategy<MutableBlock<Target, Locality, Metadata>>,
    // Check that the target block is writable.
    MutableBlock<Target, Locality, Metadata>: WritableBlock<StorageType = Target>,
    // Check that the source and target blocks have the same locality.
    ImmutableBlock<Source, Locality, Metadata>: BlockDataProvider<Locality = Locality>,
    MutableBlock<Target, Locality, Metadata>: BlockDataProviderMut<Locality = Locality>,
{
    async fn enqueue_transfer(
        &self,
        mut pending_transfer: PendingTransfer<Source, Target, Locality, Metadata>,
    ) -> Result<()> {
        let notify = pending_transfer
            .sources
            .write_to(&mut pending_transfer.targets, self.transfer_ctx.clone())?;

        let completion_future = async move {
            let _ = notify.await;
            pending_transfer
        };

        // Futures_(tx/rx) has a capacity of 1. If the queue worker has received another future and is awaiting next() due to a full `FuturesUnordered`,
        // this call will block until the worker has processed the prior future.
        self.futures_tx.send(Box::pin(completion_future)).await?;

        Ok(())
    }
}

/// A transfer manager that enforces a max batch size for transfers.
pub struct TransferBatcher<Source, Target, Locality, Metadata, Manager>
where
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider,
    Metadata: BlockMetadata,
    Manager: TransferManager<Source, Target, Locality, Metadata>,
{
    transfer_manager: Manager,
    max_transfer_batch_size: usize,
    runtime: Handle,
    cancellation_token: CancellationToken,
    _phantom: PhantomData<(Source, Target, Locality, Metadata)>,
}

impl<Source, Target, Locality, Metadata, Manager>
    TransferBatcher<Source, Target, Locality, Metadata, Manager>
where
    Source: Storage,
    Target: Storage,
    Locality: LocalityProvider + 'static,
    Metadata: BlockMetadata + 'static,
    Manager: TransferManager<Source, Target, Locality, Metadata> + 'static,
{
    pub fn new(
        transfer_manager: Manager,
        max_transfer_batch_size: usize,
        runtime: &Handle,
        cancellation_token: CancellationToken,
    ) -> Self {
        Self {
            transfer_manager,
            max_transfer_batch_size,
            runtime: runtime.clone(),
            cancellation_token,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<Source, Target, Locality, Metadata, Manager>
    TransferManager<Source, Target, Locality, Metadata>
    for TransferBatcher<Source, Target, Locality, Metadata, Manager>
where
    Source: Storage + 'static,
    Target: Storage + 'static,
    Locality: LocalityProvider + 'static,
    Metadata: BlockMetadata,
    Manager: TransferManager<Source, Target, Locality, Metadata>,
{
    async fn enqueue_transfer(
        &self,
        pending_transfer: PendingTransfer<Source, Target, Locality, Metadata>,
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
            CriticalTaskExecutionHandle::new_with_runtime(
                move |cancel_token| async move {
                    let mut results = Vec::new();

                    for indicator in indicators.into_iter() {
                        // Await each sub-transfer, and append the results to our final results.
                        tokio::select! {
                            _ = cancel_token.cancelled() => {
                                return Ok(());
                            }

                            Ok(indicator) = indicator => {
                                let result = match indicator {
                                    Ok(result) => result,
                                    Err(e) => {
                                        tracing::error!("Error receiving transfer results: {:?}", e);
                                        let _ = completion_indicator.send(Err(e));
                                        return Ok(());
                                    }
                                };
                                results.extend(result);
                            }
                        }
                    }

                    // Send the final results to the top-level completion indicator.
                    let _ = completion_indicator.send(Ok(results));

                    Ok(())
                },
                self.cancellation_token.clone(),
                "Transfer Batcher",
                &self.runtime,
            )?.detach();
        }

        Ok(())
    }
}
