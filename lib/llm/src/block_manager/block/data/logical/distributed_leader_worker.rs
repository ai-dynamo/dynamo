// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader};

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::sync::{mpsc, oneshot};
use tokio_util::sync::CancellationToken;

type TransferRequest = (BlockTransferRequest, oneshot::Sender<()>);

#[derive(Clone)]
pub struct DistributedLeaderWorkerResources {
    /// Make this an option to make testing easier.
    // TODO(jthomson04): We should be using NullResources for this.
    transfer_tx: Option<mpsc::UnboundedSender<TransferRequest>>,
}

impl std::fmt::Debug for DistributedLeaderWorkerResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedLeaderWorkerResources").finish()
    }
}

impl DistributedLeaderWorkerResources {
    pub fn new(
        leader: Option<Arc<KvbmLeader>>,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<Self> {
        if let Some(leader) = leader {
            let (transfer_tx, transfer_rx) = mpsc::unbounded_channel();

            CriticalTaskExecutionHandle::new(
                move |cancel_token| async move {
                    Self::worker(leader, transfer_rx, cancel_token).await
                },
                cancel_token,
                "DistributedLeaderWorkerResources",
            )
            .map_err(|e| anyhow::anyhow!("Failed to create DistributedLeaderWorkerResources: {}", e))?.detach();

            Ok(Self {
                transfer_tx: Some(transfer_tx),
            })
        } else {
            Ok(Self { transfer_tx: None })
        }
    }

    fn get_pool<S: Storage>(data: &impl BlockDataExt<S>) -> BlockTransferPool {
        match data.storage_type() {
            StorageType::Device(_) => BlockTransferPool::Device,
            StorageType::Pinned => BlockTransferPool::Host,
            StorageType::Disk(_) => BlockTransferPool::Disk,
            _ => panic!("Invalid storage type"),
        }
    }

    async fn worker(
        leader: Arc<KvbmLeader>,
        mut transfer_rx: mpsc::UnboundedReceiver<TransferRequest>,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        loop {
            tokio::select! {
                Some(request) = transfer_rx.recv() => {
                    let (request, notify_tx) = request;

                    let rx = leader.transfer_blocks_request(request).await?;

                    tokio::spawn(async move {
                        rx.await.unwrap();
                        let _ = notify_tx.send(());
                    });
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }
}

impl LogicalResources for DistributedLeaderWorkerResources {
    fn handle_transfer<RB, WB>(
        &self,
        sources: &[RB],
        targets: &mut [WB],
        // TODO: This transfer context is only ever used in the `Local` locality.
        _ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Logical<Self>>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Logical<Self>>,
    {
        // Check for empty slices and length mismatch early
        if sources.is_empty() && targets.is_empty() {
            tracing::warn!(
                "DistributedLeaderWorkerResources::handle_transfer called with both sources and targets empty, skipping transfer"
            );
            let (tx, rx) = oneshot::channel();
            tx.send(()).unwrap();
            return Ok(rx);
        }

        if sources.len() != targets.len() {
            return Err(TransferError::CountMismatch(sources.len(), targets.len()));
        }

        if let Some(transfer_tx) = &self.transfer_tx {
            let source_pool = Self::get_pool(sources[0].block_data());
            let target_pool = Self::get_pool(targets[0].block_data());

            let source_idxs = sources.iter().map(|source| source.block_data().block_id());
            let target_idxs = targets.iter().map(|target| target.block_data().block_id());

            let request = BlockTransferRequest::new(
                source_pool,
                target_pool,
                source_idxs.zip(target_idxs).collect(),
            );

            let (tx, rx) = oneshot::channel();
            transfer_tx.send((request, tx)).unwrap();

            Ok(rx)
        } else {
            panic!("Block transfer functionality is disabled.");
        }
    }
}
