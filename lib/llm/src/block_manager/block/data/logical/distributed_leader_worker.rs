use super::*;

use crate::block_manager::distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader};

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::sync::{mpsc, oneshot::Receiver};
use tokio_util::sync::CancellationToken;

type TransferRequest = (
    BlockTransferRequest,
    ::oneshot::Sender<Result<Receiver<()>, anyhow::Error>>,
);

#[derive(Clone)]
pub struct DistributedLeaderWorkerResources {
    transfer_tx: mpsc::UnboundedSender<TransferRequest>,
    cancel_token: CancellationToken,
}

impl std::fmt::Debug for DistributedLeaderWorkerResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistributedLeaderWorkerResources").finish()
    }
}

impl DistributedLeaderWorkerResources {
    pub fn new(leader: KvbmLeader, cancel_token: CancellationToken) -> anyhow::Result<Self> {
        let (transfer_tx, transfer_rx) = mpsc::unbounded_channel();

        CriticalTaskExecutionHandle::new(
            move |cancel_token| async move {
                Self::worker(leader, transfer_rx, cancel_token).await
            },
            cancel_token.clone(),
            "DistributedLeaderWorkerResources",
        )
        .map_err(|e| anyhow::anyhow!("Failed to create DistributedLeaderWorkerResources: {}", e))?.detach();

        Ok(Self {
            transfer_tx,
            cancel_token,
        })
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
        leader: KvbmLeader,
        mut transfer_rx: mpsc::UnboundedReceiver<TransferRequest>,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        loop {
            tokio::select! {
                Some(request) = transfer_rx.recv() => {
                    let (request, sender) = request;
                    let _ = sender.send(leader.transfer_blocks_request(request).await);
                }
                _ = cancel_token.cancelled() => {
                    break;
                }
            }
        }

        Ok(())
    }
}

impl Drop for DistributedLeaderWorkerResources {
    fn drop(&mut self) {
        self.cancel_token.cancel();
    }
}

impl LogicalResources for DistributedLeaderWorkerResources {
    fn handle_transfer<RB, WB>(
        &self,
        sources: &[RB],
        targets: &mut [WB],
        notify: bool,
        // TODO: This transfer context is only ever used in the `Local` locality.
        _ctx: Arc<TransferContext>,
    ) -> Result<Option<oneshot::Receiver<()>>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Logical<Self>>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Logical<Self>>,
    {
        let source_pool = Self::get_pool(sources[0].block_data());
        let target_pool = Self::get_pool(targets[0].block_data());

        let source_idxs = sources.iter().map(|source| source.block_data().block_id());
        let target_idxs = targets.iter().map(|target| target.block_data().block_id());

        let request = BlockTransferRequest::new(
            source_pool,
            target_pool,
            source_idxs.zip(target_idxs).collect(),
        );

        let (tx, rx) = ::oneshot::channel();
        self.transfer_tx.send((request, tx)).unwrap();

        if notify {
            Ok(Some(
                rx.recv()
                    .map_err(|e| anyhow::anyhow!("Failed to receive transfer result: {}", e))?
                    .unwrap(),
            ))
        } else {
            Ok(None)
        }
    }
}
