use std::sync::{Arc, Weak};
use tokio::sync::{Mutex, Notify};

use super::storage::Storage;
use crate::block_manager::block::{
    transfer::WriteTo, BlockExt, BlockMetadata, BlockState, ImmutableBlock, MutableBlock,
};
use crate::block_manager::{BlockPool, CacheLevel, DeviceStorage, PinnedStorage};

use anyhow::Result;
use std::any::Any;
use std::cmp::Ordering;
use std::collections::BTreeSet;

struct OffloadRequest<S: Storage, M: BlockMetadata> {
    priority: u64,
    block: Weak<MutableBlock<S, M>>,
    sequence_hash: u64,
    location: CacheLevel,
}

impl<S: Storage, M: BlockMetadata> PartialOrd for OffloadRequest<S, M> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Order high to low.
        Some(other.priority.partial_cmp(&self.priority).unwrap())
    }
}

impl<S: Storage, M: BlockMetadata> Ord for OffloadRequest<S, M> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Order high to low.
        other.priority.cmp(&self.priority)
    }
}

// Manually implement PartialEq based on sequence_hash
impl<S: Storage, M: BlockMetadata> PartialEq for OffloadRequest<S, M> {
    fn eq(&self, other: &Self) -> bool {
        self.sequence_hash == other.sequence_hash
            && self.priority == other.priority
            && self.location == other.location
    }
}

// Manually implement Eq
impl<S: Storage, M: BlockMetadata> Eq for OffloadRequest<S, M> {}

/// For now, only support offloading from G1 to G2.
pub struct OffloadManager<Metadata: BlockMetadata> {
    device_offload_tx: tokio::sync::mpsc::UnboundedSender<OffloadRequest<DeviceStorage, Metadata>>,

    ingress_handle: Option<tokio::task::JoinHandle<Result<()>>>,
    process_handle: Option<tokio::task::JoinHandle<Result<()>>>,
}

impl<Metadata: BlockMetadata> OffloadManager<Metadata> {
    pub fn new(
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
    ) -> Arc<Self> {
        let (device_offload_tx, mut device_offload_rx) = tokio::sync::mpsc::unbounded_channel();

        // Create self, then create the offload handle that references self.
        let mut this = Self {
            device_offload_tx,
            ingress_handle: None,
            process_handle: None,
        };

        let device_offload_queue = Arc::new(Mutex::new(BTreeSet::new()));
        let device_offload_notify = Arc::new(Notify::new());

        let device_offload_queue_clone = device_offload_queue.clone();
        let device_offload_notify_clone = device_offload_notify.clone();

        let ingress_handle = tokio::spawn(async move {
            while let Some(request) = device_offload_rx.recv().await {
                device_offload_queue.lock().await.insert(request);
                device_offload_notify.notify_one();
            }

            Ok(())
        });

        let process_handle = tokio::spawn(async move {
            let device_offload_queue = device_offload_queue_clone;
            let device_offload_notify = device_offload_notify_clone;
            loop {
                let request = device_offload_queue.lock().await.pop_first();

                if let Some(request) = request {
                    // Only consider offloads from G1 to G2.
                    if request.location != CacheLevel::G2 {
                        continue;
                    }

                    let block = match request.block.upgrade() {
                        Some(block) => Some(block),
                        None => {
                            if let Some(device) = device.as_ref() {
                                device
                                    .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                                    .await?
                                    .pop()
                                    .map(|block| block.mutable_block().clone())
                            } else {
                                None
                            }
                        }
                    };

                    // If we've found the block, offload it to the host.
                    if let Some(block) = block {
                        // Get a block from the host pool
                        if let Some(host) = host.as_ref() {
                            // Allocate a single block from the host pool

                            let host_blocks = host.allocate_blocks(1).await?;

                            if let Some(mut host_block) = host_blocks.into_iter().next() {
                                block.write_to(&mut host_block, None)?;

                                OffloadManager::handle_offload(&block, host_block, host).await?;
                            }
                        }
                    }
                } else {
                    device_offload_notify.notified().await;
                }
            }
        });

        this.ingress_handle = Some(ingress_handle);
        this.process_handle = Some(process_handle);
        Arc::new(this)
    }

    fn build_request<S: Storage>(
        block: &ImmutableBlock<S, Metadata>,
        location: CacheLevel,
        priority: u64,
    ) -> Result<OffloadRequest<S, Metadata>> {
        Ok(OffloadRequest {
            block: Arc::downgrade(block.mutable_block()),
            sequence_hash: block.sequence_hash()?,
            location,
            priority,
        })
    }

    async fn handle_offload<Source: Storage, Target: Storage>(
        source: &Arc<MutableBlock<Source, Metadata>>,
        mut target: MutableBlock<Target, Metadata>,
        target_pool: &BlockPool<Target, Metadata>,
    ) -> Result<()> {
        let target_mut = &mut target;

        if let BlockState::Registered(reg_handle) = source.state() {
            target_mut.reset();
            target_mut.update_metadata(source.metadata().clone());
            target_mut.apply_token_block(reg_handle.token_block().clone())?;

            target_pool.register_blocks(vec![target]).await?;
        } else {
            panic!("Invalid block state for offload! This should never happen.")
        }

        Ok(())
    }

    pub fn offload<S: Storage>(
        &self,
        block: &ImmutableBlock<S, Metadata>,
        location: CacheLevel,
        priority: u64,
    ) -> Result<()> {
        match block.state() {
            BlockState::Registered(_) => {}
            _ => {
                return Err(anyhow::anyhow!("Only registered blocks may be offloaded."));
            }
        }

        let any_block = block as &dyn Any;

        // For now, only consider offloads from G1 to G2.
        if let Some(device_block) =
            any_block.downcast_ref::<ImmutableBlock<DeviceStorage, Metadata>>()
        {
            self.device_offload_tx.send(OffloadManager::build_request(
                device_block,
                location,
                priority,
            )?)?;
        }

        Ok(())
    }
}
