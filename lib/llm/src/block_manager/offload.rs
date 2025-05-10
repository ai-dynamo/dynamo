use std::sync::{Arc, Weak};
use tokio::sync::{Mutex, Notify};

use super::storage::Storage;
use crate::block_manager::block::{
    transfer::WriteTo, BlockExt, BlockMetadata, BlockState, ImmutableBlock, MutableBlock,
};
use crate::block_manager::state::TransferContext;
use crate::block_manager::{BlockPool, CacheLevel, DeviceStorage, PinnedStorage};

use anyhow::Result;
use cudarc::driver::CudaContext;
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
        Some(self.cmp(other))
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
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
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
                println!("Received offload request.");
                device_offload_queue.lock().await.insert(request);
                device_offload_notify.notify_one();
            }

            Ok(())
        });

        let cuda_ctx = CudaContext::new(0).unwrap();
        let transfer_ctx = TransferContext::new(None, cuda_ctx.default_stream());

        let process_handle = tokio::spawn(async move {
            let device_offload_queue = device_offload_queue_clone;
            let device_offload_notify = device_offload_notify_clone;
            loop {
                let request = device_offload_queue.lock().await.pop_first();

                if let Some(request) = request {
                    println!("Processing offload request.");
                    let block = match request.block.upgrade() {
                        Some(block) => Some(block),
                        None => {
                            println!("No block found, trying to match sequence hash.");
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
                                block.write_to(&mut host_block, None, &transfer_ctx)?;

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
            if location != CacheLevel::G2 {
                return Err(anyhow::anyhow!("Only offloads to G2 are supported."));
            }

            println!("Enqueuing offload.");

            self.device_offload_tx.send(OffloadManager::build_request(
                device_block,
                location,
                priority,
            )?)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::block_manager::{
        block::{BasicMetadata, Blocks},
        layout::FullyContiguous,
        pool::BlockPool,
        storage::{DeviceAllocator, DeviceStorage, PinnedAllocator, PinnedStorage},
        DType, LayoutConfig,
    };

    const BLOCK_SIZE: usize = 4;

    fn build_pools(
        device_blocks: usize,
        host_blocks: usize,
    ) -> Result<(
        Arc<OffloadManager<BasicMetadata>>,
        Arc<Option<BlockPool<DeviceStorage, BasicMetadata>>>,
        Arc<Option<BlockPool<PinnedStorage, BasicMetadata>>>,
    )> {
        let mut config = LayoutConfig {
            num_blocks: device_blocks,
            num_layers: 8,
            page_size: BLOCK_SIZE,
            inner_dim: 1024,
            alignment: 1,
            dtype: DType::FP16,
        };

        let device = FullyContiguous::allocate(config.clone(), &DeviceAllocator::default())?;

        config.num_blocks = host_blocks;

        let host = FullyContiguous::allocate(config, &PinnedAllocator::default())?;

        let device_blocks = Blocks::<_, BasicMetadata>::new(device, 42, 0)?.into_blocks()?;
        let host_blocks = Blocks::<_, BasicMetadata>::new(host, 42, 0)?.into_blocks()?;

        let device_pool = Arc::new(Some(BlockPool::builder().blocks(device_blocks).build()?));

        let host_pool = Arc::new(Some(BlockPool::builder().blocks(host_blocks).build()?));

        let manager = OffloadManager::new(device_pool.clone(), host_pool.clone());

        Ok((manager, device_pool, host_pool))
    }

    async fn reset_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
    ) -> Result<MutableBlock<S, Metadata>> {
        Ok(pool
            .allocate_blocks(1)
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to allocate block"))?)
    }

    async fn partial_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        token: u32,
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = reset_block(pool).await?;
        block.init_sequence(42)?;
        block.add_token(token)?;
        Ok(block)
    }

    async fn completed_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        tokens: [u32; BLOCK_SIZE],
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = reset_block(pool).await?;
        block.init_sequence(42)?;
        for token in tokens {
            block.add_token(token)?;
        }
        block.commit()?;
        Ok(block)
    }

    #[tokio::test]
    async fn test_offload_invalid_blocks() -> Result<()> {
        let (offload_manager, device_pool, _) = build_pools(4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();

        // Check blocks in the 'RESET' state.
        let immutable_block = ImmutableBlock::new(Arc::new(reset_block(device_pool).await?));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .is_err());

        // Check blocks in the 'PARTIAL' state.
        let immutable_block = ImmutableBlock::new(Arc::new(partial_block(device_pool, 0).await?));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .is_err());

        // Check blocks in the 'COMPLETED' state.
        let immutable_block = ImmutableBlock::new(Arc::new(
            completed_block(device_pool, [0; BLOCK_SIZE]).await?,
        ));
        assert!(offload_manager
            .offload(&immutable_block, CacheLevel::G2, 0)
            .is_err());

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_registered_blocks() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, 4)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        // Create a block and register it with the offload manager
        let block = completed_block(device_pool, [0, 1, 2, 3]).await?;

        let immutable_device_block = device_pool
            .register_blocks(vec![block])
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to register block"))?;

        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G1, 0)
            .is_err());
        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G3, 0)
            .is_err());
        assert!(offload_manager
            .offload(&immutable_device_block, CacheLevel::G4, 0)
            .is_err());

        // Offloads should only go to G2 (for now)
        offload_manager.offload(&immutable_device_block, CacheLevel::G2, 0)?;

        // Wait for it to be processed
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool
        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(host_blocks.len(), 1);
        assert_eq!(
            host_blocks[0].sequence_hash()?,
            immutable_device_block.sequence_hash()?
        );

        Ok(())
    }
}
