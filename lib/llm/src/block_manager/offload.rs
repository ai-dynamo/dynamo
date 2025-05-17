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

use std::sync::Arc;
use tokio::sync::{mpsc, Mutex, Notify};

use super::block::{
    transfer::WriteTo, BlockError, BlockExt, BlockMetadata, BlockState, ImmutableBlock,
    MutableBlock,
};
use super::pool::BlockPoolError;
use super::state::TransferContext;
use super::storage::{Cuda, Storage};
use super::{BlockPool, DeviceStorage, PinnedStorage};

use anyhow::Result;
use cudarc::driver::sys::CUevent_flags;
use std::any::Any;

use std::collections::BTreeSet;

mod pending;
mod request;

use pending::{PendingTransfer, TransferManager};
use request::{OffloadRequest, OffloadRequestKey, OnboardRequest};

const MAX_OFFLOAD_STREAM_DEPTH: usize = 4;

/// The offload manager handles all block transfers between different cache levels.
pub struct OffloadManager<Metadata: BlockMetadata> {
    // Handles to the device and host pools.
    device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
    host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,

    /// Priority queue of pending offloads
    dtoh_offload_queue: Arc<Mutex<BTreeSet<OffloadRequest<DeviceStorage, Metadata>>>>,
    /// Used to notify the offload worker that an item has been added to the priority queue
    dtoh_offload_notify: Arc<Notify>,
    /// An incrementing counter for offloaded blocks. Within the same priority, blocks with lower tick values are processed first.
    tick: Arc<Mutex<u64>>,

    /// Queue of pending onboarding requests.
    htod_onboard_tx: mpsc::UnboundedSender<OnboardRequest<PinnedStorage, DeviceStorage, Metadata>>,
}

impl<Metadata: BlockMetadata> OffloadManager<Metadata> {
    pub fn new(
        device: Arc<Option<BlockPool<DeviceStorage, Metadata>>>,
        host: Arc<Option<BlockPool<PinnedStorage, Metadata>>>,
    ) -> Result<Arc<Self>> {
        let dtoh_offload_queue = Arc::new(Mutex::new(BTreeSet::new()));
        let dtoh_offload_notify = Arc::new(Notify::new());
        let (htod_onboard_tx, htod_onboard_rx) = mpsc::unbounded_channel();

        let this = Arc::new(Self {
            device,
            host,
            dtoh_offload_queue,
            dtoh_offload_notify,
            tick: Arc::new(Mutex::new(0)),
            htod_onboard_tx,
        });

        let this_clone = this.clone();
        // The offload and onboard workers must run in separate streams.
        // Otherwise, we'd only be doing either an offload or onboard at a time, cutting our effective transfer bandwidth in half.
        tokio::spawn(async move { this_clone.offload_worker().await });

        let this_clone = this.clone();
        tokio::spawn(async move { this_clone.onboard_worker(htod_onboard_rx).await });

        Ok(this)
    }

    async fn update_target_metadata<Source: Storage, Target: Storage>(
        source: &Arc<MutableBlock<Source, Metadata>>,
        target: &mut MutableBlock<Target, Metadata>,
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

    async fn offload_worker(&self) -> Result<()> {
        // Since cuda memcpys in streams are async, this gets a bit tricky.
        // We can't just consume the queue normally, otherwise the stream would become very backlogged.
        // From the point when the a transfer is put into the stream until the transfer corresponding to the block is complete, we need to hold a strong reference to the block.
        // If we don't do this, the block may be evicted and overwritten before the transfer is complete.
        // To do this, we use a queue to track blocks currently being offloaded. Once the offload is complete (as indicated by a CudaEvent), the reference to the block is dropped.

        if self.device.is_none() || self.host.is_none() {
            return Ok(());
        }

        let cuda_ctx = Cuda::device_or_create(0)?;

        let transfer_ctx = TransferContext::new(None, cuda_ctx.new_stream()?);

        let device = self.device.as_ref().as_ref().unwrap();
        let host = self.host.as_ref().as_ref().unwrap();

        // We don't want to hold too many strong references to blocks in the device pool, since it would limit our effective KV Cache capacity.
        // In this case, we limit it to just enough to ensure that a transfer is always occurring.
        let dtoh_pending_offload_manager = TransferManager::new(MAX_OFFLOAD_STREAM_DEPTH);

        loop {
            // Try to check the offload queue.
            let request = self.dtoh_offload_queue.lock().await.pop_first();

            // If there is a request, process it.
            if let Some(request) = request {
                // Try to upgrade the block to a strong reference.
                let block = match request.block.upgrade() {
                    Some(block) => Some(block),
                    // If unable to upgrade, the block may have been moved to the inactive pool.
                    None => device
                        .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                        .await?
                        .pop()
                        .map(|block| block.mutable_block().clone()),
                };

                // If we've found the block, offload it to the host.
                if let Some(block) = block {
                    // Allocate a block from the host pool.
                    // TODO: The most likely error here is that the host pool is full.
                    // It's probably not a good idea to keep consuming queue elements in the meantime.
                    let host_blocks = match host.allocate_blocks(1).await {
                        Ok(blocks) => blocks,
                        Err(_) => {
                            continue;
                        }
                    };

                    if let Some(mut host_block) = host_blocks.into_iter().next() {
                        // Enqueue the offload into the stream.
                        block.write_to(&mut host_block, None, &transfer_ctx)?;

                        // Record an event after the transfer is complete. Use the BLOCKING_SYNC flag to ensure the event is recorded synchronously on the host.
                        let event = transfer_ctx
                            .stream()
                            .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

                        // Update block metadata and register with host pool.
                        OffloadManager::update_target_metadata(&block, &mut host_block).await?;

                        // Record the pending offload. This may block if too many offloads are already pending.
                        dtoh_pending_offload_manager
                            .handle_pending_transfer(PendingTransfer::new(
                                vec![block],
                                vec![host_block],
                                event,
                                None,
                                self.host.clone(),
                            ))
                            .await?;
                    }
                }
            } else {
                // If the queue is empty, wait to be notified.
                self.dtoh_offload_notify.notified().await;
            }
        }
    }

    async fn onboard_worker(
        &self,
        mut htod_onboard_rx: mpsc::UnboundedReceiver<
            OnboardRequest<PinnedStorage, DeviceStorage, Metadata>,
        >,
    ) -> Result<()> {
        if self.device.is_none() || self.host.is_none() {
            return Ok(());
        }

        let cuda_ctx = Cuda::device_or_create(0)?;
        let transfer_ctx = TransferContext::new(None, cuda_ctx.new_stream()?);

        // For the onboarding manager, we can get away with a much bigger queue, since any onboardings would get triggered by an upcoming prefill.
        let htod_pending_onboard_manager = TransferManager::new(16384);
        let device = self.device.as_ref().as_ref().unwrap();

        while let Some(request) = htod_onboard_rx.recv().await {
            let mut device_blocks = match device.allocate_blocks(request.blocks.len()).await {
                Ok(blocks) => blocks,
                Err(err) => {
                    request.response_tx.send(Err(err))?;
                    continue;
                }
            };

            for (host_block, device_block) in request.blocks.iter().zip(device_blocks.iter_mut()) {
                host_block.write_to(device_block, None, &transfer_ctx)?;
                OffloadManager::update_target_metadata(host_block.mutable_block(), device_block)
                    .await?;
            }

            // Record an event after all transfers are complete. See use of CU_EVENT_BLOCKING_SYNC in offload_worker.
            let event = transfer_ctx
                .stream()
                .record_event(Some(CUevent_flags::CU_EVENT_BLOCKING_SYNC))?;

            let sources = request
                .blocks
                .iter()
                .map(|b| b.mutable_block().clone())
                .collect();

            htod_pending_onboard_manager
                .handle_pending_transfer(PendingTransfer::new(
                    sources,
                    device_blocks,
                    event,
                    Some(request.response_tx),
                    self.device.clone(),
                ))
                .await?;
        }
        Ok(())
    }

    pub async fn offload<S: Storage>(
        &self,
        block: &ImmutableBlock<S, Metadata>,
        priority: u64,
    ) -> core::result::Result<(), BlockPoolError> {
        match block.state() {
            BlockState::Registered(_, _) => {}
            _ => {
                return Err(BlockPoolError::BlockError(BlockError::InvalidState(
                    "Block is not registered.".to_string(),
                )));
            }
        }
        // This can get called by all pools, regardless of whether or not they have a place to offload to.
        // Because of this, we need to check the block type here.
        let any_block = block as &dyn Any;

        // For now, only consider offloads from G1 (device) to G2 (host).
        // TODO: What's the performance penalty of this runtime type-checking?
        if let Some(device_block) =
            any_block.downcast_ref::<ImmutableBlock<DeviceStorage, Metadata>>()
        {
            let mut tick = self.tick.lock().await;
            let key = OffloadRequestKey {
                priority,
                timestamp: *tick,
            };
            // Increment a counter for each block. Within the same priority, blocks with lower counter values are processed first.
            *tick += 1;
            drop(tick);

            let request = OffloadRequest {
                block: Arc::downgrade(device_block.mutable_block()),
                sequence_hash: device_block.sequence_hash()?,
                key,
            };

            self.dtoh_offload_queue.lock().await.insert(request);
            self.dtoh_offload_notify.notify_one();
        }

        Ok(())
    }

    pub async fn onboard(
        &self,
        blocks: Vec<ImmutableBlock<PinnedStorage, Metadata>>,
    ) -> core::result::Result<Vec<ImmutableBlock<DeviceStorage, Metadata>>, BlockPoolError> {
        for block in &blocks {
            match block.state() {
                BlockState::Registered(_, _) => {}
                _ => {
                    return Err(BlockPoolError::BlockError(BlockError::InvalidState(
                        "Block is not registered.".to_string(),
                    )));
                }
            }
        }

        let (tx, rx) = oneshot::channel();

        self.htod_onboard_tx
            .send(OnboardRequest::new(blocks, tx))
            .map_err(|_| BlockPoolError::ProgressEngineShutdown)?;
        match rx.await {
            Ok(res) => res,
            Err(_) => Err(BlockPoolError::ProgressEngineShutdown),
        }
    }
}

#[cfg(all(test, feature = "testing-cuda"))]
mod tests {
    use super::*;
    use crate::block_manager::block::test_utils::get_private_token;

    use crate::block_manager::{
        block::{BasicMetadata, BlockDataExt, BlockDataProvider, Blocks},
        layout::FullyContiguous,
        pool::BlockPool,
        storage::{
            cuda::CudaAccessible, DeviceAllocator, DeviceStorage, PinnedAllocator, PinnedStorage,
        },
        DType, LayoutConfig,
    };
    use nixl_sys::NixlDescriptor;

    use cudarc::runtime::sys::{cudaMemcpy, cudaMemcpyKind, cudaMemset};

    const BLOCK_SIZE: usize = 4;

    type DevicePool = Arc<Option<BlockPool<DeviceStorage, BasicMetadata>>>;
    type HostPool = Arc<Option<BlockPool<PinnedStorage, BasicMetadata>>>;

    fn build_pools(
        device_blocks: usize,
        host_blocks: Option<usize>,
    ) -> Result<(Arc<OffloadManager<BasicMetadata>>, DevicePool, HostPool)> {
        let mut config = LayoutConfig {
            num_blocks: device_blocks,
            num_layers: 8,
            page_size: BLOCK_SIZE,
            inner_dim: 1024,
            alignment: 1,
            dtype: DType::FP16,
        };

        let device = FullyContiguous::allocate(config.clone(), &DeviceAllocator::default())?;
        let device_blocks = Blocks::<_, BasicMetadata>::new(device, 42, 0)?.into_blocks()?;
        let device_pool = Arc::new(Some(BlockPool::builder().blocks(device_blocks).build()?));

        let host_pool = if let Some(host_blocks) = host_blocks {
            config.num_blocks = host_blocks;
            let host = FullyContiguous::allocate(config, &PinnedAllocator::default())?;
            let host_blocks = Blocks::<_, BasicMetadata>::new(host, 42, 0)?.into_blocks()?;
            Arc::new(Some(BlockPool::builder().blocks(host_blocks).build()?))
        } else {
            Arc::new(None)
        };

        let manager = OffloadManager::new(device_pool.clone(), host_pool.clone())?;

        Ok((manager, device_pool, host_pool))
    }

    /// Create a block in the 'RESET' state.
    async fn get_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
    ) -> Result<MutableBlock<S, Metadata>> {
        pool.allocate_blocks(1)
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to allocate block"))
    }

    /// Create a block in the 'PARTIAL' state.
    async fn partial_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        token: u32,
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = get_block(pool).await?;
        block.init_sequence(42)?;
        block.add_token(token)?;
        Ok(block)
    }

    /// Create a block in the 'COMPLETED' state.
    async fn completed_block<S: Storage, Metadata: BlockMetadata>(
        pool: &BlockPool<S, Metadata>,
        tokens: [u32; BLOCK_SIZE],
    ) -> Result<MutableBlock<S, Metadata>> {
        let mut block = get_block(pool).await?;
        block.init_sequence(42)?;
        for token in tokens {
            block.add_token(token)?;
        }
        block.commit()?;
        Ok(block)
    }

    fn populate_cuda_block<S: Storage + CudaAccessible + NixlDescriptor>(
        block: &impl BlockDataProvider<StorageType = S>,
        value: i32,
    ) -> Result<()> {
        let block_data = block.block_data(get_private_token()).block_view()?;
        let block_size = block_data.size();

        unsafe {
            cudaMemset(
                block_data.as_ptr() as *mut std::ffi::c_void,
                value,
                block_size,
            )
            .result()?;
        }
        Ok(())
    }

    /// Compare the contents of a device block and a host block.
    async fn compare_block_contents(
        device_block: &impl BlockDataProvider<StorageType = DeviceStorage>,
        host_block: &impl BlockDataProvider<StorageType = PinnedStorage>,
    ) -> Result<()> {
        let host_data = host_block.block_data(get_private_token()).block_view()?;
        let device_data = device_block.block_data(get_private_token()).block_view()?;

        let size = host_data.size();

        assert_eq!(size, device_data.size());

        let mut host_buffer = vec![0u8; size];
        let host_slice;

        unsafe {
            cudaMemcpy(
                host_buffer.as_mut_ptr() as *mut std::ffi::c_void,
                device_data.as_ptr() as *const std::ffi::c_void,
                size,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
            .result()?;
            host_slice = std::slice::from_raw_parts(host_buffer.as_ptr(), size);
        }

        assert_eq!(host_buffer, host_slice);

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_invalid_blocks() -> Result<()> {
        let (offload_manager, device_pool, _) = build_pools(4, Some(4))?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();

        // Check blocks in the 'RESET' state.
        let immutable_block = ImmutableBlock::new(Arc::new(get_block(device_pool).await?));

        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        // Check blocks in the 'PARTIAL' state.
        let immutable_block = ImmutableBlock::new(Arc::new(partial_block(device_pool, 0).await?));
        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        // Check blocks in the 'COMPLETED' state.
        let immutable_block = ImmutableBlock::new(Arc::new(
            completed_block(device_pool, [0; BLOCK_SIZE]).await?,
        ));
        assert!(matches!(
            offload_manager.offload(&immutable_block, 0).await,
            Err(BlockPoolError::BlockError(BlockError::InvalidState(_)))
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_registered_blocks() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, Some(4))?;

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

        populate_cuda_block(&immutable_device_block, 42)?;

        // Offloads should only go to G2 (for now)
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for it to be processed.
        // TODO: This is a bit of a hack, and may lead to non-deterministic behavior.
        // In theory, the offload + memcpy should take much less time than this.
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

        compare_block_contents(&immutable_device_block, &host_blocks[0]).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_no_host_blocks_available() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, Some(4))?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let host_blocks = host_pool.allocate_blocks(4).await?;
        assert_eq!(host_blocks.len(), 4);

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // The offload should fail gracefuly due to a lack of host blocks
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 0);

        // Wait for blocks to be returned to the pool.
        drop(host_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Try the offload again.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // This time, the offload should succeed.
        let matched_host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, Some(4))?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        // Allocate and fill a block on the host.
        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_cuda_block(&immutable_host_block, 42)?;

        // Onboard the block.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await?;

        assert_eq!(onboarded_blocks.len(), 1);
        // Check that the sequence hash is the same.
        assert_eq!(
            onboarded_blocks[0].sequence_hash()?,
            immutable_host_block.sequence_hash()?
        );
        // Check that the block is registered.
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_, _)
        ));

        compare_block_contents(&onboarded_blocks[0], &immutable_host_block).await?;

        // Wait for the new value to show up in the device pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let device_blocks = device_pool
            .match_sequence_hashes(vec![onboarded_blocks[0].sequence_hash()?].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 1);
        assert_eq!(
            device_blocks[0].sequence_hash()?,
            onboarded_blocks[0].sequence_hash()?
        );

        // Check that this is the same block.
        compare_block_contents(&device_blocks[0], &immutable_host_block).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_onboard() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, Some(4))?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_cuda_block(&immutable_device_block, 42)?;
        // Offload the block to the host.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool.
        let immutable_host_block = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()?].as_slice())
            .await?
            .into_iter()
            .next()
            .unwrap();

        compare_block_contents(&immutable_device_block, &immutable_host_block).await?;

        // Remove the device block from the pool by dropping it and allocating more blocks.
        drop(immutable_device_block);

        // Wait for the block to be returned to the pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let device_blocks = device_pool.allocate_blocks(4).await?;
        assert_eq!(device_blocks.len(), 4);

        drop(device_blocks);
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block is not in the device pool.
        let device_blocks = device_pool
            .match_sequence_hashes(vec![immutable_host_block.sequence_hash()?].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 0);

        // Onboard the block back to the device pool.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await?;
        assert_eq!(onboarded_blocks.len(), 1);
        assert_eq!(
            onboarded_blocks[0].sequence_hash()?,
            immutable_host_block.sequence_hash()?
        );
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_, _)
        ));

        compare_block_contents(&onboarded_blocks[0], &immutable_host_block).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_err_handling() -> Result<()> {
        let (offload_manager, device_pool, host_pool) = build_pools(4, Some(4))?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();
        let host_pool = host_pool.as_ref().as_ref().unwrap();

        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        let device_blocks = device_pool.allocate_blocks(4).await?;
        assert_eq!(device_blocks.len(), 4);

        let res = offload_manager
            .onboard(vec![immutable_host_block.clone()])
            .await;
        assert!(matches!(
            res.err().unwrap(),
            BlockPoolError::NotEnoughBlocksAvailable(_, _)
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_onboard_no_host_blocks() -> Result<()> {
        let (offload_manager, device_pool, _) = build_pools(4, None)?;

        let device_pool = device_pool.as_ref().as_ref().unwrap();

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        offload_manager.offload(&immutable_device_block, 0).await?;

        Ok(())
    }
}
