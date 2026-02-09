// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    BasicMetadata, Storage,
    block::{
        Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock, WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    offload::MAX_TRANSFER_BATCH_SIZE,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc};

#[cfg(feature = "nccl")]
use cudarc::nccl::sys::ncclComm_t;

/// Transfer execution mode for distributed workers
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferMode {
    /// Each rank manages its own shard independently (default)
    #[default]
    Sharded,
    /// All ranks replicate Device data via NCCL broadcast
    Replicated,
}

/// Thread-safe wrapper for NCCL communicator handle.
///
/// # Safety
/// NCCL communicators are thread-safe once created. All NCCL operations using the same
/// communicator will be serialized internally by NCCL. The raw pointer is safe to send
/// between threads as long as the communicator is not destroyed while in use.
#[cfg(feature = "nccl")]
#[derive(Clone, Copy)]
pub struct NcclCommHandle(ncclComm_t);

#[cfg(feature = "nccl")]
impl NcclCommHandle {
    /// Create a new NcclCommHandle from a raw ncclComm_t.
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `comm` is a valid NCCL communicator
    /// - The communicator will not be destroyed while this handle exists
    pub unsafe fn new(comm: ncclComm_t) -> Self {
        Self(comm)
    }

    /// Get the raw ncclComm_t handle.
    pub fn as_raw(&self) -> ncclComm_t {
        self.0
    }
}

// Safety: NCCL communicators are thread-safe once created
#[cfg(feature = "nccl")]
unsafe impl Send for NcclCommHandle {}
#[cfg(feature = "nccl")]
unsafe impl Sync for NcclCommHandle {}

/// Inner NCCL configuration (only available with nccl feature)
#[cfg(feature = "nccl")]
#[derive(Clone, Copy)]
struct NcclConfigInner {
    comm: NcclCommHandle,
    rank: i32,
    world_size: i32,
}

/// Transfer mode configuration for replicated transfers.
/// Always available regardless of NCCL feature - use is_enabled() to check.
#[derive(Clone, Copy, Default)]
pub struct NcclConfig {
    #[cfg(feature = "nccl")]
    inner: Option<NcclConfigInner>,
    #[cfg(not(feature = "nccl"))]
    _phantom: (),
}

impl NcclConfig {
    /// Create a disabled/empty config (sharded mode)
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Create an enabled config for replicated mode (only with nccl feature)
    ///
    /// # Safety
    /// The caller must ensure that:
    /// - `comm` is a valid NCCL communicator
    /// - The communicator will not be destroyed while this config exists
    #[cfg(feature = "nccl")]
    pub unsafe fn enabled(comm: ncclComm_t, rank: i32, world_size: i32) -> Self {
        Self {
            inner: Some(NcclConfigInner {
                comm: NcclCommHandle::new(comm),
                rank,
                world_size,
            }),
        }
    }

    /// Returns true if NCCL is enabled and configured
    pub fn is_enabled(&self) -> bool {
        #[cfg(feature = "nccl")]
        {
            self.inner.is_some()
        }
        #[cfg(not(feature = "nccl"))]
        {
            false
        }
    }

    /// Get rank (panics if not enabled)
    pub fn rank(&self) -> i32 {
        #[cfg(feature = "nccl")]
        {
            self.inner.as_ref().expect("NCCL not enabled").rank
        }
        #[cfg(not(feature = "nccl"))]
        {
            panic!("NCCL feature not enabled")
        }
    }

    /// Get world size (panics if not enabled)
    pub fn world_size(&self) -> i32 {
        #[cfg(feature = "nccl")]
        {
            self.inner.as_ref().expect("NCCL not enabled").world_size
        }
        #[cfg(not(feature = "nccl"))]
        {
            panic!("NCCL feature not enabled")
        }
    }

    /// Get the NCCL communicator handle (panics if not enabled)
    #[cfg(feature = "nccl")]
    pub fn comm(&self) -> NcclCommHandle {
        self.inner.as_ref().expect("NCCL not enabled").comm
    }
}

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        Self {
            max_batch_size: MAX_TRANSFER_BATCH_SIZE,
        }
    }

    pub async fn execute_batched_transfer(
        &self,
        handler: &BlockTransferHandler,
        request: BlockTransferRequest,
    ) -> Result<()> {
        // In replicated mode, execute sequentially (all ranks must participate together)
        // to ensure proper NCCL collective synchronization
        if handler.transfer_mode() == TransferMode::Replicated {
            return handler.execute_transfer_direct(request).await;
        }

        let blocks = request.blocks();
        let num_blocks = blocks.len();

        if num_blocks <= self.max_batch_size {
            return handler.execute_transfer_direct(request).await;
        }

        let batches = blocks.chunks(self.max_batch_size);

        let batch_futures: Vec<_> = batches
            .map(|batch| {
                let batch_request = BlockTransferRequest {
                    from_pool: *request.from_pool(),
                    to_pool: *request.to_pool(),
                    blocks: batch.to_vec(),
                    connector_req: None,
                };
                handler.execute_transfer_direct(batch_request)
            })
            .collect();

        // Execute all batches concurrently
        tracing::debug!("Executing {} batches concurrently", batch_futures.len());

        match try_join_all(batch_futures).await {
            Ok(_) => Ok(()),
            Err(e) => {
                tracing::error!("Batched connector transfer failed: {}", e);
                Err(e)
            }
        }
    }
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
#[derive(Clone)]
pub struct BlockTransferHandler {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    /// Transfer mode: sharded (default) or replicated
    transfer_mode: TransferMode,
    /// NCCL config (required for replicated mode)
    nccl_config: NcclConfig,
}

impl BlockTransferHandler {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        nccl_config: NcclConfig,
    ) -> Result<Self> {
        let transfer_mode = if nccl_config.is_enabled() {
            TransferMode::Replicated
        } else {
            TransferMode::Sharded
        };

        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            transfer_mode,
            nccl_config,
        })
    }

    /// Returns the transfer mode (sharded or replicated)
    pub fn transfer_mode(&self) -> TransferMode {
        self.transfer_mode
    }

    fn get_local_data<S: Storage>(
        blocks: Option<Vec<LocalBlock<S, BasicMetadata>>>,
    ) -> Option<LocalBlockDataList<S>> {
        blocks.map(|blocks| {
            blocks
                .into_iter()
                .map(|b| {
                    let block_data = b.block_data() as &dyn Any;

                    block_data
                        .downcast_ref::<LocalBlockData<S>>()
                        .unwrap()
                        .clone()
                })
                .collect()
        })
    }

    /// Initiate a transfer between two pools.
    async fn begin_transfer<Source, Target>(
        &self,
        source_pool_list: &Option<LocalBlockDataList<Source>>,
        target_pool_list: &Option<LocalBlockDataList<Target>>,
        request: BlockTransferRequest,
    ) -> Result<tokio::sync::oneshot::Receiver<()>>
    where
        Source: Storage + NixlDescriptor,
        Target: Storage + NixlDescriptor,
        // Check that the source block is readable, local, and writable to the target block.
        LocalBlockData<Source>:
            ReadableBlock<StorageType = Source> + Local + WriteToStrategy<LocalBlockData<Target>>,
        // Check that the target block is writable.
        LocalBlockData<Target>: WritableBlock<StorageType = Target>,
        LocalBlockData<Source>: BlockDataProvider<Locality = locality::Local>,
        LocalBlockData<Target>: BlockDataProviderMut<Locality = locality::Local>,
    {
        let Some(source_pool_list) = source_pool_list else {
            return Err(anyhow::anyhow!("Source pool manager not initialized"));
        };
        let Some(target_pool_list) = target_pool_list else {
            return Err(anyhow::anyhow!("Target pool manager not initialized"));
        };

        // Extract the `from` and `to` indices from the request.
        let source_idxs = request.blocks().iter().map(|(from, _)| *from);
        let target_idxs = request.blocks().iter().map(|(_, to)| *to);

        // Get the blocks corresponding to the indices.
        let sources: Vec<LocalBlockData<Source>> = source_idxs
            .map(|idx| source_pool_list[idx].clone())
            .collect();
        let mut targets: Vec<LocalBlockData<Target>> = target_idxs
            .map(|idx| target_pool_list[idx].clone())
            .collect();

        // Perform the transfer, and return the notifying channel.
        match sources.write_to(&mut targets, self.context.clone()) {
            Ok(channel) => Ok(channel),
            Err(e) => {
                tracing::error!("Failed to write to blocks: {:?}", e);
                Err(e.into())
            }
        }
    }

    /// Execute transfer with batching to prevent resource exhaustion
    pub async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    /// Execute transfer directly without batching (used by the batcher)
    pub async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        match self.transfer_mode {
            TransferMode::Sharded => self.execute_transfer_spmd_sharded(request).await,
            #[cfg(feature = "nccl")]
            TransferMode::Replicated => self.execute_transfer_spmd_replicated(request).await,
            #[cfg(not(feature = "nccl"))]
            TransferMode::Replicated => {
                Err(anyhow::anyhow!("Replicated mode requires NCCL feature"))
            }
        }
    }

    /// Execute transfer using sharded mode (each rank manages its own shard independently)
    async fn execute_transfer_spmd_sharded(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing sharded transfer of {} blocks from {:?} to {:?}",
            request.blocks().len(),
            request.from_pool(),
            request.to_pool()
        );

        tracing::debug!("request: {request:#?}");

        let notify = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => self.begin_transfer(&self.device, &self.host, request).await,
            (Device, Disk) => self.begin_transfer(&self.device, &self.disk, request).await,
            (Host, Device) => self.begin_transfer(&self.host, &self.device, request).await,
            (Host, Disk) => self.begin_transfer(&self.host, &self.disk, request).await,
            (Disk, Device) => self.begin_transfer(&self.disk, &self.device, request).await,
            _ => {
                return Err(anyhow::anyhow!("Invalid transfer type."));
            }
        }?;

        notify.await?;
        Ok(())
    }

    /// Execute transfer using replicated mode (NCCL broadcast for Device blocks)
    #[cfg(feature = "nccl")]
    async fn execute_transfer_spmd_replicated(&self, request: BlockTransferRequest) -> Result<()> {
        assert!(
            self.nccl_config.is_enabled(),
            "NCCL config required for replicated mode"
        );
        let rank = self.nccl_config.rank();
        let is_rank0 = rank == 0;
        let use_bcast = request.to_pool() == &Device && request.from_pool() != &Device;

        if use_bcast {
            tracing::info!(
                "NCCL replicated transfer: {} blocks from {:?} to {:?}, rank={}, \
                 rank0 will load from storage then broadcast to all GPUs",
                request.blocks().len(),
                request.from_pool(),
                request.to_pool(),
                rank
            );
        } else {
            tracing::debug!(
                "Replicated transfer: {} blocks from {:?} to {:?} (rank={}, bcast={})",
                request.blocks().len(),
                request.from_pool(),
                request.to_pool(),
                rank,
                use_bcast
            );
        }

        // Device → Device: all ranks do local transfer (no broadcast)
        if request.from_pool() == &Device && request.to_pool() == &Device {
            return self.execute_transfer_spmd_sharded(request).await;
        }

        // Non-rank0 with no broadcast needed: no-op
        if !is_rank0 && !use_bcast {
            return Ok(());
        }

        // Rank 0 does the actual copy
        if is_rank0 {
            let notify = match (request.from_pool(), request.to_pool()) {
                (Device, Host) => {
                    self.begin_transfer(&self.device, &self.host, request.clone())
                        .await
                }
                (Device, Disk) => {
                    self.begin_transfer(&self.device, &self.disk, request.clone())
                        .await
                }
                (Host, Device) => {
                    self.begin_transfer(&self.host, &self.device, request.clone())
                        .await
                }
                (Host, Disk) => {
                    self.begin_transfer(&self.host, &self.disk, request.clone())
                        .await
                }
                (Disk, Device) => {
                    self.begin_transfer(&self.disk, &self.device, request.clone())
                        .await
                }
                _ => {
                    return Err(anyhow::anyhow!("Invalid transfer type."));
                }
            }?;
            notify.await?;
        }

        // Broadcast Device blocks if needed (all ranks participate)
        if use_bcast {
            self.broadcast_device_blocks(&request).await?;
        }

        Ok(())
    }

    /// Broadcast Device blocks to all ranks using NCCL
    #[cfg(feature = "nccl")]
    async fn broadcast_device_blocks(&self, request: &BlockTransferRequest) -> Result<()> {
        use crate::block_manager::block::transfer::{NcclGroup, bcast_block};

        let device_blocks = self
            .device
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Device blocks required for broadcast"))?;

        // Get raw CUstream from the CudaStream wrapper
        let stream = self.context.stream().cu_stream();
        let comm = self.nccl_config.comm();

        // Get destination block indices (the Device blocks to broadcast)
        let dst_indices: Vec<usize> = request.blocks().iter().map(|(_, to)| *to).collect();

        let rank = self.nccl_config.rank();
        let world_size = self.nccl_config.world_size();
        tracing::info!(
            "NCCL broadcast starting: rank={}/{}, num_blocks={}, block_indices={:?}",
            rank,
            world_size,
            dst_indices.len(),
            dst_indices
        );

        // Create NCCL group and broadcast all blocks
        let group = unsafe { NcclGroup::new()? };

        for &block_idx in &dst_indices {
            let block = &device_blocks[block_idx];
            unsafe {
                bcast_block(block, 0, comm.as_raw(), stream)?;
            }
        }

        drop(group); // Ends the NCCL group, submits operations

        // Synchronize: wait for all NCCL operations to complete on the stream
        let (tx, rx) = tokio::sync::oneshot::channel();
        self.context.cuda_event(tx)?;
        rx.await
            .map_err(|_| anyhow::anyhow!("CUDA event channel closed"))?;

        tracing::info!(
            "NCCL broadcast completed: rank={}/{}, num_blocks={}",
            rank,
            world_size,
            dst_indices.len()
        );

        Ok(())
    }
}

#[async_trait]
impl Handler for BlockTransferHandler {
    async fn handle(&self, mut message: MessageHandle) -> Result<()> {
        if message.data.len() != 1 {
            return Err(anyhow::anyhow!(
                "Block transfer request must have exactly one data element"
            ));
        }

        let mut request: BlockTransferRequest = serde_json::from_slice(&message.data[0])?;

        let result = if let Some(req) = request.connector_req.take() {
            let operation_id = req.uuid;

            tracing::debug!(
                request_id = %req.request_id,
                operation_id = %operation_id,
                "scheduling transfer"
            );

            let client = self
                .scheduler_client
                .as_ref()
                .expect("scheduler client is required")
                .clone();

            let handle = client.schedule_transfer(req).await?;

            // we don't support cancellation yet
            assert_eq!(handle.scheduler_decision(), SchedulingDecision::Execute);

            match self.execute_transfer(request).await {
                Ok(_) => {
                    handle.mark_complete(Ok(())).await;
                    Ok(())
                }
                Err(e) => {
                    handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await;
                    Err(e)
                }
            }
        } else {
            self.execute_transfer(request).await
        };

        // we always ack regardless of if we error or not
        message.ack().await?;

        // the error may trigger a cancellation
        result
    }
}
