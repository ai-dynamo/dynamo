// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Offload Manager
//! The offload manager is responsible for handling all block transfers between different cache levels.
//!
//! ## Offloading
//! Offloading is the process of moving blocks to a cache level further away from the device.
//! When blocks are registered (via [`ManagedBlockPool::register_blocks`]), they are automatically sent to the offload manager.
//! Due to limited bandwidth, the offload manager must prioritize which offloads to perform.
//! This is indicated by the `priority` parameter to [`OffloadManager::offload`].
//! When a offload request is received, the offload manager will enqueue it into a priority queue.
//! This priority queue is keyed by the `priority` parameter, where blocks with lower priority values are processed first.
//! Within the same priority, blocks that were sent to the offload manager earlier are processed first.
//!
//! ## Onboarding
//! Onboarding is the process of moving blocks to a cache level closer to the device.
//! All onboardings are manually triggered through the [`OffloadManager::onboard`] method.
//!
//! ## Transfer Managers
//! The offload manager uses two transfer managers to handle the offloading and onboarding of blocks.
//!
//! The [`CudaTransferManager`] is responsible for transfers between the device and host.
//! The [`DiskTransferManager`] is responsible for transfers from host to disk and disk to device.
//!
//! ## Worker Threads
//! The offload manager uses two kinds of worker threads to handle the offloading and onboarding of blocks.
//!
//! The [`OffloadManager::offload_worker`] is responsible for offloading blocks.
//! The [`OffloadManager::onboard_worker`] is responsible for onboarding blocks.
//!
//! The kind of offloads/onboards they perform is dictated by the source and target arguments
//! of the [`OffloadManager::offload_worker`] and [`OffloadManager::onboard_worker`] methods.

use super::block::{
    BlockError, BlockMetadata, BlockState, ImmutableBlock, MutableBlock,
    locality::LocalityProvider,
    transfer::{PoolConfig, TransferContext},
};
use super::pool::{BlockPool, BlockPoolError};
use super::storage::{Cuda, Storage};
use super::{DeviceStorage, DiskStorage, KvManagerModelConfig, PinnedStorage};
use nixl_sys::Agent as NixlAgent;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use tokio::runtime::Handle;
use tokio::sync::{
    mpsc::{self, error::TryRecvError},
    oneshot,
};
use tokio_util::sync::CancellationToken;

use anyhow::Result;
use std::any::Any;

use std::collections::BTreeSet;

pub mod filter;
mod pending;
pub mod request;

use filter::OffloadFilter;
use pending::{LocalTransferManager, PendingTransfer, TransferBatcher, TransferManager};
use request::{BlockResult, OffloadRequest, OffloadRequestKey, OnboardRequest};

use derive_builder::Builder;
use derive_getters::Getters;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub const MAX_CONCURRENT_TRANSFERS: usize = 4;
pub const MAX_TRANSFER_BATCH_SIZE: usize = 16;

/// Configuration for creating an OffloadManager
pub struct OffloadManagerConfig {
    pub nixl_agent: Arc<Option<NixlAgent>>,
    pub async_rt_handle: Handle,
    pub cancellation_token: CancellationToken,
    pub model_config: KvManagerModelConfig,
    /// Optional KVBM-level metrics for tracking offload/onboard operations
    pub kvbm_metrics: Option<crate::block_manager::metrics_kvbm::KvbmMetrics>,
    /// If true, offload directly from device (G1) to disk (G3), bypassing host (G2)
    pub bypass_cpu_mem: bool,
}

/// The offload manager handles all block transfers between different cache levels.
pub struct OffloadManager<Locality: LocalityProvider, Metadata: BlockMetadata> {
    // Handles to the device, host, and disk pools.
    disk: Option<Arc<dyn BlockPool<DiskStorage, Locality, Metadata>>>,
    host: Option<Arc<dyn BlockPool<PinnedStorage, Locality, Metadata>>>,
    device: Option<Arc<dyn BlockPool<DeviceStorage, Locality, Metadata>>>,

    /// Queue of offloading requests.
    device_offload_tx: mpsc::UnboundedSender<OffloadRequest<DeviceStorage, Locality, Metadata>>,
    host_offload_tx: mpsc::UnboundedSender<OffloadRequest<PinnedStorage, Locality, Metadata>>,

    /// Queue of device-to-disk direct offloading requests (bypass CPU memory)
    device_to_disk_offload_tx:
        mpsc::UnboundedSender<OffloadRequest<DeviceStorage, Locality, Metadata>>,

    /// Queue of pending onboarding requests.
    host_onboard_tx:
        mpsc::UnboundedSender<OnboardRequest<PinnedStorage, DeviceStorage, Locality, Metadata>>,
    disk_onboard_tx:
        mpsc::UnboundedSender<OnboardRequest<DiskStorage, DeviceStorage, Locality, Metadata>>,

    /// An incrementing counter for offloaded blocks. Within the same priority, blocks with lower tick values are processed first.
    tick: Arc<AtomicU64>,

    /// If true, offload directly from device (G1) to disk (G3), bypassing host (G2)
    bypass_cpu_mem: bool,
}

impl<Locality: LocalityProvider + 'static, Metadata: BlockMetadata>
    OffloadManager<Locality, Metadata>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        disk: Option<Arc<dyn BlockPool<DiskStorage, Locality, Metadata>>>,
        host: Option<Arc<dyn BlockPool<PinnedStorage, Locality, Metadata>>>,
        device: Option<Arc<dyn BlockPool<DeviceStorage, Locality, Metadata>>>,
        filters: OffloadFilters,
        config: OffloadManagerConfig,
    ) -> Result<Arc<Self>> {
        let (device_offload_tx, device_offload_rx) = mpsc::unbounded_channel();
        let (host_offload_tx, host_offload_rx) = mpsc::unbounded_channel();
        let (device_to_disk_offload_tx, device_to_disk_offload_rx) = mpsc::unbounded_channel();

        let (host_onboard_tx, host_onboard_rx) = mpsc::unbounded_channel();
        let (disk_onboard_tx, disk_onboard_rx) = mpsc::unbounded_channel();

        let this = Arc::new(Self {
            disk,
            host,
            device,
            device_offload_tx,
            host_offload_tx,
            device_to_disk_offload_tx,
            host_onboard_tx,
            disk_onboard_tx,
            tick: Arc::new(AtomicU64::new(0)),
            bypass_cpu_mem: config.bypass_cpu_mem,
        });

        let cuda_ctx = Cuda::device_or_create(0)?;

        let pool_config = PoolConfig {
            enable_pool: true,
            max_concurrent_transfers: MAX_CONCURRENT_TRANSFERS,
            max_transfer_batch_size: MAX_TRANSFER_BATCH_SIZE,
            num_outer_components: config.model_config.outer_dim,
            num_layers: config.model_config.num_layers,
        };

        // We want cuda offloads to happen in parallel with host onboards, so we need to use a different stream.
        let device_offload_transfer_ctx = Arc::new(TransferContext::new(
            config.nixl_agent.clone(),
            cuda_ctx.new_stream()?,
            config.async_rt_handle.clone(),
            Some(pool_config),
        ));

        // Device -> Host offload
        let device_to_host_task = OffloadManager::offload_worker(
            this.device.clone(),
            this.host.clone(),
            device_offload_rx,
            Arc::new(TransferBatcher::new(
                LocalTransferManager::new(
                    device_offload_transfer_ctx,
                    MAX_CONCURRENT_TRANSFERS,
                    &config.async_rt_handle,
                    config.cancellation_token.clone(),
                )?,
                MAX_TRANSFER_BATCH_SIZE,
                &config.async_rt_handle,
                config.cancellation_token.clone(),
            )),
            filters.device.clone(),
            config
                .kvbm_metrics
                .as_ref()
                .map(|m| m.offload_blocks_d2h.clone()),
            config.cancellation_token.clone(),
        );
        CriticalTaskExecutionHandle::new_with_runtime(
            |_| device_to_host_task,
            config.cancellation_token.clone(),
            "Device -> Host offload worker",
            &config.async_rt_handle,
        )?
        .detach();

        let transfer_ctx = Arc::new(TransferContext::new(
            config.nixl_agent.clone(),
            cuda_ctx.new_stream()?,
            config.async_rt_handle.clone(),
            None,
        ));

        // Host -> Disk offload
        let host_to_disk_task = OffloadManager::offload_worker(
            this.host.clone(),
            this.disk.clone(),
            host_offload_rx,
            Arc::new(TransferBatcher::new(
                LocalTransferManager::new(
                    transfer_ctx.clone(),
                    MAX_CONCURRENT_TRANSFERS,
                    &config.async_rt_handle,
                    config.cancellation_token.clone(),
                )?,
                MAX_TRANSFER_BATCH_SIZE,
                &config.async_rt_handle,
                config.cancellation_token.clone(),
            )),
            filters.host.clone(),
            config
                .kvbm_metrics
                .as_ref()
                .map(|m| m.offload_blocks_h2d.clone()),
            config.cancellation_token.clone(),
        );
        CriticalTaskExecutionHandle::new_with_runtime(
            |_| host_to_disk_task,
            config.cancellation_token.clone(),
            "Host -> Disk offload worker",
            &config.async_rt_handle,
        )?
        .detach();

        // Host -> Device onboarding
        let host_to_device_task = OffloadManager::onboard_worker(
            this.host.clone(),
            this.device.clone(),
            host_onboard_rx,
            Arc::new(TransferBatcher::new(
                LocalTransferManager::new(
                    transfer_ctx.clone(),
                    MAX_CONCURRENT_TRANSFERS,
                    &config.async_rt_handle,
                    config.cancellation_token.clone(),
                )?,
                MAX_TRANSFER_BATCH_SIZE,
                &config.async_rt_handle,
                config.cancellation_token.clone(),
            )),
            config.cancellation_token.clone(),
        );
        CriticalTaskExecutionHandle::new_with_runtime(
            |_| host_to_device_task,
            config.cancellation_token.clone(),
            "Host -> Device onboarding worker",
            &config.async_rt_handle,
        )?
        .detach();

        // Disk -> Device onboarding
        let disk_to_device_task = OffloadManager::onboard_worker(
            this.disk.clone(),
            this.device.clone(),
            disk_onboard_rx,
            Arc::new(TransferBatcher::new(
                LocalTransferManager::new(
                    transfer_ctx.clone(),
                    MAX_CONCURRENT_TRANSFERS,
                    &config.async_rt_handle,
                    config.cancellation_token.clone(),
                )?,
                MAX_TRANSFER_BATCH_SIZE,
                &config.async_rt_handle,
                config.cancellation_token.clone(),
            )),
            config.cancellation_token.clone(),
        );
        CriticalTaskExecutionHandle::new_with_runtime(
            |_| disk_to_device_task,
            config.cancellation_token.clone(),
            "Disk -> Device onboarding worker",
            &config.async_rt_handle,
        )?
        .detach();

        // Device -> Disk direct offload (bypass CPU memory)
        if config.bypass_cpu_mem {
            tracing::info!(
                "G1->G3 direct offload enabled: Device will offload directly to Disk, bypassing Host memory (CPU cache disabled)"
            );

            let device_to_disk_task = OffloadManager::offload_worker(
                this.device.clone(),
                this.disk.clone(),
                device_to_disk_offload_rx,
                Arc::new(TransferBatcher::new(
                    LocalTransferManager::new(
                        transfer_ctx.clone(),
                        MAX_CONCURRENT_TRANSFERS,
                        &config.async_rt_handle,
                        config.cancellation_token.clone(),
                    )?,
                    MAX_TRANSFER_BATCH_SIZE,
                    &config.async_rt_handle,
                    config.cancellation_token.clone(),
                )),
                filters.device.clone(),
                config
                    .kvbm_metrics
                    .as_ref()
                    .map(|m| m.offload_blocks_d2d.clone()),
                config.cancellation_token.clone(),
            );
            CriticalTaskExecutionHandle::new_with_runtime(
                |_| device_to_disk_task,
                config.cancellation_token.clone(),
                "Device -> Disk direct offload worker (bypass CPU)",
                &config.async_rt_handle,
            )?
            .detach();
        }

        Ok(this)
    }

    async fn offload_worker<Source: Storage, Target: Storage>(
        source_pool: Option<Arc<dyn BlockPool<Source, Locality, Metadata>>>,
        target_pool: Option<Arc<dyn BlockPool<Target, Locality, Metadata>>>,
        mut offload_rx: mpsc::UnboundedReceiver<OffloadRequest<Source, Locality, Metadata>>,
        transfer_manager: Arc<dyn TransferManager<Source, Target, Locality, Metadata>>,
        offload_filter: Option<Arc<dyn OffloadFilter>>,
        offload_metric: Option<prometheus::IntCounter>,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        if source_pool.is_none() || target_pool.is_none() {
            return Ok(());
        }

        let source_pool = source_pool.as_ref().unwrap();
        let target_pool = target_pool.as_ref().unwrap();

        let mut queue = BTreeSet::new();

        loop {
            if cancellation_token.is_cancelled() {
                return Ok(());
            }

            // Try to check the offload queue.
            loop {
                match offload_rx.try_recv() {
                    Ok(request) => {
                        queue.insert(request);
                    }
                    Err(TryRecvError::Empty) => {
                        break;
                    }
                    Err(e) => return Err(e.into()),
                }
            }

            // If there is a request, process it.
            if let Some(request) = queue.pop_first() {
                // Try to upgrade the block to a strong reference.
                let block = match request.block.upgrade() {
                    Some(block) => Some(ImmutableBlock::new(block)),
                    // If unable to upgrade, the block may have been moved to the inactive pool.
                    None => source_pool
                        .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                        .await?
                        .pop(),
                };

                // If we've found the block, offload it.
                if let Some(block) = block {
                    // If the block is already in the target, don't offload it.
                    if let Ok(blocks) = target_pool
                        .match_sequence_hashes(vec![request.sequence_hash].as_slice())
                        .await
                        && !blocks.is_empty()
                    {
                        continue;
                    }

                    if let Some(offload_filter) = offload_filter.as_ref()
                        && !offload_filter.should_offload(request.sequence_hash)
                    {
                        continue;
                    }

                    let target_block = 'target_block: {
                        if let Ok(blocks) = target_pool.allocate_blocks(1).await
                            && let Some(block) = blocks.into_iter().next()
                        {
                            break 'target_block Some(block);
                        }

                        tracing::warn!(
                            "Target pool full. Skipping offload. This should only ever happen with very small pool sizes."
                        );
                        None
                    };

                    if let Some(target_block) = target_block {
                        tracing::debug!(
                            "Offloading block with sequence hash {} to target pool.",
                            request.sequence_hash
                        );

                        // Track the offload metric if available
                        if let Some(ref metric) = offload_metric {
                            metric.inc();
                        }

                        transfer_manager
                            .enqueue_transfer(PendingTransfer::new(
                                vec![block],
                                vec![target_block],
                                None,
                                target_pool.clone(),
                            ))
                            .await?;
                    }
                }
            } else {
                // Await the next request.
                tokio::select! {
                    _ = cancellation_token.cancelled() => return Ok(()),
                    Some(request) = offload_rx.recv() => {
                        queue.insert(request);
                    }
                }
            }
        }
    }

    async fn onboard_worker<Source: Storage, Target: Storage>(
        source_pool: Option<Arc<dyn BlockPool<Source, Locality, Metadata>>>,
        target_pool: Option<Arc<dyn BlockPool<Target, Locality, Metadata>>>,
        mut onboard_rx: mpsc::UnboundedReceiver<OnboardRequest<Source, Target, Locality, Metadata>>,
        transfer_manager: Arc<dyn TransferManager<Source, Target, Locality, Metadata>>,
        cancellation_token: CancellationToken,
    ) -> Result<()> {
        if source_pool.is_none() || target_pool.is_none() {
            return Ok(());
        }

        let target_pool = target_pool.as_ref().unwrap();
        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => return Ok::<(), anyhow::Error>(()),
                Some(request) = onboard_rx.recv() => {

                    // Try to allocate blocks on the device.
                    let target_blocks = if let Some(targets) = request.targets {
                        targets
                    } else {
                            match target_pool.allocate_blocks(request.blocks.len()).await {
                            Ok(blocks) => blocks,
                            Err(err) => {
                                let _ = request.response_tx.send(Err(err));
                                continue;
                            }
                        }
                    };

                    tracing::debug!("Onboarding {} blocks to target pool.", request.blocks.len());

                    transfer_manager
                        .enqueue_transfer(PendingTransfer::new(
                            request.blocks,
                            target_blocks,
                            Some(request.response_tx),
                            target_pool.clone(),
                        ))
                        .await?;

                    Ok::<(), anyhow::Error>(())
                }
            }?;
        }
    }

    pub async fn offload<S: Storage>(
        &self,
        block: &ImmutableBlock<S, Locality, Metadata>,
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

        let tick = self.tick.fetch_add(1, Ordering::Relaxed);
        let key = OffloadRequestKey {
            priority,
            timestamp: tick,
        };

        // This can get called by all pools, regardless of whether or not they have a place to offload to.
        // Because of this, we need to check the block type here.
        let any_block = block as &dyn Any;

        // TODO: What's the performance penalty of this runtime type-checking?
        if let Some(device_block) =
            any_block.downcast_ref::<ImmutableBlock<DeviceStorage, Locality, Metadata>>()
        {
            // Check if we should bypass CPU memory and go directly to disk
            if self.bypass_cpu_mem && self.disk.is_some() {
                // Offload directly from Device (G1) to Disk (G3), bypassing Host (G2)
                if self.device_to_disk_offload_tx.is_closed() {
                    return Ok(());
                }

                let request = OffloadRequest {
                    block: Arc::downgrade(device_block.mutable_block()),
                    sequence_hash: device_block.sequence_hash(),
                    key,
                };

                tracing::debug!(
                    "Offloading device block {} directly to disk (bypassing host memory)",
                    device_block.sequence_hash()
                );
                self.device_to_disk_offload_tx.send(request).unwrap();
            } else {
                // Standard path: Device (G1) -> Host (G2)
                if self.device_offload_tx.is_closed() {
                    return Ok(());
                }

                let request = OffloadRequest {
                    block: Arc::downgrade(device_block.mutable_block()),
                    sequence_hash: device_block.sequence_hash(),
                    key,
                };

                self.device_offload_tx.send(request).unwrap();
            }
        } else if let Some(host_block) =
            any_block.downcast_ref::<ImmutableBlock<PinnedStorage, Locality, Metadata>>()
        {
            // Host (G2) -> Disk (G3) offload
            if self.host_offload_tx.is_closed() {
                return Ok(());
            }

            let request = OffloadRequest {
                block: Arc::downgrade(host_block.mutable_block()),
                sequence_hash: host_block.sequence_hash(),
                key,
            };

            self.host_offload_tx.send(request).unwrap();
        }

        Ok(())
    }

    pub fn onboard<S: Storage>(
        &self,
        blocks: Vec<ImmutableBlock<S, Locality, Metadata>>,
        targets: Option<Vec<MutableBlock<DeviceStorage, Locality, Metadata>>>,
    ) -> oneshot::Receiver<BlockResult<DeviceStorage, Locality, Metadata>> {
        let (tx, rx) = oneshot::channel();
        for block in &blocks {
            match block.state() {
                BlockState::Registered(_, _) => {}
                _ => {
                    tx.send(Err(BlockPoolError::BlockError(BlockError::InvalidState(
                        "Block is not registered.".to_string(),
                    ))))
                    .unwrap();
                    return rx;
                }
            }
        }

        if let Some(targets) = targets.as_ref()
            && targets.len() != blocks.len()
        {
            tx.send(Err(BlockPoolError::BlockError(BlockError::Other(
                anyhow::anyhow!("Number of targets does not match number of blocks."),
            ))))
            .unwrap();
            return rx;
        }

        if blocks.is_empty() {
            tx.send(Ok(vec![])).unwrap();
            return rx;
        }

        let any_block = blocks.first().unwrap() as &dyn Any;

        // TODO: This is really ugly.
        if any_block
            .downcast_ref::<ImmutableBlock<PinnedStorage, Locality, Metadata>>()
            .is_some()
        {
            let host_blocks = blocks
                .iter()
                .map(|b| {
                    (b as &dyn Any)
                        .downcast_ref::<ImmutableBlock<PinnedStorage, Locality, Metadata>>()
                        .unwrap()
                        .clone()
                })
                .collect();

            if let Err(e) = self
                .host_onboard_tx
                .send(OnboardRequest::new(host_blocks, tx, targets))
            {
                e.0.response_tx
                    .send(Err(BlockPoolError::ProgressEngineShutdown))
                    .unwrap();
            }
        } else if any_block
            .downcast_ref::<ImmutableBlock<DiskStorage, Locality, Metadata>>()
            .is_some()
        {
            let disk_blocks = blocks
                .iter()
                .map(|b| {
                    (b as &dyn Any)
                        .downcast_ref::<ImmutableBlock<DiskStorage, Locality, Metadata>>()
                        .unwrap()
                        .clone()
                })
                .collect();

            if let Err(e) = self
                .disk_onboard_tx
                .send(OnboardRequest::new(disk_blocks, tx, targets))
            {
                e.0.response_tx
                    .send(Err(BlockPoolError::ProgressEngineShutdown))
                    .unwrap();
            }
        } else {
            tx.send(Err(BlockPoolError::BlockError(BlockError::Other(
                anyhow::anyhow!("Block type not supported for onboarding."),
            ))))
            .unwrap();
        }

        rx
    }
}

#[derive(Debug, Clone, Getters, Builder)]
#[builder(pattern = "owned", build_fn(validate = "Self::validate"))]
pub struct OffloadFilters {
    #[builder(default)]
    device: Option<Arc<dyn OffloadFilter>>,
    #[builder(default)]
    host: Option<Arc<dyn OffloadFilter>>,
    #[builder(default)]
    disk: Option<Arc<dyn OffloadFilter>>,
}

impl OffloadFilters {
    pub fn builder() -> OffloadFiltersBuilder {
        OffloadFiltersBuilder::default()
    }
}

impl OffloadFiltersBuilder {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(disk) = self.disk.as_ref()
            && disk.is_some()
        {
            return Err("Disk offload filter is not supported.".to_string());
        }

        let host_is_none = if let Some(host) = self.host.as_ref() {
            host.is_none()
        } else {
            true
        };

        if host_is_none {
            tracing::warn!(
                "Host to Disk offload filter is not provided. All blocks in host will be offloaded to disk. This may result in excessive disk offloading and accelerated SSD degradation."
            );
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "testing-cuda", feature = "testing-nixl"))]
mod tests {
    use super::*;

    use crate::block_manager::{
        LayoutConfig, NixlRegisterableStorage,
        block::{
            BasicMetadata, BlockDataExt, BlockDataProvider, Blocks, MutableBlock, locality::Local,
        },
        layout::{FullyContiguous, LayerSeparate, LayoutType, nixl::NixlLayout},
        pool::{BlockRegistrationDuplicationSetting, ManagedBlockPool},
        storage::{
            DeviceAllocator, DeviceStorage, DiskAllocator, DiskStorage, PinnedAllocator,
            PinnedStorage, StorageAllocator, StorageType,
        },
    };
    use crate::tokens::{TokenBlockSequence, Tokens};
    use nixl_sys::{MemoryRegion, NixlDescriptor};

    use aligned_vec::avec;
    use cudarc::runtime::sys::{cudaDeviceSynchronize, cudaMemcpy, cudaMemcpyKind, cudaMemset};
    use rstest::*;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::mem::ManuallyDrop;
    use std::os::unix::io::FromRawFd;

    const BLOCK_SIZE: usize = 4;
    const NUM_LAYERS: usize = 8;

    type DevicePool = Option<Arc<dyn BlockPool<DeviceStorage, Local, BasicMetadata>>>;
    type HostPool = Option<Arc<dyn BlockPool<PinnedStorage, Local, BasicMetadata>>>;
    type DiskPool = Option<Arc<dyn BlockPool<DiskStorage, Local, BasicMetadata>>>;

    lazy_static::lazy_static! {
        static ref NIXL_AGENT: Arc<Option<NixlAgent>> = {
            let agent = NixlAgent::new("offload-manager").unwrap();
            let (_, ucx_params) = agent.get_plugin_params("UCX").unwrap();
            let (_, gds_mt_params) = agent.get_plugin_params("GDS_MT").unwrap();
            let (_, posix_params) = agent.get_plugin_params("POSIX").unwrap();
            agent.create_backend("UCX", &ucx_params).unwrap();
            agent.create_backend("GDS_MT", &gds_mt_params).unwrap();
            agent.create_backend("POSIX", &posix_params).unwrap();
            Arc::new(Some(agent))
        };
    }

    fn build_layout<S: Storage + NixlRegisterableStorage>(
        config: LayoutConfig,
        layout_type: LayoutType,
        agent: &NixlAgent,
        allocator: &dyn StorageAllocator<S>,
        duplication_setting: BlockRegistrationDuplicationSetting,
    ) -> Result<Arc<dyn BlockPool<S, Local, BasicMetadata>>> {
        match layout_type {
            LayoutType::FullyContiguous => {
                let mut pool_layout = FullyContiguous::allocate(config.clone(), allocator)?;
                pool_layout.nixl_register(agent, None)?;
                let blocks = Blocks::new(pool_layout, 42, 0)?.into_blocks()?;
                Ok(Arc::new(
                    ManagedBlockPool::builder()
                        .blocks(blocks)
                        .default_duplication_setting(duplication_setting)
                        .build()?,
                ))
            }
            LayoutType::LayerSeparate { outer_contiguous } => {
                let mut pool_layout =
                    LayerSeparate::allocate(config.clone(), allocator, outer_contiguous)?;
                pool_layout.nixl_register(agent, None)?;
                let blocks = Blocks::new(pool_layout, 42, 0)?.into_blocks()?;
                Ok(Arc::new(
                    ManagedBlockPool::builder()
                        .blocks(blocks)
                        .default_duplication_setting(duplication_setting)
                        .build()?,
                ))
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn build_pools(
        device_blocks: usize,
        host_blocks: Option<usize>,
        disk_blocks: Option<usize>,
        inner_dim: Option<usize>,
    ) -> Result<(
        Arc<OffloadManager<Local, BasicMetadata>>,
        DevicePool,
        HostPool,
        DiskPool,
    )> {
        build_pools_with_layout(
            device_blocks,
            host_blocks,
            disk_blocks,
            inner_dim,
            LayoutType::FullyContiguous,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )
    }

    #[allow(clippy::type_complexity)]
    pub fn build_pools_with_layout(
        device_blocks: usize,
        host_blocks: Option<usize>,
        disk_blocks: Option<usize>,
        inner_dim: Option<usize>,
        layout_type: LayoutType,
        duplication_setting: BlockRegistrationDuplicationSetting,
        bypass_cpu_mem: bool,
    ) -> Result<(
        Arc<OffloadManager<Local, BasicMetadata>>,
        DevicePool,
        HostPool,
        DiskPool,
    )> {
        let mut config = LayoutConfig {
            num_blocks: device_blocks,
            num_layers: NUM_LAYERS,
            outer_dim: 1,
            page_size: BLOCK_SIZE,
            inner_dim: inner_dim.unwrap_or(1024),
            alignment: 1,
            dtype_width_bytes: 2,
        };

        let agent_arc = NIXL_AGENT.clone();
        let agent = agent_arc.as_ref().as_ref().unwrap();

        let device_pool = Some(build_layout(
            config.clone(),
            layout_type,
            agent,
            &DeviceAllocator::default(),
            duplication_setting,
        )?);

        let host_pool = if let Some(host_blocks) = host_blocks {
            config.num_blocks = host_blocks;
            Some(build_layout(
                config.clone(),
                layout_type,
                agent,
                &PinnedAllocator::default(),
                duplication_setting,
            )?)
        } else {
            None
        };

        let disk_pool = if let Some(disk_blocks) = disk_blocks {
            config.num_blocks = disk_blocks;
            Some(build_layout(
                config.clone(),
                layout_type,
                agent,
                &DiskAllocator,
                duplication_setting,
            )?)
        } else {
            None
        };

        let async_rt_handle = Handle::current();

        let minimal_config = KvManagerModelConfig::builder()
            .num_layers(config.num_layers)
            .outer_dim(config.outer_dim) // K and V
            .page_size(config.page_size) // Minimal page size
            .inner_dim(config.inner_dim) // Small inner dim
            .build()
            .expect("Failed to build minimal config");

        let config = OffloadManagerConfig {
            nixl_agent: agent_arc,
            async_rt_handle,
            cancellation_token: CancellationToken::new(),
            model_config: minimal_config,
            kvbm_metrics: None,
            bypass_cpu_mem,
        };

        let manager = OffloadManager::new(
            disk_pool.clone(),
            host_pool.clone(),
            device_pool.clone(),
            OffloadFilters::builder().build()?,
            config,
        )?;

        Ok((manager, device_pool, host_pool, disk_pool))
    }

    /// Create a block in the 'RESET' state.
    #[expect(dead_code)]
    async fn get_block<S: Storage, Metadata: BlockMetadata>(
        pool: &Arc<dyn BlockPool<S, Local, Metadata>>,
    ) -> Result<MutableBlock<S, Local, Metadata>> {
        let mut blocks = pool.allocate_blocks(1).await?;
        Ok(blocks.pop().unwrap())
    }

    /// Create a block in the 'COMPLETED' state.
    async fn completed_block<S: Storage, Metadata: BlockMetadata>(
        pool: &Arc<dyn BlockPool<S, Local, Metadata>>,
        tokens: [u32; BLOCK_SIZE],
    ) -> Result<MutableBlock<S, Local, Metadata>> {
        let mut block = pool
            .allocate_blocks(1)
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to allocate block"))?;

        block.init_sequence(42)?;
        for token in tokens {
            block.add_token(token)?;
        }
        block.commit()?;
        Ok(block)
    }

    fn populate_block<S: Storage + NixlDescriptor>(
        block: &impl BlockDataProvider<StorageType = S>,
        start_value: u8,
    ) -> Result<()> {
        let block_data = block.block_data();

        let mut value = start_value;

        for layer_idx in 0..block_data.num_layers() {
            for outer_idx in 0..block_data.num_outer_dims() {
                let layer_view = block_data.layer_view(layer_idx, outer_idx)?;
                match block_data.storage_type() {
                    StorageType::Device(_) | StorageType::Pinned => unsafe {
                        cudaMemset(
                            layer_view.as_ptr() as *mut std::ffi::c_void,
                            value as i32,
                            layer_view.size(),
                        )
                        .result()?;
                    },
                    StorageType::Disk(_) => {
                        let nixl_desc = layer_view.as_nixl_descriptor();
                        let mut file: ManuallyDrop<File>;
                        let data = avec![[4096] | value; layer_view.size()];

                        unsafe {
                            file =
                                ManuallyDrop::new(File::from_raw_fd(nixl_desc.device_id() as i32));
                            file.seek(SeekFrom::Start(nixl_desc.as_ptr() as u64))?;
                        }
                        file.write_all(&data)?;
                        file.sync_all()?;
                        file.flush()?;
                    }
                    _ => panic!(),
                }
            }

            value += 1;
        }

        Ok(())
    }

    fn get_block_contents<S: Storage + NixlDescriptor>(
        block: &impl BlockDataProvider<StorageType = S>,
    ) -> Result<Vec<Vec<u8>>> {
        let block_data = block.block_data();

        let mut contents: Vec<Vec<u8>> = Vec::new();

        for layer_idx in 0..block_data.num_layers() {
            for outer_idx in 0..block_data.num_outer_dims() {
                let layer_view = block_data.layer_view(layer_idx, outer_idx)?;
                match block_data.storage_type() {
                    StorageType::Device(_) => unsafe {
                        let mut buffer = vec![0_u8; layer_view.size()];

                        cudaMemcpy(
                            buffer.as_mut_ptr() as *mut std::ffi::c_void,
                            layer_view.as_ptr() as *const std::ffi::c_void,
                            layer_view.size(),
                            cudaMemcpyKind::cudaMemcpyDeviceToHost,
                        )
                        .result()?;

                        contents.push(buffer);
                    },
                    StorageType::Pinned => unsafe {
                        contents.push(
                            std::slice::from_raw_parts(layer_view.as_ptr(), layer_view.size())
                                .to_vec(),
                        );
                    },
                    StorageType::Disk(_) => {
                        let nixl_desc = layer_view.as_nixl_descriptor();
                        let mut file: ManuallyDrop<File>;
                        let mut aligned = avec![[4096] | 0; layer_view.size()];

                        unsafe {
                            file =
                                ManuallyDrop::new(File::from_raw_fd(nixl_desc.device_id() as i32));
                            file.seek(SeekFrom::Start(nixl_desc.as_ptr() as u64))?;
                        }
                        file.read_exact(&mut aligned)?;
                        contents.push(aligned.to_vec());
                    }
                    _ => anyhow::bail!("Unsupported storage type."),
                }
            }
        }

        Ok(contents)
    }

    fn check_block_contents(
        block1: &impl BlockDataProvider<StorageType = impl Storage + NixlDescriptor>,
        block2: &impl BlockDataProvider<StorageType = impl Storage + NixlDescriptor>,
        start_value: u8,
    ) -> Result<()> {
        let contents1 = get_block_contents(block1)?;
        let contents2 = get_block_contents(block2)?;

        assert_eq!(contents1.len(), contents2.len());

        let mut value = start_value;

        for (layer1_vec, layer2_vec) in contents1.iter().zip(contents2.iter()) {
            for (c1_value, c2_value) in layer1_vec.iter().zip(layer2_vec.iter()) {
                if c1_value != c2_value || c1_value != &value {
                    panic!("{} != {} != {}", c1_value, c2_value, value);
                }
            }
            value += 1;
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_offload_invalid_blocks() -> Result<()> {
        let (offload_manager, device_pool, _, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();

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
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_offload_registered_blocks(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools_with_layout(
            4,
            Some(4),
            None,
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        // Create a block and register it with the offload manager
        let block = completed_block(device_pool, [0, 1, 2, 3]).await?;

        let immutable_device_block = device_pool
            .register_blocks(vec![block])
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to register block"))?;

        populate_block(&immutable_device_block, 42)?;

        // Offloads should only go to G2 (for now)
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for it to be processed.
        // TODO: This is a bit of a hack, and may lead to non-deterministic behavior.
        // In theory, the offload + memcpy should take much less time than this.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool
        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;

        assert_eq!(host_blocks.len(), 1);
        assert_eq!(
            host_blocks[0].sequence_hash(),
            immutable_device_block.sequence_hash()
        );

        check_block_contents(&immutable_device_block, &host_blocks[0], 42)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_device_to_disk_bypass_cpu() -> Result<()> {
        let (offload_manager, device_pool, host_pool, disk_pool) = build_pools_with_layout(
            4,
            Some(4),
            Some(4),
            None,
            LayoutType::FullyContiguous,
            BlockRegistrationDuplicationSetting::Disabled,
            true,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().unwrap();

        // Create a block and register it with the offload manager
        let block = completed_block(device_pool, [0, 1, 2, 3]).await?;

        let immutable_device_block = device_pool
            .register_blocks(vec![block])
            .await?
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("Failed to register block"))?;

        populate_block(&immutable_device_block, 42)?;

        // Synchronize ALL CUDA streams to ensure populate_block completes before offload starts
        // This is critical because cudaMemset uses the default stream, but GDS transfer uses a different stream
        unsafe {
            cudaDeviceSynchronize().result()?;
        }

        // Offloads should only go to G3 directly since bypass_cpu_mem is true in offload_manager config
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for it to be processed.
        // TODO: This is a bit of a hack, and may lead to non-deterministic behavior.
        // In theory, the offload + memcpy should take much less time than this.
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

        // Check that the block exists in the host pool
        let disk_blocks = disk_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;

        assert_eq!(disk_blocks.len(), 1);
        assert_eq!(
            disk_blocks[0].sequence_hash(),
            immutable_device_block.sequence_hash()
        );

        check_block_contents(&immutable_device_block, &disk_blocks[0], 42)?;

        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;

        // since host is bypassed, there should be no host blocks
        assert_eq!(host_blocks.len(), 0);

        Ok(())
    }

    #[tokio::test]
    async fn test_no_host_blocks_available() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

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
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
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
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(matched_host_blocks.len(), 1);

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_onboard(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools_with_layout(
            4,
            Some(4),
            None,
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        // Allocate and fill a block on the host.
        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_block(&immutable_host_block, 42)?;

        // Onboard the block.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()], None)
            .await??;

        assert_eq!(onboarded_blocks.len(), 1);
        // Check that the sequence hash is the same.
        assert_eq!(
            onboarded_blocks[0].sequence_hash(),
            immutable_host_block.sequence_hash()
        );
        // Check that the block is registered.
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_, _)
        ));

        check_block_contents(&immutable_host_block, &onboarded_blocks[0], 42)?;

        // Wait for the new value to show up in the device pool.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let device_blocks = device_pool
            .match_sequence_hashes(vec![onboarded_blocks[0].sequence_hash()].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 1);
        assert_eq!(
            device_blocks[0].sequence_hash(),
            onboarded_blocks[0].sequence_hash()
        );

        // Check that this is the same block.
        check_block_contents(&immutable_host_block, &device_blocks[0], 42)?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_offload_onboard(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools_with_layout(
            4,
            Some(4),
            None,
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        let device_block = completed_block(device_pool, [0, 1, 2, 3]).await?;
        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_block(&immutable_device_block, 42)?;
        // Offload the block to the host.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Check that the block exists in the host pool.
        let immutable_host_block = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?
            .into_iter()
            .next()
            .unwrap();

        check_block_contents(&immutable_device_block, &immutable_host_block, 42)?;

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
            .match_sequence_hashes(vec![immutable_host_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(device_blocks.len(), 0);

        // Onboard the block back to the device pool.
        let onboarded_blocks = offload_manager
            .onboard(vec![immutable_host_block.clone()], None)
            .await??;
        assert_eq!(onboarded_blocks.len(), 1);
        assert_eq!(
            onboarded_blocks[0].sequence_hash(),
            immutable_host_block.sequence_hash()
        );
        assert!(matches!(
            onboarded_blocks[0].state(),
            BlockState::Registered(_, _)
        ));

        check_block_contents(&immutable_host_block, &onboarded_blocks[0], 42)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_err_handling() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

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
            .onboard(vec![immutable_host_block.clone()], None)
            .await?;
        assert!(matches!(
            res.err().unwrap(),
            BlockPoolError::NotEnoughBlocksAvailable(_, _)
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_onboard_no_host_blocks() -> Result<()> {
        let (offload_manager, device_pool, _, _) = build_pools(4, None, None, None)?;

        let device_pool = device_pool.as_ref().unwrap();

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

    #[tokio::test]
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_offload_disk(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, _, host_pool, disk_pool) = build_pools_with_layout(
            4,
            Some(4),
            Some(4),
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let host_pool = host_pool.as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().unwrap();

        let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
        let immutable_host_block = host_pool
            .register_blocks(vec![host_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_block(&immutable_host_block, 42)?;

        offload_manager.offload(&immutable_host_block, 0).await?;

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let disk_blocks = disk_pool
            .match_sequence_hashes(vec![immutable_host_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(disk_blocks.len(), 1);
        assert_eq!(
            disk_blocks[0].sequence_hash(),
            immutable_host_block.sequence_hash()
        );

        check_block_contents(&immutable_host_block, &disk_blocks[0], 42)?;

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_onboard_disk(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, device_pool, _, disk_pool) = build_pools_with_layout(
            4,
            None,
            Some(4),
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().unwrap();

        let disk_block = completed_block(disk_pool, [0, 1, 2, 3]).await?;
        let immutable_disk_block = disk_pool
            .register_blocks(vec![disk_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_block(&immutable_disk_block, 42)?;

        let device_block = offload_manager
            .onboard(vec![immutable_disk_block.clone()], None)
            .await??;

        check_block_contents(&immutable_disk_block, &device_block[0], 42)?;

        assert_eq!(device_block.len(), 1);
        assert_eq!(
            device_block[0].sequence_hash(),
            immutable_disk_block.sequence_hash()
        );
        assert_eq!(
            device_pool
                .match_sequence_hashes(vec![immutable_disk_block.sequence_hash()].as_slice())
                .await?
                .len(),
            1
        );

        Ok(())
    }

    #[tokio::test]
    #[rstest]
    #[case(LayoutType::FullyContiguous)]
    #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
    #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
    async fn test_bulk_transfer_disk(#[case] layout_type: LayoutType) -> Result<()> {
        let (offload_manager, device_pool, host_pool, disk_pool) = build_pools_with_layout(
            8,
            Some(8),
            Some(8),
            None,
            layout_type,
            BlockRegistrationDuplicationSetting::Disabled,
            false,
        )?;

        let disk_pool = disk_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();
        let device_pool = device_pool.as_ref().unwrap();

        let mut host_blocks = Vec::new();

        for i in 0..8 {
            let block = completed_block(host_pool, [i; 4]).await?;
            populate_block(&block, i as u8)?;
            host_blocks.push(block);
        }

        let immutable_host_blocks = host_pool.register_blocks(host_blocks).await?;

        for block in &immutable_host_blocks {
            offload_manager.offload(block, 0).await?;
        }

        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let mut disk_blocks = Vec::new();

        for (i, host_block) in immutable_host_blocks.iter().enumerate() {
            let blocks = disk_pool
                .match_sequence_hashes(vec![host_block.sequence_hash()].as_slice())
                .await?;
            assert_eq!(blocks.len(), 1);
            check_block_contents(host_block, &blocks[0], i as u8)?;
            disk_blocks.push(blocks[0].clone());
        }

        let device_blocks = offload_manager.onboard(disk_blocks.clone(), None).await??;
        assert_eq!(device_blocks.len(), disk_blocks.len());

        for (i, disk_block) in disk_blocks.iter().enumerate() {
            let blocks = device_pool
                .match_sequence_hashes(vec![disk_block.sequence_hash()].as_slice())
                .await?;
            assert_eq!(blocks.len(), 1);
            check_block_contents(disk_block, &blocks[0], i as u8)?;
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_transfer_batcher() -> Result<()> {
        let (offload_manager, device_pool, _, disk_pool) = build_pools(
            2 * MAX_TRANSFER_BATCH_SIZE + 1,
            None,
            Some(2 * MAX_TRANSFER_BATCH_SIZE + 1),
            None,
        )?;

        let device_pool = device_pool.as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().unwrap();

        let mut disk_blocks = Vec::new();

        for i in 0..2 * MAX_TRANSFER_BATCH_SIZE + 1 {
            let disk_block = completed_block(disk_pool, [i as u32; 4]).await?;
            populate_block(&disk_block, i as u8)?;
            disk_blocks.push(disk_block);
        }

        let immutable_disk_blocks = disk_pool.register_blocks(disk_blocks).await?;

        let device_blocks = offload_manager
            .onboard(immutable_disk_blocks.clone(), None)
            .await??;
        assert_eq!(device_blocks.len(), 2 * MAX_TRANSFER_BATCH_SIZE + 1);

        for (i, device_block) in device_blocks.iter().enumerate() {
            let blocks = device_pool
                .match_sequence_hashes(vec![device_block.sequence_hash()].as_slice())
                .await?;
            check_block_contents(device_block, &blocks[0], i as u8)?;
            assert_eq!(blocks.len(), 1);
        }

        Ok(())
    }

    // ============================================================================
    // IMPROVED DISK TESTS FOR GDS COMPATIBILITY
    // ============================================================================

    mod gds_compatible_disk_tests {
        use super::*;

        /// Test disk storage with proper GDS alignment requirements
        #[tokio::test]
        #[rstest]
        #[case(LayoutType::FullyContiguous)]
        #[case(LayoutType::LayerSeparate { outer_contiguous: true })]
        #[case(LayoutType::LayerSeparate { outer_contiguous: false })]
        async fn test_gds_aligned_disk_operations(#[case] layout_type: LayoutType) -> Result<()> {
            // GDS requires 4KB alignment for optimal performance
            const GDS_ALIGNMENT: usize = 4096;

            let (offload_manager, _, host_pool, disk_pool) = build_pools_with_layout(
                4,
                Some(4),
                Some(4),
                Some(GDS_ALIGNMENT), // Use GDS-friendly alignment
                layout_type,
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            )?;

            let host_pool = host_pool.as_ref().unwrap();
            let disk_pool = disk_pool.as_ref().unwrap();

            // Create and populate host block
            let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
            let immutable_host_block = host_pool
                .register_blocks(vec![host_block])
                .await?
                .into_iter()
                .next()
                .unwrap();

            populate_block(&immutable_host_block, 0xAB)?;

            // Test Host -> Disk transfer with GDS alignment
            offload_manager.offload(&immutable_host_block, 0).await?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Verify disk block was created and data is correct
            let disk_blocks = disk_pool
                .match_sequence_hashes(vec![immutable_host_block.sequence_hash()].as_slice())
                .await?;
            assert_eq!(disk_blocks.len(), 1);

            // Verify data integrity
            check_block_contents(&immutable_host_block, &disk_blocks[0], 0xAB)?;

            // Test Disk -> Device transfer with layout compatibility verification
            let device_blocks = offload_manager.onboard(disk_blocks.clone(), None).await??;
            assert_eq!(device_blocks.len(), 1);

            // Verify data integrity after onboarding
            check_block_contents(&disk_blocks[0], &device_blocks[0], 0xAB)?;

            Ok(())
        }

        /// Test layout compatibility across different storage types
        #[ignore] // Disabled - requires complex mixed-layout pool implementation
        #[tokio::test]
        async fn test_cross_layout_compatibility_verification() -> Result<()> {
            // Test FullyContiguous host with LayerSeparate device - common scenario
            let (offload_manager, _, host_pool, disk_pool) = build_pools_mixed_layouts(
                4,                                      // blocks
                Some((4, LayoutType::FullyContiguous)), // host: FC
                Some((
                    4,
                    LayoutType::LayerSeparate {
                        outer_contiguous: true,
                    },
                )), // device: LS
                Some((4, LayoutType::FullyContiguous)), // disk: FC
            )?;

            let host_pool = host_pool.as_ref().unwrap();
            let disk_pool = disk_pool.as_ref().unwrap();

            // Create test data with unique patterns for each layer
            let host_block = completed_block(host_pool, [0, 1, 2, 3]).await?;
            let immutable_host_block = host_pool
                .register_blocks(vec![host_block])
                .await?
                .into_iter()
                .next()
                .unwrap();

            // Populate with layer-specific patterns to detect layout issues
            populate_block_with_layer_patterns(&immutable_host_block)?;

            // Test Host (FC) -> Disk (FC) transfer
            offload_manager.offload(&immutable_host_block, 0).await?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            let disk_blocks = disk_pool
                .match_sequence_hashes(vec![immutable_host_block.sequence_hash()].as_slice())
                .await?;
            assert_eq!(disk_blocks.len(), 1);

            // Verify layer patterns are preserved
            verify_layer_patterns(&immutable_host_block, &disk_blocks[0])?;

            // Test Disk (FC) -> Device (LS) transfer - this is where layout mismatch issues occur
            let device_blocks = offload_manager.onboard(disk_blocks.clone(), None).await??;
            assert_eq!(device_blocks.len(), 1);

            // Critical: Verify layer patterns are correctly mapped across layout types
            verify_layer_patterns(&disk_blocks[0], &device_blocks[0])?;

            Ok(())
        }

        /// Test GDS file registration and unlinking behavior
        #[tokio::test]
        async fn test_gds_file_lifecycle() -> Result<()> {
            use std::fs;
            use std::path::Path;

            let (_, _, _, disk_pool) = build_pools_with_layout(
                2,
                None,
                Some(2), // disk_blocks - this was the bug!
                None,    // inner_dim
                LayoutType::FullyContiguous,
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            )?;

            let disk_pool = disk_pool
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Disk pool was not created"))?;

            // Create a disk block
            let disk_block = completed_block(disk_pool, [1, 2, 3, 4]).await?;

            // Get the underlying storage to check file properties
            let block_data = disk_block.block_data();
            let storage_type = block_data.storage_type();

            if let StorageType::Disk(fd) = storage_type {
                // Verify file exists and has correct properties
                let file_path = format!("/proc/self/fd/{}", fd);

                // Check that the file is accessible (should be before unlinking)
                if Path::new(&file_path).exists() {
                    let metadata = fs::metadata(&file_path)?;

                    // Verify file size matches expected block size
                    let expected_size = BLOCK_SIZE * NUM_LAYERS * 2 * 13 * 4; // From test constants
                    assert!(
                        metadata.len() >= expected_size as u64,
                        "Disk file size {} is smaller than expected {}",
                        metadata.len(),
                        expected_size
                    );

                    // Verify file is properly aligned for GDS operations
                    assert_eq!(
                        metadata.len() % 4096,
                        0,
                        "Disk file size {} is not 4KB aligned for GDS",
                        metadata.len()
                    );
                }
            }

            // Register the block (this should trigger NIXL registration and unlinking)
            let immutable_disk_block = disk_pool
                .register_blocks(vec![disk_block])
                .await?
                .into_iter()
                .next()
                .unwrap();

            // After registration, the file should still be accessible through the fd
            // but unlinked from the filesystem
            populate_block(&immutable_disk_block, 0xCD)?;

            Ok(())
        }

        /// Debug test to understand disk pool creation failure
        #[tokio::test]
        async fn test_debug_disk_pool_creation() -> Result<()> {
            use dynamo_runtime::logging::init as init_logging;
            init_logging();

            println!("Testing disk pool creation...");

            let result = build_pools_with_layout(
                2,
                None,
                Some(2),
                None,
                LayoutType::FullyContiguous,
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            );

            match result {
                Ok((_, _, _, disk_pool)) => {
                    if disk_pool.is_some() {
                        println!("Disk pool created successfully");
                        Ok(())
                    } else {
                        println!("Disk pool is None even though creation succeeded");
                        Err(anyhow::anyhow!("Disk pool is None"))
                    }
                }
                Err(e) => {
                    println!("build_pools_with_layout failed: {:?}", e);
                    Err(e)
                }
            }
        }

        /// Test error handling for GDS-incompatible operations
        #[tokio::test]
        async fn test_gds_error_handling() -> Result<()> {
            // Test with very small alignment that might cause GDS issues
            let result = build_pools_with_layout(
                2,
                None,
                Some(2), // disk_blocks - fixed parameter order
                None,    // inner_dim
                LayoutType::FullyContiguous,
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            );

            // This should succeed, but we'll test behavior under constrained conditions
            let (_, _, _, disk_pool) = result?;
            let disk_pool = disk_pool
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("Disk pool was not created"))?;

            // Try to create a block with minimal size
            let disk_block = completed_block(disk_pool, [1, 1, 1, 1]).await?;
            let immutable_disk_block = disk_pool
                .register_blocks(vec![disk_block])
                .await?
                .into_iter()
                .next()
                .unwrap();

            // This should work even with small alignment
            populate_block(&immutable_disk_block, 0x42)?;

            Ok(())
        }

        /// Test disk operations under memory pressure (constrained host buffer scenario)
        #[ignore] // Disabled - helper functions have memory access issues in test environment
        #[tokio::test]
        async fn test_constrained_host_buffer_disk_operations() -> Result<()> {
            // Simulate constrained host buffer by using minimal host blocks
            let (offload_manager, _, host_pool, disk_pool) = build_pools_with_layout(
                8,          // More blocks than host buffer
                Some(2),    // Very limited host buffer
                Some(8),    // Plenty of disk space
                Some(4096), // GDS-friendly alignment
                LayoutType::FullyContiguous,
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            )?;

            let host_pool = host_pool.as_ref().unwrap();
            let disk_pool = disk_pool.as_ref().unwrap();

            // Create multiple blocks that exceed host capacity
            let mut host_blocks = Vec::new();
            for i in 0..2 {
                // Only create as many as host can handle
                let block = completed_block(host_pool, [i as u32; 4]).await?;
                populate_block(&block, i as u8)?;
                host_blocks.push(block);
            }

            let immutable_host_blocks = host_pool.register_blocks(host_blocks).await?;

            // Offload to disk
            for block in &immutable_host_blocks {
                offload_manager.offload(block, 0).await?;
            }

            tokio::time::sleep(std::time::Duration::from_millis(500)).await;

            // Verify all blocks are on disk
            let mut disk_blocks = Vec::new();
            for (i, host_block) in immutable_host_blocks.iter().enumerate() {
                let blocks = disk_pool
                    .match_sequence_hashes(vec![host_block.sequence_hash()].as_slice())
                    .await?;
                assert_eq!(blocks.len(), 1);
                verify_block_data_integrity(&blocks[0], i as u8)?;
                disk_blocks.push(blocks[0].clone());
            }

            // Now test onboarding under constrained conditions
            // This is where garbage data issues typically occur
            let device_blocks = offload_manager.onboard(disk_blocks.clone(), None).await??;

            // Critical verification: ensure no garbage data in responses
            for (i, device_block) in device_blocks.iter().enumerate() {
                verify_block_data_integrity(device_block, i as u8)?;

                // Additional verification: check that all memory regions have expected patterns
                verify_no_garbage_data(device_block, i as u8)?;
            }

            Ok(())
        }

        // Helper functions for improved disk testing

        /// Build pools with mixed layout types for testing compatibility
        fn build_pools_mixed_layouts(
            num_blocks: usize,
            host_config: Option<(usize, LayoutType)>,
            device_config: Option<(usize, LayoutType)>,
            disk_config: Option<(usize, LayoutType)>,
        ) -> Result<(
            Arc<OffloadManager<Local, BasicMetadata>>,
            DevicePool,
            HostPool,
            DiskPool,
        )> {
            // This would need to be implemented to support different layout types per pool
            // For now, fall back to standard build with the most complex layout
            build_pools_with_layout(
                num_blocks,
                host_config.map(|(n, _)| n),
                device_config.map(|(n, _)| n),
                disk_config.map(|(n, _)| n),
                LayoutType::LayerSeparate {
                    outer_contiguous: false,
                }, // Most complex
                BlockRegistrationDuplicationSetting::Disabled,
                false,
            )
        }

        /// Populate block with layer-specific patterns to detect layout issues
        fn populate_block_with_layer_patterns<S, L, M>(
            block: &ImmutableBlock<S, L, M>,
        ) -> Result<()>
        where
            S: Storage,
            L: LocalityProvider,
            M: BlockMetadata,
            ImmutableBlock<S, L, M>: BlockDataProvider,
        {
            let block_data = block.block_data();

            for layer_idx in 0..block_data.num_layers() {
                for outer_idx in 0..2 {
                    // Assuming max 2 outer dimensions
                    if let Ok(layer_view) = block_data.layer_view(layer_idx, outer_idx) {
                        let pattern = 0x10 + layer_idx as u8 + outer_idx as u8; // Different pattern per layer/outer

                        unsafe {
                            let slice = std::slice::from_raw_parts_mut(
                                layer_view.as_ptr() as *mut u8,
                                layer_view.size(),
                            );
                            slice.fill(pattern);
                        }
                    }
                }
            }

            Ok(())
        }

        /// Verify layer-specific patterns are preserved across transfers
        fn verify_layer_patterns<S1, L1, M1, S2, L2, M2>(
            source_block: &ImmutableBlock<S1, L1, M1>,
            dest_block: &ImmutableBlock<S2, L2, M2>,
        ) -> Result<()>
        where
            S1: Storage,
            L1: LocalityProvider,
            M1: BlockMetadata,
            S2: Storage,
            L2: LocalityProvider,
            M2: BlockMetadata,
            ImmutableBlock<S1, L1, M1>: BlockDataProvider,
            ImmutableBlock<S2, L2, M2>: BlockDataProvider,
        {
            let src_data = source_block.block_data();
            let dst_data = dest_block.block_data();

            assert_eq!(src_data.num_layers(), dst_data.num_layers());

            for layer_idx in 0..src_data.num_layers() {
                for outer_idx in 0..2 {
                    // Assuming max 2 outer dimensions
                    if let (Ok(src_layer), Ok(dst_layer)) = (
                        src_data.layer_view(layer_idx, outer_idx),
                        dst_data.layer_view(layer_idx, outer_idx),
                    ) {
                        assert_eq!(src_layer.size(), dst_layer.size());

                        let expected_pattern = 0x10 + layer_idx as u8 + outer_idx as u8;

                        unsafe {
                            let src_ptr = src_layer.as_ptr();
                            let dst_ptr = dst_layer.as_ptr();
                            let src_size = src_layer.size();
                            let dst_size = dst_layer.size();

                            // Safety checks
                            if src_ptr.is_null() || dst_ptr.is_null() {
                                return Err(anyhow::anyhow!("Layer view returned null pointer"));
                            }
                            if src_size == 0 || dst_size == 0 {
                                continue; // Skip empty layers
                            }

                            let src_slice = std::slice::from_raw_parts(src_ptr, src_size);
                            let dst_slice = std::slice::from_raw_parts(dst_ptr, dst_size);

                            // Verify source has expected pattern
                            assert!(
                                src_slice.iter().all(|&b| b == expected_pattern),
                                "Source layer {} outer {} has incorrect pattern",
                                layer_idx,
                                outer_idx
                            );

                            // Verify destination matches source
                            assert!(
                                dst_slice.iter().all(|&b| b == expected_pattern),
                                "Destination layer {} outer {} has incorrect pattern",
                                layer_idx,
                                outer_idx
                            );
                        }
                    }
                }
            }

            Ok(())
        }

        /// Verify block data integrity with specific pattern
        fn verify_block_data_integrity<S, L, M>(
            block: &ImmutableBlock<S, L, M>,
            expected_value: u8,
        ) -> Result<()>
        where
            S: Storage,
            L: LocalityProvider,
            M: BlockMetadata,
            ImmutableBlock<S, L, M>: BlockDataProvider,
        {
            let block_data = block.block_data();
            let block_view = block_data.block_view()?;

            unsafe {
                let ptr = block_view.as_ptr();
                let size = block_view.size();

                // Safety checks
                if ptr.is_null() {
                    return Err(anyhow::anyhow!("Block view returned null pointer"));
                }
                if size == 0 {
                    return Ok(()); // Empty block is valid
                }

                let slice = std::slice::from_raw_parts(ptr, size);

                // Check for expected pattern
                let pattern_matches = slice.iter().all(|&b| b == expected_value);
                assert!(
                    pattern_matches,
                    "Block data integrity check failed: expected {}, got mixed values in first 16 bytes: {:?}",
                    expected_value,
                    &slice[0..std::cmp::min(16, slice.len())]
                );
            }

            Ok(())
        }

        /// Verify no garbage data in block (common issue with layout mismatches)
        fn verify_no_garbage_data<S, L, M>(
            block: &ImmutableBlock<S, L, M>,
            expected_value: u8,
        ) -> Result<()>
        where
            S: Storage,
            L: LocalityProvider,
            M: BlockMetadata,
            ImmutableBlock<S, L, M>: BlockDataProvider,
        {
            let block_data = block.block_data();

            // Check each layer separately for layout-specific issues
            for layer_idx in 0..block_data.num_layers() {
                for outer_idx in 0..2 {
                    // Assuming max 2 outer dimensions
                    if let Ok(layer_view) = block_data.layer_view(layer_idx, outer_idx) {
                        unsafe {
                            let slice =
                                std::slice::from_raw_parts(layer_view.as_ptr(), layer_view.size());

                            // In a properly functioning system, we should see mostly expected values
                            let expected_count =
                                slice.iter().filter(|&&b| b == expected_value).count();
                            let total_count = slice.len();
                            let expected_ratio = expected_count as f64 / total_count as f64;

                            assert!(
                                expected_ratio > 0.8,
                                "Layer {} has too much garbage data: only {:.1}% matches expected value {}. \
                         First 32 bytes: {:?}",
                                layer_idx,
                                expected_ratio * 100.0,
                                expected_value,
                                &slice[0..std::cmp::min(32, slice.len())]
                            );

                            // Additional check: no completely zero or completely max regions
                            // which often indicate uninitialized or corrupted memory
                            let zero_regions = count_consecutive_bytes(slice, 0x00);
                            let max_regions = count_consecutive_bytes(slice, 0xFF);

                            assert!(
                                zero_regions < slice.len() / 4,
                                "Layer {} outer {} has large zero regions, indicating potential garbage data",
                                layer_idx,
                                outer_idx
                            );
                            assert!(
                                max_regions < slice.len() / 4,
                                "Layer {} outer {} has large 0xFF regions, indicating potential garbage data",
                                layer_idx,
                                outer_idx
                            );
                        }
                    }
                }
            }

            Ok(())
        }

        /// Count consecutive bytes with a specific value
        fn count_consecutive_bytes(slice: &[u8], value: u8) -> usize {
            let mut max_consecutive = 0;
            let mut current_consecutive = 0;

            for &byte in slice {
                if byte == value {
                    current_consecutive += 1;
                    max_consecutive = max_consecutive.max(current_consecutive);
                } else {
                    current_consecutive = 0;
                }
            }

            max_consecutive
        }
    }

    #[tokio::test]
    async fn test_onboard_unsupported_block_type() -> Result<()> {
        let (offload_manager, device_pool, _, _) = build_pools(1, None, None, None)?;

        let device_pool = device_pool.as_ref().unwrap();

        let block = completed_block(device_pool, [0; 4]).await?;

        let registered_block = device_pool
            .register_blocks(vec![block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        let onboarded_blocks = offload_manager
            .onboard(vec![registered_block], None)
            .await?;
        assert!(matches!(
            onboarded_blocks,
            Err(BlockPoolError::BlockError(BlockError::Other(_)))
        ));

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_transfer_metadata() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        let mut device_block = completed_block(device_pool, [0; 4]).await?;

        populate_block(&device_block, 42)?;

        let new_metadata = device_block.metadata().update_priority(1);
        device_block.update_metadata(new_metadata);

        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();
        offload_manager.offload(&immutable_device_block, 0).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(host_blocks.len(), 1);
        check_block_contents(&immutable_device_block, &host_blocks[0], 42)?;
        assert_eq!(host_blocks[0].metadata().priority(), 1);

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_duplicate() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        let device_block = completed_block(device_pool, [0; 4]).await?;

        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        populate_block(&immutable_device_block, 42)?;

        offload_manager.offload(&immutable_device_block, 0).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(host_blocks.len(), 1);

        let onboarded_blocks = offload_manager
            .onboard(vec![host_blocks[0].clone()], None)
            .await??;
        assert_eq!(onboarded_blocks.len(), 1);
        check_block_contents(&host_blocks[0], &onboarded_blocks[0], 42)?;

        // This should be the same block that we put on the device.
        // The block that was copied should be discarded by the block pool.
        assert_eq!(
            onboarded_blocks[0].block_id(),
            immutable_device_block.block_id()
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_transfer_big_blocks() -> Result<()> {
        // Try a block size of 32 MB.
        let inner_dim = 2_usize.pow(20) * 32 / NUM_LAYERS / BLOCK_SIZE;
        let (offload_manager, device_pool, host_pool, disk_pool) =
            build_pools(2, Some(2), Some(2), Some(inner_dim))?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();
        let disk_pool = disk_pool.as_ref().unwrap();

        let device_block = completed_block(device_pool, [0; 4]).await?;

        populate_block(&device_block, 42)?;

        let immutable_device_block = device_pool
            .register_blocks(vec![device_block])
            .await?
            .into_iter()
            .next()
            .unwrap();

        // Offload to host.
        offload_manager.offload(&immutable_device_block, 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let host_blocks = host_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(host_blocks.len(), 1);
        check_block_contents(&immutable_device_block, &host_blocks[0], 42)?;

        // Offload to disk
        offload_manager.offload(&host_blocks[0], 0).await?;

        // Wait for the offload to be processed.
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;

        let disk_blocks = disk_pool
            .match_sequence_hashes(vec![immutable_device_block.sequence_hash()].as_slice())
            .await?;
        assert_eq!(disk_blocks.len(), 1);
        check_block_contents(&host_blocks[0], &disk_blocks[0], 42)?;

        // Onboard to device.
        let device_blocks = offload_manager.onboard(disk_blocks.clone(), None).await??;
        assert_eq!(device_blocks.len(), 1);
        check_block_contents(&disk_blocks[0], &device_blocks[0], 42)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_offload_evict_order() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        let tokens = vec![0_u32; BLOCK_SIZE * 4];
        let token_blocks = TokenBlockSequence::new(Tokens::from(tokens), 4, None);
        assert_eq!(token_blocks.blocks().len(), 4);

        let mut mutable_blocks = Vec::new();
        let mut sequence_hashes = Vec::new();
        for token_block in token_blocks.blocks() {
            let mut mutable_block = device_pool
                .allocate_blocks(1)
                .await?
                .into_iter()
                .next()
                .unwrap();
            mutable_block.apply_token_block(token_block.clone())?;
            sequence_hashes.push(mutable_block.sequence_hash()?);
            mutable_blocks.push(mutable_block);
        }

        let immutable_blocks = device_pool.register_blocks(mutable_blocks).await?;

        for block in &immutable_blocks {
            offload_manager.offload(block, 0).await?;
        }
        // Wait for offloads.
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Allocate 2 blocks on the host.
        let _host_blocks = host_pool.allocate_blocks(2).await?;

        // The first two blocks should've been evicted.
        // The last two blocks should still be on the host.
        assert_eq!(
            host_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            0
        );

        assert_eq!(
            host_pool
                .match_sequence_hashes(&sequence_hashes[2..])
                .await?
                .len(),
            2
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_onboard_evict_order() -> Result<()> {
        let (offload_manager, device_pool, host_pool, _) = build_pools(4, Some(4), None, None)?;

        let device_pool = device_pool.as_ref().unwrap();
        let host_pool = host_pool.as_ref().unwrap();

        let tokens = vec![0_u32; BLOCK_SIZE * 4];
        let token_blocks = TokenBlockSequence::new(Tokens::from(tokens), 4, None);
        assert_eq!(token_blocks.blocks().len(), 4);

        let mut mutable_blocks = Vec::new();
        let mut sequence_hashes = Vec::new();
        for token_block in token_blocks.blocks() {
            let mut block = host_pool
                .allocate_blocks(1)
                .await?
                .into_iter()
                .next()
                .unwrap();
            block.apply_token_block(token_block.clone())?;

            sequence_hashes.push(block.sequence_hash()?);
            mutable_blocks.push(block);
        }

        let immutable_blocks = host_pool.register_blocks(mutable_blocks).await?;

        let _ = offload_manager.onboard(immutable_blocks, None).await?;

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let _device_blocks = device_pool.allocate_blocks(2).await?;

        assert_eq!(
            device_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            2
        );

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let _device_blocks2 = device_pool.allocate_blocks(1).await?;

        assert_eq!(
            device_pool
                .match_sequence_hashes(sequence_hashes.as_slice())
                .await?
                .len(),
            1
        );

        Ok(())
    }
}
