// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use futures::future::try_join_all;
use nixl_sys::NixlDescriptor;
use utils::*;
use zmq::*;

use BlockTransferPool::*;

use crate::block_manager::{
    Storage,
    block::{
        BasicMetadata, Block, BlockDataProvider, BlockDataProviderMut, ReadableBlock,
        WritableBlock,
        data::local::LocalBlockData,
        locality,
        transfer::{TransferContext, WriteTo, WriteToStrategy},
    },
    connector::scheduler::{SchedulingDecision, TransferSchedulerClient},
    distributed::utils::ObjectStorageConfig,
    offload::max_transfer_batch_size,
    storage::{DeviceStorage, DiskStorage, Local, PinnedStorage},
    v2::physical::{
        layout::{LayoutConfig, PhysicalLayout, builder::PhysicalLayoutBuilder},
        manager::TransportManager,
        transfer::LayoutHandle,
        transfer::BounceBufferSpec,
        transfer::options::TransferOptions,
    },
};

use anyhow::Result;
use async_trait::async_trait;
use std::{any::Any, sync::Arc, time::Duration};
use tokio::time::timeout;

/// Default timeout for object storage offload (GPU → S3).
/// Offloads can be slower due to S3 write latency.
/// Can be overridden via DYN_KVBM_OBJECT_OFFLOAD_TIMEOUT_SECS environment variable.
const DEFAULT_OBJECT_OFFLOAD_TIMEOUT_SECS: u64 = 60;

/// Default timeout for object storage onboard (S3 → GPU).
/// Onboards are typically faster as reads are often cached.
/// Can be overridden via DYN_KVBM_OBJECT_ONBOARD_TIMEOUT_SECS environment variable.
const DEFAULT_OBJECT_ONBOARD_TIMEOUT_SECS: u64 = 10;

/// Get the object offload timeout from environment or use default.
fn object_offload_timeout() -> Duration {
    let secs = std::env::var("DYN_KVBM_OBJECT_OFFLOAD_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_OBJECT_OFFLOAD_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

/// Get the object onboard timeout from environment or use default.
fn object_onboard_timeout() -> Duration {
    let secs = std::env::var("DYN_KVBM_OBJECT_ONBOARD_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_OBJECT_ONBOARD_TIMEOUT_SECS);
    Duration::from_secs(secs)
}

type LocalBlock<S, M> = Block<S, locality::Local, M>;
type LocalBlockDataList<S> = Vec<LocalBlockData<S>>;

// ============================================================================
// Object Storage Transfer Helpers
// ============================================================================

/// Log object storage transfer completion with throughput calculation.
fn log_object_transfer_complete(
    operation: &str,
    num_blocks: usize,
    layout_config: Option<&LayoutConfig>,
    transfer_elapsed: std::time::Duration,
    total_elapsed: std::time::Duration,
    needs_bounce: bool,
) {
    let transfer_ms = transfer_elapsed.as_secs_f64() * 1000.0;
    let total_ms = total_elapsed.as_secs_f64() * 1000.0;

    if let Some(lc) = layout_config {
        // bytes_per_block = layers * outer_dim * page_size * inner_dim * dtype_bytes
        let bytes_per_block = lc.num_layers * lc.outer_dim * lc.page_size
            * lc.inner_dim * lc.dtype_width_bytes;
        let total_bytes = num_blocks * bytes_per_block;
        let mb_transferred = total_bytes as f64 / (1024.0 * 1024.0);
        let throughput_mbps = if transfer_elapsed.as_secs_f64() > 0.0 {
            mb_transferred / transfer_elapsed.as_secs_f64()
        } else {
            0.0
        };

        tracing::debug!(
            target: "object_transfer_timing",
            num_blocks = num_blocks,
            mb = mb_transferred,
            transfer_ms = transfer_ms,
            total_ms = total_ms,
            throughput_mbps = throughput_mbps,
            bounce = needs_bounce,
            "{}: {} blocks, {:.2} MB | transfer={:.2}ms | total={:.2}ms | {:.2} MB/s (bounce={})",
            operation,
            num_blocks,
            mb_transferred,
            transfer_ms,
            total_ms,
            throughput_mbps,
            needs_bounce
        );
    } else {
        tracing::debug!(
            target: "object_transfer_timing",
            num_blocks = num_blocks,
            transfer_ms = transfer_ms,
            total_ms = total_ms,
            "{}: {} blocks | transfer={:.2}ms | total={:.2}ms (bounce={})",
            operation,
            num_blocks,
            transfer_ms,
            total_ms,
            needs_bounce
        );
    }
}

// Dedicated bounce buffer allocator removed - we now use host pool blocks only for object transfers.
// This simplifies resource management and avoids blocking the vLLM scheduler.

/// A batching wrapper for connector transfers to prevent resource exhaustion.
/// Splits large transfers into smaller batches that can be handled by the resource pools.
#[derive(Clone, Debug)]
pub struct ConnectorTransferBatcher {
    max_batch_size: usize,
}

impl ConnectorTransferBatcher {
    pub fn new() -> Self {
        let batch_size = max_transfer_batch_size();
        tracing::debug!(
            target: "object_transfer_timing",
            batch_size = batch_size,
            "ConnectorTransferBatcher initialized with batch_size={}",
            batch_size
        );
        Self {
            max_batch_size: batch_size,
        }
    }

    pub async fn execute_batched_transfer<T: BlockTransferDirectHandler>(
        &self,
        handler: &T,
        request: BlockTransferRequest,
    ) -> Result<()> {
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
                    bounce_block_ids: None,
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

#[async_trait]
pub trait BlockTransferHandler: Send + Sync {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()>;

    fn scheduler_client(&self) -> Option<TransferSchedulerClient>;
}

#[async_trait]
pub trait BlockTransferDirectHandler {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()>;
}

/// A handler for all block transfers. Wraps a group of [`BlockTransferPoolManager`]s.
#[derive(Clone)]
pub struct BlockTransferHandlerV1 {
    device: Option<LocalBlockDataList<DeviceStorage>>,
    host: Option<LocalBlockDataList<PinnedStorage>>,
    disk: Option<LocalBlockDataList<DiskStorage>>,
    context: Arc<TransferContext>,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    // add worker-connector scheduler client here
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV1 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV1 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        tracing::debug!(
            "Performing transfer of {} blocks from {:?} to {:?}",
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
}

impl BlockTransferHandlerV1 {
    pub fn new(
        device_blocks: Option<Vec<LocalBlock<DeviceStorage, BasicMetadata>>>,
        host_blocks: Option<Vec<LocalBlock<PinnedStorage, BasicMetadata>>>,
        disk_blocks: Option<Vec<LocalBlock<DiskStorage, BasicMetadata>>>,
        context: Arc<TransferContext>,
        scheduler_client: Option<TransferSchedulerClient>,
        // add worker-connector scheduler client here
    ) -> Result<Self> {
        Ok(Self {
            device: Self::get_local_data(device_blocks),
            host: Self::get_local_data(host_blocks),
            disk: Self::get_local_data(disk_blocks),
            context,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
        })
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
}

/// Async-friendly bounce buffer allocator for Device ↔ Object transfers.
///
/// Uses a semaphore to limit concurrent transfers and provides non-overlapping
/// block ID ranges for each transfer. When the permit is dropped, the slot
/// becomes available for other transfers.
///
/// Layout:
/// ```text
/// ┌─────────────────────────────────────────────────────────────┐
/// A simple bounce buffer spec created from externally-provided block IDs.
/// Use this when the caller (e.g., slot.rs) dynamically allocates bounce blocks
/// from the host pool and needs to pass them to the transfer layer.
pub struct ExternalBounceBuffer {
    layout: PhysicalLayout,
    block_ids: Vec<usize>,
}

impl ExternalBounceBuffer {
    /// Create a bounce buffer spec from externally-allocated block IDs.
    ///
    /// # Arguments
    /// * `layout` - The host layout these blocks belong to
    /// * `block_ids` - Block IDs allocated from the host pool
    pub fn new(layout: PhysicalLayout, block_ids: Vec<usize>) -> Self {
        Self { layout, block_ids }
    }
}

impl BounceBufferSpec for ExternalBounceBuffer {
    fn layout(&self) -> &PhysicalLayout {
        &self.layout
    }

    fn block_ids(&self) -> &[usize] {
        &self.block_ids
    }
}

#[derive(Clone)]
pub struct BlockTransferHandlerV2 {
    device_handle: Option<LayoutHandle>,
    host_handle: Option<LayoutHandle>,
    disk_handle: Option<LayoutHandle>,
    transport_manager: TransportManager,
    scheduler_client: Option<TransferSchedulerClient>,
    batcher: ConnectorTransferBatcher,
    /// Configuration for dynamic object layout creation
    object_storage_config: Option<ObjectStorageConfig>,
    /// Layout config template for creating object layouts
    layout_config: Option<LayoutConfig>,
    /// Worker ID for resolving bucket template
    worker_id: u32,
}

impl BlockTransferHandlerV2 {
    pub fn new(
        device_layout: Option<PhysicalLayout>,
        host_layout: Option<PhysicalLayout>,
        disk_layout: Option<PhysicalLayout>,
        transport_manager: TransportManager,
        scheduler_client: Option<TransferSchedulerClient>,
        object_storage_config: Option<ObjectStorageConfig>,
        layout_config: Option<LayoutConfig>,
        worker_id: u32,
    ) -> Result<Self> {
        // Object storage transfers now use host pool blocks as bounce buffers.
        // No dedicated bounce buffer allocator - this avoids blocking the vLLM scheduler.
        if object_storage_config.is_some() {
            tracing::info!(
                "Object storage enabled. Bounce buffers will be allocated from host pool. \
                 Offloads: GPU→Host→Object. Onboards: Object→Host→GPU."
            );
        }

        Ok(Self {
            device_handle: device_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            host_handle: host_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            disk_handle: disk_layout
                .map(|layout| transport_manager.register_layout(layout).unwrap()),
            transport_manager,
            scheduler_client,
            batcher: ConnectorTransferBatcher::new(),
            object_storage_config,
            layout_config,
            worker_id,
        })
    }

    /// Create a dynamic ObjectLayout for the given object keys (sequence hashes).
    ///
    /// Each key becomes one block in the layout. The block_id in the transfer
    /// maps directly to the object key in S3.
    fn create_object_layout(&self, object_keys: Vec<u64>) -> Result<PhysicalLayout> {
        let config = self.object_storage_config.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Object storage config not set"))?;

        let mut layout_config = self.layout_config.clone()
            .ok_or_else(|| anyhow::anyhow!("Layout config not set for object storage"))?;

        // Set num_blocks to match the number of objects we're transferring
        layout_config.num_blocks = object_keys.len();

        let agent = self.transport_manager.context().nixl_agent().clone();

        // Resolve bucket name with worker_id substitution
        let bucket = config.resolve_bucket(self.worker_id);

        PhysicalLayoutBuilder::new(agent)
            .with_config(layout_config)
            .object_layout()
            .allocate_objects(bucket, object_keys)?
            .build()
    }
}

#[async_trait]
impl BlockTransferHandler for BlockTransferHandlerV2 {
    async fn execute_transfer(&self, request: BlockTransferRequest) -> Result<()> {
        self.batcher.execute_batched_transfer(self, request).await
    }

    fn scheduler_client(&self) -> Option<TransferSchedulerClient> {
        self.scheduler_client.clone()
    }
}

#[async_trait]
impl BlockTransferDirectHandler for BlockTransferHandlerV2 {
    async fn execute_transfer_direct(&self, request: BlockTransferRequest) -> Result<()> {
        // Check if this involves object storage - requires dynamic layout creation
        let involves_object = matches!(request.from_pool(), Object)
            || matches!(request.to_pool(), Object);

        if involves_object {
            return self.execute_object_transfer(&request).await;
        }

        // Standard transfer path for non-object storage
        let (src, dst) = match (request.from_pool(), request.to_pool()) {
            (Device, Host) => (self.device_handle.as_ref(), self.host_handle.as_ref()),
            (Device, Disk) => (self.device_handle.as_ref(), self.disk_handle.as_ref()),
            (Host, Device) => (self.host_handle.as_ref(), self.device_handle.as_ref()),
            (Host, Disk) => (self.host_handle.as_ref(), self.disk_handle.as_ref()),
            (Disk, Device) => (self.disk_handle.as_ref(), self.device_handle.as_ref()),
            _ => return Err(anyhow::anyhow!("Invalid transfer type.")),
        };

        if let (Some(src), Some(dst)) = (src, dst) {
            let src_block_ids = request
                .blocks()
                .iter()
                .map(|(from, _)| *from)
                .collect::<Vec<_>>();
            let dst_block_ids = request
                .blocks()
                .iter()
                .map(|(_, to)| *to)
                .collect::<Vec<_>>();

            let num_blocks = src_block_ids.len();
            tracing::info!(
                target: "blocking_ops",
                num_blocks = num_blocks,
                from = ?request.from_pool(),
                to = ?request.to_pool(),
                "BLOCKING: Starting standard NIXL transfer"
            );

            self.transport_manager
                .execute_transfer(
                    *src,
                    &src_block_ids,
                    *dst,
                    &dst_block_ids,
                    TransferOptions::default(),
                )?
                .await?;

            tracing::info!(
                target: "blocking_ops",
                num_blocks = num_blocks,
                from = ?request.from_pool(),
                to = ?request.to_pool(),
                "UNBLOCKED: Completed standard NIXL transfer"
            );
        } else {
            return Err(anyhow::anyhow!("Invalid transfer type."));
        }

        Ok(())
    }
}

impl BlockTransferHandlerV2 {
    /// Execute a transfer involving object storage.
    ///
    /// Object storage transfers require dynamic layout creation because:
    /// - Block IDs for objects ARE the sequence hashes (object keys)
    /// - Objects are not pre-allocated; layouts are created on-demand
    ///
    /// Device ↔ Object transfers require a bounce buffer (pinned host memory)
    /// since there's no direct GPU → S3 path. Bounce buffers are acquired from
    /// a pool to allow concurrent transfers without overlap.
    async fn execute_object_transfer(&self, request: &BlockTransferRequest) -> Result<()> {
        let blocks = request.blocks();
        if blocks.is_empty() {
            return Ok(());
        }

        let transfer_start = std::time::Instant::now();
        let num_blocks = blocks.len();

        // Check if we need a bounce buffer (Device ↔ Object)
        let needs_bounce = matches!(request.from_pool(), Device)
            || matches!(request.to_pool(), Device);

        // Determine transfer direction
        let is_offload = matches!(request.to_pool(), Object);

        // Get bounce buffer from request (must be provided by caller from host pool)
        let bounce_spec: Option<Arc<dyn BounceBufferSpec>> = if needs_bounce {
            if let Some(ref bounce_ids) = request.bounce_block_ids {
                // Use external bounce blocks allocated by caller from host pool
                let host_layout = self.transport_manager
                    .get_layout(*self.host_handle.as_ref().ok_or_else(|| {
                        anyhow::anyhow!("Bounce blocks provided but no host layout available")
                    })?)
                    .ok_or_else(|| anyhow::anyhow!("Host layout not found in transport manager"))?;

                tracing::debug!(
                    "Using bounce blocks from host pool: {:?}",
                    bounce_ids
                );

                Some(Arc::new(ExternalBounceBuffer::new(
                    host_layout.clone(),
                    bounce_ids.clone(),
                )))
            } else {
                // No bounce blocks provided - this transfer cannot proceed
                // For Device↔Object transfers, caller must provide bounce_block_ids
                return Err(anyhow::anyhow!(
                    "Device↔Object transfer requires bounce blocks from host pool. \
                     Caller must provide bounce_block_ids (direction={})",
                    if is_offload { "offload" } else { "onboard" }
                ));
            }
        } else {
            None
        };

        tracing::debug!(
            target: "object_transfer_timing",
            direction = ?request.from_pool(),
            num_blocks = num_blocks,
            needs_bounce = needs_bounce,
            "OBJECT_XFER_START: {:?}→{:?} {} blocks",
            request.from_pool(),
            request.to_pool(),
            num_blocks,
        );

        match (request.from_pool(), request.to_pool()) {
            (Device, Object) | (Host, Object) => {
                // Offload: Device/Host → Object Storage
                // dst_block_ids are the object keys (sequence hashes)
                let src_block_ids: Vec<usize> = blocks.iter().map(|(from, _)| *from).collect();
                let object_keys: Vec<u64> = blocks.iter().map(|(_, to)| *to as u64).collect();

                // Create dynamic ObjectLayout for these specific keys
                let object_layout = self.create_object_layout(object_keys)?;
                let object_handle = self.transport_manager.register_layout(object_layout)?;

                // Get source handle
                let src_handle = match request.from_pool() {
                    Device => self.device_handle.as_ref(),
                    Host => self.host_handle.as_ref(),
                    _ => None,
                }.ok_or_else(|| anyhow::anyhow!("Source handle not available"))?;

                // Execute transfer: src blocks → object storage (blocks 0..N in object layout)
                let dst_block_ids: Vec<usize> = (0..blocks.len()).collect();

                // Build transfer options with bounce buffer if needed
                let options = if let Some(ref bounce) = bounce_spec {
                    TransferOptions::builder()
                        .bounce_buffer(bounce.clone())
                        .build()
                        .unwrap_or_default()
                } else {
                    TransferOptions::default()
                };

                let start = std::time::Instant::now();
                let num_blocks = blocks.len();
                let transfer_timeout = object_offload_timeout();

                tracing::info!(
                    target: "blocking_ops",
                    num_blocks = num_blocks,
                    timeout_secs = transfer_timeout.as_secs(),
                    from = ?request.from_pool(),
                    needs_bounce = needs_bounce,
                    "BLOCKING: Starting object OFFLOAD transfer (S3 write)"
                );

                let transfer_future = self.transport_manager
                    .execute_transfer(
                        *src_handle,
                        &src_block_ids,
                        object_handle,
                        &dst_block_ids,
                        options,
                    )?;

                match timeout(transfer_timeout, transfer_future).await {
                    Ok(Ok(())) => {
                        let elapsed = start.elapsed();
                        let total_elapsed = transfer_start.elapsed();

                        tracing::info!(
                            target: "blocking_ops",
                            num_blocks = num_blocks,
                            elapsed_ms = elapsed.as_millis(),
                            "UNBLOCKED: Completed object OFFLOAD transfer"
                        );

                        log_object_transfer_complete(
                            "OBJECT_OFFLOAD_DONE",
                            num_blocks,
                            self.layout_config.as_ref(),
                            elapsed,
                            total_elapsed,
                            needs_bounce,
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::error!(
                            num_blocks = num_blocks,
                            elapsed_ms = start.elapsed().as_millis(),
                            "OBJECT_OFFLOAD_FAILED: transfer error: {}",
                            e
                        );
                        return Err(e);
                    }
                    Err(_) => {
                        tracing::error!(
                            num_blocks = num_blocks,
                            timeout_secs = transfer_timeout.as_secs(),
                            "OBJECT_OFFLOAD_TIMEOUT: transfer exceeded {}s timeout",
                            transfer_timeout.as_secs()
                        );
                        return Err(anyhow::anyhow!(
                            "Object offload timed out after {}s for {} blocks",
                            transfer_timeout.as_secs(),
                            num_blocks
                        ));
                    }
                }
            }
            (Object, Device) | (Object, Host) => {
                // Onboard: Object Storage → Device/Host
                // src_block_ids are the object keys (sequence hashes)
                let object_keys: Vec<u64> = blocks.iter().map(|(from, _)| *from as u64).collect();
                let dst_block_ids: Vec<usize> = blocks.iter().map(|(_, to)| *to).collect();

                // Clone data needed for spawn_blocking (blocking ops must run off async runtime)
                let object_storage_config = self.object_storage_config.clone();
                let layout_config = self.layout_config.clone();
                let worker_id = self.worker_id;
                let transport_manager = self.transport_manager.clone();
                let num_keys = object_keys.len();

                // Move blocking layout creation + registration off the async runtime
                // This prevents 100s of concurrent registrations from blocking all tokio workers
                let object_handle = tokio::task::spawn_blocking(move || -> Result<LayoutHandle> {
                    let reg_start = std::time::Instant::now();

                    let config = object_storage_config.as_ref()
                        .ok_or_else(|| anyhow::anyhow!("Object storage config not set"))?;

                    let mut lc = layout_config
                        .ok_or_else(|| anyhow::anyhow!("Layout config not set for object storage"))?;
                    lc.num_blocks = num_keys;

                    let agent = transport_manager.context().nixl_agent().clone();
                    let bucket = config.resolve_bucket(worker_id);

                    let object_layout = PhysicalLayoutBuilder::new(agent)
                        .with_config(lc)
                        .object_layout()
                        .allocate_objects(bucket, object_keys)?
                        .build()?;

                    let handle = transport_manager.register_layout(object_layout)?;

                    tracing::debug!(
                        target: "blocking_ops",
                        elapsed_ms = reg_start.elapsed().as_millis(),
                        num_blocks = num_keys,
                        "Object layout created and registered (spawn_blocking)"
                    );

                    Ok(handle)
                })
                .await
                .map_err(|e| anyhow::anyhow!("spawn_blocking join error: {}", e))??;

                // Get destination handle
                let dst_handle = match request.to_pool() {
                    Device => self.device_handle.as_ref(),
                    Host => self.host_handle.as_ref(),
                    _ => None,
                }.ok_or_else(|| anyhow::anyhow!("Destination handle not available"))?;

                // Execute transfer: object storage (blocks 0..N) → dst blocks
                let src_block_ids: Vec<usize> = (0..blocks.len()).collect();

                // Build transfer options with bounce buffer if needed
                let options = if let Some(ref bounce) = bounce_spec {
                    TransferOptions::builder()
                        .bounce_buffer(bounce.clone())
                        .build()
                        .unwrap_or_default()
                } else {
                    TransferOptions::default()
                };

                let start = std::time::Instant::now();
                let num_blocks = blocks.len();
                let transfer_timeout = object_onboard_timeout();

                tracing::info!(
                    target: "blocking_ops",
                    num_blocks = num_blocks,
                    timeout_secs = transfer_timeout.as_secs(),
                    to = ?request.to_pool(),
                    needs_bounce = needs_bounce,
                    "BLOCKING: Starting object ONBOARD transfer (S3 read)"
                );

                let transfer_future = self.transport_manager
                    .execute_transfer(
                        object_handle,
                        &src_block_ids,
                        *dst_handle,
                        &dst_block_ids,
                        options,
                    )?;

                match timeout(transfer_timeout, transfer_future).await {
                    Ok(Ok(())) => {
                        let elapsed = start.elapsed();
                        let total_elapsed = transfer_start.elapsed();

                        tracing::info!(
                            target: "blocking_ops",
                            num_blocks = num_blocks,
                            elapsed_ms = elapsed.as_millis(),
                            "UNBLOCKED: Completed object ONBOARD transfer"
                        );

                        log_object_transfer_complete(
                            "OBJECT_ONBOARD_DONE",
                            num_blocks,
                            self.layout_config.as_ref(),
                            elapsed,
                            total_elapsed,
                            needs_bounce,
                        );
                    }
                    Ok(Err(e)) => {
                        tracing::error!(
                            num_blocks = num_blocks,
                            elapsed_ms = start.elapsed().as_millis(),
                            "OBJECT_ONBOARD_FAILED: transfer error: {}",
                            e
                        );
                        return Err(e);
                    }
                    Err(_) => {
                        tracing::error!(
                            num_blocks = num_blocks,
                            timeout_secs = transfer_timeout.as_secs(),
                            "OBJECT_ONBOARD_TIMEOUT: transfer exceeded {}s timeout",
                            transfer_timeout.as_secs()
                        );
                        return Err(anyhow::anyhow!(
                            "Object onboard timed out after {}s for {} blocks",
                            transfer_timeout.as_secs(),
                            num_blocks
                        ));
                    }
                }
            }
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported object storage transfer: {:?} -> {:?}",
                    request.from_pool(),
                    request.to_pool()
                ));
            }
        }

        Ok(())
    }
}

#[async_trait]
impl<T: ?Sized + BlockTransferHandler> Handler for T {
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
                .scheduler_client()
                .expect("scheduler client is required");

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
