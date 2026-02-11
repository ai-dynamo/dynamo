// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use async_trait::async_trait;
use transfer::*;
use utils::*;
use zmq::*;

use crate::block_manager::{
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
    block::{
        Block, layout_to_blocks, locality,
        transfer::{PoolConfig, TransferContext},
    },
    connector::scheduler::TransferSchedulerClient,
    layout::LayoutType,
    offload::{MAX_CONCURRENT_TRANSFERS, MAX_TRANSFER_BATCH_SIZE},
    storage::{DeviceAllocator, DeviceStorage, DiskAllocator, PinnedAllocator, torch::TorchTensor},
};

use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use transfer::CacheGroup;

use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::sync::{Mutex, RwLock, oneshot};

struct WorkerState {
    ready_for_ping: AtomicBool,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            ready_for_ping: AtomicBool::new(false),
        }
    }
    fn mark_ready(&self) {
        self.ready_for_ping.store(true, Ordering::SeqCst);
    }
    fn is_ready(&self) -> bool {
        self.ready_for_ping.load(Ordering::SeqCst)
    }
}

/// Configuration for an additional cache group (e.g. DSA indexer k cache).
/// Each additional group has its own tensors and dtype but shares `num_blocks` and `page_size`
/// with the primary cache.
#[derive(Clone)]
pub struct AdditionalCacheGroupConfig {
    pub name: String,
    pub tensors: Vec<Arc<dyn TorchTensor>>,
    pub dtype_width_bytes: usize,
}

pub fn load_and_validate_tensors(
    tensors: &[Arc<dyn TorchTensor>],
    device_id: usize,
) -> anyhow::Result<(Vec<DeviceStorage>, Vec<usize>)> {
    let mut shape = None;

    let mut device_tensors = Vec::with_capacity(tensors.len());
    let allocator = DeviceAllocator::new(device_id)?;

    for tensor in tensors {
        // Check the stride, and ensure our tensor is contiguous.
        // TODO: We eventually need to be able to handle this.
        let stride = tensor.stride();
        tracing::debug!("stride: {:?}", stride);
        tracing::debug!("stride is monotonically decreasing for NHD layout");
        tracing::debug!("stride is NOT monotonically decreasing for HND layout");

        // Check that all layer tensors have the same shape.
        // TODO: We eventually need to support the weirder models with heterogenous layers.
        if let Some(shape) = shape.as_ref() {
            if *shape != tensor.shape() {
                return Err(anyhow::anyhow!(
                    "All tensors must have the same shape! Got {:?} and {:?}",
                    *shape,
                    tensor.shape()
                ));
            }
        } else {
            shape = Some(tensor.shape());
        }

        // Build the storage object from the tensor.
        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), tensor.clone())?;

        device_tensors.push(device_tensor);
    }

    Ok((device_tensors, shape.unwrap()))
}

fn build_agent(worker_id: usize, use_gds: bool) -> anyhow::Result<NixlAgent> {
    let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id))?;
    if use_gds {
        let (_, gds_params) = agent.get_plugin_params("GDS_MT")?;
        agent.create_backend("GDS_MT", &gds_params)?;
    }
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    agent.create_backend("POSIX", &posix_params)?;

    Ok(agent)
}

// Helper: perform allocation and build transfer handler (factored from previous code)
#[allow(clippy::too_many_arguments)]
async fn perform_allocation_and_build_handler(
    device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
    mut layout_builder: LayoutConfigBuilder,
    worker_config: KvbmWorkerConfig,
    leader_meta: LeaderMetadata,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
    additional_device_layouts: Vec<(
        String,
        Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        LayoutConfigBuilder,
    )>,
) -> anyhow::Result<BlockTransferHandler> {
    // Determine if this rank should allocate G2/G3 (host/disk)
    // - Sharded mode (rank=None): all ranks allocate
    // - Replicated mode (rank=Some(r)): only rank 0 allocates
    let should_allocate_offload = match worker_config.rank {
        None => true,     // Sharded mode: all ranks allocate
        Some(0) => true,  // Replicated mode rank 0: allocate
        Some(_) => false, // Replicated mode non-rank0: skip
    };

    if !should_allocate_offload {
        tracing::info!(
            "Rank {} skipping host/disk allocation (replicated mode)",
            worker_config.rank.unwrap_or(-1)
        );
    }

    // Only create NIXL agent if we need disk blocks AND we should allocate
    let need_disk = should_allocate_offload && leader_meta.num_disk_blocks > 0;
    let agent = build_agent(worker_id, need_disk)?;
    let pool_config = PoolConfig {
        enable_pool: true,
        max_concurrent_transfers: MAX_CONCURRENT_TRANSFERS,
        max_transfer_batch_size: MAX_TRANSFER_BATCH_SIZE,
        num_outer_components: device_layout.config().outer_dim,
        num_layers: device_layout.config().num_layers,
    };
    let transfer_context = Arc::new(
        TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(device_id)?.ctx().new_stream()?,
            Handle::current(),
            Some(pool_config),
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "Failed to create transfer context for worker {} with CUDA memory pool: {}. \
                 This is a critical error - the worker cannot start without CUDA memory pools. \
                 Please ensure sufficient GPU memory is available on device {}.",
                worker_id,
                e,
                device_id
            )
        })?,
    );

    // --- Primary cache group ---
    let primary_device_blocks = Some(KvbmWorker::make_layout::<_, BasicMetadata>(
        device_layout,
        transfer_context.nixl_agent().as_ref(),
        0,
        worker_id,
    )?);

    let primary_host_blocks = if should_allocate_offload && leader_meta.num_host_blocks > 0 {
        let host_allocator = Arc::new(PinnedAllocator::default());
        let host_layout = layout_builder
            .num_blocks(leader_meta.num_host_blocks)
            .build()?
            .allocate_layout(worker_config.host_layout_type, host_allocator)?;
        Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            host_layout,
            transfer_context.nixl_agent().as_ref(),
            1,
            worker_id,
        )?)
    } else {
        None
    };

    let primary_disk_blocks = if should_allocate_offload && leader_meta.num_disk_blocks > 0 {
        let disk_allocator = Arc::new(DiskAllocator);
        let disk_layout = layout_builder
            .num_blocks(leader_meta.num_disk_blocks)
            .build()?
            .allocate_layout(worker_config.disk_layout_type, disk_allocator)?;
        Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            disk_layout,
            transfer_context.nixl_agent().as_ref(),
            2,
            worker_id,
        )?)
    } else {
        None
    };

    let primary_group = CacheGroup {
        name: "primary".to_string(),
        device: BlockTransferHandler::get_local_data(primary_device_blocks),
        host: BlockTransferHandler::get_local_data(primary_host_blocks),
        disk: BlockTransferHandler::get_local_data(primary_disk_blocks),
    };

    let mut cache_groups = vec![primary_group];

    // --- Additional cache groups (e.g. DSA indexer k cache) ---
    // block_set_idx starts at 3 (0=primary device, 1=primary host, 2=primary disk)
    let mut block_set_idx = 3;
    for (name, additional_device_layout, mut additional_layout_builder) in
        additional_device_layouts
    {
        tracing::info!(
            "Building additional cache group '{}': num_layers={}, outer_dim={}, inner_dim={}, dtype_bytes={}",
            name,
            additional_device_layout.config().num_layers,
            additional_device_layout.config().outer_dim,
            additional_device_layout.config().inner_dim,
            additional_device_layout.config().dtype_width_bytes
        );

        let add_device_blocks = Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            additional_device_layout,
            transfer_context.nixl_agent().as_ref(),
            block_set_idx,
            worker_id,
        )?);
        block_set_idx += 1;

        let add_host_blocks = if should_allocate_offload && leader_meta.num_host_blocks > 0 {
            let host_allocator = Arc::new(PinnedAllocator::default());
            let host_layout = additional_layout_builder
                .num_blocks(leader_meta.num_host_blocks)
                .build()?
                .allocate_layout(worker_config.host_layout_type, host_allocator)?;
            Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                host_layout,
                transfer_context.nixl_agent().as_ref(),
                block_set_idx,
                worker_id,
            )?)
        } else {
            None
        };
        block_set_idx += 1;

        let add_disk_blocks = if should_allocate_offload && leader_meta.num_disk_blocks > 0 {
            let disk_allocator = Arc::new(DiskAllocator);
            let disk_layout = additional_layout_builder
                .num_blocks(leader_meta.num_disk_blocks)
                .build()?
                .allocate_layout(worker_config.disk_layout_type, disk_allocator)?;
            Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                disk_layout,
                transfer_context.nixl_agent().as_ref(),
                block_set_idx,
                worker_id,
            )?)
        } else {
            None
        };
        block_set_idx += 1;

        cache_groups.push(CacheGroup {
            name,
            device: BlockTransferHandler::get_local_data(add_device_blocks),
            host: BlockTransferHandler::get_local_data(add_host_blocks),
            disk: BlockTransferHandler::get_local_data(add_disk_blocks),
        });
    }

    let handler = BlockTransferHandler::new_multi(
        cache_groups,
        transfer_context,
        scheduler_client,
        worker_config.nccl_config,
    )?;
    Ok(handler)
}

struct WorkerMetadataHandler {
    num_device_blocks: usize,
    bytes_per_block: usize,
}

#[async_trait]
impl Handler for WorkerMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        let payload = bincode::serde::encode_to_vec(
            &WorkerMetadata {
                num_device_blocks: self.num_device_blocks,
                bytes_per_block: self.bytes_per_block,
            },
            bincode::config::standard(),
        )?;
        message
            .reply(ZMQ_WORKER_METADATA_MESSAGE, &[payload])
            .await?;
        Ok(())
    }
}

// Leader sends allocation config -> allocate -> publish handler -> mark ready -> ACK
struct LeaderMetadataHandler {
    state: Arc<WorkerState>,
    device_layout: Mutex<Option<Box<dyn NixlLayout<StorageType = DeviceStorage>>>>,
    layout_builder: LayoutConfigBuilder,
    worker_config: KvbmWorkerConfig,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
    handler_cell: Arc<RwLock<Option<BlockTransferHandler>>>,
    handler_tx: Arc<Mutex<Option<oneshot::Sender<BlockTransferHandler>>>>,
    started: AtomicBool,
    additional_device_layouts: Mutex<
        Option<
            Vec<(
                String,
                Box<dyn NixlLayout<StorageType = DeviceStorage>>,
                LayoutConfigBuilder,
            )>,
        >,
    >,
}

#[async_trait]
impl Handler for LeaderMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        // Always ACK ASAP so Drop can't panic and leader can finish the round.
        if let Err(e) = message.ack().await {
            tracing::error!("leader_metadata: failed to ACK: {e:#}");
        }

        // Validate payload; if bad, ignore.
        if message.data.len() != 1 {
            tracing::error!(
                "leader_metadata expects 1 payload frame (got {})",
                message.data.len()
            );
            return Ok(());
        }
        let leader_meta: LeaderMetadata = match bincode::serde::decode_from_slice(
            &message.data[0],
            bincode::config::standard(),
        ) {
            Ok((m, _)) => m,
            Err(e) => {
                tracing::error!("leader_metadata: bad payload: {e:#}");
                return Ok(());
            }
        };

        // Single-flight: only the first message triggers allocation.
        if self
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::debug!("leader_metadata: allocation already started; dropping duplicate");
            return Ok(());
        }

        // Take device_layout once.
        let dev_layout = {
            let mut guard = self.device_layout.lock().await;
            match guard.take() {
                Some(d) => d,
                None => {
                    tracing::warn!("leader_metadata: device_layout already consumed; dropping");
                    return Ok(());
                }
            }
        };

        // Take additional device layouts once.
        let additional_layouts = {
            let mut guard = self.additional_device_layouts.lock().await;
            guard.take().unwrap_or_default()
        };

        // Capture what we need and run allocation in the background.
        let layout_builder = self.layout_builder.clone();
        let worker_config = self.worker_config.clone();
        let worker_id = self.worker_id;
        let device_id = self.device_id;
        let scheduler_client = self.scheduler_client.clone();
        let handler_cell = self.handler_cell.clone();
        let handler_tx = self.handler_tx.clone();
        let state = self.state.clone();

        tokio::spawn(async move {
            match perform_allocation_and_build_handler(
                dev_layout,
                layout_builder,
                worker_config,
                leader_meta,
                worker_id,
                device_id,
                scheduler_client,
                additional_layouts,
            )
            .await
            {
                Ok(handler) => {
                    // Install transfer handler
                    {
                        let mut w = handler_cell.write().await;
                        *w = Some(handler.clone());
                    }
                    // Return handler to creator (once)
                    {
                        let mut g = handler_tx.lock().await;
                        if let Some(tx) = g.take() {
                            let _ = tx.send(handler);
                        }
                    }
                    // Now the worker can ACK pings
                    state.mark_ready();
                    tracing::info!("allocation finished; worker is ping-ACK-able");
                }
                Err(e) => {
                    tracing::error!("allocation failed: {e:#}");
                    // leave ready=false so pings keep being ignored
                }
            }
        });

        Ok(())
    }
}

// Gated ping: the worker can only response to ping after the state is ready
struct GatedPing {
    state: Arc<WorkerState>,
    // fired exactly once after the first successful ping ACK
    layout_ready_tx: Mutex<Option<oneshot::Sender<String>>>,
}

#[async_trait]
impl Handler for GatedPing {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        if !self.state.is_ready() {
            tracing::info!(
                "KVBM worker is under initialization. It could take a while if set with large CPU or DISK cache size. Please wait..."
            );
            tracing::debug!("Ping received but worker not ready; deferring ACK");
            // Prevent Drop panic; leader won't get an ACK for this round and will retry.
            message.mark_handled();
            return Ok(());
        }

        message.ack().await?;

        // After a successful ACK, flip the readiness oneshot exactly once
        let mut guard = self.layout_ready_tx.lock().await;
        if let Some(tx) = guard.take() {
            let _ = tx.send("ping-acked".to_string());
            tracing::info!("Reported ping-ready after first ACK");
        }

        Ok(())
    }
}

// Transfer dispatcher that waits until block transfer handler exists
struct BlockTransferDispatch {
    cell: Arc<RwLock<Option<BlockTransferHandler>>>,
}

#[async_trait]
impl Handler for BlockTransferDispatch {
    async fn handle(&self, message: MessageHandle) -> anyhow::Result<()> {
        let maybe = { self.cell.read().await.clone() };
        if let Some(inner) = maybe {
            inner.handle(message).await
        } else {
            Err(anyhow::anyhow!("transfer handler not ready yet"))
        }
    }
}

#[derive(Builder, Clone)]
#[builder(pattern = "owned")]
pub struct KvbmWorkerConfig {
    cancel_token: CancellationToken,

    num_device_blocks: usize,

    #[builder(default = "32")]
    page_size: usize,

    #[builder(default = "Vec::new()")]
    tensors: Vec<Arc<dyn TorchTensor>>,

    #[builder(default = "0")]
    device_id: usize,

    #[builder(default = "2")]
    dtype_width_bytes: usize,

    #[builder(default = "LayoutType::FullyContiguous")]
    device_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    host_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    disk_layout_type: LayoutType,

    #[builder(default = "None")]
    scheduler_client: Option<TransferSchedulerClient>,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56001\")")]
    leader_pub_url: String,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56002\")")]
    leader_ack_url: String,

    /// Rank for replicated mode (None = sharded mode)
    #[builder(default = "None")]
    rank: Option<i32>,

    /// World size for replicated mode
    #[builder(default = "None")]
    world_size: Option<i32>,

    /// NCCL configuration for replicated mode
    #[builder(default = "transfer::NcclConfig::disabled()")]
    nccl_config: transfer::NcclConfig,

    /// Additional cache groups (e.g. DSA indexer k cache) that move in lockstep with primary
    #[builder(default = "Vec::new()")]
    additional_cache_groups: Vec<AdditionalCacheGroupConfig>,
}

impl KvbmWorkerConfig {
    pub fn builder() -> KvbmWorkerConfigBuilder {
        KvbmWorkerConfigBuilder::default()
    }
}

pub struct KvbmWorker {
    task: Option<CriticalTaskExecutionHandle>,
    block_transfer_handler_rx: Option<oneshot::Receiver<transfer::BlockTransferHandler>>,
}

impl KvbmWorker {
    pub async fn new(config: KvbmWorkerConfig, layout_blocking: bool) -> anyhow::Result<Self> {
        tracing::info!(
            "Initializing KvbmWorker with params: num_device_blocks={}, page_size={}, dtype_width_bytes={}",
            config.num_device_blocks,
            config.page_size,
            config.dtype_width_bytes
        );

        if config.num_device_blocks == 0 {
            return Err(anyhow::anyhow!("num_device_blocks must be greater than 0"));
        }

        let (device_tensors, shape) = load_and_validate_tensors(&config.tensors, config.device_id)?;

        if shape.len() < 3 {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        }

        let (layout_type, num_layers, outer_dim, inner_dim) = match config.device_layout_type {
            LayoutType::FullyContiguous => {
                let num_layers = shape[1];
                let outer_dim = shape[2];
                let inner_dim = shape[3..].iter().product::<usize>() / config.page_size;
                tracing::info!(
                    "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
                    num_layers,
                    outer_dim,
                    config.page_size,
                    inner_dim
                );

                (
                    LayoutType::FullyContiguous,
                    num_layers,
                    outer_dim,
                    inner_dim,
                )
            }
            LayoutType::LayerSeparate { outer_contiguous } => {
                // Use the already-detected layout type from config (no re-detection needed)
                let layout_type = config.device_layout_type;

                // Extract outer_dim based on the provided outer_contiguous value
                let outer_dim = if outer_contiguous {
                    shape[0] // Outer contiguous: [outer_dim, n_blocks, ...]
                } else {
                    shape[1] // Block contiguous: [n_blocks, outer_dim, ...]
                };

                let num_layers = device_tensors.len();
                let inner_dim = shape[2..].iter().product::<usize>() / config.page_size;

                tracing::info!(
                    "Inferred layout: num_layers={}, outer_dim={}, outer_contiguous={}, page_size={}, inner_dim={}",
                    num_layers,
                    outer_dim,
                    outer_contiguous,
                    config.page_size,
                    inner_dim
                );

                (layout_type, num_layers, outer_dim, inner_dim)
            }
        };

        let mut bytes_per_block =
            num_layers * outer_dim * config.page_size * inner_dim * config.dtype_width_bytes;

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(num_layers)
            .outer_dim(outer_dim)
            .page_size(config.page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(config.dtype_width_bytes);

        let device_layout = layout_builder
            .num_blocks(config.num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors)?;

        let layout_builder = layout_builder.clone();

        // Process additional cache groups (e.g. DSA indexer k cache)
        let mut additional_device_layouts: Vec<(
            String,
            Box<dyn NixlLayout<StorageType = DeviceStorage>>,
            LayoutConfigBuilder,
        )> = Vec::new();

        for additional_group in &config.additional_cache_groups {
            let (add_device_tensors, add_shape) =
                load_and_validate_tensors(&additional_group.tensors, config.device_id)?;

            // Validate same num_device_blocks (shape[0])
            let add_num_device_blocks = add_shape[0];
            if add_num_device_blocks != config.num_device_blocks {
                return Err(anyhow::anyhow!(
                    "Additional cache group '{}' has {} device blocks, but primary has {}. They must match.",
                    additional_group.name,
                    add_num_device_blocks,
                    config.num_device_blocks
                ));
            }

            if add_shape.len() < 3 {
                return Err(anyhow::anyhow!(
                    "Additional cache group '{}' has unsupported shape: {:?}",
                    additional_group.name,
                    add_shape
                ));
            }

            // Infer layout dimensions for the additional cache.
            // FullyContiguous layout: [num_blocks, num_layers, outer_dim, page_size * inner_dim]
            let add_num_layers = add_shape[1];
            let add_outer_dim = add_shape[2];
            let add_inner_dim =
                add_shape[3..].iter().product::<usize>() / config.page_size;

            tracing::info!(
                "Additional cache group '{}': num_layers={}, outer_dim={}, page_size={}, inner_dim={}, dtype_bytes={}",
                additional_group.name,
                add_num_layers,
                add_outer_dim,
                config.page_size,
                add_inner_dim,
                additional_group.dtype_width_bytes
            );

            let add_bytes = add_num_layers
                * add_outer_dim
                * config.page_size
                * add_inner_dim
                * additional_group.dtype_width_bytes;
            bytes_per_block += add_bytes;

            let mut add_layout_builder_instance = LayoutConfigBuilder::default();
            let add_layout_builder = add_layout_builder_instance
                .num_layers(add_num_layers)
                .outer_dim(add_outer_dim)
                .page_size(config.page_size)
                .inner_dim(add_inner_dim)
                .dtype_width_bytes(additional_group.dtype_width_bytes);

            let add_device_layout = add_layout_builder
                .num_blocks(config.num_device_blocks)
                .build()?
                .create_layout(LayoutType::FullyContiguous, add_device_tensors)?;

            let add_layout_builder = add_layout_builder.clone();

            additional_device_layouts.push((
                additional_group.name.clone(),
                add_device_layout,
                add_layout_builder,
            ));
        }

        tracing::info!(
            "Total bytes_per_block (primary + {} additional groups): {}",
            additional_device_layouts.len(),
            bytes_per_block
        );

        let (task, handler_rx) = if layout_blocking {
            Self::run_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
                additional_device_layouts,
            )
            .await?
        } else {
            Self::run_non_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
                additional_device_layouts,
            )
            .await?
        };

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
        })
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
        additional_device_layouts: Vec<(
            String,
            Box<dyn NixlLayout<StorageType = DeviceStorage>>,
            LayoutConfigBuilder,
        )>,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.cancel_token.clone();

        // establish a oneshot channel to get back the raw BlockTransferHandler
        let (handler_tx, handler_rx) = oneshot::channel();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // establish a oneshot channel to block on the main routine to wait for layout allocation readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        let scheduler_client = config.scheduler_client.clone();

        let worker_config = config.clone();
        // start background worker task to do layout allocation for host or disk
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder,
                    layout_type,
                    worker_config,
                    cancel_token,
                    handler_tx_cell,
                    layout_ready_tx_cell,
                    scheduler_client,
                    bytes_per_block,
                    additional_device_layouts,
                )
            },
            cancel_token.clone(),
            "kvbm-worker-task",
        )?;

        // waiting for the worker layout allocation ready
        match layout_ready_rx.await {
            Ok(_) => tracing::info!("worker layout allocation finished."),
            Err(_) => tracing::error!("Worker layout dropped without sending"),
        }

        Ok((task, handler_rx))
    }

    #[allow(clippy::too_many_arguments)]
    async fn run_non_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage> + Send + 'static>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
        additional_device_layouts: Vec<(
            String,
            Box<dyn NixlLayout<StorageType = DeviceStorage>>,
            LayoutConfigBuilder,
        )>,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.cancel_token.clone();
        let scheduler_client = config.scheduler_client.clone();

        // channel to get BlockTransferHandler back to the caller
        let (handler_tx, handler_rx) = oneshot::channel::<transfer::BlockTransferHandler>();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // channel that the worker will use to signal layout readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        // clone what we need inside the orchestrator
        let worker_config = config.clone();
        let cancel_token_for_task = cancel_token.clone();

        // Single task that orchestrates everything in-order.
        let task = CriticalTaskExecutionHandle::new(
            move |ct| {
                let cfg = worker_config.clone();
                let scheduler = scheduler_client.clone();

                async move {
                    // Start the long-running worker.
                    let dev_layout = device_layout; // moved in
                    let lb = layout_builder; // moved in
                    let lt = layout_type; // moved in

                    let worker_fut = KvbmWorker::worker_task(
                        dev_layout,
                        lb,
                        lt,
                        cfg.clone(),
                        ct.clone(),
                        handler_tx_cell,
                        layout_ready_tx_cell,
                        scheduler,
                        bytes_per_block,
                        additional_device_layouts,
                    );

                    // If worker_task returns Result, handle/log it inside the spawned task.
                    tokio::spawn(async move {
                        if let Err(e) = worker_fut.await {
                            tracing::error!("worker_task exited with error: {e:#}");
                        }
                    });

                    // 3) wait for the worker's layout allocation readiness
                    match layout_ready_rx.await {
                        Ok(_) => tracing::info!("worker layout allocation finished."),
                        Err(_) => tracing::warn!("worker layout readiness channel dropped"),
                    }

                    Ok::<(), anyhow::Error>(())
                }
            },
            cancel_token_for_task,
            "kvbm-worker-task",
        )?;

        Ok((task, handler_rx))
    }

    /// One-time use method to extract the block transfer handler from the worker.
    ///
    /// This is a bit of a hack. Improve the API design around this in the future.
    pub fn block_transfer_handler_rx(
        &mut self,
    ) -> Option<tokio::sync::oneshot::Receiver<BlockTransferHandler>> {
        self.block_transfer_handler_rx.take()
    }

    fn make_layout<S: Storage, M: BlockMetadata>(
        mut layout: Box<dyn NixlLayout<StorageType = S>>,
        agent: &Option<NixlAgent>,
        block_set_idx: usize,
        worker_id: usize,
    ) -> anyhow::Result<Vec<Block<S, locality::Local, M>>> {
        // Register with NIXL, if applicable.
        if let Some(agent) = agent {
            layout.nixl_register(agent, None)?;
        }

        // Convert the layout into blocks.
        let layout: Arc<dyn NixlLayout<StorageType = S>> = Arc::from(layout);
        let blocks = layout_to_blocks::<_, M>(layout, block_set_idx, worker_id as u64)?;
        Ok(blocks)
    }

    #[allow(clippy::too_many_arguments)]
    async fn worker_task(
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        _device_layout_type: LayoutType,
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        handler_tx: Arc<Mutex<Option<oneshot::Sender<BlockTransferHandler>>>>,
        layout_ready_tx: tokio::sync::Mutex<Option<oneshot::Sender<String>>>,
        scheduler_client: Option<TransferSchedulerClient>,
        bytes_per_block: usize,
        additional_device_layouts: Vec<(
            String,
            Box<dyn NixlLayout<StorageType = DeviceStorage>>,
            LayoutConfigBuilder,
        )>,
    ) -> anyhow::Result<()> {
        let worker_id = config.device_id;
        // Readiness gating for ping
        let state = Arc::new(WorkerState::new());

        // Cell to publish the transfer handler
        let transfer_handler_cell: Arc<RwLock<Option<BlockTransferHandler>>> =
            Arc::new(RwLock::new(None));

        // Build handlers map
        let mut handlers: HashMap<String, Arc<dyn Handler>> = HashMap::new();

        handlers.insert(
            ZMQ_PING_MESSAGE.to_string(),
            Arc::new(GatedPing {
                state: state.clone(),
                layout_ready_tx,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_WORKER_METADATA_MESSAGE.to_string(),
            Arc::new(WorkerMetadataHandler {
                num_device_blocks: config.num_device_blocks,
                bytes_per_block,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_LEADER_METADATA_MESSAGE.to_string(),
            Arc::new(LeaderMetadataHandler {
                state: state.clone(),
                device_layout: tokio::sync::Mutex::new(Some(device_layout)), // moved in
                layout_builder,                                              // moved
                worker_config: config.clone(),
                worker_id,
                device_id: config.device_id,
                scheduler_client,
                handler_cell: transfer_handler_cell.clone(),
                handler_tx, // sends BlockTransferHandler to caller
                started: AtomicBool::new(false),
                additional_device_layouts: tokio::sync::Mutex::new(Some(
                    additional_device_layouts,
                )),
            }) as Arc<dyn Handler>,
        );

        // transfer requests get dispatched to built handler (after allocation)
        handlers.insert(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(BlockTransferDispatch {
                cell: transfer_handler_cell.clone(),
            }) as Arc<dyn Handler>,
        );

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &config.leader_pub_url,
            &config.leader_ack_url,
            handlers,
            cancel_token.clone(),
        )?;

        // TODO: Some sort of fancy loop here.
        // For now, just wait for cancellation.
        cancel_token.cancelled().await;

        Ok(())
    }
}

impl Drop for KvbmWorker {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.cancel();
            task.detach();
        }
    }
}
