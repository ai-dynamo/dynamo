// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;

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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use tokio::runtime::Handle;
use tokio::sync::oneshot;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    DistributedRuntime,
    utils::{leader_worker_barrier::WorkerBarrier, task::CriticalTaskExecutionHandle},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvbmWorkerData {
    pub num_device_blocks: usize,
    pub bytes_per_block: usize,
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
        for i in 1..stride.len() {
            if stride[i] > stride[i - 1] {
                return Err(anyhow::anyhow!(
                    "Tensor strides must be monotonically decreasing! Got {:?}",
                    stride
                ));
            }
        }

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

#[derive(Builder, Clone)]
#[builder(pattern = "owned")]
pub struct KvbmWorkerConfig {
    drt: DistributedRuntime,

    num_device_blocks: usize,

    #[builder(default = "32")]
    page_size: usize,

    #[builder(default = "Vec::new()")]
    tensors: Vec<Arc<dyn TorchTensor>>,

    #[builder(default = "0")]
    device_id: usize,

    #[builder(default = "2")]
    dtype_width_bytes: usize,

    #[builder(default = false)]
    is_fully_contiguous_layout: bool,

    #[builder(default = "String::from(\"kvbm\")")]
    barrier_id_prefix: String,

    #[builder(default = "None")]
    scheduler_client: Option<TransferSchedulerClient>,
}

impl KvbmWorkerConfig {
    pub fn builder() -> KvbmWorkerConfigBuilder {
        KvbmWorkerConfigBuilder::default()
    }
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

        let (layout_type, num_layers, outer_dim, inner_dim) = if !config.is_fully_contiguous_layout
        {
            let (outer_contiguous, outer_dim) = if shape[0] >= config.num_device_blocks {
                (false, shape[1])
            } else if shape[1] >= config.num_device_blocks {
                (true, shape[0])
            } else {
                return Err(anyhow::anyhow!(format!(
                    "Unsupported kv cache layout. Got shape: {:?}",
                    shape
                )));
            };
            let num_layers = device_tensors.len();
            let inner_dim = shape[2..].iter().product::<usize>() / config.page_size;

            tracing::info!(
                "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
                device_tensors.len(),
                outer_dim,
                config.page_size,
                inner_dim
            );

            (
                LayoutType::LayerSeparate { outer_contiguous },
                num_layers,
                outer_dim,
                inner_dim,
            )
        } else {
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
        };

        let bytes_per_block =
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

        let (task, handler_rx) = if layout_blocking {
            Self::run_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        } else {
            Self::run_non_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        };

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
        })
    }

    async fn run_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.drt.primary_token().clone();

        // barrier sync with leader to get the leader data
        let leader_data = tokio::task::block_in_place(|| {
            // This is now synchronous blocking code
            // We need a separate current-thread runtime to block_on async calls here
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                KvbmWorker::leader_barrier_sync(
                    config.clone(),
                    cancel_token.clone(),
                    bytes_per_block,
                )
                .await
            })
        })?;

        // establish a oneshot channel to get back the raw BlockTransferHandler
        let (handler_tx, handler_rx) = oneshot::channel();

        // establish a oneshot channel to block on the main routine to wait for layout allocation readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();

        let scheduler_client = config.scheduler_client.clone();

        let worker_config = config.clone();
        // start background worker task to do layout allocation for host or disk
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder,
                    leader_data,
                    layout_type,
                    worker_config,
                    cancel_token,
                    handler_tx,
                    layout_ready_tx,
                    scheduler_client,
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

        let worker_config = config.clone();
        let cancel_for_barrier = cancel_token.clone();
        // wait until the leader finished the initialization of all components
        tokio::task::block_in_place(|| {
            // This is now synchronous blocking code
            // We need a separate current-thread runtime to block_on async calls here
            let rt = tokio::runtime::Handle::current();
            rt.block_on(async {
                KvbmWorker::leader_readiness_sync(worker_config, cancel_for_barrier).await
            })
        })?;

        Ok((task, handler_rx))
    }

    async fn run_non_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage> + Send + 'static>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<transfer::BlockTransferHandler>,
    )> {
        let cancel_token = config.drt.primary_token().clone();
        let scheduler_client = config.scheduler_client.clone();

        // channel to get BlockTransferHandler back to the caller
        let (handler_tx, handler_rx) = oneshot::channel::<transfer::BlockTransferHandler>();

        // channel that the worker will use to signal layout readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();

        // clone what we need inside the orchestrator
        let worker_config = config.clone();
        let cancel_token_for_task = cancel_token.clone();

        // Single task that orchestrates everything in-order.
        let task = CriticalTaskExecutionHandle::new(
            move |ct| {
                let cfg = worker_config.clone();
                let scheduler = scheduler_client.clone();

                async move {
                    // 1) barrier (must finish before worker_task starts)
                    let leader_data =
                        KvbmWorker::leader_barrier_sync(cfg.clone(), ct.clone(), bytes_per_block)
                            .await?;

                    // 2) start the long-running worker (after barrier)
                    //    Spawn it so the orchestrator can continue with readiness + waiting.
                    let dev_layout = device_layout; // moved in
                    let lb = layout_builder; // moved in
                    let lt = layout_type; // moved in

                    let worker_fut = KvbmWorker::worker_task(
                        dev_layout,
                        lb,
                        leader_data,
                        lt,
                        cfg.clone(),
                        ct.clone(),
                        handler_tx,
                        layout_ready_tx,
                        scheduler,
                    );

                    // If worker_task returns Result, handle/log it inside the spawned task.
                    tokio::spawn(async move {
                        if let Err(e) = worker_fut.await {
                            tracing::error!("worker_task exited with error: {e:#}");
                        }
                    });

                    // 3) wait for the worker’s layout allocation readiness
                    match layout_ready_rx.await {
                        Ok(_) => tracing::info!("worker layout allocation finished."),
                        Err(_) => tracing::warn!("worker layout readiness channel dropped"),
                    }

                    // 4) wait for leader to finish its side of initialization
                    KvbmWorker::leader_readiness_sync(cfg.clone(), ct.clone()).await?;

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

    async fn leader_barrier_sync(
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        bytes_per_block: usize,
    ) -> anyhow::Result<KvbmLeaderData> {
        let drt = config.drt.clone();

        let worker_id = drt
            .primary_lease()
            .ok_or(anyhow::anyhow!(
                "unable to get primary lease; check that drt is not static"
            ))?
            .id() as usize;

        let barrier_id_worker_to_leader =
            format!("{}{}", config.barrier_id_prefix, "-worker-to-leader");
        tracing::info!(
            "Worker {} waiting on barrier {}",
            worker_id,
            barrier_id_worker_to_leader
        );

        let worker_to_leader_barrier = WorkerBarrier::<(), KvbmWorkerData>::new(
            barrier_id_worker_to_leader,
            worker_id.to_string(),
        );

        let worker_data = KvbmWorkerData {
            num_device_blocks: config.num_device_blocks,
            bytes_per_block,
        };

        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(anyhow::anyhow!("Cancelled"))
            }
            _leader_data = worker_to_leader_barrier.sync(&drt, &worker_data) => {
                _leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync worker to leader barrier: {:?}", e))?;

        tracing::debug!(
            "Worker {} sent the worker data in worker to leader phase",
            worker_id
        );

        let barrier_id_leader_to_worker =
            format!("{}{}", config.barrier_id_prefix, "-leader-to-worker");
        tracing::info!(
            "Worker {} waiting on barrier {}",
            worker_id,
            barrier_id_leader_to_worker
        );

        let leader_to_worker_barrier = WorkerBarrier::<KvbmLeaderData, ()>::new(
            barrier_id_leader_to_worker,
            worker_id.to_string(),
        );

        let leader_data = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(anyhow::anyhow!("Cancelled"))
            }
            leader_data = leader_to_worker_barrier.sync(&drt, &()) => {
                leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync worker to leader barrier: {:?}", e))?;

        tracing::info!(
            "Worker {} received leader data: {:?}",
            worker_id,
            leader_data
        );

        Ok(leader_data)
    }

    async fn leader_readiness_sync(
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let drt = config.drt.clone();

        let worker_id = drt
            .primary_lease()
            .ok_or(anyhow::anyhow!(
                "unable to get primary lease; check that drt is not static"
            ))?
            .id() as usize;

        let barrier_id_leader_readiness =
            format!("{}{}", config.barrier_id_prefix, "-leader-ready");
        tracing::info!(
            "Worker {} waiting on barrier {}",
            worker_id,
            barrier_id_leader_readiness
        );

        let leader_readiness_barrier =
            WorkerBarrier::<(), ()>::new(barrier_id_leader_readiness, worker_id.to_string());

        // leader_data is not important in the leader readiness case
        tokio::select! {
            _ = cancel_token.cancelled() => {
                return Err(anyhow::anyhow!("Cancelled"))
            }
            _leader_data = leader_readiness_barrier.sync(&drt, &()) => {
                _leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync leader readiness barrier: {:?}", e))?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn worker_task(
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        mut layout_builder: LayoutConfigBuilder,
        leader_data: KvbmLeaderData,
        layout_type: LayoutType,
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        handler_tx: oneshot::Sender<BlockTransferHandler>,
        layout_ready_tx: oneshot::Sender<String>,
        scheduler_client: Option<TransferSchedulerClient>,
    ) -> anyhow::Result<()> {
        let drt = config.drt.clone();

        let worker_id = drt
            .primary_lease()
            .ok_or(anyhow::anyhow!(
                "unable to get primary lease; check that drt is not static"
            ))?
            .id() as usize;

        let agent = build_agent(worker_id, leader_data.num_disk_blocks > 0)?;

        let pool_config = PoolConfig {
            enable_pool: true,
            max_concurrent_transfers: MAX_CONCURRENT_TRANSFERS,
            max_transfer_batch_size: MAX_TRANSFER_BATCH_SIZE,
            num_outer_components: device_layout.config().outer_dim,
            num_layers: device_layout.config().num_layers,
        };

        let transfer_context = Arc::new(TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(config.device_id)
                .unwrap()
                .ctx()
                .new_stream()
                .unwrap(),
            Handle::current(),
            Some(pool_config),
        ));

        // Build our device, host, and disk block lists.
        let device_blocks = Some(Self::make_layout::<_, BasicMetadata>(
            device_layout,
            transfer_context.nixl_agent().as_ref(),
            0,
            worker_id,
        )?);

        let host_blocks = if leader_data.num_host_blocks > 0 {
            let host_allocator = Arc::new(PinnedAllocator::default());
            let host_layout = layout_builder
                .num_blocks(leader_data.num_host_blocks)
                .build()?
                .allocate_layout(layout_type, host_allocator)?;

            Some(Self::make_layout::<_, BasicMetadata>(
                host_layout,
                transfer_context.nixl_agent().as_ref(),
                1,
                worker_id,
            )?)
        } else {
            None
        };

        let disk_blocks = if leader_data.num_disk_blocks > 0 {
            let disk_allocator = Arc::new(DiskAllocator);
            let disk_layout = layout_builder
                .num_blocks(leader_data.num_disk_blocks)
                .build()?
                .allocate_layout(layout_type, disk_allocator)?;

            Some(Self::make_layout::<_, BasicMetadata>(
                disk_layout,
                transfer_context.nixl_agent().as_ref(),
                2,
                worker_id,
            )?)
        } else {
            None
        };

        let block_transfer_handler = BlockTransferHandler::new(
            device_blocks,
            host_blocks,
            disk_blocks,
            transfer_context,
            scheduler_client,
        )?;

        tracing::debug!("sending block transfer handler to worker");
        handler_tx
            .send(block_transfer_handler.clone())
            .map_err(|_| {
                anyhow::anyhow!("Failed to send block transfer handler over oneshot channel")
            })?;
        tracing::debug!("sent block transfer handler to worker");

        let handlers = HashMap::from([(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(block_transfer_handler) as Arc<dyn Handler>,
        )]);

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &leader_data.pub_url,
            &leader_data.ack_url,
            handlers,
            cancel_token.clone(),
        )?;

        if layout_ready_tx.send("finished".to_string()).is_err() {
            tracing::error!("worker receiver dropped before result was sent");
        }

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
