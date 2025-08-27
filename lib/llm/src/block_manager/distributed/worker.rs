// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;

use transfer::*;
use utils::*;
use zmq::*;

use crate::block_manager::{
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
    block::{Block, layout_to_blocks, locality, transfer::TransferContext},
    connector::scheduler::TransferSchedulerClient,
    layout::LayoutType,
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

#[derive(Builder)]
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

    #[builder(default = "String::from(\"kvbm\")")]
    barrier_id: String,

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
    pub async fn new(config: KvbmWorkerConfig) -> anyhow::Result<Self> {
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

        let inner_dim = shape[2..].iter().product::<usize>() / config.page_size;

        tracing::info!(
            "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
            device_tensors.len(),
            outer_dim,
            config.page_size,
            inner_dim
        );

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(device_tensors.len())
            .outer_dim(outer_dim)
            .page_size(config.page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(config.dtype_width_bytes);

        let layout_type = LayoutType::LayerSeparate { outer_contiguous };

        let device_layout = layout_builder
            .num_blocks(config.num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors)?;

        let layout_builder_clone = layout_builder.clone();

        // add worker-connector scheduler here
        // let scheduler = KvbmWorkerScheduler::new(config.scheduler.clone());
        let cancel_token = config.drt.primary_token().clone();

        // establish a oneshot channel to get back the raw BlockTransferHandler
        let (handler_tx, handler_rx) = oneshot::channel();

        let scheduler_client = config.scheduler_client.clone();

        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder_clone,
                    layout_type,
                    config,
                    cancel_token,
                    handler_tx,
                    scheduler_client,
                )
            },
            cancel_token.clone(),
            "kvbm-worker-task",
        )?;

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
        })
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

    async fn worker_task(
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        mut layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        handler_tx: oneshot::Sender<BlockTransferHandler>,
        scheduler_client: Option<TransferSchedulerClient>,
    ) -> anyhow::Result<()> {
        let drt = config.drt.clone();

        let worker_id = drt
            .primary_lease()
            .ok_or(anyhow::anyhow!(
                "unable to get primary lease; check that drt is not static"
            ))?
            .id() as usize;

        tracing::info!(
            "Worker {} waiting on barrier {}",
            worker_id,
            config.barrier_id
        );

        let worker_barrier = WorkerBarrier::<KvbmLeaderData, KvbmWorkerData>::new(
            config.barrier_id,
            worker_id.to_string(),
        );

        let worker_data = KvbmWorkerData {
            num_device_blocks: config.num_device_blocks,
        };

        let leader_data = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Ok(())
            }
            leader_data = worker_barrier.sync(&drt, &worker_data) => {
                leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync worker barrier: {:?}", e))?;

        tracing::info!(
            "Worker {} received leader data: {:?}",
            worker_id,
            leader_data
        );

        let agent = build_agent(worker_id, leader_data.num_disk_blocks > 0)?;

        let transfer_context = Arc::new(TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(config.device_id)
                .unwrap()
                .ctx()
                .new_stream()
                .unwrap(),
            Handle::current(),
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

        // Create the handler for our active message worker.
        // Determine if host blocks will be contiguous based on layout type
        let host_blocks_contiguous = match layout_type {
            LayoutType::FullyContiguous => {
                tracing::info!("Host layout: FullyContiguous - enabling single-base scatter kernel optimization");
                true
            }
            LayoutType::LayerSeparate { outer_contiguous } => {
                tracing::info!(
                    "Host layout: LayerSeparate (outer_contiguous: {}) - enabling multi-base scatter kernel",
                    outer_contiguous
                );
                false  // LayerSeparate is never contiguous across blocks for our purposes
            }
        };

        let block_transfer_handler = BlockTransferHandler::new(
            device_blocks,
            host_blocks,
            disk_blocks,
            transfer_context,
            scheduler_client,
            host_blocks_contiguous,
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
