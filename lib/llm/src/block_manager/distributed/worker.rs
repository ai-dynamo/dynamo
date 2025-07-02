// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;

use transfer::*;
use utils::*;
use zmq::*;

use crate::block_manager::{
    block::{layout_to_blocks, locality, transfer::TransferContext, Block},
    layout::LayoutType,
    storage::{torch::TorchTensor, DeviceAllocator, DeviceStorage, DiskAllocator, PinnedAllocator},
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
};

use derive_builder::Builder;
use nixl_sys::Agent as NixlAgent;
use std::collections::HashMap;
use std::sync::Arc;

use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{
    utils::{leader_worker_barrier::WorkerBarrier, task::CriticalTaskExecutionHandle},
    DistributedRuntime, Runtime,
};

fn load_and_validate_tensors(
    tensors: Vec<Box<dyn TorchTensor>>,
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
        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), tensor)?;

        device_tensors.push(device_tensor);
    }

    Ok((device_tensors, shape.unwrap()))
}

#[derive(Builder, Debug)]
#[builder(pattern = "owned")]
pub struct KvbmWorkerConfig {
    num_device_blocks: usize,

    #[builder(default = "32")]
    page_size: usize,

    #[builder(default = "Vec::new()")]
    tensors: Vec<Box<dyn TorchTensor>>,

    #[builder(default = "0")]
    device_id: usize,

    #[builder(default = "1")]
    worker_id: usize,

    #[builder(default = "2")]
    dtype_width_bytes: usize,

    #[builder(default = "String::from(\"kvbm\")")]
    barrier_id: String,
}

impl KvbmWorkerConfig {
    pub fn builder() -> KvbmWorkerConfigBuilder {
        KvbmWorkerConfigBuilder::default()
    }
}

fn build_agent(worker_id: usize) -> anyhow::Result<NixlAgent> {
    // TODO: Get GDS enabled here.
    // There seems to be some issue with NIXL that causes errors if a large amount of GDS backends are instantiated all at once.

    let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id))?;
    // let (_, gds_params) = agent.get_plugin_params("GDS")?;
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    // agent.create_backend("GDS", &gds_params)?;
    agent.create_backend("POSIX", &posix_params)?;

    Ok(agent)
}

pub struct KvbmWorker {
    task: Option<CriticalTaskExecutionHandle>,
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

        let (device_tensors, shape) = load_and_validate_tensors(config.tensors, config.device_id)?;

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

        let agent = build_agent(config.worker_id)?;

        let transfer_context = Arc::new(TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(config.device_id)
                .unwrap()
                .ctx()
                .new_stream()
                .unwrap(),
            Handle::current(),
        ));

        let cancel_token = CancellationToken::new();
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder_clone,
                    layout_type,
                    config.barrier_id,
                    config.worker_id,
                    transfer_context,
                    cancel_token,
                )
            },
            cancel_token,
            "kvbm-worker-task",
        )?;

        Ok(Self { task: Some(task) })
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
        barrier_id: String,
        worker_id: usize,
        transfer_context: Arc<TransferContext>,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        // Build our device, host, and disk block lists.
        let device_blocks = Some(Self::make_layout::<_, BasicMetadata>(
            device_layout,
            transfer_context.nixl_agent().as_ref(),
            0,
            worker_id,
        )?);

        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::from_settings(runtime).await?;

        tracing::info!("Worker {} waiting on barrier {}", worker_id, barrier_id);

        let worker_barrier =
            WorkerBarrier::<KvbmLeaderData, ()>::new(barrier_id, worker_id.to_string());

        let leader_data = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Ok(())
            }
            leader_data = worker_barrier.sync(&drt, &()) => {
                leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync worker barrier: {:?}", e))?;

        tracing::info!(
            "Worker {} received leader data: {:?}",
            worker_id,
            leader_data
        );

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
        let block_transfer_handler =
            BlockTransferHandler::new(device_blocks, host_blocks, disk_blocks, transfer_context)?;

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
