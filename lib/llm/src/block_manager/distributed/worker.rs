// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;

use transfer::*;
use utils::*;
use zmq::*;

use crate::block_manager::{
    block::{layout_to_blocks, transfer::TransferContext, Block},
    layout::LayoutType,
    storage::{
        torch::TorchTensor, DeviceAllocator, DeviceStorage, DiskAllocator, DiskStorage,
        PinnedAllocator, PinnedStorage,
    },
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
};
use crate::common::dtype::DType;

use nixl_sys::Agent as NixlAgent;
use std::collections::HashMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{utils::leader_worker_barrier::WorkerBarrier, DistributedRuntime, Runtime};

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

pub struct KvbmWorker {
    cancel_token: CancellationToken,
    task: Option<std::thread::JoinHandle<anyhow::Result<()>>>,
}

impl KvbmWorker {
    fn register_layout<S: Storage, M: BlockMetadata>(
        mut layout: Box<dyn NixlLayout<StorageType = S>>,
        agent: &Option<NixlAgent>,
        block_set_idx: usize,
        worker_id: usize,
    ) -> anyhow::Result<Vec<Block<S, M>>> {
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
        host_layout: Option<Box<dyn NixlLayout<StorageType = PinnedStorage>>>,
        disk_layout: Option<Box<dyn NixlLayout<StorageType = DiskStorage>>>,
        barrier_id: String,
        worker_id: usize,
        transfer_context: Arc<TransferContext>,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        // Build our device, host, and disk block lists.
        let device_blocks = Some(Self::register_layout::<_, BasicMetadata>(
            device_layout,
            transfer_context.nixl_agent().as_ref(),
            0,
            worker_id,
        )?);
        let host_blocks = host_layout
            .map(|layout| {
                Self::register_layout::<_, BasicMetadata>(
                    layout,
                    transfer_context.nixl_agent().as_ref(),
                    1,
                    worker_id,
                )
            })
            .transpose()?;
        let disk_blocks = disk_layout
            .map(|layout| {
                Self::register_layout::<_, BasicMetadata>(
                    layout,
                    transfer_context.nixl_agent().as_ref(),
                    2,
                    worker_id,
                )
            })
            .transpose()?;

        // Create the handler for our active message worker.
        let block_transfer_handler =
            BlockTransferHandler::new(device_blocks, host_blocks, disk_blocks, transfer_context)?;

        let handlers = HashMap::from([(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(block_transfer_handler) as Arc<dyn Handler>,
        )]);

        let runtime = Runtime::from_current()?;
        let drt = DistributedRuntime::from_settings(runtime).await?;

        tracing::info!("Worker {} waiting on barrier {}", worker_id, barrier_id);

        let worker_barrier =
            WorkerBarrier::<KvbmLeaderData>::new(barrier_id, worker_id.to_string());

        let leader_data = tokio::select! {
            _ = cancel_token.cancelled() => {
                return Ok(())
            }
            leader_data = worker_barrier.sync(&drt) => {
                leader_data
            }
        }
        .map_err(|e| anyhow::anyhow!("Failed to sync worker barrier: {:?}", e))?;

        tracing::info!(
            "Worker {} received leader data: {:?}",
            worker_id,
            leader_data
        );

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &leader_data.zmq_url,
            leader_data.broadcast_port,
            leader_data.ack_port,
            handlers,
            cancel_token,
        )?;

        // TODO: Some sort of fancy loop here.
        std::future::pending::<()>().await;

        Ok(())
    }
}

impl KvbmWorker {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_device_blocks: usize,
        num_host_blocks: usize,
        num_disk_blocks: usize,
        page_size: usize,
        tensors: Vec<Box<dyn TorchTensor>>,
        device_id: usize,
        worker_id: usize,
        dtype: DType,
        barrier_id: String,
    ) -> anyhow::Result<Self> {
        tracing::info!("Initializing KvbmWorker with params: num_device_blocks={}, num_host_blocks={}, num_disk_blocks={}, page_size={}, dtype={:?}", num_device_blocks, num_host_blocks, num_disk_blocks, page_size, dtype);

        if num_device_blocks == 0 {
            return Err(anyhow::anyhow!("num_device_blocks must be greater than 0"));
        } else if num_disk_blocks > 0 && num_host_blocks == 0 {
            return Err(anyhow::anyhow!(
                "Host offloading is required for disk offloading to be enabled."
            ));
        }

        let (device_tensors, shape) = load_and_validate_tensors(tensors, device_id)?;

        if shape.len() < 3 {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        }

        let (outer_contiguous, outer_dim) = if shape[0] == num_device_blocks {
            (false, shape[1])
        } else if shape[1] == num_device_blocks {
            (true, shape[0])
        } else {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        };

        let inner_dim = shape[2..].iter().product::<usize>() / page_size;

        tracing::info!(
            "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
            device_tensors.len(),
            outer_dim,
            page_size,
            inner_dim
        );

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(device_tensors.len())
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype(dtype);

        let layout_type = LayoutType::LayerSeparate { outer_contiguous };

        let device_layout = layout_builder
            .num_blocks(num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors, true)?;

        let host_layout = if num_host_blocks > 0 {
            let host_allocator = Arc::new(PinnedAllocator::default());
            Some(
                layout_builder
                    .num_blocks(num_host_blocks)
                    .build()?
                    .allocate_layout(layout_type, host_allocator)?,
            )
        } else {
            None
        };

        let disk_layout = if num_disk_blocks > 0 {
            if num_host_blocks == 0 {
                return Err(anyhow::anyhow!(
                    "num_host_blocks must be greater than 0 if num_disk_blocks is greater than 0"
                ));
            }
            let disk_allocator = Arc::new(DiskAllocator);
            Some(
                layout_builder
                    .num_blocks(num_disk_blocks)
                    .build()?
                    .allocate_layout(layout_type, disk_allocator)?,
            )
        } else {
            None
        };

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let task = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id)).unwrap();

            let transfer_context = Arc::new(TransferContext::new(
                Arc::new(Some(agent)),
                DeviceAllocator::new(device_id)
                    .unwrap()
                    .ctx()
                    .new_stream()
                    .unwrap(),
                runtime.handle().clone(),
            ));

            runtime.block_on(async move {
                KvbmWorker::worker_task(
                    device_layout,
                    host_layout,
                    disk_layout,
                    barrier_id,
                    worker_id,
                    transfer_context,
                    cancel_token_clone,
                )
                .await
            })
        });

        Ok(Self {
            cancel_token,
            task: Some(task),
        })
    }
}

impl Drop for KvbmWorker {
    fn drop(&mut self) {
        self.cancel_token.cancel();
        if let Some(task) = self.task.take() {
            task.join().unwrap().unwrap();
        }
    }
}
