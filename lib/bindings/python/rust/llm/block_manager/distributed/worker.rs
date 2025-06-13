// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;

use transfer::*;
use utils::*;
use zmq::*;

use llm_rs::block_manager::{
    block::{layout_to_blocks, transfer::TransferContext, Block},
    layout::LayoutType,
    storage::{
        torch::TorchTensor, DeviceAllocator, DeviceStorage, DiskAllocator, DiskStorage,
        PinnedAllocator, PinnedStorage,
    },
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
};

use nixl_sys::Agent as NixlAgent;
use std::collections::HashMap;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::{utils::leader_worker_barrier::WorkerBarrier, DistributedRuntime, Runtime};

fn load_and_validate_tensors(
    tensors: Vec<Py<PyAny>>,
    device_id: usize,
) -> anyhow::Result<(Vec<DeviceStorage>, Vec<usize>)> {
    let mut device_tensors = Vec::with_capacity(tensors.len());

    let allocator = DeviceAllocator::new(device_id)?;

    let mut shape = None;
    for tensor in tensors {
        let vllm_tensor = VllmTensor::new(tensor)?;

        // Check the stride, and ensure our tensor is contiguous.
        // TODO: We eventually need to be able to handle this.
        let stride = vllm_tensor.stride();
        for i in 1..stride.len() {
            if stride[i] >= stride[i - 1] {
                return Err(anyhow::anyhow!(
                    "Tensor strides must be monotonically decreasing"
                ));
            }
        }

        // Check that all layer tensors have the same shape.
        // TODO: We eventually need to support the weirder models with heterogenous layers.
        if let Some(shape) = shape.as_ref() {
            if *shape != vllm_tensor.shape() {
                return Err(anyhow::anyhow!("All tensors must have the same shape"));
            }
        } else {
            shape = Some(vllm_tensor.shape());
        }

        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), Box::new(vllm_tensor))?;

        device_tensors.push(device_tensor);
    }

    Ok((device_tensors, shape.unwrap()))
}

#[pyclass]
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
        if let Some(agent) = agent {
            layout.nixl_register(agent, None)?;
        }

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
        let device_blocks = Some(Self::register_layout::<_, BasicMetadata>(
            device_layout,
            &transfer_context.nixl_agent().as_ref(),
            0,
            worker_id,
        )?);
        let host_blocks = host_layout
            .map(|layout| {
                Self::register_layout::<_, BasicMetadata>(
                    layout,
                    &transfer_context.nixl_agent().as_ref(),
                    1,
                    worker_id,
                )
            })
            .transpose()?;
        let disk_blocks = disk_layout
            .map(|layout| {
                Self::register_layout::<_, BasicMetadata>(
                    layout,
                    &transfer_context.nixl_agent().as_ref(),
                    2,
                    worker_id,
                )
            })
            .transpose()?;

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
        .map_err(|e| {
            to_pyerr(anyhow::anyhow!(format!(
                "Failed to sync worker barrier: {:?}",
                e
            )))
        })?;

        tracing::info!(
            "Worker {} received leader data: {:?}",
            worker_id,
            leader_data
        );

        let block_transfer_handler = BlockTransferHandler::new(
            device_blocks,
            host_blocks,
            disk_blocks,
            transfer_context,
            cancel_token.clone(),
        )?;

        let handlers = HashMap::from([(
            "transfer_blocks".to_string(),
            Arc::new(block_transfer_handler) as Arc<dyn Handler>,
        )]);

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

#[pymethods]
impl KvbmWorker {
    #[new]
    #[pyo3(signature = (num_layers, num_device_blocks, num_host_blocks, num_disk_blocks, outer_dim, page_size, inner_dim, tensors, device_id=0, worker_id=0, dtype=None, barrier_id="kvbm".to_string()))]
    fn new(
        num_layers: usize,
        num_device_blocks: usize,
        num_host_blocks: usize,
        num_disk_blocks: usize,
        outer_dim: usize,
        page_size: usize,
        inner_dim: usize,
        tensors: Vec<Py<PyAny>>,
        device_id: usize,
        worker_id: usize,
        dtype: Option<String>,
        barrier_id: String,
    ) -> PyResult<Self> {
        if num_layers == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_layers must be greater than 0",
            ));
        }

        if num_layers != tensors.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_layers must match the number of tensors",
            ));
        }

        let dtype = dtype.unwrap_or("fp16".to_string());

        tracing::info!("Initializing KvbmWorker with params: num_layers={}, num_device_blocks={}, num_host_blocks={}, num_disk_blocks={}, outer_dim={}, page_size={}, inner_dim={}, dtype={}", num_layers, num_device_blocks, num_host_blocks, num_disk_blocks, outer_dim, page_size, inner_dim, dtype);

        let (device_tensors, shape) =
            load_and_validate_tensors(tensors, device_id).map_err(to_pyerr)?;

        let outer_contiguous = shape[0] == outer_dim;

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(num_layers)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype(map_dtype(&dtype).map_err(to_pyerr)?);

        let layout_type = LayoutType::LayerSeparate { outer_contiguous };

        if num_device_blocks == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "num_device_blocks must be greater than 0",
            ));
        }

        let device_layout = layout_builder
            .num_blocks(num_device_blocks)
            .build()
            .map_err(to_pyerr)?
            .create_layout(layout_type, device_tensors, true)
            .map_err(to_pyerr)?;

        let host_layout = if num_host_blocks > 0 {
            let host_allocator = Arc::new(PinnedAllocator::default());
            Some(
                layout_builder
                    .num_blocks(num_host_blocks)
                    .build()
                    .map_err(to_pyerr)?
                    .allocate_layout(layout_type, host_allocator)
                    .map_err(to_pyerr)?,
            )
        } else {
            None
        };

        let disk_layout = if num_disk_blocks > 0 {
            if num_host_blocks == 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "num_host_blocks must be greater than 0 if num_disk_blocks is greater than 0",
                ));
            }
            let disk_allocator = Arc::new(DiskAllocator::default());
            Some(
                layout_builder
                    .num_blocks(num_disk_blocks)
                    .build()
                    .map_err(to_pyerr)?
                    .allocate_layout(layout_type, disk_allocator)
                    .map_err(to_pyerr)?,
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
