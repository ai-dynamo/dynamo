// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use leader::KvbmLeaderData;
use utils::*;
use zmq::*;

use llm_rs::block_manager::{
    block::layout_to_blocks,
    layout::LayoutType,
    storage::{torch::TorchTensor, DeviceAllocator, DeviceStorage},
    BasicMetadata, LayoutConfig, NixlLayout,
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
    // The DistributedRuntime only stores a handle, so we need to keep the runtime around.
    _runtime: tokio::runtime::Runtime,
}

impl KvbmWorker {
    async fn worker_task(
        mut device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        barrier_id: String,
        worker_id: usize,
        cancel_token: CancellationToken,
    ) -> anyhow::Result<()> {
        let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id)).map_err(to_pyerr)?;

        device_layout
            .nixl_register(&agent, None)
            .map_err(to_pyerr)?;
        let device_layout: Arc<dyn NixlLayout<StorageType = DeviceStorage>> =
            Arc::from(device_layout);
        let _device_blocks =
            layout_to_blocks::<_, BasicMetadata>(device_layout, 0, worker_id as u64)?;

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
        }.map_err(|e| {
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

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &leader_data.zmq_url,
            leader_data.broadcast_port,
            leader_data.ack_port,
            HashMap::new(),
        )?;

        // TODO: Some sort of fancy loop here.
        std::future::pending::<()>().await;

        Ok(())
    }
}

#[pymethods]
impl KvbmWorker {
    #[new]
    #[pyo3(signature = (num_layers, num_blocks, outer_dim, page_size, inner_dim, tensors, device_id=0, worker_id=0, dtype=None, barrier_id="kvbm".to_string()))]
    fn new(
        num_layers: usize,
        num_blocks: usize,
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

        tracing::info!("Initializing KvbmWorker with params: num_layers={}, num_blocks={}, outer_dim={}, page_size={}, inner_dim={}, dtype={}", num_layers, num_blocks, outer_dim, page_size, inner_dim, dtype);

        let (device_tensors, shape) =
            load_and_validate_tensors(tensors, device_id).map_err(to_pyerr)?;

        let outer_contiguous = shape[0] == outer_dim;

        let layout_builder = LayoutConfig::builder()
            .num_layers(num_layers)
            .num_blocks(num_blocks)
            .outer_dim(outer_dim)
            .page_size(page_size)
            .inner_dim(inner_dim)
            .dtype(map_dtype(&dtype).map_err(to_pyerr)?)
            .build()
            .map_err(to_pyerr)?;

        let layout = layout_builder
            .create_layout(
                LayoutType::LayerSeparate { outer_contiguous },
                device_tensors,
                true,
            )
            .map_err(to_pyerr)?;

        let runtime = tokio::runtime::Runtime::new().map_err(to_pyerr)?;

        let cancel_token = CancellationToken::new();

        let cancel_token_clone = cancel_token.clone();
        let task = std::thread::spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                KvbmWorker::worker_task(layout, barrier_id, worker_id, cancel_token_clone).await
            })
        });

        Ok(Self {
            cancel_token,
            task: Some(task),
            _runtime: runtime,
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
