// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::TransferType;
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::llm::block_manager::distributed::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};
use crate::{
    DistributedRuntime as PyDistributedRuntime, llm::block_manager::distributed::VllmTensor,
    to_pyerr,
};

use crate::llm::block_manager::distributed::PyLayoutType;
use anyhow;
use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig};
use dynamo_llm::block_manager::layout::LayoutType;
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

pub trait Worker: Send + Sync {
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
    ) -> anyhow::Result<()>;

    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()>;

    fn clear_connector_metadata(&mut self);

    fn save_kv_layer(&mut self, layer_name: String) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>);
}

pub struct KvConnectorWorker {
    drt: DistributedRuntime,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    kv_cache_layers: Vec<(String, Arc<VllmTensor>)>,

    /// Map of request id to inflight load requests
    maybe_finished_onboarding: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_finished_offloading: HashSet<String>,

    /// For now, offloading operations will be enqueued at the end of the forward pass
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,

    /// cuda events created by the python side
    layer_events: Vec<u64>,
}

impl KvConnectorWorker {
    fn new(py_drt: PyDistributedRuntime, vllm_worker_id: String) -> anyhow::Result<Self> {
        let drt = py_drt.inner.clone();
        let runtime = drt.runtime().primary();

        let (scheduler, worker_client, transfer_client) = Scheduler::new(drt.primary_token());

        CriticalTaskExecutionHandle::new_with_runtime(
            move |_| {
                let mut scheduler = scheduler;
                async move { scheduler.run().await }
            },
            drt.primary_token(),
            "kv-connector-scheduler-task",
            &runtime,
        )?
        .detach();

        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            vllm_worker_id
        );

        Ok(Self {
            drt,
            kvbm_worker: OnceLock::new(),
            connector: worker_client,
            transfer_client,
            maybe_finished_onboarding: HashSet::new(),
            maybe_finished_offloading: HashSet::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
            kv_cache_layers: Vec::new(),
            layer_events: Vec::new(),
        })
    }
}

impl Worker for KvConnectorWorker {
    /// Registers the KV caches with the KVBM worker.
    ///
    /// The Dynamo KVBM worker is lazily initialized when the first KV cache is registered.
    /// This process establishes a connection between all KVBM workers and the leader.
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Arc<VllmTensor>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<LayoutType>,
        host_layout_type: Option<LayoutType>,
        disk_layout_type: Option<LayoutType>,
    ) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }

        assert_eq!(
            kv_caches.len(),
            raw_event_handles.len(),
            "kv_caches and raw_event_handles must have the same length"
        );

        // Process kv_caches in layer execution order (already sorted by layer index)
        let mut vllm_tensors = Vec::new();
        let mut first_tensor_shape: Option<Vec<usize>> = None;

        for (layer_name, vllm_tensor) in kv_caches {
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");

            // Capture the shape of the first tensor for layout detection
            if first_tensor_shape.is_none() {
                first_tensor_shape = Some(vllm_tensor.shape());
            }

            // Store for later lookup by name
            self.kv_cache_layers.push((layer_name, vllm_tensor.clone()));

            // Build ordered tensor list for worker config
            vllm_tensors.push(vllm_tensor as Arc<dyn TorchTensor>);
        }

        self.layer_events = raw_event_handles;

        // Auto-detect device layout type if not explicitly provided
        let detected_device_layout_type = match device_layout_type {
            Some(layout) => layout,
            None => {
                if let Some(ref shape) = first_tensor_shape {
                    match LayoutType::layer_separate_auto(shape, num_device_blocks) {
                        Ok(detected) => {
                            tracing::info!(
                                "Auto-detected device layout from tensor shape: {:?}",
                                detected
                            );
                            detected
                        }
                        Err(e) => {
                            tracing::warn!(
                                "Failed to auto-detect layout from shape {:?}: {}. Using default.",
                                shape,
                                e
                            );
                            LayoutType::layer_separate_auto_default()
                        }
                    }
                } else {
                    tracing::warn!("No tensors available for layout detection. Using default.");
                    LayoutType::layer_separate_auto_default()
                }
            }
        };

        let config = KvbmWorkerConfig::builder()
            .drt(self.drt.clone())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .scheduler_client(Some(self.transfer_client.clone()))
            .device_layout_type(detected_device_layout_type)
            .host_layout_type(host_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .disk_layout_type(disk_layout_type.unwrap_or(LayoutType::FullyContiguous))
            .build()?;

        let worker = self.drt.runtime().primary().block_on(async move {
            let worker = KvbmWorker::new(config, false).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> anyhow::Result<()> {
        // debug_assert!(!self.bound, "connector metadata already bound");
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata)?;
        self.bound = true;
        self.iteration = metadata.iteration;
        self.layers_complete = 0;
        tracing::debug!(
            iteration = self.iteration,
            "bound new metadata: {metadata:#?}"
        );

        self.connector.start_next_iteration()?;

        debug_assert_eq!(
            self.connector.iteration(),
            metadata.iteration,
            "iteration mismatch"
        );

        // self.engine_tx
        //     .send(EngineMessage::UpdateIteration(self.iteration))
        //     .map_err(to_pyerr)?;

        // local actions
        // - create a request slot for each new request
        // - for each action in the metadata, add the action to the request slot
        // - send the list of actions to the engine to track completion

        for slot in metadata.new_slots {
            debug_assert!(!self.connector.has_slot(&slot), "slot already exists");
            self.connector.create_slot(slot)?;
        }

        let mut onboarding_operations = Vec::new();
        let mut offloading_operations = Vec::new();

        for operation in metadata.operations {
            tracing::debug!(
                request_id = operation.request_id, operation_id = %operation.uuid,
                "adding operation to slot: {operation:#?}"
            );

            match operation.transfer_type {
                TransferType::Load => onboarding_operations.push(operation),
                TransferType::Store => offloading_operations.push(operation),
            }
        }

        // immediately enqueue the onboarding operations
        for operation in onboarding_operations {
            let request_id = operation.request_id.clone();
            self.connector.enqueue_request(operation);
            self.maybe_finished_onboarding.insert(request_id);
        }

        // delay offloading operations until the end of the forward pass
        debug_assert!(
            self.offloading_operations.is_empty(),
            "offloading operations should be empty"
        );
        self.offloading_operations = offloading_operations;

        Ok(())
    }

    /// Clears the connector metadata and marks the iteration as complete.
    fn clear_connector_metadata(&mut self) {
        tracing::debug!(iteration = self.iteration, "clearing connector metadata");
        debug_assert!(self.bound, "connector metadata not bound");
        self.bound = false;
        self.iteration = 0; // always reset; leader drives the counter
        self.layers_complete = 0;
        self.connector
            .mark_iteration_complete()
            .expect("failed to mark iteration complete");
    }

    /// Trigger layer-wise completion signals.
    /// Trigger block-wise completion signals afer last layer.
    fn save_kv_layer(&mut self, _layer_name: String) -> anyhow::Result<()> {
        self.layers_complete += 1;
        if self.layers_complete == self.kv_cache_layers.len() {
            let offloading_operations = std::mem::take(&mut self.offloading_operations);

            // block on the the completion of the last layer
            // todo(ryan): capture the context, pass this to the scheduler to do the await on another thread
            // or put the event on a stream and use stream waits to keep it all on device.
            event_sync_blocking(self.layer_events[self.layers_complete - 1]);
            for operation in offloading_operations {
                self.connector.enqueue_request(operation);
            }
        }
        Ok(())
    }

    fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        tracing::debug!(
            iteration = self.iteration,
            "Getting finished requests: {finished_requests:?}"
        );

        // we do not have to visit every slot on every pass, just slots we are waiting on
        //
        // there are two conditions where we would be waiting:
        // 1. if we have requested a load, we need to wait for it to complete
        //    - the load request would come in via the metadata this is processsed in the bind
        // 2. if we have requested a finished event, then we need to await for all outstanding
        //    operations to complete -- either by finishing or being cancelled
        //    - the finish request is triggered by this function, it is not seen in the metadata
        //
        // under each scenario, we mark the `maybe_loading_finished` and `maybe_finished_offloading` hashsets with
        // the request id
        //
        // on each forward pass we visit the maybe slots to see if they are finished

        let mut is_finished_offloading = HashSet::new();
        let mut is_finished_onboarding = HashSet::new();

        // before we process the maybes, add any newly annotated finished requests
        // to the maybe finished set
        for request_id in finished_requests {
            tracing::debug!(request_id, "marking request as finished");

            if !self.connector.has_slot(&request_id) {
                tracing::warn!(
                    request_id,
                    "finished request received for unknown request_id; assuming never started"
                );
                continue;
            }

            if self.maybe_finished_onboarding.contains(&request_id) {
                tracing::info!(
                    request_id,
                    "got a finished warning for a request that is onboarding"
                );
            } else if self.maybe_finished_offloading.contains(&request_id) {
                tracing::warn!(
                    request_id,
                    "possibly got a duplicate finished request; request_id already in the maybe_finished_offloading set"
                );
            } else {
                tracing::debug!(
                    request_id,
                    "received finished request; adding to maybe_finished_offloading set"
                );
                self.maybe_finished_offloading.insert(request_id.clone());
            }
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_finished_offloading.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    tracing::debug!(request_id, "request slot is finished");
                    is_finished_offloading.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished");
                }
            } else {
                // made this condition more strict slot existence checks were added as a prerequesite
                // to be added to the maybe_finished_offloading set.
                panic!(
                    "request slot missing for {request_id}; however, it was present when added to the maybe finished offloading set"
                );
            }
        }

        // remove the finished requests from the maybe finished set
        // note: when storing is finished we also remove the request from the engine state
        for request_id in &is_finished_offloading {
            self.maybe_finished_offloading.remove(request_id);

            // currently chomping the error as the engine is closed and we are shutting down
            if self.connector.has_slot(request_id) {
                self.connector.remove_slot(request_id);
            } else {
                tracing::debug!(
                    request_id,
                    "is_finished_offloading: request slot is not found - likely aborted, removing from is finished offloading set"
                );
            }
        }

        // visit each request slot in the maybe finished set to see if it is finished
        for request_id in self.maybe_finished_onboarding.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    tracing::debug!(request_id, "request slot is finished");
                    is_finished_onboarding.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished");
                }
            } else {
                panic!(
                    "request slot missing for {request_id}; however, it was present when added to the maybe finished onboarding set"
                );
            }
        }

        // remove the finished requests from the maybe finished set
        for request_id in &is_finished_onboarding {
            self.maybe_finished_onboarding.remove(request_id);
            if self.connector.has_slot(request_id) {
                self.connector.remove_slot(request_id);
            }
        }

        (is_finished_offloading, is_finished_onboarding)
    }
}

#[pyclass]
pub struct PyKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, vllm_worker_id))]
    pub fn new(py_drt: PyDistributedRuntime, vllm_worker_id: String) -> PyResult<Self> {
        let connector_worker: Box<dyn Worker> =
            Box::new(KvConnectorWorker::new(py_drt, vllm_worker_id).map_err(to_pyerr)?);
        Ok(Self { connector_worker })
    }

    #[pyo3(signature = (num_device_blocks, page_size, device_id, dtype_width_bytes, kv_caches, raw_event_handles, device_layout_type=None, host_layout_type=None, disk_layout_type=None))]
    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Py<PyAny>)>,
        raw_event_handles: Vec<u64>,
        device_layout_type: Option<PyLayoutType>,
        host_layout_type: Option<PyLayoutType>,
        disk_layout_type: Option<PyLayoutType>,
    ) -> PyResult<()> {
        // Convert Python tensors to Rust VllmTensor objects
        let mut rust_kv_caches = Vec::new();
        for (layer_name, py_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(py_tensor).map_err(to_pyerr)?);
            rust_kv_caches.push((layer_name, vllm_tensor));
        }

        self.connector_worker
            .register_kv_caches(
                num_device_blocks,
                page_size,
                device_id,
                dtype_width_bytes,
                rust_kv_caches,
                raw_event_handles,
                device_layout_type.map(|py_layout| py_layout.into()),
                host_layout_type.map(|py_layout| py_layout.into()),
                disk_layout_type.map(|py_layout| py_layout.into()),
            )
            .map_err(to_pyerr)
    }

    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        self.connector_worker
            .bind_connector_metadata(metadata)
            .map_err(to_pyerr)
    }

    pub fn clear_connector_metadata(&mut self) {
        self.connector_worker.clear_connector_metadata()
    }

    pub fn save_kv_layer(&mut self, layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        // Note: kv_layer is not used in the current implementation
        self.connector_worker
            .save_kv_layer(layer_name)
            .map_err(to_pyerr)
    }

    pub fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        self.connector_worker.get_finished(finished_requests)
    }
}

use cudarc::driver::sys::{
    CUcontext, CUevent, cuCtxGetCurrent, cuEventSynchronize, cudaError_enum,
};
use std::ptr;

// todo(ryan): we will need this if we farm off the cuEventSynchronize to another thread
fn _get_current_context() -> CUcontext {
    let mut ctx: CUcontext = ptr::null_mut();
    let status = unsafe { cuCtxGetCurrent(&mut ctx) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuCtxGetCurrent failed"
    );
    assert!(!ctx.is_null(), "Torch has not set a CUDA context");
    ctx
}

pub fn event_sync_blocking(event: u64) {
    let status = unsafe { cuEventSynchronize(event as CUevent) };
    assert_eq!(
        status,
        cudaError_enum::CUDA_SUCCESS,
        "cuEventSynchronize failed"
    );
}
