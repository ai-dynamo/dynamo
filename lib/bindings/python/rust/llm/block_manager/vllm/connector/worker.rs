// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::TransferType;
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use super::*;
use crate::llm::block_manager::distributed::get_barrier_id;
use crate::{
    llm::block_manager::distributed::VllmTensor, to_pyerr,
    DistributedRuntime as PyDistributedRuntime,
};

use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig};
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use dynamo_runtime::DistributedRuntime;

#[pyclass]
pub struct KvConnectorWorker {
    drt: DistributedRuntime,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    // request_slots: HashMap<String, WorkerSlot>,
    kv_caches: HashMap<String, Arc<VllmTensor>>,

    /// Map of request id to inflight load requests
    maybe_finished_onboarding: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_finished_offloading: HashSet<String>,

    /// For now, offloading operations will be enqueued at the end of the forward pass
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,
}

#[pymethods]
impl KvConnectorWorker {
    #[new]
    fn new(py_drt: PyDistributedRuntime, vllm_worker_id: String) -> PyResult<Self> {
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
        )
        .map_err(to_pyerr)?
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
            kv_caches: HashMap::new(),
            maybe_finished_onboarding: HashSet::new(),
            maybe_finished_offloading: HashSet::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
        })
    }

    /// Registers the KV caches with the KVBM worker.
    ///
    /// The Dynamo KVBM worker is lazily initialized when the first KV cache is registered.
    /// This process establishes a connection between all KVBM workers and the leader.
    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_caches: Vec<(String, Py<PyAny>)>,
    ) -> PyResult<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(to_pyerr(anyhow::anyhow!("kvbm worker already registered")));
        }

        // Process kv_caches in layer execution order (already sorted by layer index)
        let mut vllm_tensors = Vec::new();
        for (layer_name, torch_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(torch_tensor).map_err(to_pyerr)?);
            tracing::trace!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");

            // Store for later lookup by name
            self.kv_caches.insert(layer_name, vllm_tensor.clone());

            // Build ordered tensor list for worker config
            vllm_tensors.push(vllm_tensor as Arc<dyn TorchTensor>);
        }

        let config = KvbmWorkerConfig::builder()
            .drt(self.drt.clone())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(vllm_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .barrier_id(get_barrier_id())
            .scheduler_client(Some(self.transfer_client.clone()))
            .build()
            .map_err(to_pyerr)?;

        let worker = self
            .drt
            .runtime()
            .primary()
            .block_on(async move {
                let worker = KvbmWorker::new(config).await?;
                anyhow::Ok(worker)
            })
            .map_err(to_pyerr)?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| to_pyerr(anyhow::anyhow!("failed to set kvbm worker")))?;

        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        debug_assert!(!self.bound, "connector metadata already bound");
        let metadata: ConnectorMetadata = serde_json::from_slice(&metadata).map_err(to_pyerr)?;
        self.bound = true;
        self.iteration = metadata.iteration;
        self.layers_complete = 0;
        tracing::debug!(
            iteration = self.iteration,
            "bound new metadata: {metadata:#?}"
        );

        self.connector.start_next_iteration().map_err(to_pyerr)?;

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
            self.connector.create_slot(slot).map_err(to_pyerr)?;
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
    pub fn clear_connector_metadata(&mut self) {
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
    pub fn save_kv_layer(&mut self, _layer_name: String, _kv_layer: Py<PyAny>) -> PyResult<()> {
        self.layers_complete += 1;
        if self.layers_complete == self.kv_caches.len() {
            let offloading_operations = std::mem::take(&mut self.offloading_operations);
            for operation in offloading_operations {
                self.connector.enqueue_request(operation);
            }
        }
        Ok(())
    }

    pub fn get_finished(
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

            debug_assert!(
                self.connector.has_slot(&request_id),
                "request slot not found"
            );

            debug_assert!(
                !self.maybe_finished_offloading.contains(&request_id),
                "request already in maybe storing finished"
            );

            // insert request into the maybe finished set
            self.maybe_finished_offloading.insert(request_id.clone());
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_finished_offloading.iter() {
            if self.connector.is_complete(request_id) {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_offloading.insert(request_id.clone());
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        // note: when storing is finished we also remove the request from the engine state
        for request_id in &is_finished_offloading {
            self.maybe_finished_offloading.remove(request_id);

            // currently chomping the error as the engine is closed and we are shutting down
            self.connector.remove_slot(request_id);
        }

        // visit each request slot in the maybe finished set to see if it is finished
        for request_id in self.maybe_finished_onboarding.iter() {
            if self.connector.is_complete(request_id) {
                tracing::debug!(request_id, "request slot is finished");
                is_finished_onboarding.insert(request_id.clone());
            } else {
                tracing::debug!(request_id, "request slot is not finished");
            }
        }

        // remove the finished requests from the maybe finished set
        for request_id in &is_finished_onboarding {
            self.maybe_finished_onboarding.remove(request_id);
            self.connector.remove_slot(request_id);
        }

        (is_finished_offloading, is_finished_onboarding)
    }
}
