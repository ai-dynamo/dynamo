// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::TransferType;
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::llm::block_manager::distributed::get_barrier_id_prefix;
use crate::llm::block_manager::vllm::connector::worker::event_sync_blocking;
use crate::{
    llm::block_manager::distributed::VllmTensor, to_pyerr,
    DistributedRuntime as PyDistributedRuntime,
};

use anyhow;
use dynamo_llm::block_manager::distributed::{KvbmWorker, KvbmWorkerConfig};
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use dynamo_runtime::DistributedRuntime;

pub trait Worker: Send + Sync {
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Arc<VllmTensor>,
        raw_event_handles: Vec<u64>,
    ) -> anyhow::Result<()>;

    fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> anyhow::Result<()>;

    fn start_load_kv(&mut self) -> anyhow::Result<()>;

    fn save_kv_layer(&mut self, layer_idx: usize) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>);
}

pub struct KvConnectorWorker {
    drt: DistributedRuntime,
    kvbm_worker: OnceLock<KvbmWorker>,
    connector: WorkerSchedulerClient,
    transfer_client: TransferSchedulerClient,

    /// Map of request id to inflight load requests
    maybe_finished_onboarding: HashSet<String>,

    /// Map of request id to inflight finished requests
    maybe_finished_offloading: HashSet<String>,

    onboarding_operations: Vec<WorkerTransferRequest>,
    offloading_operations: Vec<WorkerTransferRequest>,

    bound: bool,
    iteration: u64,
    layers_complete: usize,

    /// cuda events created by the python side
    layer_events: Vec<u64>,
}

impl KvConnectorWorker {
    fn new(py_drt: PyDistributedRuntime, trtllm_rank: String) -> anyhow::Result<Self> {
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
            "KvConnectorWorker initialized with worker_rank: {}",
            trtllm_rank
        );

        Ok(Self {
            drt,
            kvbm_worker: OnceLock::new(),
            connector: worker_client,
            transfer_client,
            maybe_finished_onboarding: HashSet::new(),
            maybe_finished_offloading: HashSet::new(),
            onboarding_operations: Vec::new(),
            offloading_operations: Vec::new(),
            bound: false,
            iteration: 0,
            layers_complete: 0,
            layer_events: Vec::new(),
        })
    }
}

impl Worker for KvConnectorWorker {
    fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Arc<VllmTensor>,
        raw_event_handles: Vec<u64>,
    ) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            tracing::warn!("kvbm worker already registered");
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }

        let kv_cache_tensors = vec![kv_cache_tensor as Arc<dyn TorchTensor>];

        let config = KvbmWorkerConfig::builder()
            .drt(self.drt.clone())
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .tensors(kv_cache_tensors)
            .device_id(device_id)
            .dtype_width_bytes(dtype_width_bytes)
            .is_fully_contiguous_layout(true)
            .barrier_id_prefix(get_barrier_id_prefix())
            .scheduler_client(Some(self.transfer_client.clone()))
            .build()?;

        self.layer_events = raw_event_handles;

        let worker = self.drt.runtime().primary().block_on(async move {
            let worker = KvbmWorker::new(config, true).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }

    fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> anyhow::Result<()> {
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

        debug_assert!(
            self.onboarding_operations.is_empty(),
            "onboarding operations should be empty"
        );
        self.onboarding_operations = onboarding_operations;

        debug_assert!(
            self.offloading_operations.is_empty(),
            "offloading operations should be empty"
        );
        self.offloading_operations = offloading_operations;

        Ok(())
    }

    fn save_kv_layer(&mut self, _layer_idx: usize) -> anyhow::Result<()> {
        self.layers_complete += 1;
        if self.layers_complete == self.layer_events.len() {
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

    fn start_load_kv(&mut self) -> anyhow::Result<()> {
        let onboarding_operations = self.onboarding_operations.clone();
        for operation in onboarding_operations {
            let request_id = operation.request_id.clone();
            self.connector.enqueue_request(operation);
            self.maybe_finished_onboarding.insert(request_id);
        }
        Ok(())
    }

    fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>) {
        // we do not have to visit every slot on every pass, just slots we are waiting on
        //
        // there are two conditions where we would be waiting:
        // 1. if we have requested a load, we need to wait for it to complete
        //    - the load request would come in via the metadata this is processsed in the bind
        // 2. if we have requested a finished event, then we need to await for all outstanding
        //    operations to complete -- either by finishing or being cancelled
        //    - the finish request is triggered by this function, it is not seen in the metadata
        //
        // under each scenario, we mark the `maybe_finished_onboarding` and `maybe_finished_offloading` hashsets with
        // the request id
        //
        // on each forward pass we visit the maybe slots to see if they are finished
        let mut is_finished_offloading = HashSet::new();
        let mut is_finished_onboarding = HashSet::new();

        // before we process the maybes, add any newly annotated finished requests
        // to the maybe finished set
        for request_id in finished_gen_req_ids {
            tracing::debug!(request_id, "marking request as finished");

            if !self.connector.has_slot(&request_id.to_string()) {
                tracing::warn!(
                    request_id,
                    "finished request received for unknown request_id; assuming never started"
                );
                continue;
            }

            if self
                .maybe_finished_offloading
                .contains(&request_id.to_string())
            {
                tracing::warn!(request_id, "possibly got a duplicate finished request; request_id already in the maybe_finished_offloading set");
            } else {
                tracing::debug!(
                    request_id,
                    "received finished request; adding to maybe_finished_offloading set"
                );
                self.maybe_finished_offloading
                    .insert(request_id.to_string());
            }
        }

        for request_id in started_loading_req_ids {
            tracing::debug!(request_id, "marking request as finished");

            if !self.connector.has_slot(&request_id.to_string()) {
                tracing::warn!(
                    request_id,
                    "finished request received for unknown request_id; assuming never started"
                );
                continue;
            }

            if self
                .maybe_finished_onboarding
                .contains(&request_id.to_string())
            {
                tracing::warn!(request_id, "possibly got a duplicate finished request; request_id already in the maybe_finished_onboarding set");
            }
        }

        // visit each request slot in the maybe finished set
        for request_id in self.maybe_finished_offloading.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    tracing::debug!(request_id, "request slot is finished offloading");
                    is_finished_offloading.insert(request_id.to_string());
                } else {
                    tracing::debug!(request_id, "request slot is not finished offloading");
                }
            } else {
                // made this condition more strict slot existence checks were added as a prerequesite
                // to be added to the maybe_finished_offloading set.
                panic!("request slot missing for {request_id}; however, it was present when added to the maybe finished offloading set");
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
                tracing::debug!(request_id, "is_finished_offloading: request slot is not found - likely aborted, removing from is finished offloading set");
            }
        }

        // visit each request slot in the maybe finished set to see if it is finished
        for request_id in self.maybe_finished_onboarding.iter() {
            if self.connector.has_slot(request_id) {
                if self.connector.is_complete(request_id) {
                    tracing::debug!(request_id, "request slot is finished onboarding");
                    is_finished_onboarding.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished onboarding");
                }
            } else {
                panic!("request slot missing for {request_id}; however, it was present when added to the maybe finished onboarding set");
            }
        }

        // remove the finished requests from the maybe finished set
        for request_id in &is_finished_onboarding {
            self.maybe_finished_onboarding.remove(request_id);
            if self.connector.has_slot(request_id) {
                self.connector.remove_slot(request_id);
            }
        }

        let finished_offloading: Vec<u64> = is_finished_offloading
            .iter()
            .filter_map(|s| s.parse::<u64>().ok()) // parse String -> u64
            .collect();

        let finished_onboarding: Vec<u64> = is_finished_onboarding
            .iter()
            .filter_map(|s| s.parse::<u64>().ok()) // parse String -> u64
            .collect();

        (finished_offloading, finished_onboarding)
    }
}

#[pyclass]
pub struct PyTrtllmKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyTrtllmKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, trtllm_rank))]
    pub fn new(py_drt: PyDistributedRuntime, trtllm_rank: String) -> PyResult<Self> {
        let connector_worker: Box<dyn Worker> =
            Box::new(KvConnectorWorker::new(py_drt, trtllm_rank).map_err(to_pyerr)?);
        Ok(Self { connector_worker })
    }

    pub fn register_kv_caches(
        &mut self,
        num_device_blocks: usize,
        page_size: usize,
        device_id: usize,
        dtype_width_bytes: usize,
        kv_cache_tensor: Py<PyAny>,
        raw_event_handles: Vec<u64>,
    ) -> PyResult<()> {
        // Convert Python tensor to Rust VllmTensor objects
        let rust_kv_cache_tensor = Arc::new(VllmTensor::new(kv_cache_tensor).map_err(to_pyerr)?);

        self.connector_worker
            .register_kv_caches(
                num_device_blocks,
                page_size,
                device_id,
                dtype_width_bytes,
                rust_kv_cache_tensor,
                raw_event_handles,
            )
            .map_err(to_pyerr)
    }

    pub fn bind_connector_meta(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        self.connector_worker
            .bind_connector_meta(metadata)
            .map_err(to_pyerr)
    }

    pub fn save_kv_layer(&mut self, layer_idx: usize) -> PyResult<()> {
        self.connector_worker
            .save_kv_layer(layer_idx)
            .map_err(to_pyerr)
    }

    pub fn start_load_kv(&mut self) -> PyResult<()> {
        self.connector_worker.start_load_kv().map_err(to_pyerr)
    }

    pub fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>) {
        self.connector_worker
            .get_finished(finished_gen_req_ids, started_loading_req_ids)
    }
}
