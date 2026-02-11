// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::connector::protocol::TransferType;
use dynamo_llm::block_manager::connector::scheduler::{
    Scheduler, SchedulerMessage, TransferSchedulerClient, WorkerSchedulerClient,
};

use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::*;
use crate::block_manager::distributed::{get_leader_zmq_ack_url, get_leader_zmq_pub_url};
use crate::block_manager::vllm::connector::worker::event_sync_blocking;
use crate::{block_manager::distributed::VllmTensor, to_pyerr};
use dynamo_runtime::DistributedRuntime;

use crate::{
    extract_distributed_runtime_from_obj, get_current_cancel_token, get_current_tokio_handle,
};
use anyhow;
use dynamo_llm::block_manager::distributed::{
    AdditionalCacheGroupConfig, KvbmWorker, KvbmWorkerConfig, NcclConfig,
};
use dynamo_llm::block_manager::layout::LayoutType;
use dynamo_llm::block_manager::storage::torch::TorchTensor;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;

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

    fn execute_offload_operations(&mut self) -> anyhow::Result<()>;

    fn save_kv_layer(&mut self, layer_idx: usize) -> anyhow::Result<()>;

    fn get_finished(
        &mut self,
        finished_gen_req_ids: Vec<u64>,
        started_loading_req_ids: Vec<u64>,
    ) -> (Vec<u64>, Vec<u64>);

    /// Submit offload operations to execute after the CUDA event completes (non-blocking).
    /// Does slot bookkeeping synchronously, then spawns an async task to poll the event
    /// and send operations to the scheduler when complete.
    fn submit_offload_on_event(&mut self, event: u64) -> anyhow::Result<()>;

    /// Register an additional indexer k cache (for DSA models).
    /// Must be called after register_kv_caches and before finalize_registration.
    fn register_indexer_k_caches(
        &mut self,
        dtype_width_bytes: usize,
        tensor: Arc<VllmTensor>,
    ) -> anyhow::Result<()>;

    /// Finalize registration: build the KvbmWorker from pending configs.
    /// Must be called exactly once after register_kv_caches (and optionally register_indexer_k_caches).
    fn finalize_registration(&mut self) -> anyhow::Result<()>;
}

/// Pending primary cache config, stored between register_kv_caches and finalize_registration.
struct PendingPrimaryConfig {
    num_device_blocks: usize,
    page_size: usize,
    device_id: usize,
    dtype_width_bytes: usize,
    kv_cache_tensor: Arc<VllmTensor>,
    raw_event_handles: Vec<u64>,
}

/// Pending indexer k cache config, stored between register_indexer_k_caches and finalize_registration.
struct PendingIndexerConfig {
    dtype_width_bytes: usize,
    tensor: Arc<VllmTensor>,
}

pub struct KvConnectorWorker {
    _drt: Option<Arc<DistributedRuntime>>,
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

    /// Rank for NCCL replicated mode (None = sharded mode)
    rank: Option<i32>,

    /// World size for NCCL replicated mode
    world_size: Option<i32>,

    /// Raw pointer to NCCL communicator for replicated mode
    nccl_comm_ptr: Option<usize>,

    /// Pending primary cache registration (deferred until finalize_registration)
    pending_primary: Option<PendingPrimaryConfig>,

    /// Pending indexer k cache registration (for DSA models)
    pending_indexer_k: Option<PendingIndexerConfig>,
}

impl KvConnectorWorker {
    fn new(
        drt: Option<Arc<DistributedRuntime>>,
        trtllm_rank: String,
        rank: Option<i32>,
        world_size: Option<i32>,
        nccl_comm_ptr: Option<usize>,
    ) -> anyhow::Result<Self> {
        let runtime = get_current_tokio_handle();

        let (scheduler, worker_client, transfer_client) =
            Scheduler::new(get_current_cancel_token());

        CriticalTaskExecutionHandle::new_with_runtime(
            move |_| {
                let mut scheduler = scheduler;
                async move { scheduler.run().await }
            },
            get_current_cancel_token(),
            "kv-connector-scheduler-task",
            &runtime,
        )?
        .detach();

        tracing::info!(
            "KvConnectorWorker initialized with worker_rank: {}, nccl_rank: {:?}, world_size: {:?}, nccl_comm_ptr: {:?}",
            trtllm_rank,
            rank,
            world_size,
            nccl_comm_ptr.map(|p| format!("{:#x}", p))
        );

        Ok(Self {
            _drt: drt,
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
            rank,
            world_size,
            nccl_comm_ptr,
            pending_primary: None,
            pending_indexer_k: None,
        })
    }
}

/// Build NcclConfig from the provided parameters.
/// Returns NcclConfig::disabled() if any required parameter is missing or if NCCL feature is not enabled.
fn build_nccl_config(
    rank: Option<i32>,
    world_size: Option<i32>,
    nccl_comm_ptr: Option<usize>,
) -> NcclConfig {
    #[cfg(feature = "nccl")]
    {
        match (rank, world_size, nccl_comm_ptr) {
            (Some(r), Some(ws), Some(ptr)) if ptr != 0 => {
                tracing::info!(
                    "Creating NCCL config for replicated mode: rank={}, world_size={}, comm_ptr={:#x}",
                    r,
                    ws,
                    ptr
                );
                // Safety: The caller (Python side) is responsible for ensuring the pointer
                // is a valid NCCL communicator that outlives this config.
                unsafe {
                    use cudarc::nccl::sys::ncclComm_t;
                    NcclConfig::enabled(ptr as ncclComm_t, r, ws)
                }
            }
            _ => {
                tracing::debug!("NCCL parameters incomplete, using sharded mode");
                NcclConfig::disabled()
            }
        }
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = (rank, world_size, nccl_comm_ptr); // suppress unused warnings
        tracing::debug!("NCCL feature not enabled, using sharded mode");
        NcclConfig::disabled()
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
            return Err(anyhow::anyhow!("kvbm worker already registered"));
        }
        if self.pending_primary.is_some() {
            return Err(anyhow::anyhow!(
                "register_kv_caches already called; call finalize_registration first"
            ));
        }

        self.layer_events = raw_event_handles.clone();
        self.pending_primary = Some(PendingPrimaryConfig {
            num_device_blocks,
            page_size,
            device_id,
            dtype_width_bytes,
            kv_cache_tensor,
            raw_event_handles,
        });

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

        for slot_info in metadata.new_slots {
            debug_assert!(
                !self.connector.has_slot(&slot_info.request_id),
                "slot already exists"
            );
            // Create slot with expected immediate ops count BEFORE any operations arrive.
            // This ensures proper completion tracking and avoids race conditions in TP>1.
            self.connector.create_slot_with_immediate_ops(
                slot_info.request_id,
                slot_info.expected_immediate_ops,
            )?;
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

    // Assumes the operations are in a valid state for offloading.
    fn execute_offload_operations(&mut self) -> anyhow::Result<()> {
        let offloading_operations = std::mem::take(&mut self.offloading_operations);
        for operation in offloading_operations {
            self.connector.enqueue_request(operation);
        }
        Ok(())
    }

    fn save_kv_layer(&mut self, _layer_idx: usize) -> anyhow::Result<()> {
        self.layers_complete += 1;
        if self.layers_complete == self.layer_events.len() {
            // block on the the completion of the last layer
            // todo(ryan): capture the context, pass this to the scheduler to do the await on another thread
            // or put the event on a stream and use stream waits to keep it all on device.
            event_sync_blocking(self.layer_events[self.layers_complete - 1]);
            if let Err(e) = self.execute_offload_operations() {
                tracing::error!("Failed to execute offload operations: {}", e);
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
                tracing::warn!(
                    request_id,
                    "possibly got a duplicate finished request; request_id already in the maybe_finished_offloading set"
                );
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
                tracing::warn!(
                    request_id,
                    "possibly got a duplicate finished request; request_id already in the maybe_finished_onboarding set"
                );
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
                    tracing::debug!(request_id, "request slot is finished onboarding");
                    is_finished_onboarding.insert(request_id.clone());
                } else {
                    tracing::debug!(request_id, "request slot is not finished onboarding");
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

    fn submit_offload_on_event(&mut self, event: u64) -> anyhow::Result<()> {
        let operations = std::mem::take(&mut self.offloading_operations);

        // Bookkeeping done synchronously while we have &mut self
        for op in &operations {
            self.connector.record_operation(&op.request_id, op.uuid);
        }

        // Clone channel for async use
        let tx = self.connector.get_scheduler_tx();

        // Use std::thread since we may be in a subprocess without tokio runtime
        std::thread::spawn(move || {
            // Block this thread until event completes (doesn't block main thread)
            event_sync_blocking(event);

            // Send operations to scheduler
            for op in operations {
                if let Err(e) = tx.send(SchedulerMessage::EnqueueRequest(op)) {
                    tracing::error!("Failed to send offload operation: {}", e);
                }
            }
        });

        Ok(())
    }

    fn register_indexer_k_caches(
        &mut self,
        dtype_width_bytes: usize,
        tensor: Arc<VllmTensor>,
    ) -> anyhow::Result<()> {
        if self.pending_primary.is_none() {
            return Err(anyhow::anyhow!(
                "register_kv_caches must be called before register_indexer_k_caches"
            ));
        }
        if self.pending_indexer_k.is_some() {
            return Err(anyhow::anyhow!(
                "register_indexer_k_caches already called"
            ));
        }

        self.pending_indexer_k = Some(PendingIndexerConfig {
            dtype_width_bytes,
            tensor,
        });

        Ok(())
    }

    fn finalize_registration(&mut self) -> anyhow::Result<()> {
        if self.kvbm_worker.get().is_some() {
            return Err(anyhow::anyhow!("kvbm worker already finalized"));
        }

        let primary = self.pending_primary.take().ok_or_else(|| {
            anyhow::anyhow!("register_kv_caches must be called before finalize_registration")
        })?;

        let kv_cache_tensors = vec![primary.kv_cache_tensor as Arc<dyn TorchTensor>];

        // Build NCCL config for replicated mode if parameters are provided
        let nccl_config = build_nccl_config(self.rank, self.world_size, self.nccl_comm_ptr);

        // Build additional cache groups from pending indexer config
        let mut additional_cache_groups = Vec::new();
        if let Some(indexer) = self.pending_indexer_k.take() {
            additional_cache_groups.push(AdditionalCacheGroupConfig {
                name: "indexer_k".to_string(),
                tensors: vec![indexer.tensor as Arc<dyn TorchTensor>],
                dtype_width_bytes: indexer.dtype_width_bytes,
            });
        }

        let config = KvbmWorkerConfig::builder()
            .cancel_token(get_current_cancel_token())
            .num_device_blocks(primary.num_device_blocks)
            .page_size(primary.page_size)
            .tensors(kv_cache_tensors)
            .device_id(primary.device_id)
            .dtype_width_bytes(primary.dtype_width_bytes)
            .device_layout_type(LayoutType::FullyContiguous)
            .host_layout_type(LayoutType::FullyContiguous)
            .disk_layout_type(LayoutType::FullyContiguous)
            .leader_pub_url(get_leader_zmq_pub_url())
            .leader_ack_url(get_leader_zmq_ack_url())
            .scheduler_client(Some(self.transfer_client.clone()))
            .rank(self.rank)
            .world_size(self.world_size)
            .nccl_config(nccl_config)
            .additional_cache_groups(additional_cache_groups)
            .build()?;

        let worker = get_current_tokio_handle().block_on(async move {
            let worker = KvbmWorker::new(config, true).await?;
            anyhow::Ok(worker)
        })?;

        self.kvbm_worker
            .set(worker)
            .map_err(|_| anyhow::anyhow!("failed to set kvbm worker"))?;

        Ok(())
    }
}

#[pyclass]
pub struct PyTrtllmKvConnectorWorker {
    connector_worker: Box<dyn Worker>,
}

#[pymethods]
impl PyTrtllmKvConnectorWorker {
    #[new]
    #[pyo3(signature = (py_drt, trtllm_rank, rank=None, world_size=None, nccl_comm_ptr=None))]
    pub fn new(
        py_drt: Option<PyObject>,
        trtllm_rank: String,
        rank: Option<i32>,
        world_size: Option<i32>,
        nccl_comm_ptr: Option<usize>,
    ) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = py_drt {
                extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_worker: Box<dyn Worker> = Box::new(
            KvConnectorWorker::new(drt, trtllm_rank, rank, world_size, nccl_comm_ptr)
                .map_err(to_pyerr)?,
        );
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

    pub fn execute_offload_operations(&mut self) -> PyResult<()> {
        self.connector_worker
            .execute_offload_operations()
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

    pub fn submit_offload_on_event(&mut self, event: u64) -> PyResult<()> {
        self.connector_worker
            .submit_offload_on_event(event)
            .map_err(to_pyerr)
    }

    pub fn register_indexer_k_caches(
        &mut self,
        dtype_width_bytes: usize,
        indexer_k_tensor: Py<PyAny>,
    ) -> PyResult<()> {
        let rust_tensor = Arc::new(VllmTensor::new(indexer_k_tensor).map_err(to_pyerr)?);
        self.connector_worker
            .register_indexer_k_caches(dtype_width_bytes, rust_tensor)
            .map_err(to_pyerr)
    }

    pub fn finalize_registration(&mut self) -> PyResult<()> {
        self.connector_worker
            .finalize_registration()
            .map_err(to_pyerr)
    }
}
