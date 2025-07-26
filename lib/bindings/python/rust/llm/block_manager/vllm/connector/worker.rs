// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::*;
use crate::{llm::block_manager::distributed::VllmTensor, to_pyerr};
use dynamo_llm::block_manager::distributed::load_and_validate_tensors;
use dynamo_llm::block_manager::storage::torch::{TorchDevice, TorchTensor};

#[derive(Debug, Clone)]
pub struct WorkerAction {
    pub action_type: WorkerActionType,
    // Add other fields as needed
}

impl WorkerAction {
    pub fn action_type(&self) -> WorkerActionType {
        self.action_type
    }
}

#[pyclass]
pub struct KvConnectorWorker {
    slots: HashMap<String, WorkerSlot>,

    /// Map of actions per layer, per slot
    forward_pass_actions: HashMap<String, HashMap<String, WorkerAction>>,

    /// Map of layer name to vllm tensor
    kv_caches: HashMap<String, Arc<VllmTensor>>,
}

#[pymethods]
impl KvConnectorWorker {
    #[new]
    fn new(worker_id: String) -> Self {
        tracing::info!(
            "KvConnectorWorker initialized with worker_id: {}",
            worker_id
        );
        Self {
            slots: HashMap::new(),
            forward_pass_actions: HashMap::new(),
            kv_caches: HashMap::new(),
        }
    }

    pub fn register_kv_caches(&mut self, kv_caches: HashMap<String, Py<PyAny>>) -> PyResult<()> {
        for (layer_name, torch_tensor) in kv_caches {
            let vllm_tensor = Arc::new(VllmTensor::new(torch_tensor).map_err(to_pyerr)?);
            tracing::debug!("Registering KV cache layer: {layer_name}, tensor: {vllm_tensor:?}");
            self.kv_caches.insert(layer_name, vllm_tensor);
        }

        let tensors: Vec<Arc<dyn TorchTensor>> = self
            .kv_caches
            .values()
            .map(|tensor| tensor.clone() as Arc<dyn TorchTensor>)
            .collect();

        let first_tensor = tensors.first().unwrap();

        let device_id = match first_tensor.device() {
            TorchDevice::Cuda(id) => id,
            TorchDevice::Other(_) => 0, // Default to 0 for non-CUDA devices
        };

        // validate all tensors are on the same device
        for tensor in &tensors {
            if tensor.device() != first_tensor.device() {
                return Err(to_pyerr(anyhow::anyhow!(
                    "All tensors must be on the same device! Got {:?} and {:?}",
                    tensor.device(),
                    first_tensor.device()
                )));
            }
        }

        load_and_validate_tensors(&tensors, device_id).map_err(to_pyerr)?;

        // refactor john's kvbm worker into just pieces
        // - build a block layout from tensors, inferring the layout dims and dtype
        // - initialize nixl agent with uxc default, if g3 enabled add gds to the agent - do this in the constructor
        // - allocate blocks for g2 and/or g3, register them with nixl
        // - construct an transfer manager (offload manager)
        // - wait for worker to send a plan - a simply map of {request_id, [(src_block_id, dst_block_id]}
        //   - count layers, when all layers are counted, enqueue block-wise transfers and hold per request the notification handles
        //   - i think we have oneshot return notificatoin channel that is optional offloads/writes, if so use t.
        //
        // - currently leader and worker are not connected
        //   - rework the zmq bits or just use nats to perform completion events on leadera
        //   - this will allow the leader to free cpu/disk blocks when partial xfers are complete.
        //   - once wired up, the host will keep a list of transfer ids associated with each action it puts in the metdata
        //   - trigger the worker -> leader notification when the transfer is complete using the transfer id
        Ok(())
    }

    /// Loads the metadata from the leader.
    /// This action translates the metadata into a set of actions that the worker will perform.
    /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    pub fn bind_connector_metadata(&mut self, metadata: Vec<u8>) -> PyResult<()> {
        let scheduler_output: SchedulerOutput =
            serde_json::from_slice(&metadata).map_err(to_pyerr)?;
        tracing::debug!("Bound metadata: {scheduler_output:#?}");
        Ok(())
    }

    pub fn clear_connector_metadata(&mut self) {
        tracing::debug!("Clearing connector metadata");
        assert!(
            self.forward_pass_actions.is_empty(),
            "All actions must be assigned to a slot before clearing metadata"
        );
    }

    pub fn save_kv_layer(&mut self, layer_name: String, kv_layer: Py<PyAny>) -> PyResult<()> {
        let tensor = VllmTensor::new(kv_layer).map_err(to_pyerr)?;
        // tracing::debug!("Saving KV layer: {layer_name}; kv_layer: {tensor:?}");
        Ok(())
    }

    pub fn get_finished(
        &mut self,
        finished_requests: HashSet<String>,
    ) -> (HashSet<String>, HashSet<String>) {
        tracing::debug!("Getting finished requests: {finished_requests:?}");
        (finished_requests, HashSet::new())
    }
}

impl KvConnectorWorker {
    // /// Loads the metadata from the leader.
    // /// This action translates the metadata into a set of actions that the worker will perform.
    // /// All actions much be assigned to a slot before [`KvConnectorWorker::clear_metadata`] is called.
    // pub fn bind_metadata(&mut self, metadata: KvConnectorMetadata) {
    //     // build forward pass actions
    //     unimplemented!()
    // }

    // pub fn clear_metadata(&mut self) {
    //     assert!(
    //         self.forward_pass_actions.is_empty(),
    //         "All actions must be assigned to a slot before clearing metadata"
    //     );
    // }

    // / This function serves two purposes:
    // / 1. To mark the slots as leader finished.
    // / 2. To report which slots have fully completed all their outstanding actions.
    // /
    // / The leader finish event can potentially trigger cancellation of best effort actions; however,
    // / all outstanding actions must be completed before the slot can report it has finished.
    // /
    // / If the implementation chooses to do so, it can trigger a cancellation token on a when provided
    // / a finished event, but this is not required.
    // /
    // / However, failure to cancel increases memory pressure on the GPU pool as there are no coarse grain
    // / API calls to release specific GPU blocks. Currently it appears it's an all or nothing approach
    // / with respect to GPU block ownership by the slot.
    //     pub fn get_finished(&mut self, finished_requests: &mut HashSet<String>) -> CompletedSlots {
    //         let mut completed_slots = CompletedSlots::default();
    //         let mut slots_to_remove = Vec::new();

    //         finished_requests.iter().for_each(|request| {
    //             if let Some(slot) = self.slots.get_mut(request) {
    //                 slot.leader_finished = true;
    //             } else {
    //                 panic!("Request not found in slots: {}", request);
    //             }
    //         });

    //         for (slot_id, slot) in self.slots.iter_mut() {
    //             if let Some(action_type) = slot.is_finished() {
    //                 match action_type {
    //                     WorkerActionType::Loading => {
    //                         completed_slots
    //                             .recv_loading_requests
    //                             .insert(slot_id.clone());
    //                     }
    //                     WorkerActionType::Saving | WorkerActionType::Idle => {
    //                         // The leader must have already informed the worker that it is finished
    //                         // per the vllm protocol/policies.
    //                         assert!(slot.leader_finished);
    //                         completed_slots.send_saving_requests.insert(slot_id.clone());
    //                         slots_to_remove.push(slot_id.clone());
    //                     }
    //                 }
    //             }
    //         }

    //         // Remove completed slots
    //         for slot_id in slots_to_remove {
    //             self.slots.remove(&slot_id);
    //         }

    //         completed_slots
    //     }
}

/// Workers can only be in one of these states.
///
/// If the worker is active loading, it can only accept [`WorkerActionType::Loading`] actions,
/// and if the worker is active saving, it can only accept [`WorkerActionType::Saving`] actions.
///
/// After all actions of a given type are completed the slot should transition to [`WorkerActionType::Idle`]
/// on during the next [`KvConnectorWorker::get_finished`] during a visit to the [`WorkerSlot::is_finished`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerActionType {
    Idle,
    Loading,
    Saving,
}

impl Default for WorkerActionType {
    fn default() -> Self {
        Self::Idle
    }
}

#[derive(Debug, Clone, Default)]
pub struct WorkerSlot {
    state: WorkerActionType,
    inflight_actions: Vec<WorkerAction>,
    pending_actions: HashMap<String, Vec<WorkerAction>>,
    leader_finished: bool,
}

impl WorkerSlot {
    pub fn is_finished(&self) -> Option<WorkerActionType> {
        // all pending actions must be triggered
        unimplemented!()
    }
}

#[derive(Debug, Clone, Default)]
pub struct CompletedSlots {
    send_saving_requests: HashSet<String>,
    recv_loading_requests: HashSet<String>,
}
