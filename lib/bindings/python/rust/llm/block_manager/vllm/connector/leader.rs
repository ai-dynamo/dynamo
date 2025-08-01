// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod slot;

use super::*;
use slot::{ConnectorSlotManager, SlotError, SlotManager, SlotState};

use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::llm::block_manager::{
    distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest, VllmBlockManager,
};

use dynamo_llm::block_manager::{
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::*,
    BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
};
use dynamo_llm::tokens::{SaltHash, TokenBlockSequence, Tokens};

use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};
use tokio::sync::mpsc;

type VllmLocality = Logical<DistributedLeaderWorkerResources>;

impl From<SlotError> for PyErr {
    fn from(err: SlotError) -> Self {
        to_pyerr(err)
    }
}

#[pyclass]
pub struct KvConnectorLeader {
    slot_manager: ConnectorSlotManager<String>,
    block_manager: VllmBlockManager,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
}

#[pymethods]
impl KvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, block_manager, leader))]
    pub fn new(worker_id: String, block_manager: PyBlockManager, leader: PyKvbmLeader) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        // if drt is none, then we must construct a runtime and distributed runtime
        let block_manager = block_manager.get_block_manager().clone();
        let block_size = block_manager.block_size();

        let leader = leader.get_inner();

        Self {
            slot_manager: ConnectorSlotManager::new(block_manager.clone(), leader),
            block_manager,
            block_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
        }
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For vllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // the number of device matched tokens should be less than or equal to the number of tokens in the request
        debug_assert!(num_computed_tokens % self.block_size == 0);

        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;

        // vllm is telling us that the tokens have been computed, since we do not have insight into the device pool
        // we accept this and advance the computed position
        slot.advance_computed_position(num_computed_tokens)?;

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_all_local_matches()?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!((num_computed_tokens + num_external_tokens) % self.block_size == 0);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// We drop the need to pass in the KvCacheBlocks and the num_external_tokens as they are captured
    /// statefully in the [`VllmLeaderKvCacheManagerAndConnector::get_num_new_matched_tokens`] function.
    ///
    /// Note: vLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    pub fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: u64,
    ) -> PyResult<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}; num_external_tokens: {}",
            block_ids.len(),
            num_external_tokens
        );

        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;

        slot.append_mutable_device_blocks(&block_ids)?;

        // the second call will show num_external_tokens == 0
        // this call is just letting us know the other blocks that are being used for the remainder of the prefill
        if num_external_tokens > 0 {
            self.onboarding_slots.insert(request_id);
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all)]
    pub fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> PyResult<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata; iteration {iteration}");
        tracing::debug!("scheduler_output: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        // Worker-side - we create a request slot for onboarding, then delete it when onboarding is finished, then
        // recreate it again when we start the prefill/decode phase.
        //
        // This is kind of a nice abstraction as it keeps the events simplier; however, we now create the request-slot
        // once for onboarding (this loop), then again for prefill/decode (new_requests loop).
        tracing::debug!("evalatuing {} onboarding slots", onboarding_slots.len());
        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager.get_slot(request_id).map_err(to_pyerr)?;
            let mut slot = shared_slot.lock().map_err(to_pyerr)?;

            tracing::debug!("marking slot as onboarding: {request_id}");
            md.create_slot(request_id.clone());
            slot.mark_as_onboarding(iteration)?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!("adding {} pending onboarding operations", pending_ops.len());
                md.add_operations(pending_ops);
            }
        }

        // vLLM provides us with "new_requests" which are "new" after onboarding, but not before or during.
        // this makes the lifecyle a potentially two-phase lifecycle.
        //
        // todo: update the code and abstraction to account for this two-phase lifecycle.
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager.get_slot(request_id).map_err(to_pyerr)?;
            let mut slot = shared_slot.lock().map_err(to_pyerr)?;

            // inform the worker that a new request-slot should be created
            md.create_slot(new_req.request_id.clone());

            slot.mark_as_scheduled(iteration)?;

            debug_assert!(
                matches!(
                    slot.state(),
                    SlotState::Initialized | SlotState::Onboarding(_)
                ),
                "current slot state: {:?}",
                slot.state()
            );

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(&[], &[], new_req.num_computed_tokens, scheduled_tokens)?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    new_req.request_id
                );
                md.add_operations(pending_ops);
            }
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager.get_slot(request_id).map_err(to_pyerr)?;
            let mut slot = shared_slot.lock().map_err(to_pyerr)?;

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                tracing::debug!(
                    "adding {} pending operations for slot {}",
                    pending_ops.len(),
                    request_id
                );
                md.add_operations(pending_ops);
            }
        }

        std::thread::sleep(std::time::Duration::from_secs(1));

        tracing::debug!("scheduler_output: {scheduler_output:#?}");
        serde_json::to_vec(&md).map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: String, block_ids: Vec<BlockId>) -> PyResult<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");
        // grab the slot
        let shared_slot = self.slot_manager.get_slot(&request_id).map_err(to_pyerr)?;

        // mark the slot as finished
        let mut slot = shared_slot.lock().map_err(to_pyerr)?;
        slot.mark_as_finished(self.iteration_counter)?;

        // remove it from the manager as we will never use it again
        self.slot_manager.remove_slot(&request_id)?;

        // if the slot has finished, we can return false to vllm, indicating all gpu blocks are free to be reused
        // otherwise, we return false, which means there are still outstanding operations on gpu blocks which
        // must be awaited before the gpu blocks can be reused. if we return true, then it is the worker side
        // of the connector api which will be used to inform vllm that the request is finished.
        if let SlotState::Finished = slot.state() {
            Ok(false)
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
            Ok(true)
        }
    }

    pub fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager.has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    pub fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.slot_manager
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }
}
