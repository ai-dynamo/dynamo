// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::BlockManagerBuilder;
use dynamo_llm::block_manager::connector::protocol::RequestType;
use dynamo_llm::block_manager::kv_consolidator::EventSource;
use crate::block_manager::vllm::connector::leader::slot::{
    ConnectorSlotManager, SlotManager, SlotState,
};
use crate::block_manager::vllm::connector::leader::{
    kvbm_metrics_endpoint_enabled, parse_kvbm_metrics_port,
};
use crate::block_manager::{distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest};
use crate::get_current_tokio_handle;
use anyhow;
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};
use tokio::runtime::Handle;

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>>;

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;
}

#[derive(Debug)]
pub struct KvConnectorLeader {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
    inflight_request_to_num_external_tokens: HashMap<String, usize>,
    kvbm_metrics: KvbmMetrics,
}

impl KvConnectorLeader {
    fn new(
        worker_id: u64,
        page_size: usize,
        leader_py: PyKvbmLeader,
        consolidator_trtllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let handle: Handle = get_current_tokio_handle();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );

        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());

        {
            let slot_manager_cell = slot_manager_cell.clone();
            // Capture consolidator endpoints for the async block
            let consolidator_trtllm_ep = consolidator_trtllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let mut block_manager_builder = BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone());

                // Add consolidator config if endpoint is provided
                // For TRTLLM: engine_endpoint is where TRTLLM publishes, output_endpoint is where consolidator publishes
                if let Some(trtllm_ep) = consolidator_trtllm_ep.clone() {
                    tracing::info!(
                        "Consolidator config: trtllm_endpoint={}, consolidated_output_endpoint={:?}",
                        trtllm_ep,
                        consolidator_output_ep
                    );

                    block_manager_builder = block_manager_builder.consolidator_config(
                        trtllm_ep,
                        consolidator_output_ep,
                        EventSource::Trtllm,
                    );
                }

                let block_manager = match block_manager_builder.build().await
                {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    kvbm_metrics_clone.clone(),
                    Some(format!("worker-{}", worker_id)),
                );

                let _ = slot_manager_cell.set(sm);

                tracing::info!("KvConnectorLeader init complete.");
            });
        }

        Self {
            slot_manager: slot_manager_cell,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            inflight_request_to_num_external_tokens: HashMap::new(),
            kvbm_metrics,
        }
    }
}

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    /// Match the tokens in the request with the available block pools.
    /// Note: the necessary details of the request are captured prior to this call. For trtllm,
    /// we make a create slot call prior to this call, so a slot is guaranteed to exist.
    ///
    /// To align with the connector interface, we must ensure that if no blocks are matched, we return (0, false).
    /// In our implementation, if we match any block, we return (num_matched_tokens, true).
    #[tracing::instrument(level = "debug", skip(self, request_num_tokens, num_computed_tokens))]
    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(usize, bool)> {
        tracing::debug!(
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );

        // TRTLLM could match partial blocks if enable_partial_reuse = True,
        // immediately return 0 to simplify things.
        if num_computed_tokens % self.block_size != 0 {
            return Ok((0, false));
        }

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // early exit if we cannot match full block
        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            let total_tokens = slot.sequence().total_tokens();
            tracing::debug!(
                "total_tokens in sequence: {total_tokens}; num_computed_tokens: {num_computed_tokens}; can not match full block."
            );
            return Ok((0, false));
        }

        // find matches for any remaining tokens
        // this will advance the computed position and hold any newly matched blocks in the slot
        slot.acquire_local_matches(num_computed_tokens)?;

        // return the number of external tokens that are ready for onboarding
        // we always return true here as we always asynchronously onboard matched blocks
        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!((num_computed_tokens + num_external_tokens) % self.block_size == 0);
            tracing::debug!(
                request_id = request_id,
                "scheduling onboarding for {} external tokens",
                num_external_tokens
            );
            // Add to the map so that onboarding can be triggered in update_state_after_alloc.
            self.inflight_request_to_num_external_tokens
                .insert(request_id, num_external_tokens);

            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            Ok((num_external_tokens, true))
        } else {
            Ok((0, false))
        }
    }

    /// Note: TRTLLM will not provide any scheduler output data for requests that are onboarding. it is entirely
    /// on the connector's implementation to handle this case.
    #[tracing::instrument(level = "debug", skip_all, fields(request_id))]
    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> anyhow::Result<()> {
        tracing::debug!(
            request_id,
            "num_device_blocks: {}, context_current_position: {}",
            block_ids.len(),
            context_current_position
        );

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        // we have not yet advanced the computed position, but now we can, since we have an indication that we have
        // necessary gpu blocks into which we will load the external tokens.

        slot.append_mutable_device_blocks(&block_ids)?;

        if let Some(&num_external_tokens) = self
            .inflight_request_to_num_external_tokens
            .get(&request_id)
        {
            if num_external_tokens > 0 {
                let num_computed_tokens = context_current_position - num_external_tokens;
                slot.record_cached_device_tokens(num_computed_tokens);
                slot.advance_computed_position(num_computed_tokens)?;

                tracing::debug!(
                    request_id = request_id,
                    "triggering onboarding for {} external tokens",
                    num_external_tokens
                );
                slot.trigger_onboarding(num_external_tokens)?;
                self.onboarding_slots.insert(request_id.clone());
            }

            self.inflight_request_to_num_external_tokens
                .remove(&request_id);
        }

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = self.iteration_counter + 1))]
    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        // the iteration counter is used to track the number of times we have built the connector metadata
        // all connetor operations have the iteration counter at which they were issued.
        // this allows operations to be lazily enqueued to the transfer engine
        // the worker side of the connector will track all operations for completion before the request is
        // allowed to be marked as finished.
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata");
        tracing::debug!("SchedulerOutput: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        // Worker-side - we create a request slot for onboarding, then delete it when onboarding is finished, then
        // recreate it again when we start the prefill/decode phase.
        //
        // This is kind of a nice abstraction as it keeps the events simplier; however, we now create the request-slot
        // once for onboarding (this loop), then again for prefill/decode (new_requests loop).
        //
        // TODO(krish): Consider a more deterministic way to count immediate ops.
        // Currently we count by filtering pending_ops at runtime. A higher-level approach
        // (e.g., tracking count when onboard_blocks is called, or deriving from architecture
        // config) might be more robust against potential timing-related issues.
        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(request_id.clone(), num_immediate);
                md.add_operations(pending_ops);
            } else {
                md.create_slot(request_id.clone(), 0);
            }
        }

        // todo: update the code and abstraction to account for this two-phase lifecycle.
        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;

            let already_created = md.new_slots.iter().any(|s| &s.request_id == request_id);

            // Skip if this slot was already created in the onboarding_slots loop above.
            // This prevents overwriting the slot with expected_immediate_ops=0 when it should have the correct count.
            if already_created {
                assert!(
                    inflight_requests.remove(request_id),
                    "request_id {request_id} not found in inflight_requests: "
                );
                continue;
            }

            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            slot.record_start_iteration(iteration)?;

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

            slot.apply_scheduler_output(
                &new_req.prompt_token_ids,
                &new_req.block_ids,
                new_req.num_computed_tokens,
                scheduled_tokens,
            )?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                // Count immediate (onboard) operations for this slot
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                // Create slot with expected immediate ops BEFORE adding operations
                md.create_slot(new_req.request_id.clone(), num_immediate);

                tracing::debug!(
                    "adding {} pending operations for slot {} ({} immediate)",
                    pending_ops.len(),
                    new_req.request_id,
                    num_immediate
                );
                md.add_operations(pending_ops);
            } else {
                md.create_slot(new_req.request_id.clone(), 0);
            }
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            // note: evicition might trigger this assert
            assert!(
                inflight_requests.remove(request_id),
                "request_id {request_id} not found in inflight_requests: "
            );

            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

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

        tracing::debug!("metadata: {md:#?}");
        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");
        // grab the slot
        let shared_slot = self.slot_manager().get_slot(&request_id)?;

        // mark the slot as finished
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
        slot.mark_as_finished(self.iteration_counter)?;

        // todo: allow the request to resolve when it should exit
        // the request may have some outstanding operations
        // we would like to inform it to shutdown, then have it signal to the work that is officially gone,
        // then we can remove the slot and trigger the worker to clean up as well.

        // remove it from the manager as we will never use it again
        self.slot_manager().remove_slot(&request_id)?;
        self.inflight_request_to_num_external_tokens
            .remove(&request_id);

        // if the slot has finished, we can return false to trtllm, indicating all gpu blocks are free to be reused
        // otherwise, we return true, which means there are still outstanding operations on gpu blocks which
        // must be awaited before the gpu blocks can be reused. if we return true, then it is the worker side
        // of the connector api which will be used to inform trtllm that the request is finished.
        if let SlotState::Finished = slot.state() {
            Ok(false)
        } else {
            debug_assert!(matches!(slot.state(), SlotState::Finishing));
            Ok(true)
        }
    }

    fn has_slot(&self, request_id: String) -> bool {
        self.slot_manager().has_slot(&request_id)
    }

    /// Create a new slot for the given request ID.
    /// This is used to create a new slot for the request.
    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        self.slot_manager()
            .create_slot(&request.request_id, tokens, request.salt_hash)?;

        self.inflight_requests.insert(request.request_id);

        Ok(())
    }
}

#[pyclass]
pub struct PyTrtllmKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
}

#[pymethods]
impl PyTrtllmKvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, drt, page_size, leader, consolidator_trtllm_endpoint=None, consolidator_output_endpoint=None))]
    pub fn new(
        worker_id: u64,
        drt: Option<PyObject>,
        page_size: usize,
        leader: PyKvbmLeader,
        consolidator_trtllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> PyResult<Self> {
        let _ = &drt; // drt is currently un-used in leader

        let connector_leader: Box<dyn Leader> = Box::new(KvConnectorLeader::new(
            worker_id,
            page_size,
            leader,
            consolidator_trtllm_endpoint,
            consolidator_output_endpoint,
        ));
        Ok(Self { connector_leader })
    }

    fn get_num_new_matched_tokens(
        &mut self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(usize, bool)> {
        self.connector_leader
            .get_num_new_matched_tokens(request_id, request_num_tokens, num_computed_tokens)
            .map_err(to_pyerr)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        context_current_position: usize,
    ) -> PyResult<()> {
        self.connector_leader
            .update_state_after_alloc(request_id, block_ids, context_current_position)
            .map_err(to_pyerr)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        self.connector_leader
            .build_connector_metadata(scheduler_output)
            .map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: &str, block_ids: Vec<BlockId>) -> PyResult<bool> {
        self.connector_leader
            .request_finished(request_id.to_string(), block_ids)
            .map_err(to_pyerr)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.connector_leader.has_slot(request_id.to_string())
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.connector_leader
            .create_slot(request, tokens)
            .map_err(to_pyerr)
    }
}
