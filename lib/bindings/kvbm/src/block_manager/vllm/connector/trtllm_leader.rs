// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::block_manager::BlockManagerBuilder;
use crate::block_manager::vllm::connector::leader::slot::{
    ConnectorSlotManager, SlotManager, SlotState,
};
use crate::block_manager::vllm::connector::leader::{
    kvbm_metrics_endpoint_enabled, parse_kvbm_metrics_port,
};
use crate::block_manager::vllm::connector::trtllm_onboarding_advisor::RemoteOnboardingAdvisor;
use crate::block_manager::{distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest};
use crate::get_current_tokio_handle;
use dynamo_llm::block_manager::connector::protocol::RequestType;
use dynamo_llm::block_manager::kv_consolidator::{EventSource, KvEventConsolidationMode};
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use dynamo_runtime::DistributedRuntime;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, OnceLock};
use tokio::runtime::Handle;

fn parse_consolidator_mode(mode: Option<String>) -> KvEventConsolidationMode {
    let Some(mode) = mode else {
        return KvEventConsolidationMode::Dedup;
    };

    match mode.parse() {
        Ok(mode) => mode,
        Err(error) => {
            tracing::warn!(
                "Invalid KV event consolidator mode {:?}: {}. Falling back to dedup.",
                mode,
                error
            );
            KvEventConsolidationMode::Dedup
        }
    }
}

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
        consolidator_mode: Option<String>,
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
            let consolidator_trtllm_ep = consolidator_trtllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();
            let consolidator_mode = parse_consolidator_mode(consolidator_mode.clone());

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
                        consolidator_mode,
                    );
                }

                let block_manager = match block_manager_builder.build().await {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

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

        if !num_computed_tokens.is_multiple_of(self.block_size) {
            return Ok((0, false));
        }

        let shared_slot = self.slot_manager().get_slot(&request_id)?;
        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            let total_tokens = slot.sequence().total_tokens();
            tracing::debug!(
                "total_tokens in sequence: {total_tokens}; num_computed_tokens: {num_computed_tokens}; can not match full block."
            );
            return Ok((0, false));
        }

        slot.acquire_local_matches(num_computed_tokens)?;

        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!(
                (num_computed_tokens + num_external_tokens).is_multiple_of(self.block_size)
            );
            tracing::debug!(
                request_id = request_id,
                "scheduling onboarding for {} external tokens",
                num_external_tokens
            );
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
        self.iteration_counter += 1;
        let iteration = self.iteration_counter;

        tracing::debug!("Building connector metadata");
        tracing::debug!("SchedulerOutput: {scheduler_output:#?}");

        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);

        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        for request_id in onboarding_slots.iter() {
            let shared_slot = self.slot_manager().get_slot(request_id)?;
            let mut slot = shared_slot
                .lock()
                .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

                md.create_slot(request_id.clone(), num_immediate);
                md.add_operations(pending_ops);
            } else {
                md.create_slot(request_id.clone(), 0);
            }
        }

        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;

            let already_created = md.new_slots.iter().any(|s| &s.request_id == request_id);

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
                .get(&new_req.request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &[],
                &new_req.block_ids,
                new_req.num_computed_tokens,
                scheduled_tokens,
                new_req.priorities.as_deref(),
                new_req.external_sequence_hashes.as_deref(),
            )?;

            let pending_ops_opt = slot.take_pending_operations();

            if let Some(pending_ops) = pending_ops_opt {
                let num_immediate = pending_ops
                    .iter()
                    .filter(|op| op.request_type == RequestType::Immediate)
                    .count() as u64;

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
                .get(&cached_req.request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
                cached_req.priorities.as_deref(),
                cached_req.external_sequence_hashes.as_deref(),
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
        let shared_slot = self.slot_manager().get_slot(&request_id)?;

        let mut slot = shared_slot
            .lock()
            .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
        slot.mark_as_finished(self.iteration_counter)?;

        self.slot_manager().remove_slot(&request_id)?;
        self.inflight_request_to_num_external_tokens
            .remove(&request_id);

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
    onboarding_advisor: RemoteOnboardingAdvisor,
}

#[pymethods]
impl PyTrtllmKvConnectorLeader {
    #[new]
    #[pyo3(signature = (rank, device_id, drt, page_size, leader, consolidator_trtllm_endpoint=None, consolidator_output_endpoint=None, consolidator_mode=None))]
    pub fn new(
        rank: u64,
        device_id: u64,
        drt: Option<PyObject>,
        page_size: usize,
        leader: PyKvbmLeader,
        consolidator_trtllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
        consolidator_mode: Option<String>,
    ) -> PyResult<Self> {
        let drt: Option<Arc<DistributedRuntime>> = Python::with_gil(|py| {
            if let Some(obj) = drt {
                crate::extract_distributed_runtime_from_obj(py, obj)
            } else {
                Ok(None)
            }
        })?;

        let connector_leader = KvConnectorLeader::new(
            rank,
            page_size,
            leader,
            consolidator_trtllm_endpoint,
            consolidator_output_endpoint,
            consolidator_mode,
        );
        let onboarding_advisor = RemoteOnboardingAdvisor::new(
            drt,
            device_id as usize,
            connector_leader.slot_manager.clone(),
            connector_leader.kvbm_metrics.clone(),
        );

        Ok(Self {
            connector_leader: Box::new(connector_leader),
            onboarding_advisor,
        })
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

    #[pyo3(signature = (request, tokens, transfer_budget_ms=100, min_blocks=10))]
    fn advise_async_loading(
        &mut self,
        request: KvbmRequest,
        tokens: Vec<u32>,
        transfer_budget_ms: u64,
        min_blocks: u64,
    ) -> PyResult<()> {
        let _ = tokens;
        self.onboarding_advisor
            .advise_async_onboarding(request, transfer_budget_ms, min_blocks)
            .map_err(to_pyerr)
    }
}
