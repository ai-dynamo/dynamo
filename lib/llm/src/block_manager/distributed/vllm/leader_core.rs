// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, OnceLock},
};

use crate::block_manager::{
    block::BlockId,
    connector::protocol::WorkerTransferRequest,
    distributed::vllm::is_dev_mode,
    metrics_kvbm::KvbmMetrics,
};
use serde::{Deserialize, Serialize};

use super::{
    ConnectorSlotManager, SlotManager, SlotState, VllmConnectorSlot,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerOutput {
    pub new_requests: Vec<NewRequestData>,
    pub cached_requests: Vec<CachedRequestData>,
    pub num_scheduled_tokens: std::collections::HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub priorities: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
    pub priorities: Option<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSlotInfo {
    pub request_id: String,
    pub expected_immediate_ops: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorMetadata {
    pub iteration: u64,
    pub new_slots: Vec<NewSlotInfo>,
    pub operations: Vec<WorkerTransferRequest>,
}

impl ConnectorMetadata {
    pub fn new(iteration: u64) -> Self {
        Self {
            iteration,
            new_slots: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn create_slot(&mut self, request_id: String, expected_immediate_ops: u64) {
        self.new_slots.push(NewSlotInfo {
            request_id,
            expected_immediate_ops,
        });
    }

    pub fn add_operations(&mut self, xfer_reqs: Vec<WorkerTransferRequest>) {
        self.operations.extend(xfer_reqs);
    }
}

#[derive(Debug)]
pub struct KvConnectorLeaderCore {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    finishing_requests: HashSet<String>,
    iteration_counter: u64,
    kvbm_metrics: KvbmMetrics,
    /// Maps request_id -> W3C traceparent so all spans for one request share a single trace ID.
    request_traces: HashMap<String, String>,
}

impl KvConnectorLeaderCore {
    pub fn new(
        slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
        block_size: usize,
        kvbm_metrics: KvbmMetrics,
    ) -> Self {
        Self {
            slot_manager,
            block_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            finishing_requests: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
            request_traces: HashMap::new(),
        }
    }

    #[inline]
    pub fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    /// Get the traceparent for a request, if one was registered at create_slot time.
    pub fn request_traceparent(&self, request_id: &str) -> Option<&str> {
        self.request_traces.get(request_id).map(|s| s.as_str())
    }

    /// Override the traceparent for a request. Called from Python when the
    /// connector can provide a traceparent from the HTTP/OTEL context.
    pub fn set_request_traceparent(&mut self, request_id: String, traceparent: String) {
        self.request_traces.insert(request_id, traceparent);
    }

    /// Enter a span linked to the request's trace. Returns the guard (drop to exit).
    /// Falls back to a standalone span if no traceparent is registered.
    fn enter_request_span(&self, request_id: &str, span_name: &'static str) -> tracing::span::EnteredSpan {
        if let Some(tp) = self.request_traces.get(request_id) {
            dynamo_runtime::logging::make_linked_span(span_name, tp).entered()
        } else {
            tracing::info_span!("kvbm_op", otel.name = span_name, request_id = request_id).entered()
        }
    }

    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        _request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(Option<usize>, bool)> {
        let _span = self.enter_request_span(&request_id, "kvbm.get_matched_tokens");
        debug_assert!(num_computed_tokens.is_multiple_of(self.block_size));

        crate::lock_slot!(self, &request_id => slot);

        debug_assert!(
            slot.state() != SlotState::Prefilling && slot.state() != SlotState::Decoding,
            "slot is in the Prefilled state or Decoding; shouldn't happen"
        );

        if slot.state() == SlotState::SkippedPrefill || slot.state() == SlotState::SkippedDecode {
            match slot.state() {
                SlotState::SkippedPrefill => {
                    slot.mark_as_prefilling(self.iteration_counter)?;
                    return Ok((Some(0), false));
                }
                SlotState::SkippedDecode => {
                    slot.mark_as_decoding(self.iteration_counter)?;
                    return Ok((Some(0), false));
                }
                _ => unreachable!("slot is not in the SkippedPrefill or SkippedDecode state"),
            }
        }

        if (slot.sequence().total_tokens() - num_computed_tokens) < self.block_size {
            return Ok((Some(0), false));
        }

        slot.acquire_local_matches(num_computed_tokens)?;

        if slot
            .as_any_mut()
            .downcast_mut::<VllmConnectorSlot>()
            .map(|s| s.has_pending_g4_lookup())
            .unwrap_or(false)
        {
            return Ok((None, false));
        }

        if let SlotState::OnboardStaged(num_external_tokens) = slot.state() {
            debug_assert!(
                (num_computed_tokens + num_external_tokens).is_multiple_of(self.block_size)
            );
            self.kvbm_metrics
                .matched_tokens
                .inc_by(num_external_tokens as u64);
            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens,
                num_computed_tokens,
                total_tokens = slot.sequence().total_tokens(),
                "get_num_new_matched_tokens → OnboardStaged (returning match)"
            );
            Ok((Some(num_external_tokens), true))
        } else {
            Ok((Some(0), false))
        }
    }

    pub fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        let _span = self.enter_request_span(&request_id, "kvbm.update_state_after_alloc");
        crate::lock_slot!(self, &request_id => slot);
        slot.append_mutable_device_blocks(&block_ids)?;

        if num_external_tokens > 0 {
            let num_computed_tokens = block_ids.len() * self.block_size - num_external_tokens;
            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                num_external_tokens,
                num_device_blocks = block_ids.len(),
                num_computed_tokens,
                block_size = self.block_size,
                "update_state_after_alloc → triggering onboarding"
            );
            slot.record_cached_device_tokens(num_computed_tokens);
            slot.advance_computed_position(num_computed_tokens)?;
            slot.trigger_onboarding(num_external_tokens)?;
            self.onboarding_slots.insert(request_id);
        }

        Ok(())
    }

    pub fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>> {
        self.kvbm_metrics.scheduler_new_requests.set(scheduler_output.new_requests.len() as f64);
        self.kvbm_metrics.scheduler_cached_requests.set(scheduler_output.cached_requests.len() as f64);
        self.kvbm_metrics.scheduler_finishing.set(self.finishing_requests.len() as f64);
        self.kvbm_metrics.scheduler_onboarding.set(self.onboarding_slots.len() as f64);
        self.kvbm_metrics.scheduler_inflight.set(self.inflight_requests.len() as f64);

        if !self.finishing_requests.is_empty() {
            let to_clean: Vec<String> = self.finishing_requests.drain().collect();
            for request_id in &to_clean {
                if self.slot_manager().has_slot(request_id) {
                    let _ = self.slot_manager().remove_slot(request_id);
                }
            }
        }

        self.iteration_counter += 1;
        let iteration = self.iteration_counter;
        let mut inflight_requests = self.inflight_requests.clone();
        let mut md = ConnectorMetadata::new(iteration);
        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);

        if !onboarding_slots.is_empty() {
            tracing::info!(
                target: "kvbm-diag",
                iteration,
                num_onboarding = onboarding_slots.len(),
                onboarding_reqs = ?onboarding_slots,
                "build_connector_metadata: flushing onboarding slots"
            );
        }

        for request_id in &onboarding_slots {
            let _req_span = self.enter_request_span(request_id, "kvbm.flush_onboarding");
            crate::lock_slot!(self, request_id => slot);
            crate::flush_slot_to_metadata!(slot, md, request_id);
            assert!(inflight_requests.remove(request_id));
        }

        for new_req in &scheduler_output.new_requests {
            let request_id = &new_req.request_id;
            let already_created = md.new_slots.iter().any(|s| &s.request_id == request_id);
            if already_created {
                assert!(inflight_requests.remove(request_id));
                continue;
            }

            let _req_span = self.enter_request_span(request_id, "kvbm.schedule_new_request");
            assert!(inflight_requests.remove(request_id));
            crate::lock_slot!(self, request_id => slot);
            slot.record_start_iteration(iteration)?;

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            tracing::info!(
                target: "kvbm-diag",
                request_id = %request_id,
                iteration,
                vllm_num_computed_tokens = new_req.num_computed_tokens,
                vllm_num_scheduled_tokens = scheduled_tokens,
                slot_state = ?slot.state(),
                slot_computed_tokens = slot.computed_tokens(),
                "build_connector_metadata: new request from vLLM scheduler"
            );

            slot.apply_scheduler_output(&[], &[], new_req.num_computed_tokens, scheduled_tokens, None)?;
            crate::flush_slot_to_metadata!(slot, md, new_req.request_id);
        }

        for cached_req in &scheduler_output.cached_requests {
            let request_id = &cached_req.request_id;

            if cached_req.resumed_from_preemption {
                let shared_slot = self.slot_manager().get_slot(request_id)?;
                let mut slot = shared_slot
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Failed to lock slot: {}", e))?;
                slot.reset_after_preemption();
            }

            assert!(inflight_requests.remove(request_id));
            crate::lock_slot!(self, request_id => slot);

            let scheduled_tokens = *scheduler_output
                .num_scheduled_tokens
                .get(request_id)
                .unwrap_or(&0);

            slot.apply_scheduler_output(
                &cached_req.new_token_ids,
                &cached_req.new_block_ids,
                cached_req.num_computed_tokens,
                scheduled_tokens,
                None,
            )?;

            if let Some(pending_ops) = slot.take_pending_operations() {
                md.add_operations(pending_ops);
            }
        }

        for unscheduled_req in &inflight_requests {
            crate::lock_slot!(self, unscheduled_req => slot_guard);
            let slot = slot_guard
                .as_any_mut()
                .downcast_mut::<VllmConnectorSlot>()
                .ok_or_else(|| anyhow::anyhow!("Expected VllmConnectorSlot, got different type"))?;
            slot.mark_as_skipped()?;
        }

        serde_json::to_vec(&md)
            .map_err(|e| anyhow::anyhow!("Failed to serialize connector metadata: {}", e))
    }

    pub fn request_finished(
        &mut self,
        request_id: String,
        _block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool> {
        self.onboarding_slots.remove(&request_id);
        self.request_traces.remove(&request_id);

        if !self.slot_manager().has_slot(&request_id) {
            self.inflight_requests.remove(&request_id);
            return Ok(true);
        }

        crate::lock_slot!(self, &request_id => slot);
        if matches!(slot.state(), SlotState::Onboarding(_))
            && let Some(vllm_slot) = slot.as_any_mut().downcast_mut::<VllmConnectorSlot>()
        {
            vllm_slot.discard_pending_operations();
        }

        slot.mark_as_finished(self.iteration_counter)?;
        self.inflight_requests.remove(&request_id);

        match slot.state() {
            SlotState::Finished => {
                self.slot_manager().remove_slot(&request_id)?;
            }
            SlotState::Finishing => {
                self.finishing_requests.insert(request_id);
            }
            _ => {
                self.slot_manager().remove_slot(&request_id)?;
            }
        }

        Ok(true)
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slot_manager().has_slot(&request_id.to_string())
    }

    pub fn create_slot(
        &mut self,
        request_id: String,
        salt_hash: u64,
        tokens: Vec<u32>,
    ) -> anyhow::Result<()> {
        self.slot_manager()
            .create_slot(&request_id, tokens, salt_hash)?;
        self.inflight_requests.insert(request_id.clone());

        if !self.request_traces.contains_key(&request_id) {
            let root_span = tracing::info_span!(
                "kvbm_request",
                otel.name = "kvbm.request",
                request_id = %request_id,
            );
            let _guard = root_span.entered();
            if let Some(ctx) = dynamo_runtime::logging::get_distributed_tracing_context() {
                self.request_traces
                    .insert(request_id, ctx.create_traceparent());
            }
        }
        Ok(())
    }

    pub fn clear_pool(&mut self, pool: String) -> anyhow::Result<()> {
        if !is_dev_mode() {
            anyhow::bail!(
                "clear_pool called but KVBM_DEV_MODE is not enabled. \
                 Set KVBM_DEV_MODE=TRUE to allow destructive pool operations."
            );
        }
        self.inflight_requests.clear();
        self.onboarding_slots.clear();
        self.finishing_requests.clear();
        self.slot_manager().clear_pool(&pool)?;
        Ok(())
    }
}
