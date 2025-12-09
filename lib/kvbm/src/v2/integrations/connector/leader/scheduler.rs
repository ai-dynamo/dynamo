// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ConnectorLeader;
use crate::{
    G1, InstanceId, SequenceHash, distributed::offload::ExternalBlock,
    integrations::connector::leader::slot::RequestSlot, v2::BlockId,
};

use dynamo_nova::events::EventHandle;
use dynamo_tokens::Tokens;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Instant};

/// Data for a newly scheduled request that hasn't been seen before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub req_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
}

/// Data for a cached request that was previously scheduled.
///
/// This represents a request that has been scheduled before and may have been
/// preempted. The `resumed` field indicates if it resumed from preemption,
/// and `all_token_ids` contains the full token sequence if resumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub req_id: String,
    /// Whether this request resumed from preemption (derived from resumed_req_ids membership).
    pub resumed: bool,
    /// New token IDs added in this scheduling step.
    pub new_token_ids: Vec<u32>,
    /// All token IDs for the request (present only if resumed from preemption).
    pub all_token_ids: Option<Vec<u32>>,
    /// New block IDs allocated in this scheduling step.
    pub new_block_ids: Vec<BlockId>,
    /// Number of computed tokens for this request.
    pub num_computed_tokens: usize,
    /// Number of output tokens generated for this request.
    pub num_output_tokens: usize,
}

/// Scheduler output containing all requests scheduled in a single iteration.
///
/// This mirrors vLLM's `SchedulerOutput` structure with the updated API that uses
/// `resumed_req_ids` and `all_token_ids` instead of deprecated per-item fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerOutput {
    /// Iteration number
    pub iteration: usize,
    /// Requests scheduled for the first time.
    pub scheduled_new_reqs: Vec<NewRequestData>,
    /// Requests that have been scheduled before (may have been preempted).
    pub scheduled_cached_reqs: Vec<CachedRequestData>,
    /// Number of tokens scheduled for each request ID.
    pub num_scheduled_tokens: HashMap<String, usize>,
    /// Total number of tokens scheduled across all requests.
    pub total_num_scheduled_tokens: usize,
}

impl SchedulerOutput {
    /// Create a new empty SchedulerOutput.
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            ..Default::default()
        }
    }

    /// Add a new request to the output.
    pub fn add_new_request(
        &mut self,
        req_id: String,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
    ) {
        self.scheduled_new_reqs.push(NewRequestData {
            req_id,
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
        });
    }

    /// Add a cached request to the output.
    ///
    /// # Arguments
    /// * `req_id` - The request ID
    /// * `resumed` - Whether this request resumed from preemption
    /// * `new_token_ids` - New token IDs added in this step
    /// * `all_token_ids` - All token IDs (if resumed, otherwise None)
    /// * `new_block_ids` - New block IDs allocated in this step
    /// * `num_computed_tokens` - Number of computed tokens
    /// * `num_output_tokens` - Number of output tokens generated
    #[allow(clippy::too_many_arguments)]
    pub fn add_cached_request(
        &mut self,
        req_id: String,
        resumed: bool,
        new_token_ids: Vec<u32>,
        all_token_ids: Option<Vec<u32>>,
        new_block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
        num_output_tokens: usize,
    ) {
        self.scheduled_cached_reqs.push(CachedRequestData {
            req_id,
            resumed,
            new_token_ids,
            all_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
        });
    }

    /// Set the number of scheduled tokens for each request.
    ///
    /// This also updates `total_num_scheduled_tokens` to be the sum of all values.
    pub fn set_num_scheduled_tokens(&mut self, num_scheduled_tokens: HashMap<String, usize>) {
        self.num_scheduled_tokens = num_scheduled_tokens;
        self.total_num_scheduled_tokens = self.num_scheduled_tokens.values().sum();
    }

    /// Get the total number of scheduled tokens.
    pub fn total_num_scheduled_tokens(&self) -> usize {
        self.total_num_scheduled_tokens
    }

    /// Get the number of scheduled tokens for a specific request.
    pub fn num_scheduled_tokens(&self, req_id: &str) -> Option<usize> {
        self.num_scheduled_tokens.get(req_id).copied()
    }

    /// Get an iterator over new requests.
    pub fn new_requests(&self) -> impl Iterator<Item = &NewRequestData> {
        self.scheduled_new_reqs.iter()
    }

    /// Get an iterator over cached requests.
    pub fn cached_requests(&self) -> impl Iterator<Item = &CachedRequestData> {
        self.scheduled_cached_reqs.iter()
    }
}

pub struct IterationSession {
    pub iteration: usize,
    pub created: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvConnectorMetadata {
    pub iteration: usize,
    /// Map of worker instance_id to event handle for forward pass completion.
    /// Workers trigger their corresponding event in clear_connector_metadata.
    pub forward_pass_events: Option<HashMap<InstanceId, EventHandle>>,
}

pub struct ForwardPassBuilder {
    pub iteration: usize,
}

impl ConnectorLeader {
    /// Process the scheduler output and return the connector metadata.
    ///
    /// On each iteration that has a at least one total_num_scheduled_tokens, we will create a IterationSession
    /// and before returning, register the IterationSession with the [`ConnectorLeader`] and spawn a tracking task
    /// to wait for the IterationSession to be completed.
    pub fn process_scheduler_output(
        &self,
        scheduler_output: &SchedulerOutput,
    ) -> Result<KvConnectorMetadata> {
        tracing::debug!(
            "processing scheduler output for iteration: {}; active slots: {}",
            scheduler_output.iteration,
            self.slots.len()
        );

        if scheduler_output.total_num_scheduled_tokens == 0 {
            tracing::debug!("no scheduled tokens, early exiting");
            return Ok(KvConnectorMetadata::new(scheduler_output.iteration));
        }

        // Create per-worker forward pass completion events
        let worker_events = self.create_worker_forward_pass_events()?;

        // Create precondition event for offload (single or merged)
        let precondition = self.create_offload_precondition(&worker_events)?;

        // create a forward pass builder, this will be used to apply actions to the forward pass
        // - we will generate from this object a session and a metadata object
        // - the session we will register with the leader so the worker can interact with it via nova
        // - the metadata object will be serialized and sent to the workers so they know what actions to perform
        let _builder = ForwardPassBuilder::new(scheduler_output.iteration);

        for req in &scheduler_output.scheduled_new_reqs {
            match self.get_slot(&req.req_id) {
                Ok(shared_slot) => {
                    let mut slot = shared_slot.lock();
                    if let Err(e) = self.process_request_offload(
                        &req.req_id,
                        &req.block_ids,
                        &mut slot,
                        precondition,
                        0, // New requests start at offset 0
                    ) {
                        tracing::error!(
                            "failed to process offload for new req_id {}: {}",
                            req.req_id,
                            e
                        );
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "unexpected event: slot not found for request id: {}",
                        req.req_id
                    );
                }
            }
        }

        for req in &scheduler_output.scheduled_cached_reqs {
            match self.get_slot(&req.req_id) {
                Ok(shared_slot) => {
                    let mut slot = shared_slot.lock();

                    // Skip if slot is marked for deletion or finished evaluating
                    if slot.is_marked_for_deletion() {
                        tracing::debug!(
                            "slot is marked for deletion, skipping cached request for req_id: {}",
                            req.req_id
                        );
                        continue;
                    }

                    if slot.is_finished_evaluating() {
                        tracing::debug!(
                            "slot is finished evaluating, skipping cached request for req_id: {}",
                            req.req_id
                        );
                        continue;
                    }

                    // If resumed from preemption, we can't handle this yet
                    if req.resumed {
                        tracing::warn!(
                            "request resumed from preemption, marking finished_evaluating for req_id: {}",
                            req.req_id
                        );
                        slot.mark_finished_evaluating();
                        // TODO: determine how to handle resumed requests
                        continue;
                    }

                    // If we have new token IDs, extend the sequence
                    if !req.new_token_ids.is_empty() {
                        let tokens = Tokens::from(
                            req.new_token_ids
                                .iter()
                                .map(|&t| t as i32)
                                .collect::<Vec<_>>(),
                        );
                        if let Err(e) = slot.sequence.extend(tokens) {
                            tracing::error!(
                                "failed to extend sequence for req_id: {}: {}",
                                req.req_id,
                                e
                            );
                            continue;
                        }
                    }

                    // Skip if no new block IDs
                    if req.new_block_ids.is_empty() {
                        continue;
                    }

                    // Get the current count of mapped blocks - this is our offset
                    let already_mapped = slot.mapped_block_count();

                    // Process offload with offset for cached requests
                    if let Err(e) = self.process_request_offload(
                        &req.req_id,
                        &req.new_block_ids,
                        &mut slot,
                        precondition,
                        already_mapped,
                    ) {
                        tracing::error!(
                            "failed to process offload for cached req_id {}: {}",
                            req.req_id,
                            e
                        );
                    }
                }
                Err(_) => {
                    tracing::warn!(
                        "unexpected event: slot not found for cached request id: {}",
                        req.req_id
                    );
                }
            }
        }

        // Convert worker events to event map for metadata
        let event_map: HashMap<InstanceId, EventHandle> = worker_events
            .into_iter()
            .map(|(instance_id, event)| (instance_id, event.handle()))
            .collect();

        Ok(KvConnectorMetadata::new(scheduler_output.iteration).with_events(event_map))
    }

    /// Create one event per worker for forward pass completion tracking.
    fn create_worker_forward_pass_events(
        &self,
    ) -> Result<HashMap<InstanceId, Arc<dynamo_nova::events::LocalEvent>>> {
        let worker_instances = self.get_worker_instance_ids()?;
        let mut events = HashMap::new();

        for instance_id in worker_instances {
            let event = Arc::new(self.runtime.nova().events().new_event()?);
            events.insert(instance_id, event);
        }

        Ok(events)
    }

    /// Create precondition event: single event if 1 worker, merge event if N>1.
    fn create_offload_precondition(
        &self,
        worker_events: &HashMap<InstanceId, Arc<dynamo_nova::events::LocalEvent>>,
    ) -> Result<Option<EventHandle>> {
        if worker_events.is_empty() {
            return Ok(None);
        }

        let handles: Vec<EventHandle> =
            worker_events.values().map(|event| event.handle()).collect();

        if handles.len() == 1 {
            Ok(Some(handles[0]))
        } else {
            // N>1: Create merge event (spawns task immediately)
            let merged = self.runtime.nova().events().merge_events(handles)?;
            Ok(Some(merged))
        }
    }

    /// Get list of worker instance IDs from WorkerClients.
    fn get_worker_instance_ids(&self) -> Result<Vec<InstanceId>> {
        let workers = self
            .workers
            .get()
            .ok_or_else(|| anyhow::anyhow!("Workers not initialized"))?;
        Ok(workers.worker_instance_ids.clone())
    }

    /// Process offload for a single request with optional offset for cached requests.
    ///
    /// This method encapsulates the common logic for enqueuing offload operations
    /// with preconditions.
    ///
    /// # Arguments
    /// * `req_id` - Request identifier for logging
    /// * `block_ids` - Block IDs to map
    /// * `slot` - Request slot containing sequence information
    /// * `precondition` - Optional event handle that must be satisfied before transfer
    /// * `offset` - Offset into token blocks for cached requests (0 for new requests)
    fn process_request_offload(
        &self,
        req_id: &str,
        block_ids: &[BlockId],
        slot: &mut RequestSlot,
        precondition: Option<EventHandle>,
        offset: usize,
    ) -> Result<()> {
        // Skip if slot is marked for deletion or finished evaluating
        if slot.is_marked_for_deletion() || slot.is_finished_evaluating() {
            return Ok(());
        }

        let block_count = block_ids.len();
        let token_block_count = slot.sequence.blocks().len();
        let available_slots = token_block_count.saturating_sub(offset);

        // Validate block counts (allow 1 extra block for partial/decoding block)
        if block_count > available_slots + 1 {
            tracing::warn!(
                "block_ids ({}) exceed available token_blocks ({} available, {} offset) by >1, marking finished for {}",
                block_count,
                available_slots,
                offset,
                req_id
            );
            slot.mark_finished_evaluating();
        }

        let mappable_count = block_count.min(available_slots);
        if mappable_count == 0 {
            return Ok(());
        }

        // Generate source blocks from token blocks at offset
        let token_blocks = slot
            .sequence
            .blocks()
            .get(offset..offset + mappable_count)
            .expect("mappable range within bounds");

        let block_mappings: Vec<(BlockId, SequenceHash)> = block_ids
            .iter()
            .take(mappable_count)
            .zip(token_blocks.iter())
            .map(|(block_id, token_block)| (*block_id, token_block.positional_sequence_hash()))
            .collect();

        let source_blocks: Vec<_> = block_mappings
            .iter()
            .map(|(block_id, seq_hash)| ExternalBlock::<G1>::new(*block_id, *seq_hash))
            .collect();

        // Enqueue with precondition
        let handle = self
            .offload_engine
            .get()
            .expect("offload engine initialized")
            .enqueue_g1_to_g2_with_precondition(source_blocks, precondition)?;

        // Record offload in slot state
        slot.record_offload(block_mappings, handle)?;

        Ok(())
    }
}

impl IterationSession {
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            created: Instant::now(),
        }
    }
}

impl KvConnectorMetadata {
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            forward_pass_events: None,
        }
    }

    pub fn with_events(mut self, events: HashMap<InstanceId, EventHandle>) -> Self {
        self.forward_pass_events = Some(events);
        self
    }
}

impl ForwardPassBuilder {
    pub fn new(iteration: usize) -> Self {
        Self { iteration }
    }
}

pub trait Oracle: Send + Sync {
    // Evaluate the new request and determine if we should offload any blocks
    // This must be a fast local decision to avoid slowing down the scheduler
    // A positive result here means we put the returne blocks into an offload
    // engine, which is out-of-band from the scheduler and can take into account
    // more global information.
    fn evaluate_new_request(&self, req: &NewRequestData) -> Result<()>;
}

#[derive(Default, Debug)]
pub struct DefaultOracle {}

impl Oracle for DefaultOracle {
    fn evaluate_new_request(&self, _req: &NewRequestData) -> Result<()> {
        Ok(())
    }
}
