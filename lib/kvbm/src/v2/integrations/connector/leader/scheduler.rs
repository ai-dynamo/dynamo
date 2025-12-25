// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ConnectorLeader;
use crate::{
    G1, G2, InstanceId, distributed::offload::ExternalBlock,
    integrations::connector::leader::slot::RequestSlot, v2::BlockId,
    v2::logical::blocks::ImmutableBlock,
};

use derive_builder::Builder;
use dynamo_nova::events::EventHandle;

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
    pub foward_pass_completion_events: Option<HashMap<InstanceId, EventHandle>>,

    /// This will hold the G2 source and G1 destination block_ids
    pub intra_pass_load: Option<IntraPassLoad>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntraPassLoad {
    pub g2_src_block_ids: Vec<BlockId>,
    pub g1_dst_block_ids: Vec<BlockId>,
}

impl KvConnectorMetadata {
    pub fn should_bind(&self) -> bool {
        self.foward_pass_completion_events.is_some() || self.intra_pass_load.is_some()
    }
}

pub struct ForwardPassBuilder {
    pub iteration: usize,
}

#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned")]
pub struct ForwardPassSample {
    pub iteration: usize,
    pub total_scheduled_tokens: usize,
    pub new_reqs: usize,
    pub cached_reqs: usize,
    pub active_slots: usize,

    #[builder(default = "std::time::Instant::now()")]
    pub forward_pass_start: Instant,
}

impl ForwardPassSample {
    pub fn builder() -> ForwardPassSampleBuilder {
        ForwardPassSampleBuilder::default()
    }
}

impl ConnectorLeader {
    /// Process the scheduler output and return the connector metadata.
    ///
    /// On each iteration that has a at least one total_num_scheduled_tokens, we will create a IterationSession
    /// and before returning, register the IterationSession with the [`ConnectorLeader`] and spawn a tracking task
    /// to wait for the IterationSession to be completed.
    pub fn process_scheduler_output(
        &self,
        scheduler_output: SchedulerOutput,
    ) -> Result<KvConnectorMetadata> {
        if let Some(forward_pass_sample) = self.forward_pass_samples.lock().take() {
            tracing::info!(
                iteration = forward_pass_sample.iteration,
                "Previous forward pass took {:?}; {:?}",
                forward_pass_sample.forward_pass_start.elapsed(),
                forward_pass_sample
            );
        }

        if scheduler_output.total_num_scheduled_tokens == 0 {
            tracing::debug!(
                iteration = scheduler_output.iteration,
                new_reqs = scheduler_output.scheduled_new_reqs.len(),
                cached_reqs = scheduler_output.scheduled_cached_reqs.len(),
                "No scheduled tokens, early exiting"
            );
            return Ok(KvConnectorMetadata::new(scheduler_output.iteration));
        }

        let forward_pass_sample = ForwardPassSample::builder()
            .iteration(scheduler_output.iteration)
            .total_scheduled_tokens(scheduler_output.total_num_scheduled_tokens)
            .new_reqs(scheduler_output.scheduled_new_reqs.len())
            .cached_reqs(scheduler_output.scheduled_cached_reqs.len())
            .active_slots(self.slots.len())
            .build()?;

        tracing::info!("Forward pass sample: {:?}", forward_pass_sample);

        *self.forward_pass_samples.lock() = Some(forward_pass_sample);

        // Create per-worker forward pass completion events
        let worker_events = self.create_worker_foward_pass_completion_events()?;

        // Create cheap local event as the "promise" for forward pass completion.
        // This is passed to process_request_offload as the precondition.
        // The actual merge event is only created at the end if any actions were scheduled.
        let forward_pass_promise = Arc::new(self.runtime.nova().events().new_event()?);
        let forward_pass_handle = Some(forward_pass_promise.handle());

        // Track if any offload actions were scheduled
        let mut scheduled_actions: bool = false;

        // Process new requests
        for req in scheduler_output.scheduled_new_reqs {
            match self.get_slot(&req.req_id) {
                Ok(shared_slot) => {
                    let mut slot = shared_slot.lock();

                    // We are given all the block_ids for the request, but we need to filter out all those that may
                    // have been applied to the slot already.
                    let filtered_block_ids = slot.filter_block_ids(req.block_ids);
                    slot.apply_new_blocks(filtered_block_ids);

                    // Get scheduled tokens and compute total for offload evaluation.
                    // We evaluate ALL tokens (computed + scheduled) so presence filters
                    // handle any blocks already in G2 from external matches.
                    let num_scheduled_tokens = scheduler_output
                        .num_scheduled_tokens
                        .get(&req.req_id)
                        .copied()
                        .unwrap_or(0);
                    let total_tokens = req.num_computed_tokens + num_scheduled_tokens;

                    tracing::debug!(
                        req_id = %req.req_id,
                        num_computed_tokens = req.num_computed_tokens,
                        num_scheduled_tokens,
                        total_tokens,
                        assigned_blocks = slot.assigned_block_count(),
                        evaluated_tokens = slot.evaluated_tokens(),
                        evaluated_blocks = slot.evaluated_blocks(),
                        "new_request: evaluating blocks for offload"
                    );

                    match self.process_request_offload(
                        &req.req_id,
                        &mut slot,
                        total_tokens,
                        forward_pass_handle,
                    ) {
                        Ok(OffloadAction::NoAction) => {
                            tracing::trace!(
                                "no offload action for new request for req_id: {}",
                                req.req_id
                            );
                        }
                        Ok(OffloadAction::InterPassOffloadScheduled) => {
                            scheduled_actions = true;
                            tracing::trace!(
                                "inter-pass offload scheduled for new request for req_id: {}",
                                req.req_id
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "failed to process offload for new req_id {}: {}",
                                req.req_id,
                                e
                            );
                        }
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

        for req in scheduler_output.scheduled_cached_reqs {
            tracing::debug!(
                req_id = %req.req_id,
                new_token_ids_count = req.new_token_ids.len(),
                new_block_ids_count = req.new_block_ids.len(),
                num_computed_tokens = req.num_computed_tokens,
                "Processing cached request"
            );

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

                    // NOTE: We do NOT extend tokens here from req.new_token_ids.
                    //
                    // Token extension is handled by Python's update_slot() method which calls
                    // extend_slot_tokens() BEFORE build_connector_metadata() is invoked.
                    // This ensures the slot's sequence is already up-to-date with the latest
                    // tokens from vLLM's Request object.
                    //
                    // The scheduler output's new_token_ids field may be None/empty during
                    // decode (vLLM doesn't always populate it), but update_slot() reliably
                    // syncs tokens by comparing slot.total_tokens() vs request.all_token_ids.
                    //
                    // Applying tokens here would risk double-application or race conditions.
                    // If vLLM ever starts populating new_token_ids during decode, we should
                    // verify Python's total_tokens matches Rust's before deciding to apply.

                    // Apply new block IDs if any (handles unassigned blocks correctly)
                    let new_block_ids_count = req.new_block_ids.len();
                    if !req.new_block_ids.is_empty() {
                        slot.apply_new_blocks(req.new_block_ids);
                    }

                    let num_scheduled_tokens = scheduler_output
                        .num_scheduled_tokens
                        .get(&req.req_id)
                        .copied()
                        .unwrap_or(0);

                    tracing::debug!(
                        req_id = %req.req_id,
                        num_scheduled_tokens,
                        new_block_ids = new_block_ids_count,
                        assigned_blocks = slot.assigned_block_count(),
                        evaluated_tokens = slot.evaluated_tokens(),
                        evaluated_blocks = slot.evaluated_blocks(),
                        "cached_request: evaluating blocks for offload"
                    );

                    // Always process offload - may have completed blocks from previous allocations
                    // even if no new block IDs were allocated this step

                    match self.process_request_offload(
                        &req.req_id,
                        &mut slot,
                        num_scheduled_tokens,
                        forward_pass_handle,
                    ) {
                        Ok(OffloadAction::NoAction) => {
                            tracing::trace!(
                                "no offload action for cached request for req_id: {}",
                                req.req_id
                            );
                        }
                        Ok(OffloadAction::InterPassOffloadScheduled) => {
                            scheduled_actions = true;
                            tracing::trace!(
                                "inter-pass offload scheduled for cached request for req_id: {}",
                                req.req_id
                            );
                        }
                        Err(e) => {
                            tracing::error!(
                                "failed to process offload for cached req_id {}: {}",
                                req.req_id,
                                e
                            );
                        }
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

        // Aggregate pending intra-pass onboarding data from all slots
        let intra_pass_load = self.aggregate_intra_pass_onboarding();

        // Take accumulated G2 blocks for cleanup
        let g2_blocks = std::mem::take(&mut *self.pending_intra_pass_g2_blocks.lock());
        if !g2_blocks.is_empty() {
            assert!(intra_pass_load.is_some());
            scheduled_actions = true;
        }

        let mut metadata = KvConnectorMetadata::new(scheduler_output.iteration);

        if scheduled_actions {
            // Convert worker events to event map for metadata
            let event_map: HashMap<InstanceId, EventHandle> = worker_events
                .iter()
                .map(|(instance_id, event)| (*instance_id, event.handle()))
                .collect();

            metadata = metadata.with_events(event_map);

            if let Some(ref load) = intra_pass_load {
                metadata.intra_pass_load = Some(load.clone());
                tracing::debug!(
                    iteration = scheduler_output.iteration,
                    g2_count = load.g2_src_block_ids.len(),
                    g1_count = load.g1_dst_block_ids.len(),
                    "Added intra-pass load to connector metadata"
                );
            }

            // Spawn unified cleanup task that:
            // 1. Awaits the merge of all worker events (lazy - only creates merge here)
            // 2. Triggers the forward_pass_promise (unblocks preconditions)
            // 3. Drops G2 blocks (releases back to cache)
            self.spawn_forward_pass_cleanup_task(worker_events, forward_pass_promise, g2_blocks);
        }

        Ok(metadata)
    }

    /// Aggregate pending intra-pass onboarding data from all active slots.
    ///
    /// This method iterates through all slots, takes any pending intra-pass data,
    /// and combines it into a single `IntraPassLoad` for the workers.
    fn aggregate_intra_pass_onboarding(&self) -> Option<IntraPassLoad> {
        let mut g2_src_all = Vec::new();
        let mut g1_dst_all = Vec::new();

        // Iterate through all slots and collect pending intra-pass data
        for entry in self.slots.iter() {
            let mut slot = entry.value().lock();
            if let Some(pending) = slot.take_pending_intra_pass() {
                g2_src_all.extend(pending.g2_block_ids);
                g1_dst_all.extend(pending.g1_block_ids);
            }
        }

        if g2_src_all.is_empty() {
            None
        } else {
            Some(IntraPassLoad {
                g2_src_block_ids: g2_src_all,
                g1_dst_block_ids: g1_dst_all,
            })
        }
    }

    /// Create one event per worker for forward pass completion tracking.
    fn create_worker_foward_pass_completion_events(
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

    /// Get list of worker instance IDs from WorkerClients.
    fn get_worker_instance_ids(&self) -> Result<Vec<InstanceId>> {
        let workers = self
            .workers
            .get()
            .ok_or_else(|| anyhow::anyhow!("Workers not initialized"))?;
        Ok(workers.worker_instance_ids.clone())
    }

    /// Spawn a unified cleanup task for forward pass completion.
    ///
    /// This task:
    /// 1. Creates and awaits a merge event from all worker events (lazy - only created here if needed)
    /// 2. Triggers the forward_pass_promise (unblocks any preconditions waiting on it)
    /// 3. Drops G2 blocks (releases them back to the G2 cache)
    ///
    /// By deferring merge event creation to this async task, we avoid spawning
    /// the merge task unless offload actions were actually scheduled.
    fn spawn_forward_pass_cleanup_task(
        &self,
        worker_events: HashMap<InstanceId, Arc<dynamo_nova::events::LocalEvent>>,
        forward_pass_promise: Arc<dynamo_nova::events::LocalEvent>,
        g2_blocks: Vec<ImmutableBlock<G2>>,
    ) {
        let nova = self.runtime.nova().clone();
        let block_count = g2_blocks.len();

        tracing::debug!(
            block_count,
            worker_count = worker_events.len(),
            "Spawning forward pass cleanup task"
        );

        self.runtime.tokio().spawn(async move {
            // Step 1: Create and await merge event (lazy - only created now)
            let handles: Vec<EventHandle> = worker_events.values().map(|e| e.handle()).collect();

            let await_result: Option<Result<(), anyhow::Error>> = if handles.len() == 1 {
                // Single worker - just await directly
                match nova.events().awaiter(handles[0]) {
                    Ok(awaiter) => Some(awaiter.await),
                    Err(e) => {
                        tracing::error!("Failed to create awaiter for single worker: {}", e);
                        None
                    }
                }
            } else if !handles.is_empty() {
                // Multiple workers - create merge and await
                match nova.events().merge_events(handles) {
                    Ok(merge_handle) => match nova.events().awaiter(merge_handle) {
                        Ok(awaiter) => Some(awaiter.await),
                        Err(e) => {
                            tracing::error!("Failed to create merge awaiter: {}", e);
                            None
                        }
                    },
                    Err(e) => {
                        tracing::error!("Failed to create merge event: {}", e);
                        None
                    }
                }
            } else {
                None
            };

            if let Some(Err(e)) = await_result {
                tracing::warn!("Forward pass completion failed: {}", e);
            }

            // Step 2: Trigger the promise (unblocks preconditions)
            if let Err(e) = forward_pass_promise.trigger() {
                tracing::warn!("Failed to trigger forward pass promise: {}", e);
            }

            // Step 3: Release G2 blocks (implicit via drop)
            tracing::debug!(
                block_count,
                "Released G2 blocks after forward pass completion"
            );
            drop(g2_blocks);
        });
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
        slot: &mut RequestSlot,
        num_scheduled_tokens: usize,
        precondition: Option<EventHandle>,
    ) -> Result<OffloadAction> {
        // Skip if slot is marked for deletion or finished evaluating
        if slot.is_marked_for_deletion() || slot.is_finished_evaluating() {
            return Ok(OffloadAction::NoAction);
        }

        // Capture state before evaluation for debug logging
        let evaluated_blocks_before = slot.evaluated_blocks();

        // Ask the slot for the next set of assigned_block_ids to offload
        let block_mappings = slot.get_next_block_mappings(num_scheduled_tokens);

        // Only advance evaluated_tokens up to assigned blocks.
        // This prevents advancing past blocks that haven't been assigned yet,
        // which can happen due to a 1-iteration lag between token extension
        // (via update_slot) and block assignment (via scheduler_output).
        let max_tokens_with_blocks = slot.assigned_block_count() * slot.block_size();
        let desired_tokens = slot.evaluated_tokens() + num_scheduled_tokens;
        let capped_tokens = desired_tokens.min(max_tokens_with_blocks);
        let actual_advance = capped_tokens.saturating_sub(slot.evaluated_tokens());
        slot.advance_evaluated_tokens(actual_advance);

        let evaluated_blocks_after = slot.evaluated_blocks();

        tracing::debug!(
            req_id,
            num_scheduled_tokens,
            evaluated_blocks_before,
            evaluated_blocks_after,
            blocks_to_offload = block_mappings.len(),
            "offload: queuing blocks"
        );

        let source_blocks: Vec<_> = block_mappings
            .iter()
            .map(|(block_id, seq_hash)| ExternalBlock::<G1>::new(*block_id, *seq_hash))
            .collect();

        if source_blocks.is_empty() {
            return Ok(OffloadAction::NoAction);
        }

        // Enqueue with precondition
        let handle = self
            .offload_engine
            .get()
            .expect("offload engine initialized")
            .enqueue_g1_to_g2_with_precondition(source_blocks, precondition)?;

        // Record offload in slot state
        slot.record_offload(block_mappings, handle)?;

        Ok(OffloadAction::InterPassOffloadScheduled)
    }
}

enum OffloadAction {
    NoAction,
    InterPassOffloadScheduled,
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
            foward_pass_completion_events: None,
            intra_pass_load: None,
        }
    }

    pub fn with_events(mut self, events: HashMap<InstanceId, EventHandle>) -> Self {
        self.foward_pass_completion_events = Some(events);
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
