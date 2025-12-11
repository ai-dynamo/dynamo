// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::num::NonZero;
use std::sync::{Arc, Mutex};

use dynamo_tokens::{TokenBlockSequence, Tokens};
use uuid::Uuid;

use super::{
    G1, G2, G3, LeaderRuntime, Result, data::*, init::InitializedState,
    messages::KvbmOnboardRequest,
};
use crate::physical::manager::LayoutHandle;
use crate::v2::integrations::IntegrationsConfig;
use crate::v2::integrations::connector::{
    ConnectorMetadataBuilder, OperationInfo, Slot, SlotCore, SlotState, TransferDirection,
};
use crate::v2::logical::executor::{
    BroadcastSlotStateBroadcaster, ChannelTransferDispatcher, MockBlockManager, SlotExecutor,
    TransferPipelineRuntime,
};
use crate::v2::logical::manager::BlockManager;
use crate::v2::logical::pools::SequenceHash;

#[cfg(feature = "console")]
use super::events::{ActionCollectorRef, AllocPurpose, ConnectorAction, NoopActionCollector};

pub struct ConnectorLeader {
    engine_id: String,
    metadata: ConnectorMetadataBuilder,

    // Configuration
    config: IntegrationsConfig,

    // Block managers (for matching and allocation)
    cpu_block_manager: Arc<BlockManager<G2>>,
    disk_block_manager: Option<Arc<BlockManager<G3>>>,

    // Worker layout handles (for distributed operations)
    g1_handles: Vec<LayoutHandle>,
    g2_handles: Vec<LayoutHandle>,
    g3_handles: Option<Vec<LayoutHandle>>,

    // Slot management
    slots: HashMap<String, Mutex<Slot>>,

    // Executor integration
    pipeline_runtime: Arc<TransferPipelineRuntime<String>>,
    slot_executor: SlotExecutor<ChannelTransferDispatcher, BroadcastSlotStateBroadcaster>,

    // Tracking state
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,

    // Keep existing
    known_slots: HashSet<String>,
    forward_seq: HashMap<String, u64>,
    pending_deletes: Vec<String>,

    // Action recording (console feature only)
    #[cfg(feature = "console")]
    action_collector: ActionCollectorRef,
}

impl std::fmt::Debug for ConnectorLeader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnectorLeader")
            .field("engine_id", &self.engine_id)
            .field("block_size", &self.config.block_size())
            .field("num_slots", &self.slots.len())
            .field("world_size", &self.config.parallel.world_size())
            .field("inflight_requests", &self.inflight_requests)
            .field("onboarding_slots", &self.onboarding_slots)
            .field("iteration_counter", &self.iteration_counter)
            .field("known_slots", &self.known_slots)
            .finish_non_exhaustive()
    }
}

impl ConnectorLeader {
    /// Create a new ConnectorLeader with mock initialization.
    ///
    /// This constructor uses mock block managers and bypasses the leader-worker
    /// handshake. Suitable for testing and development.
    pub fn new(engine_id: impl Into<String>, config: IntegrationsConfig) -> Self {
        // Create mock initialized state
        let init_state = InitializedState::mock_from_config(&config)
            .expect("Failed to create mock initialized state");

        Self::new_with_state(engine_id, config, init_state)
    }

    /// Create a new ConnectorLeader with pre-initialized state.
    ///
    /// This constructor accepts an `InitializedState` from the initialization
    /// facade, which contains block managers and worker layout handles.
    pub fn new_with_state(
        engine_id: impl Into<String>,
        config: IntegrationsConfig,
        state: InitializedState,
    ) -> Self {
        // Create executor infrastructure
        let block_manager = Arc::new(MockBlockManager::new());
        let pipeline_runtime = Arc::new(TransferPipelineRuntime::new(block_manager));

        let dispatcher = pipeline_runtime.dispatcher();
        let broadcaster =
            BroadcastSlotStateBroadcaster::new(tokio::sync::broadcast::channel(1024).0);
        let slot_executor = SlotExecutor::new(dispatcher, broadcaster);

        Self {
            engine_id: engine_id.into(),
            metadata: ConnectorMetadataBuilder::new(1),
            config,
            cpu_block_manager: state.cpu_block_manager,
            disk_block_manager: state.disk_block_manager,
            g1_handles: state.g1_handles,
            g2_handles: state.g2_handles,
            g3_handles: state.g3_handles,
            slots: HashMap::new(),
            pipeline_runtime,
            slot_executor,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            known_slots: HashSet::new(),
            forward_seq: HashMap::new(),
            pending_deletes: Vec::new(),
            #[cfg(feature = "console")]
            action_collector: NoopActionCollector::shared(),
        }
    }

    /// Set an action collector for recording connector actions (console feature only).
    #[cfg(feature = "console")]
    pub fn with_action_collector(mut self, collector: ActionCollectorRef) -> Self {
        self.action_collector = collector;
        self
    }

    /// Emit a connector action to the collector (no-op when console disabled).
    #[cfg(feature = "console")]
    fn emit(&self, action: ConnectorAction) {
        self.action_collector.record_action(action);
    }

    #[cfg(not(feature = "console"))]
    #[allow(unused_variables)]
    fn emit(&self, action: impl std::fmt::Debug) {
        // No-op when console feature disabled
    }

    /// Get the block size from the configuration.
    pub fn block_size(&self) -> usize {
        self.config.block_size()
    }

    fn queue_forward_event(&mut self, request_id: &str) {
        if !self.known_slots.contains(request_id) {
            return;
        }
        let counter = self.forward_seq.entry(request_id.to_string()).or_insert(0);
        let event = format!("evt.forward.{}.{}", request_id, counter);
        self.metadata
            .queue_forward_event(request_id.to_string(), 0, event);
        *counter += 1;
    }

    // Slot management helper methods
    fn get_slot(&self, request_id: &str) -> Result<&Mutex<Slot>> {
        self.slots
            .get(request_id)
            .ok_or_else(|| anyhow::anyhow!("Slot not found for request_id: {}", request_id))
    }

    fn has_slot_registered(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    fn remove_slot(&mut self, request_id: &str) -> Result<()> {
        self.slots
            .remove(request_id)
            .ok_or_else(|| anyhow::anyhow!("Slot not found for removal: {}", request_id))?;
        Ok(())
    }

    /// Enqueue onboarding request to worker cohort (placeholder).
    ///
    /// This will eventually send a message via:
    /// `cohort_leader.bcast().am_send(consts::KVBM_ONBOARD, request)`
    ///
    /// For now, just constructs the message structure and logs.
    fn enqueue_onboarding_request(
        &mut self,
        request_id: &str,
        g2_blocks: BlocksView<G2>,
        g3_blocks: Option<BlocksView<G3>>,
        g1_blocks: BlocksView<G1>,
    ) -> Result<()> {
        // TODO: Create completion events (one per worker rank)
        let completion_events = vec![];

        tracing::debug!(
            request_id = request_id,
            num_g2_blocks = g2_blocks.len(),
            num_g3_blocks = g3_blocks.as_ref().map(|v| v.len()).unwrap_or(0),
            num_g1_blocks = g1_blocks.len(),
            "Enqueued onboarding request (placeholder - awaiting active message integration)"
        );

        // Build onboard request
        let _onboard_request = KvbmOnboardRequest::builder()
            .request_id(request_id.to_string())
            .g1_layout(self.g1_handles.clone())
            .g1_block_ids(g1_blocks)
            .g2_layout(self.g2_handles.clone())
            .g2_block_ids(g2_blocks)
            .g3_layout(self.g3_handles.clone())
            .g3_block_ids(g3_blocks)
            .completion_events(completion_events)
            .build()?;

        // TODO: Send via active message library
        // cohort_leader.bcast().am_send(consts::KVBM_ONBOARD, onboard_request)?;

        // Placeholder: log the request

        Ok(())
    }

    /// Handle finished offloading for a request.
    ///
    /// Called when workers report they've finished offloading (device->host/disk).
    /// This means a request that returned `Pending` from request_finished can now
    /// be safely cleaned up.
    pub fn handle_finished_offload(&mut self, request_id: &str) -> Result<()> {
        // Gracefully handle missing slot
        let Some(slot_mutex) = self.slots.get(request_id) else {
            tracing::debug!(request_id, "Offload finished for unknown slot");
            return Ok(());
        };

        let mut slot = slot_mutex.lock().unwrap();

        // Verify slot is in Finishing state
        if !matches!(slot.state(), SlotState::Finishing) {
            tracing::warn!(
                request_id,
                state = ?slot.state(),
                "Offload finished but slot not in Finishing state"
            );
        }

        // Mark as finished
        slot.set_state(SlotState::Finished);
        slot.mark_finished_sending();

        // Drop lock
        drop(slot);

        // Queue for deletion (will be processed in next build_connector_metadata)
        self.pending_deletes.push(request_id.to_string());

        tracing::debug!(
            request_id,
            "Offload finished, slot marked Finished and queued for deletion"
        );

        Ok(())
    }

    /// Handle finished onboarding for a request.
    ///
    /// Called when workers report they've finished onboarding (host/disk->device).
    /// Clears onboarding state so request can continue to prefill/decode.
    pub fn handle_finished_onboard(&mut self, request_id: &str) -> Result<()> {
        // Gracefully handle missing slot
        let Some(slot_mutex) = self.slots.get(request_id) else {
            tracing::debug!(request_id, "Onboard finished for unknown slot");
            return Ok(());
        };

        let mut slot = slot_mutex.lock().unwrap();

        // Verify onboarding was in progress
        if !slot.is_onboarding() {
            tracing::warn!(
                request_id,
                "Onboard finished but slot not marked as onboarding"
            );
        }

        // Clear onboarding state
        slot.mark_finished_receiving();

        // Drop lock
        drop(slot);

        // Remove from onboarding tracking
        self.onboarding_slots.remove(request_id);

        tracing::debug!(
            request_id,
            "Onboard finished, cleared onboarding state (slot continues to prefill/decode)"
        );

        // NOTE: Slot stays in HashMap - will continue normal execution

        Ok(())
    }
}

impl LeaderRuntime for ConnectorLeader {
    fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<MatchResult> {
        // 1. Validate num_computed_tokens is evenly divisible by block_size
        let block_size = self.block_size();
        if num_computed_tokens % block_size != 0 {
            return Err(anyhow::anyhow!(
                "num_computed_tokens {} not divisible by block_size {}",
                num_computed_tokens,
                block_size
            ));
        }

        // 2. Get and lock slot
        let slot_mutex = self.get_slot(request_id)?;
        let mut slot = slot_mutex.lock().unwrap();

        // 3. Mark search started
        slot.set_search_started();

        // 4. Calculate starting block index for G2 search
        let num_g1_blocks = num_computed_tokens / block_size;
        let sequence = slot.core().sequence();
        let total_blocks = sequence.blocks().len();

        if num_g1_blocks >= total_blocks {
            // All blocks already in G1
            return Ok(MatchResult::NoMatches);
        }

        // 5. Extract sequence hashes from block[num_g1_blocks..] for G2 search
        let search_hashes: Vec<SequenceHash> = sequence.blocks()[num_g1_blocks..]
            .iter()
            .map(|block| block.positional_sequence_hash())
            .collect();

        if search_hashes.is_empty() {
            return Ok(MatchResult::NoMatches);
        }

        // 6. Search G2 for matches
        let g2_matches = self.cpu_block_manager.match_blocks(&search_hashes);
        let num_g2_matched = g2_matches.len();

        // 7. Search G3 if G2 didn't match all and G3 exists
        let mut num_g3_matched = 0;
        if num_g2_matched < search_hashes.len() {
            if let Some(disk_manager) = &self.disk_block_manager {
                let g3_search_hashes = &search_hashes[num_g2_matched..];
                let g3_matches = disk_manager.match_blocks(g3_search_hashes);
                num_g3_matched = g3_matches.len();
                slot.store_g3_matches(g3_matches);
            }
        }

        // 8. Store matched blocks in slot
        slot.store_g2_matches(g2_matches);

        // 9. Calculate total matched tokens
        let total_matched_blocks = num_g2_matched + num_g3_matched;
        let total_matched_tokens = total_matched_blocks * block_size;

        // Emit action
        #[cfg(feature = "console")]
        if total_matched_tokens > 0 {
            self.emit(ConnectorAction::BlocksMatched {
                request_id: request_id.to_string(),
                num_g2: num_g2_matched,
                num_g3: num_g3_matched,
                total_tokens: total_matched_tokens,
            });
        }

        // 10. Return result
        if total_matched_tokens > 0 {
            Ok(MatchResult::Matched(
                NonZero::new(total_matched_tokens).unwrap(),
            ))
        } else {
            Ok(MatchResult::NoMatches)
        }
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: &str,
        block_ids: BlocksView<G1>,
        num_external_tokens: usize,
    ) -> Result<()> {
        // Scope the slot borrow to release it before mutating self
        let (should_enqueue, onboard_data) = {
            // CRITICAL: Acquire slot lock
            let slot_mutex = self
                .slots
                .get(request_id)
                .ok_or_else(|| anyhow::anyhow!("Slot not found for request_id: {}", request_id))?;
            let mut slot = slot_mutex.lock().unwrap();

            // Check if slot is finishing/finished - if yes, return immediately
            if matches!(slot.state(), SlotState::Finishing | SlotState::Finished) {
                // Slot is being torn down, don't start new operations
                tracing::warn!(
                    request_id,
                    "Slot is being torn down, not starting new operations"
                );
                return Ok(());
            }

            // Check if we have matched g2/g3 blocks
            let num_matched_blocks =
                slot.matched_g2_blocks().len() + slot.matched_g3_blocks().len();

            assert_eq!(num_matched_blocks * self.block_size(), num_external_tokens);

            if num_matched_blocks > 0 && num_external_tokens > 0 {
                // === FIRST CALL: Onboarding Allocation ===

                // // Validate block count matches matched blocks
                // if block_ids.len() != num_matched_blocks  {
                //     return Err(anyhow::anyhow!(
                //         "Block count mismatch for {}: expected {}, got {}",
                //         request_id,
                //         num_matched_blocks,
                //         block_ids.len()
                //     ));
                // }

                assert!(slot.device_blocks().is_empty());
                slot.device_blocks_mut().extend(block_ids.clone());

                // CRITICAL: Mark inflight operations BEFORE enqueuing message

                // Get views of the source and destination blocks
                let g1_blocks = block_ids;
                let g2_blocks = slot.matched_g2_blocks().as_blocks_view();
                let g3_blocks = slot.matched_g3_blocks().as_blocks_view();

                slot.onboarding_blocks(&g1_blocks, &g2_blocks, &g3_blocks);

                // Return data for onboarding (slot lock will be dropped at end of scope)
                (true, Some((g2_blocks, g3_blocks, g1_blocks)))
            } else if num_external_tokens == 0 {
                // === SECOND CALL: Remaining Prefill Allocation ===

                // Store remaining prefill blocks in device_blocks
                // slot.core_mut().push_device_blocks(&block_ids);

                (false, None)
            } else {
                // No matched blocks but has external tokens - unexpected state
                return Err(anyhow::anyhow!(
                    "Unexpected state for {}: num_external_tokens={} but no matched blocks",
                    request_id,
                    num_external_tokens
                ));
            }
        }; // slot lock dropped here

        // Now we can mutate self without borrow conflicts
        if should_enqueue {
            let (g2_blocks, g3_blocks, g1_blocks) = onboard_data.unwrap();

            // Track in leader onboarding set (requires &mut self)
            self.onboarding_slots.insert(request_id.to_string());

            // Enqueue onboarding message to workers (placeholder for now)
            self.enqueue_onboarding_request(
                request_id,
                g2_blocks,
                if g3_blocks.len() > 0 {
                    Some(g3_blocks)
                } else {
                    None
                },
                g1_blocks,
            )?;
        }

        Ok(())
    }

    fn request_finished(
        &mut self,
        request_id: &str,
        _block_ids: Blocks<G1>,
    ) -> Result<FinishedStatus> {
        // Gracefully handle missing slot (edge case: request cancelled before slot created)
        let Some(slot_mutex) = self.slots.get(request_id) else {
            tracing::debug!(
                request_id = request_id,
                "request_finished called for unknown slot (likely cancelled before creation)"
            );
            return Ok(FinishedStatus::Finished);
        };

        // CRITICAL: Lock slot and HOLD for entire call
        let mut slot = slot_mutex.lock().unwrap();

        // 1. Mark request as finishing (state transition)
        slot.set_state(SlotState::Finishing);

        // 2. Remove from inflight tracking (under lock)
        // Ryan: Not sure we want to clear this yet
        self.inflight_requests.remove(request_id);

        // 3. Check for outstanding operations
        let outstanding_ops = slot.outstanding_operations();

        if outstanding_ops.is_empty() {
            // Safe to finish immediately
            slot.set_state(SlotState::Finished);

            // Queue for deletion in next metadata build
            // NOTE: Slot remains in HashMap until build_connector_metadata() processes this
            // This ensures deletion metadata is emitted before slot is dropped
            self.pending_deletes.push(request_id.to_string());

            tracing::debug!(
                request_id = request_id,
                "Request finished, no outstanding operations, queued for deletion"
            );

            Ok(FinishedStatus::Finished)
        } else {
            // Have outstanding operations - must wait
            // Keep slot in HashMap for later cleanup (deferred)
            tracing::debug!(
                request_id = request_id,
                num_outstanding = outstanding_ops.len(),
                "Request finishing, waiting for outstanding operations (slot cleanup deferred)"
            );

            Ok(FinishedStatus::Pending)
        }

        // Lock held until end of function (or explicitly dropped)
    }

    fn update_connector_output(&mut self, connector_output: KVConnectorOutput) -> Result<()> {
        // Handle finished offloading (requests that returned Pending from request_finished)
        // finished_sending = worker finished sending to host/disk (offload complete)
        if let Some(finished_offloading) = connector_output.finished_sending {
            for request_id in finished_offloading {
                self.handle_finished_offload(&request_id)?;
            }
        }

        // Handle finished onboarding (requests that had external matches)
        // finished_recving = worker finished receiving from host/disk (onboard complete)
        if let Some(finished_onboarding) = connector_output.finished_recving {
            for request_id in finished_onboarding {
                self.handle_finished_onboard(&request_id)?;
            }
        }

        Ok(())
    }

    fn build_connector_metadata(&mut self, output: &SchedulerOutput) -> Result<Vec<u8>> {
        // Use current iteration (incremented at END of this method)
        let current_iteration = self.iteration_counter;

        // Clone inflight tracking to detect unscheduled requests
        let mut inflight_requests = self.inflight_requests.clone();

        tracing::debug!(
            iteration = current_iteration,
            num_new = output.new_requests().len(),
            num_cached = output.cached_requests().len(),
            num_onboarding = self.onboarding_slots.len(),
            "Building connector metadata"
        );

        // Emit action
        #[cfg(feature = "console")]
        self.emit(ConnectorAction::MetadataBuildStarted {
            iteration: current_iteration,
            num_new: output.new_requests().len(),
            num_cached: output.cached_requests().len(),
            num_onboarding: self.onboarding_slots.len(),
        });

        // 1. Process onboarding slots
        let onboarding_slots = std::mem::take(&mut self.onboarding_slots);
        for request_id in onboarding_slots.iter() {
            // Create slot on worker for onboarding
            let event = format!("evt.create.onboard.{}.{}", request_id, current_iteration);
            self.metadata.queue_slot_create(request_id.clone(), event);

            // Queue forward event
            self.queue_forward_event(request_id);

            // TODO (Phase 4): Drain pending operations via SlotExecutor
            // if let Some(actions) = slot.compute_actions() {
            //     self.slot_executor.execute(&mut slot, actions)?;
            // }

            // Remove from inflight tracking
            inflight_requests.remove(request_id);
        }

        // 2. Process new requests
        for new_req in output.new_requests() {
            let request_id = &new_req.request_id;

            // Create slot on worker
            let event = format!("evt.create.{}.{}", request_id, current_iteration);
            self.metadata.queue_slot_create(request_id.clone(), event);

            // Queue forward event
            self.queue_forward_event(request_id);

            // TODO (Phase 5): slot.record_start_iteration(current_iteration)
            // TODO (Phase 5): slot.apply_scheduler_output(&[], &[], new_req.num_computed_tokens, scheduled_tokens)
            // TODO (Phase 4): Drain pending operations via SlotExecutor

            // Remove from inflight tracking
            inflight_requests.remove(request_id);
        }

        // 3. Process cached requests (continuing requests from previous iterations)
        for cached_req in output.cached_requests() {
            let request_id = &cached_req.request_id;

            // Handle preemption recovery if needed
            if cached_req.resumed_from_preemption {
                tracing::info!(request_id, "Request resumed from preemption");
                // TODO (Phase 5): slot.reset_after_preemption()
            }

            // Queue forward event
            self.queue_forward_event(request_id);

            // TODO (Phase 5): slot.apply_scheduler_output(new_token_ids, new_block_ids, ...)
            // TODO (Phase 4): Drain pending operations via SlotExecutor

            // Remove from inflight tracking
            inflight_requests.remove(request_id);
        }

        // 4. Process unscheduled requests (skipped this iteration)
        for unscheduled_req in inflight_requests.iter() {
            tracing::debug!(
                request_id = unscheduled_req,
                iteration = current_iteration,
                "Request not scheduled this iteration (will be marked as skipped)"
            );
            // TODO (Phase 5): slot.mark_as_skipped()
        }

        // 5. Process pending deletions
        let pending_deletes: Vec<_> = self.pending_deletes.drain(..).collect();
        for request_id in pending_deletes {
            // Emit slot deletion in metadata
            self.metadata.queue_slot_delete(request_id.clone());

            // Remove slot from HashMap (RAII cleanup happens here)
            if let Some(_) = self.slots.remove(&request_id) {
                tracing::debug!(request_id, "Removed slot from HashMap");

                // Emit action
                #[cfg(feature = "console")]
                self.emit(ConnectorAction::SlotRemoved {
                    request_id: request_id.clone(),
                });
            }
        }

        // 6. Build metadata bytes
        let bytes = self.metadata.build_bytes()?;

        // 7. CRITICAL: Increment iteration counter at END
        self.iteration_counter += 1;

        tracing::debug!(
            prev_iteration = current_iteration,
            next_iteration = self.iteration_counter,
            "Completed metadata build, incremented iteration"
        );

        // Emit action
        #[cfg(feature = "console")]
        self.emit(ConnectorAction::MetadataBuildCompleted {
            iteration: current_iteration,
            bytes_len: bytes.len(),
        });

        Ok(bytes)
    }

    // ================================
    // KVBM Specific Methods
    // ================================

    fn engine_id(&self) -> &str {
        &self.engine_id
    }

    fn is_ready(&self) -> bool {
        unimplemented!("is_ready is not implemented");
    }

    fn wait_ready(&self) -> Result<()> {
        unimplemented!("wait_ready is not implemented");
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.known_slots.contains(request_id)
    }

    fn create_slot(&mut self, request: Request, all_token_ids: Vec<Vec<i64>>) -> Result<()> {
        let request_id = request.request_id.clone();

        // Flatten all token IDs into a single vector and convert i64 to i32
        let flat_tokens: Vec<i32> = all_token_ids
            .into_iter()
            .flatten()
            .map(|id| {
                if id < 0 {
                    Err(anyhow::anyhow!(
                        "Negative token ID {} found for request_id: {}",
                        id,
                        request_id
                    ))
                } else {
                    Ok(id as i32)
                }
            })
            .collect::<Result<Vec<_>>>()?;

        // Convert to Tokens and create TokenBlockSequence
        let tokens = Tokens::from(flat_tokens);
        let block_size = self.config.block_size();
        let sequence = TokenBlockSequence::new(tokens, block_size as u32, Some(request.salt_hash));

        // Create SlotCore
        let core = SlotCore::new(request_id.clone(), sequence, block_size);

        // Wrap in Slot
        let slot = Slot::new(core);

        // Register in slots HashMap
        self.slots.insert(request_id.clone(), Mutex::new(slot));

        // Track in inflight_requests
        self.inflight_requests.insert(request_id.clone());

        // Keep existing metadata queue
        let inserted = self.known_slots.insert(request_id.clone());
        if inserted {
            let event = format!("evt.create.{}", request_id);
            self.metadata.queue_slot_create(request_id.clone(), event);
        }

        // Emit action
        #[cfg(feature = "console")]
        self.emit(ConnectorAction::SlotCreated {
            request_id: request_id.clone(),
            num_tokens,
            iteration: self.iteration_counter,
        });

        Ok(())
    }
}
