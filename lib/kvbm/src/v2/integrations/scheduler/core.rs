// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core scheduler implementation.

use super::config::SchedulerConfig;
use super::connector_shim::SchedulerConnectorShim;
use super::kv_cache::KVCacheManager;
use super::policy::{FCFSPolicy, SchedulingPolicy};
use super::projection::{GlobalProjectionState, PlannedEvictionTracker};
use super::queues::{PausedRequests, RunningRequests, WaitingQueue};
use super::request::{OnboardingStatus, RequestStatus, SchedulerRequest};
use crate::v2::KvbmSequenceHashProvider;
use crate::v2::integrations::common::{
    BlockAssignmentOps, BlockAssignmentStorage, Request, SchedulerConnectorState, SchedulerOutput,
};
use crate::v2::integrations::connector::leader::{ConnectorLeader, FinishedStatus};

use derive_builder::Builder;
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Error type for SchedulerBuilder.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SchedulerBuilderError {
    #[error("Uninitialized field: {0}")]
    UninitializedField(&'static str),
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl From<derive_builder::UninitializedFieldError> for SchedulerBuilderError {
    fn from(e: derive_builder::UninitializedFieldError) -> Self {
        Self::UninitializedField(e.field_name())
    }
}

impl From<String> for SchedulerBuilderError {
    fn from(s: String) -> Self {
        Self::ValidationError(s)
    }
}

/// The main scheduler for G1 block management.
///
/// This scheduler manages the allocation of GPU (G1) blocks to requests,
/// handling scheduling decisions, preemption, and request lifecycle.
///
/// # Block Management
///
/// The scheduler uses `KVCacheManager` to allocate real RAII blocks from
/// `BlockManager<G1>`. Blocks are stored in each `SchedulerRequest`'s
/// `block_state` field, which manages pending (mutable) and registered
/// (immutable) blocks.
///
/// # Block Lifecycle
///
/// 1. **Scheduling**: `KVCacheManager::allocate()` returns `MutableBlock<G1>`
///    which are stored in `request.block_state.pending`.
///
/// 2. **Forward Pass**: After the model computes token data, blocks are
///    transitioned via `KVCacheManager::complete_and_register()`.
///
/// 3. **Cleanup**: When requests finish or are preempted, blocks are dropped
///    via RAII, returning them to the appropriate pools.
///
/// # Integration with Connector
///
/// When a [`ConnectorLeader`] is attached via the builder, the scheduler gains:
///
/// - **External KV cache lookup**: Query G2/G3/remote for cached tokens via
///   `get_num_new_matched_tokens()`, reducing prefill computation.
///
/// - **Async KV loading**: When external tokens require async transfer (inter-pass
///   mode), requests are moved to the `onboarding` collection until the G2→G1
///   transfer completes.
///
/// - **Intelligent eviction**: Check `can_evict()` before preemption to avoid
///   evicting requests with inflight RDMA transfers. Score candidates by G2
///   coverage via `get_eviction_score()`.
///
/// - **Delayed block freeing**: When requests finish with pending offloads,
///   blocks are held in `pending_block_free` until `finished_sending` signal
///   arrives, preventing data corruption during G1→G2 transfers.
///
/// The connector integration is completely optional - the scheduler works
/// independently without it.
///
/// # Construction
///
/// Use [`Scheduler::builder()`] to construct a scheduler with custom options:
///
/// ```ignore
/// let scheduler = Scheduler::builder()
///     .config(config)
///     .kv_cache(kv_cache)
///     .policy(Box::new(CustomPolicy::new()))
///     .connector(connector)
///     .build()?;
/// ```
///
/// For the common case with default policy and no connector, use [`Scheduler::new()`]:
///
/// ```ignore
/// let scheduler = Scheduler::new(config, kv_cache);
/// ```
#[derive(Builder)]
#[builder(
    pattern = "owned",
    build_fn(private, name = "build_inner", error = "SchedulerBuilderError")
)]
pub struct Scheduler {
    /// Scheduler configuration.
    config: SchedulerConfig,

    /// KV cache manager for block allocation.
    kv_cache: KVCacheManager,

    /// Currently running requests.
    #[builder(setter(skip), default = "RunningRequests::new()")]
    running: RunningRequests,

    /// Queue of requests waiting to be scheduled.
    #[builder(setter(skip), default = "WaitingQueue::new()")]
    waiting: WaitingQueue,

    /// Paused requests that hold blocks but are not scheduled.
    ///
    /// Used by the projection system for proactive pause/resume.
    #[builder(setter(skip), default = "PausedRequests::new()")]
    paused: PausedRequests,

    /// Scheduling policy for request prioritization.
    ///
    /// If not set, defaults to [`FCFSPolicy`] configured with `config.max_num_seqs`.
    #[builder(setter(strip_option), default)]
    policy: Option<Box<dyn SchedulingPolicy>>,

    /// Optional shared state with connector (completely optional).
    #[builder(setter(strip_option), default)]
    shared_state: Option<Arc<Mutex<dyn SchedulerConnectorState>>>,

    /// Optional connector shim for intelligent eviction and KV cache offloading.
    ///
    /// The shim wraps a [`ConnectorLeader`] and manages slot lifecycle automatically:
    /// - Creates slots on first `get_num_new_matched_tokens` call
    /// - Tracks inflight requests
    /// - Deletes slots on `request_finished`
    ///
    /// When present, the scheduler can:
    /// - Check for inflight offloads before preemption (`can_evict()`)
    /// - Score eviction candidates by G2 availability (`get_eviction_score()`)
    /// - Coordinate block freeing on request completion (`request_finished()`)
    ///
    /// This is set via the builder's `.connector()` method, which accepts an
    /// `Arc<ConnectorLeader>` and internally creates the shim.
    #[builder(setter(skip), default)]
    connector_shim: Option<SchedulerConnectorShim>,

    /// The underlying ConnectorLeader (used by builder to create shim).
    ///
    /// Note: Use `connector_shim` for all connector operations. This field is
    /// only used during builder construction.
    #[builder(setter(strip_option, name = "connector"), default)]
    connector_leader: Option<Arc<ConnectorLeader>>,

    /// Current iteration number.
    #[builder(setter(skip), default = "0")]
    iteration: usize,

    // =========================================================================
    // Projection System Fields
    // =========================================================================
    /// Global projection state with schedule-based block budgeting.
    ///
    /// Created when `config.enable_projection` is true.
    /// Maintains per-request block schedules and provides incremental updates
    /// instead of full recomputation each iteration.
    #[builder(setter(skip), default)]
    projector: Option<GlobalProjectionState>,

    /// Tracker for requests planned for eviction with priority G2 offload.
    ///
    /// Requests are added here when they're selected for eviction but need
    /// to wait for their blocks to be offloaded to G2 first.
    #[builder(setter(skip), default = "PlannedEvictionTracker::new()")]
    planned_evictions: PlannedEvictionTracker,

    // =========================================================================
    // Async KV Loading Fields
    // =========================================================================
    /// Requests actively onboarding (async KV load in progress).
    ///
    /// When `get_num_new_matched_tokens` returns `load_kv_async=true`, the request
    /// is moved here to wait for the async G2→G1 transfer to complete. When
    /// `finished_recving` signal arrives via `update_connector_signals()`, the
    /// request's `OnboardingStatus` is set to `Complete`. Then `schedule()` moves
    /// completed requests back to the waiting queue.
    #[builder(setter(skip), default = "HashMap::new()")]
    onboarding: HashMap<String, SchedulerRequest>,

    /// Requests whose blocks are held pending offload completion.
    ///
    /// When a request finishes and `request_finished()` returns `FinishedStatus::Pending`,
    /// the request is held here instead of immediately freeing blocks. When
    /// `finished_sending` signal arrives via `update_connector_signals()`, blocks
    /// are freed via RAII.
    #[builder(setter(skip), default = "HashMap::new()")]
    pending_block_free: HashMap<String, SchedulerRequest>,
}

impl SchedulerBuilder {
    /// Build the scheduler, applying default policy if not explicitly set.
    ///
    /// If no policy was specified via [`policy()`](Self::policy), this will
    /// create a default [`FCFSPolicy`] configured with `config.max_num_seqs`.
    ///
    /// If a connector was specified via [`connector()`](Self::connector), this
    /// will create a [`SchedulerConnectorShim`] that wraps it for automatic
    /// slot lifecycle management.
    pub fn build(self) -> Result<Scheduler, SchedulerBuilderError> {
        let mut scheduler = self.build_inner()?;

        // Apply default policy if none was provided
        if scheduler.policy.is_none() {
            scheduler.policy = Some(Box::new(FCFSPolicy::new(scheduler.config.max_num_seqs)));
        }

        // Create connector shim if connector was provided
        if let Some(leader) = scheduler.connector_leader.take() {
            scheduler.connector_shim = Some(SchedulerConnectorShim::new(leader));
        }

        // Initialize projector if projection is enabled
        if scheduler.config.enable_projection {
            let total_blocks = scheduler.kv_cache.total_blocks();
            scheduler.projector = Some(GlobalProjectionState::with_config(
                scheduler.config.block_size,
                scheduler.config.max_seq_len,
                total_blocks,
                scheduler.config.max_prefill_chunk_size,
                scheduler.config.min_guaranteed_blocks,
            ));
        }

        Ok(scheduler)
    }
}

impl Scheduler {
    /// Create a new scheduler with the given configuration and KV cache manager.
    ///
    /// This is a convenience constructor that uses the default FCFS policy and
    /// no connector or shared state. For more control, use [`Scheduler::builder()`].
    pub fn new(config: SchedulerConfig, kv_cache: KVCacheManager) -> Self {
        let policy =
            Some(Box::new(FCFSPolicy::new(config.max_num_seqs)) as Box<dyn SchedulingPolicy>);

        // Initialize projector if projection is enabled
        let projector = if config.enable_projection {
            let total_blocks = kv_cache.total_blocks();
            Some(GlobalProjectionState::with_config(
                config.block_size,
                config.max_seq_len,
                total_blocks,
                config.max_prefill_chunk_size,
                config.min_guaranteed_blocks,
            ))
        } else {
            None
        };

        Self {
            config,
            kv_cache,
            waiting: WaitingQueue::new(),
            running: RunningRequests::new(),
            policy,
            shared_state: None,
            connector_shim: None,
            connector_leader: None,
            iteration: 0,
            paused: PausedRequests::new(),
            projector,
            planned_evictions: PlannedEvictionTracker::new(),
            onboarding: HashMap::new(),
            pending_block_free: HashMap::new(),
        }
    }

    /// Create a new builder for constructing a Scheduler.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let scheduler = Scheduler::builder()
    ///     .config(config)
    ///     .kv_cache(kv_cache)
    ///     .policy(Box::new(CustomPolicy::new()))
    ///     .connector(connector)
    ///     .build()?;
    /// ```
    ///
    /// # Connector Integration
    ///
    /// When attaching a connector via `.connector()`, the scheduler gains:
    ///
    /// - **External KV cache lookup**: Query G2/G3/remote for cached tokens via
    ///   `get_num_new_matched_tokens()`, reducing prefill computation.
    ///
    /// - **Async KV loading**: When external tokens require async transfer (`load_kv_async=true`),
    ///   requests are moved to the `onboarding` collection and resume after transfer completes.
    ///
    /// - **Inflight transfer awareness**: Before preempting, check `can_evict()` to ensure
    ///   no active G1→G2 transfers are reading from the request's blocks.
    ///
    /// - **G2 availability scoring**: Query `get_eviction_score()` to prefer evicting
    ///   requests with more G2 coverage, minimizing prefill overhead on resume.
    ///
    /// - **Delayed block freeing**: When requests finish with pending offloads, blocks
    ///   are held in `pending_block_free` until the `finished_sending` signal arrives.
    ///
    /// # Connector API Integration
    ///
    /// The scheduler integrates with the connector at these points:
    ///
    /// | Scheduler Operation | Connector Call |
    /// |---------------------|----------------|
    /// | New request scheduling | `get_num_new_matched_tokens()` |
    /// | After block allocation | `update_state_after_alloc()` |
    /// | Request completion | `request_finished()` |
    /// | Preemption candidate selection | `can_evict()`, `get_eviction_score()` |
    /// | After forward pass | `update_connector_output()` (via `update_connector_signals()`) |
    pub fn builder() -> SchedulerBuilder {
        SchedulerBuilder::default()
    }

    /// Get a reference to the underlying ConnectorLeader, if attached.
    ///
    /// For most operations, prefer using the shim methods which handle
    /// slot lifecycle automatically.
    pub fn connector(&self) -> Option<&Arc<ConnectorLeader>> {
        self.connector_shim.as_ref().map(|s| s.leader())
    }

    /// Get a reference to the connector shim, if attached.
    ///
    /// The shim provides automatic slot lifecycle management:
    /// - Creates slots on first `get_num_new_matched_tokens` call
    /// - Deletes slots on `request_finished`
    pub fn connector_shim(&self) -> Option<&SchedulerConnectorShim> {
        self.connector_shim.as_ref()
    }

    /// Get the current iteration number.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    /// Get the number of running requests.
    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    /// Get the KV cache usage as a fraction.
    pub fn cache_usage(&self) -> f32 {
        self.kv_cache.usage()
    }

    /// Get the number of requests actively onboarding (async KV load in progress).
    pub fn num_onboarding(&self) -> usize {
        self.onboarding.len()
    }

    /// Get the number of requests with blocks held pending offload completion.
    pub fn num_pending_block_free(&self) -> usize {
        self.pending_block_free.len()
    }

    /// Get a reference to the global projection state.
    ///
    /// Returns `Some` if projection is enabled, `None` otherwise.
    /// Useful for testing and debugging projection behavior.
    pub fn projection_state(&self) -> Option<&GlobalProjectionState> {
        self.projector.as_ref()
    }

    /// Add a new request to the scheduler.
    ///
    /// The request's TokenBlockSequence is initialized with the prompt tokens
    /// and the scheduler's block size for computing block hashes.
    pub fn add_request(&mut self, request: Request) {
        let scheduler_request = SchedulerRequest::new(request, self.config.block_size);
        self.waiting.push_back(scheduler_request);
    }

    /// Abort a request by ID.
    ///
    /// Removes the request from whichever collection it's in (waiting, onboarding,
    /// running, or pending_block_free) and cleans up associated resources.
    ///
    /// # Request Locations
    ///
    /// The request may be in one of several locations:
    ///
    /// - **Waiting**: No blocks allocated, immediate cleanup.
    /// - **Onboarding**: Async KV load in progress, blocks allocated.
    /// - **Running**: Actively processing, blocks allocated.
    /// - **Pending Block Free**: Already finished, awaiting offload completion.
    ///
    /// # Connector Interaction
    ///
    /// When a connector is attached, this method coordinates block cleanup:
    ///
    /// 1. Calls `connector.request_finished()` to notify the connector and check
    ///    for inflight offload operations.
    ///
    /// 2. If the connector returns [`FinishedStatus::Pending`], blocks are held in
    ///    the `pending_block_free` collection until `finished_sending` signal arrives
    ///    via [`update_connector_signals()`](Self::update_connector_signals).
    ///
    /// 3. If the connector returns [`FinishedStatus::Finished`] or the request is
    ///    untracked, blocks are freed immediately via RAII.
    ///
    /// # Block Safety
    ///
    /// This method ensures blocks are not freed while RDMA transfers are reading
    /// from them, preventing data corruption during G1→G2 offload operations.
    ///
    /// # Note
    ///
    /// Requests in `pending_block_free` cannot be truly aborted - they've already
    /// finished and are just waiting for offload completion. The abort is logged
    /// but the offload continues to completion.
    pub fn abort_request(&mut self, request_id: &str) {
        // Try to remove from waiting queue.
        // Waiting requests have no blocks allocated, so no connector coordination needed.
        if let Some(mut request) = self.waiting.remove(request_id) {
            request.finish(RequestStatus::FinishedAborted);
            return;
        }

        // Try to remove from onboarding collection.
        // These requests have allocated blocks for async KV load.
        if let Some(mut request) = self.onboarding.remove(request_id) {
            if let Some(shim) = &self.connector_shim {
                let status = shim.request_finished(request_id);
                tracing::debug!(
                    request_id = %request_id,
                    ?status,
                    "Connector notified of aborted onboarding request"
                );
                if matches!(status, FinishedStatus::Pending) {
                    // Even for aborted requests, if offload is in progress, hold blocks
                    request.finish(RequestStatus::FinishedAborted);
                    self.pending_block_free
                        .insert(request_id.to_string(), request);
                    return;
                }
            }
            request.finish(RequestStatus::FinishedAborted);
            return;
        }

        // Try to remove from pending_block_free (request already finished, waiting for offload).
        // Nothing to do - just let it complete naturally. We can't abort the offload.
        if self.pending_block_free.contains_key(request_id) {
            tracing::debug!(
                request_id = %request_id,
                "Cannot abort request pending block free - offload in progress"
            );
            return;
        }

        // Try to remove from running.
        //
        // NOTE: We do NOT update projector here. Projection state is updated via
        // the normal scheduling cycle (finish_requests, update_from_output, etc.) which
        // handles block cleanup atomically with projection updates.
        if let Some(mut request) = self.running.remove(request_id) {
            // Notify connector of request finish before freeing blocks.
            if let Some(shim) = &self.connector_shim {
                let status = shim.request_finished(request_id);
                tracing::debug!(
                    request_id = %request_id,
                    ?status,
                    "Connector notified of aborted request"
                );
                if matches!(status, FinishedStatus::Pending) {
                    // Offload in progress - hold blocks until complete
                    request.finish(RequestStatus::FinishedAborted);
                    self.pending_block_free
                        .insert(request_id.to_string(), request);
                    return;
                }
            }
            request.finish(RequestStatus::FinishedAborted);
        }
    }

    /// Finish requests by ID with the given status.
    ///
    /// Marks requests as finished and coordinates block cleanup with the connector.
    /// This is the normal completion path for requests that have finished generation.
    ///
    /// # Connector Interaction
    ///
    /// When a connector is attached, this method coordinates block cleanup:
    ///
    /// 1. Removes the request from the projection state (if enabled).
    ///
    /// 2. Calls `connector.request_finished()` to notify the connector and check
    ///    for inflight offload operations (G1→G2 transfers).
    ///
    /// 3. Based on the connector's response:
    ///    - [`FinishedStatus::Pending`]: Blocks are held in `pending_block_free`
    ///      until `finished_sending` signal arrives via
    ///      [`update_connector_signals()`](Self::update_connector_signals).
    ///    - [`FinishedStatus::Finished`]: Blocks are freed immediately.
    ///    - [`FinishedStatus::UntrackedRequest`]: Blocks are freed immediately
    ///      (request wasn't tracked by connector).
    ///
    /// # Block Safety
    ///
    /// This method ensures blocks are not freed while RDMA transfers are reading
    /// from them, preventing data corruption during G1→G2 offload operations.
    ///
    /// # Without Connector
    ///
    /// When no connector is attached, blocks are freed immediately via RAII when
    /// the request is dropped.
    pub fn finish_requests(&mut self, request_ids: &[String], status: RequestStatus) {
        for request_id in request_ids {
            if let Some(mut request) = self.running.remove(request_id) {
                // Remove from global projection state if enabled
                if let Some(proj) = &mut self.projector {
                    proj.remove_request(request_id);
                }

                // Notify connector of request finish before freeing blocks.
                // If connector has inflight offloads, hold blocks until finished_sending.
                if let Some(shim) = &self.connector_shim {
                    let finished_status = shim.request_finished(request_id);
                    match finished_status {
                        FinishedStatus::Pending => {
                            // Connector has inflight offloads - blocks must be held.
                            // Move to pending_block_free collection until finished_sending arrives.
                            tracing::debug!(
                                request_id = %request_id,
                                blocks = request.block_state.num_assigned(),
                                "Request has pending offloads, holding blocks until offload completes"
                            );
                            request.finish(status);
                            self.pending_block_free
                                .insert(request_id.to_string(), request);
                            continue; // Skip normal drop - blocks held
                        }
                        FinishedStatus::Finished | FinishedStatus::UntrackedRequest => {
                            // Safe to free blocks immediately
                            tracing::debug!(
                                request_id = %request_id,
                                ?finished_status,
                                "Connector cleared request for block release"
                            );
                        }
                    }
                }

                request.finish(status);
                // Request is dropped here, RAII frees blocks
            }
        }
    }

    /// Run the scheduler to produce a scheduling decision.
    ///
    /// This is the main scheduling loop that orchestrates request scheduling,
    /// block allocation, and connector coordination.
    ///
    /// # Scheduling Phases
    ///
    /// The scheduler runs through several phases each iteration:
    ///
    /// 1. **Projection Update**: Analyze future block requirements and detect choke points.
    ///
    /// 2. **Onboarding Completion**: Move requests that finished async KV loading (G2→G1)
    ///    back to the waiting queue. These get priority since they've allocated blocks.
    ///
    /// 3. **Proactive Eviction**: Pause requests predicted to cause memory pressure.
    ///
    /// 4. **Running Allocation**: Allocate blocks for decode phase (1 token/step).
    ///
    /// 5. **Resume Paused**: Resume paused requests if headroom exists.
    ///
    /// 6. **Schedule Waiting**: Move new requests from waiting to running (prefill phase).
    ///
    /// # Connector Integration
    ///
    /// When a connector is attached, this method integrates several connector operations:
    ///
    /// - **External KV lookup**: During waiting request scheduling, queries the connector
    ///   for cached tokens in G2/G3/remote via `get_num_new_matched_tokens()`.
    ///
    /// - **Async KV loading**: If external tokens require async transfer (`load_kv_async=true`),
    ///   the request is moved to the `onboarding` collection instead of running.
    ///
    /// - **Block allocation notification**: After allocating blocks, notifies the connector
    ///   via `update_state_after_alloc()`.
    ///
    /// - **Eviction safety**: Before preemption, checks `can_evict()` to avoid evicting
    ///   requests with inflight RDMA transfers.
    ///
    /// # Block State After Scheduling
    ///
    /// After `schedule()` returns, allocated blocks are in `pending` state. They
    /// transition to `registered` state after the forward pass completes and
    /// `complete_and_register()` is called with token data.
    ///
    /// # Post-Scheduling
    ///
    /// After the forward pass, call:
    /// - [`update_from_output()`](Self::update_from_output) - Register blocks and process tokens
    /// - [`update_connector_signals()`](Self::update_connector_signals) - Process async transfer signals
    ///
    /// See `STATE_TRANSITIONS.md` for the complete scheduling flow.
    pub fn schedule(&mut self) -> SchedulerOutput {
        self.iteration += 1;
        let mut output = SchedulerOutput::new(self.iteration);
        let mut num_scheduled_tokens: HashMap<String, usize> = HashMap::new();

        // Phase 0: Update projections (if enabled)
        // This analyzes future block requirements and detects choke points.
        // todo: we should really update projections in two places
        // here, but also at the end of the process model output.
        // currently we are recomputing teh projections from scratch; however, that is wasteful
        // we should all of our future chock points and worse-case free events once computed
        // are valid until there is a free event.
        // thus, using applying an invaidate or an update when we know a request is finished
        // during the processing of the model output is valuable.
        // similarly, there are per-request projections that are valid for that request until is
        // is either finished, paused or evicted.
        // therefore, we should preserve that state when possible and only recompute the necessary
        // bits when doing basic updates.
        self.update_projections();

        // Phase 0.3: Process completed onboarding requests
        // Requests that finished async KV loading (G2→G1) are moved back to waiting.
        // They get scheduling priority since they've already allocated blocks.
        self.process_completed_onboarding();

        // Phase 0.5: Proactive pause/eviction based on choke point predictions
        // This pauses requests that are eligible for eviction before we run out
        // of blocks, enabling smoother scheduling without emergency preemption.
        self.process_proactive_evictions();

        // Phase 1: Allocate blocks for running requests (decode phase)
        // Running requests continue from their current state, needing blocks
        // only for newly generated tokens (typically 1 token per decode step).
        self.allocate_for_running(&mut output, &mut num_scheduled_tokens);

        // Phase 2: Resume paused requests first (if projection allows)
        // Paused requests already made progress and hold blocks; resuming them
        // is more efficient than starting new requests. Skip if resuming would
        // create or worsen a choke point.
        if self.should_evaluate_paused() {
            self.try_resume_paused(&mut output, &mut num_scheduled_tokens);
        }

        // Phase 3: Schedule new requests from waiting queue (prefill phase)
        // Only schedule if backfilling is possible (no active multi-chunk prefill
        // or current prefill is on final chunk).
        if self.should_evaluate_waiting() {
            self.schedule_waiting(&mut output, &mut num_scheduled_tokens);
        }

        // Update totals
        output.set_num_scheduled_tokens(num_scheduled_tokens);

        // Build connector metadata for workers
        // After scheduling is complete, build metadata that workers need for
        // KV cache operations during the forward pass.
        if let Some(shim) = &self.connector_shim {
            match shim.build_connector_meta(output.clone()) {
                Ok(meta) => {
                    output.kv_connector_metadata = Some(meta);
                }
                Err(e) => {
                    tracing::error!(
                        iteration = self.iteration,
                        error = %e,
                        "Failed to build connector metadata"
                    );
                }
            }
        }

        // Validate block allocations match projections (debug/development check)
        self.validate_allocation_vs_projection();

        output
    }

    /// Validate that actual block allocations match projections.
    ///
    /// This is a debugging/validation check that compares:
    /// - Actual blocks held by each scheduled request
    /// - Projected blocks for that request at the current iteration
    ///
    /// Emits `tracing::warn!` if they differ, which helps identify:
    /// - Bugs in the projection system
    /// - Edge cases where projection model doesn't match reality
    fn validate_allocation_vs_projection(&self) {
        let Some(projector) = &self.projector else {
            return;
        };

        let mut total_projected = 0usize;
        let mut total_actual = 0usize;
        let mut mismatches = Vec::new();

        // Check running requests
        for (request_id, request) in self.running.iter() {
            let actual_blocks = request.block_state.total_blocks();
            total_actual += actual_blocks;

            if let Some(schedule) = projector.get_schedule(request_id) {
                let projected_blocks = schedule.blocks_at_iteration(self.iteration);
                total_projected += projected_blocks;

                if actual_blocks != projected_blocks {
                    mismatches.push((
                        request_id.clone(),
                        actual_blocks,
                        projected_blocks,
                        "running",
                    ));
                }
            }
        }

        // Check paused requests (they shouldn't grow, but verify)
        for (_request_id, request) in self.paused.iter() {
            let actual_blocks = request.block_state.total_blocks();
            total_actual += actual_blocks;
            // Paused requests are removed from projection, so we just count actuals
        }

        // Emit warnings for mismatches with detailed schedule info
        if !mismatches.is_empty() {
            for (request_id, actual, projected, state) in &mismatches {
                // Get schedule details for debugging
                let schedule_info = projector.get_schedule(request_id).map(|s| {
                    format!(
                        "base={}, starting={}, offset={}, events={:?}",
                        s.base_iteration,
                        s.starting_blocks,
                        self.iteration.saturating_sub(s.base_iteration),
                        s.block_events
                            .iter()
                            .take(5) // Limit to first 5 events
                            .map(|e| (e.iteration_offset, e.delta))
                            .collect::<Vec<_>>()
                    )
                });

                // Get request details
                let request_info = self.running.get(request_id).map(|r| {
                    format!(
                        "computed={}, pending={}, registered={}",
                        r.num_computed_tokens,
                        r.block_state.num_pending(),
                        r.block_state.num_registered()
                    )
                });

                tracing::warn!(
                    iteration = self.iteration,
                    request_id = %request_id,
                    actual_blocks = actual,
                    projected_blocks = projected,
                    state = state,
                    schedule = ?schedule_info,
                    request = ?request_info,
                    "Block allocation mismatch: actual != projected"
                );
            }
        }

        // Also log aggregate mismatch if significant
        // Use total_projected (sum of blocks_at_iteration) for consistency with individual checks
        if total_actual != total_projected && !mismatches.is_empty() {
            tracing::warn!(
                iteration = self.iteration,
                total_actual_blocks = total_actual,
                projected_demand = total_projected,
                mismatch_count = mismatches.len(),
                "Aggregate block allocation mismatch"
            );
        }
    }

    /// Allocate blocks for running requests (decode phase).
    fn allocate_for_running(
        &mut self,
        output: &mut SchedulerOutput,
        num_scheduled_tokens: &mut HashMap<String, usize>,
    ) {
        // Collect request IDs first to avoid borrow issues
        let request_ids: Vec<String> = self.running.request_ids().cloned().collect();

        for request_id in request_ids {
            // First, get the info we need without holding the mutable borrow
            let (blocks_needed, tokens_to_compute, resumed, all_tokens, computed, output_count) = {
                let request = match self.running.get(&request_id) {
                    Some(r) => r,
                    None => continue,
                };
                (
                    request.num_new_blocks_needed(self.config.block_size),
                    request.tokens_to_compute(),
                    request.resumed_from_preemption,
                    if request.resumed_from_preemption {
                        // Get ALL tokens (prompt + output) for resumed requests
                        // so workers can resync their state after preemption
                        Some(request.all_tokens_for_resume())
                    } else {
                        None
                    },
                    request.num_computed_tokens,
                    request.num_output_tokens,
                )
            };

            if blocks_needed > 0 {
                // Try to allocate new blocks from the KV cache manager
                if let Some(new_blocks) = self.kv_cache.allocate(blocks_needed) {
                    // Extract block IDs before moving blocks into request
                    let new_block_ids: Vec<_> = new_blocks.iter().map(|b| b.block_id()).collect();

                    // Now get mutable access to update the request
                    if let Some(request) = self.running.get_mut(&request_id) {
                        request.add_pending_blocks(new_blocks);
                        request.clear_resumed_flag();
                    }

                    // Record in output
                    num_scheduled_tokens.insert(request_id.clone(), tokens_to_compute);

                    output.add_cached_request(
                        request_id.clone(),
                        resumed,
                        vec![], // new_token_ids - populated by model output
                        all_tokens,
                        new_block_ids,
                        computed,
                        output_count,
                    );
                }
                // else: Need to preempt - handled in preemption phase
            } else {
                // No new blocks needed, just decode one token
                let tokens_to_schedule = 1; // Single decode token
                num_scheduled_tokens.insert(request_id.clone(), tokens_to_schedule);

                // Clear resumed flag
                if let Some(request) = self.running.get_mut(&request_id) {
                    request.clear_resumed_flag();
                }

                output.add_cached_request(
                    request_id.clone(),
                    resumed,
                    vec![],
                    None,
                    vec![],
                    computed,
                    output_count,
                );
            }
        }
    }

    /// Schedule new requests from the waiting queue.
    ///
    /// # Prefix Caching (G1)
    ///
    /// When prefix caching is enabled, this method searches for cached blocks
    /// in G1 before allocating new blocks. The flow mirrors vLLM's scheduling:
    ///
    /// 1. Get locally-cached tokens via `kv_cache.get_computed_blocks()`
    /// 2. (TODO) Get externally-cached tokens via `connector.get_num_new_matched_tokens()`
    /// 3. Calculate tokens to schedule = total - computed
    /// 4. Allocate only the new blocks needed (not cached portion)
    /// 5. (TODO) Call `connector.update_state_after_alloc()` to start loading
    ///
    /// # Connector Integration (TODO)
    ///
    /// The connector APIs are stubbed out with detailed comments showing where
    /// they will be called when full integration is implemented.
    fn schedule_waiting(
        &mut self,
        output: &mut SchedulerOutput,
        num_scheduled_tokens: &mut HashMap<String, usize>,
    ) {
        let mut total_scheduled = output.total_num_scheduled_tokens;

        while !self.waiting.is_empty() {
            // Check budget limits
            if total_scheduled >= self.config.max_num_batched_tokens {
                break;
            }
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            // Check backfill eligibility: only allow new prefills if:
            // - No active chunked prefill, OR
            // - Active prefill is on final chunk
            if self.config.enable_chunked_prefill && !self.can_backfill_prefill() {
                // There's an active multi-chunk prefill that hasn't reached final pass
                // Don't start new requests - complete the current prefill first
                break;
            }

            // Calculate available blocks for policy
            let available_blocks = self.kv_cache.free_blocks();

            // Collect waiting requests as references for policy
            let waiting_refs: Vec<&SchedulerRequest> = self.waiting.iter().collect();

            // Ask policy which request to schedule next
            // SAFETY: policy is always initialized by new() or build()
            let next_idx = self
                .policy
                .as_ref()
                .expect("policy always initialized")
                .select_next(
                    &waiting_refs,
                    self.running.len(),
                    available_blocks,
                    self.config.block_size,
                );

            let Some(idx) = next_idx else {
                // Policy says don't schedule anything
                break;
            };

            // Remove the selected request from waiting queue
            // Note: We need to handle index correctly since we're using a VecDeque
            let mut request = match self.waiting.pop_front() {
                Some(r) => r,
                None => break,
            };

            // If not the first one, we need to re-add and pop the right one
            // For simplicity, FCFS always returns 0, so this works
            if idx != 0 {
                // Put it back and skip for now (complex case)
                self.waiting.push_front(request);
                break;
            }

            // =========================================================================
            // PHASE 1: Prefix Cache Lookup (G1 Local + External via Connector)
            // =========================================================================
            //
            // Get already-cached tokens to avoid redundant computation.
            // This mirrors vLLM's scheduler.py lines 447-480.

            // Get locally-cached tokens from G1 prefix cache.
            //
            // Note on prefix caching optionality: get_computed_blocks() returns (vec![], 0)
            // when prefix caching is disabled, so no explicit check is needed here.
            //
            // Note on prefix match validity: The prefix match is only valid at evaluation
            // time. Matched blocks may be evicted or freed between scheduling iterations.
            // We re-evaluate prefix matches each time a request is scheduled from waiting.
            let (matched_blocks, num_local_computed_tokens) = if request.num_computed_tokens == 0 {
                // First time scheduling - check prefix cache
                let seq_hashes = request.get_sequence_hashes();
                tracing::debug!(
                    request_id = %request.request_id(),
                    num_hashes = seq_hashes.len(),
                    seq_hashes = ?seq_hashes,
                    "Looking up prefix cache with sequence hashes"
                );
                let result = self.kv_cache.get_computed_blocks(&seq_hashes);
                tracing::debug!(
                    request_id = %request.request_id(),
                    num_matched = result.0.len(),
                    computed_tokens = result.1,
                    "Prefix cache lookup result"
                );
                result
            } else {
                // This should be unreachable for requests from the waiting queue.
                // Requests in waiting queue have not been scheduled yet, so they
                // should always have num_computed_tokens == 0.
                //
                // If this is reached, it indicates a bug in state management:
                // - A request was added to waiting with computed tokens already set
                // - Or a preempted request wasn't properly reset
                tracing::error!(
                    request_id = %request.request_id(),
                    num_computed_tokens = request.num_computed_tokens,
                    "Request in waiting queue has non-zero computed tokens - this is a bug"
                );
                debug_assert!(
                    false,
                    "Request in waiting queue should have num_computed_tokens == 0"
                );
                // In release builds, treat as if no prefix cache hit
                (vec![], 0)
            };

            // Update request's cached token count for metrics
            if !request.has_checked_prefix_cache() {
                request.set_num_cached_tokens(num_local_computed_tokens);
            }

            // -------------------------------------------------------------------------
            // KV Connector - Get externally-cached tokens (G2/G3/remote)
            // -------------------------------------------------------------------------
            // Query the connector for external KV cache hits.
            // The connector checks G2 (host memory), G3 (remote storage), and
            // potentially other nodes for matching blocks.
            //
            // vLLM reference: scheduler.py lines 454-469
            //
            // get_num_new_matched_tokens returns:
            // - (None, false) = search still in progress, skip this request
            // - (Some(0), false) = no external matches found
            // - (Some(n), true) = n tokens available, need async load (inter-pass)
            // - (Some(n), false) = n tokens available, sync load (intra-pass)
            let (mut num_external_computed_tokens, mut load_kv_async) = (0usize, false);
            if let Some(shim) = &self.connector_shim {
                match shim.get_num_new_matched_tokens(&request, num_local_computed_tokens) {
                    Ok((None, _)) => {
                        // Connector still searching - skip this request for now
                        self.waiting.push_front(request);
                        continue;
                    }
                    Ok((Some(ext_tokens), async_load)) => {
                        num_external_computed_tokens = ext_tokens;
                        load_kv_async = async_load;
                        tracing::debug!(
                            request_id = %request.request_id(),
                            ext_tokens,
                            async_load,
                            "Got external cached tokens from connector"
                        );
                    }
                    Err(e) => {
                        tracing::warn!(
                            request_id = %request.request_id(),
                            error = %e,
                            "Connector get_num_new_matched_tokens failed, proceeding without external cache"
                        );
                    }
                }
            }
            // -------------------------------------------------------------------------

            // Total computed tokens = local G1 cache + external (G2/G3/remote)
            let num_computed_tokens = num_local_computed_tokens + num_external_computed_tokens;

            // -------------------------------------------------------------------------
            // KV Connector - Handle async KV loading (inter-pass mode)
            // -------------------------------------------------------------------------
            // If the connector indicates async loading is needed, move the request to
            // the onboarding collection. The blocks will be allocated and the connector
            // will start the async G2→G1 transfer. When finished_recving signal arrives,
            // the request will be moved back to waiting for scheduling.
            //
            // vLLM reference: scheduler.py lines 582-587
            if load_kv_async && num_external_computed_tokens > 0 {
                // Calculate total blocks needed for external tokens
                let blocks_for_external = (num_external_computed_tokens + self.config.block_size
                    - 1)
                    / self.config.block_size;

                // Allocate blocks for the async KV load
                let allocated_for_async = if blocks_for_external > matched_blocks.len() {
                    let new_blocks_needed = blocks_for_external - matched_blocks.len();
                    match self.kv_cache.allocate(new_blocks_needed) {
                        Some(blocks) => blocks,
                        None => {
                            // Allocation failed - put request back and try later
                            tracing::debug!(
                                request_id = %request.request_id(),
                                blocks_needed = new_blocks_needed,
                                free_blocks = self.kv_cache.free_blocks(),
                                "Insufficient blocks for async KV load, will retry"
                            );
                            drop(matched_blocks);
                            self.waiting.push_front(request);
                            break;
                        }
                    }
                } else {
                    Vec::new()
                };

                // Collect all block IDs for connector
                let matched_block_ids: Vec<_> =
                    matched_blocks.iter().map(|b| b.block_id()).collect();
                let new_block_ids: Vec<_> =
                    allocated_for_async.iter().map(|b| b.block_id()).collect();
                let all_block_ids: Vec<_> = matched_block_ids
                    .iter()
                    .chain(new_block_ids.iter())
                    .copied()
                    .collect();

                // Add matched G1 blocks as registered (they have token data)
                request.add_registered_blocks(matched_blocks);
                // Add newly allocated blocks as pending (waiting for async load)
                request.add_pending_blocks(allocated_for_async);

                // Update computed tokens to reflect what will be loaded
                request.num_computed_tokens = num_computed_tokens;
                // Set onboarding status to Loading
                request.set_onboarding_status(OnboardingStatus::Loading);

                // Notify connector of block allocation
                if let Some(shim) = &self.connector_shim {
                    if let Err(e) = shim.update_state_after_alloc(
                        request.request_id(),
                        all_block_ids,
                        num_external_computed_tokens,
                    ) {
                        tracing::error!(
                            request_id = %request.request_id(),
                            error = %e,
                            "Failed to update connector state after alloc for async load"
                        );
                    }
                }

                tracing::info!(
                    request_id = %request.request_id(),
                    external_tokens = num_external_computed_tokens,
                    blocks = blocks_for_external,
                    "Moving request to onboarding for async KV load"
                );

                // Move to onboarding collection
                self.onboarding
                    .insert(request.request_id().to_string(), request);
                continue;
            }
            // -------------------------------------------------------------------------

            // =========================================================================
            // PHASE 2: Calculate Tokens and Blocks to Schedule
            // =========================================================================

            // Tokens to schedule = total request tokens - already computed tokens
            let total_request_tokens = request.total_known_tokens();
            let tokens_remaining = total_request_tokens.saturating_sub(num_computed_tokens);

            // Apply chunked prefill limits if enabled
            let tokens_to_schedule =
                self.calculate_prefill_tokens_with_computed(tokens_remaining, total_scheduled);

            if tokens_to_schedule == 0 {
                // Can't fit any tokens - drop matched blocks and put request back
                drop(matched_blocks);
                // todo: do we need to reset the state of the request here?
                self.waiting.push_front(request);
                break;
            }

            // Calculate how many new blocks we need (beyond cached blocks)
            let total_blocks_needed =
                (num_computed_tokens + tokens_to_schedule + self.config.block_size - 1)
                    / self.config.block_size;
            let cached_blocks = matched_blocks.len();
            let new_blocks_needed = total_blocks_needed.saturating_sub(cached_blocks);

            // =========================================================================
            // PHASE 3: Allocate New Blocks
            // =========================================================================

            let allocated_blocks = if new_blocks_needed > 0 {
                match self.kv_cache.allocate(new_blocks_needed) {
                    Some(blocks) => blocks,
                    None => {
                        if !request.resumed_from_preemption {
                            tracing::error!(
                                request_id = %request.request_id(),
                                new_blocks_needed,
                                free_blocks = self.kv_cache.free_blocks(),
                                "Insufficient blocks for new request; skipping preemption"
                            );
                            drop(matched_blocks);
                            self.waiting.push_front(request);
                            break;
                        }

                        if !self.try_preempt(new_blocks_needed) {
                            // Can't preempt enough - drop matched blocks and put request back
                            drop(matched_blocks);
                            self.waiting.push_front(request);
                            break;
                        }
                        // Try allocation again after preemption
                        match self.kv_cache.allocate(new_blocks_needed) {
                            Some(blocks) => blocks,
                            None => {
                                // Still not enough - drop matched blocks and put request back
                                drop(matched_blocks);
                                self.waiting.push_front(request);
                                break;
                            }
                        }
                    }
                }
            } else {
                Vec::new()
            };

            // =========================================================================
            // PHASE 4: Update Request State and Record Output
            // =========================================================================

            // Collect all block IDs for output (matched + newly allocated)
            let matched_block_ids: Vec<_> = matched_blocks.iter().map(|b| b.block_id()).collect();
            let new_block_ids: Vec<_> = allocated_blocks.iter().map(|b| b.block_id()).collect();
            let all_block_ids: Vec<_> = matched_block_ids
                .iter()
                .chain(new_block_ids.iter())
                .copied()
                .collect();

            // Add matched blocks as registered (they already have token data)
            request.add_registered_blocks(matched_blocks);
            // Add newly allocated blocks as pending (waiting for forward pass)
            request.add_pending_blocks(allocated_blocks);

            // Update computed tokens to reflect cached portion
            request.num_computed_tokens = num_computed_tokens;

            // Start running
            request.start_running();

            // -------------------------------------------------------------------------
            // KV Connector - Notify of allocation for external tokens
            // -------------------------------------------------------------------------
            // After successful allocation, notify the connector so it can:
            // - Start loading external blocks (inter-pass mode)
            // - Prepare sync transfer metadata (intra-pass mode)
            //
            // vLLM reference: scheduler.py lines 569-577
            if let Some(shim) = &self.connector_shim {
                if num_external_computed_tokens > 0 {
                    if let Err(e) = shim.update_state_after_alloc(
                        request.request_id(),
                        all_block_ids.clone(),
                        num_external_computed_tokens,
                    ) {
                        tracing::error!(
                            request_id = %request.request_id(),
                            error = %e,
                            "Failed to update connector state after allocation"
                        );
                    }
                }
            }
            // -------------------------------------------------------------------------

            // Record in output
            output.add_new_request(
                request.request_id().to_string(),
                request.request.tokens.to_vec(),
                all_block_ids,
                request.num_computed_tokens,
            );

            num_scheduled_tokens.insert(request.request_id().to_string(), tokens_to_schedule);
            total_scheduled += tokens_to_schedule;

            tracing::debug!(
                request_id = %request.request_id(),
                num_local_cached = num_local_computed_tokens,
                num_external_cached = num_external_computed_tokens,
                tokens_to_schedule,
                cached_blocks,
                new_blocks = new_blocks_needed,
                "Scheduled new request"
            );

            // Add to global projection state if enabled
            // Use resumed_from_preemption to detect restored requests (full block guarantee)
            if let Some(proj) = &mut self.projector {
                proj.add_request(&request, request.resumed_from_preemption);
            }

            // Move to running
            self.running.insert(request);
        }
    }

    /// Calculate prefill tokens accounting for already-computed tokens.
    ///
    /// This is similar to `calculate_prefill_tokens` but works with the
    /// remaining tokens (after subtracting cached tokens).
    fn calculate_prefill_tokens_with_computed(
        &self,
        tokens_remaining: usize,
        current_total: usize,
    ) -> usize {
        let remaining_budget = self
            .config
            .max_num_batched_tokens
            .saturating_sub(current_total);

        if self.config.enable_chunked_prefill {
            let max_chunk = self
                .config
                .max_prefill_chunk_size
                .unwrap_or(self.config.max_num_batched_tokens);
            tokens_remaining.min(remaining_budget).min(max_chunk)
        } else {
            // Without chunked prefill, we need to fit the whole prefill
            if tokens_remaining <= remaining_budget {
                tokens_remaining
            } else {
                0 // Can't fit, don't schedule
            }
        }
    }

    /// Calculate how many tokens to prefill for a request.
    #[allow(dead_code)]
    fn calculate_prefill_tokens(&self, request: &SchedulerRequest, current_total: usize) -> usize {
        let remaining_budget = self
            .config
            .max_num_batched_tokens
            .saturating_sub(current_total);
        let tokens_to_compute = request.tokens_to_compute();

        if self.config.enable_chunked_prefill {
            let max_chunk = self
                .config
                .max_prefill_chunk_size
                .unwrap_or(self.config.max_num_batched_tokens);
            tokens_to_compute.min(remaining_budget).min(max_chunk)
        } else {
            // Without chunked prefill, we need to fit the whole prefill
            if tokens_to_compute <= remaining_budget {
                tokens_to_compute
            } else {
                0 // Can't fit, don't schedule
            }
        }
    }

    /// Try to preempt running requests to free up blocks.
    ///
    /// This preempts the lowest priority running request(s) to free up blocks.
    /// When a request is preempted, its RAII blocks are dropped, returning
    /// them to the appropriate pools.
    ///
    /// # Eviction Criteria (with Connector)
    ///
    /// When a connector is attached, the scheduler uses intelligent victim selection:
    ///
    /// 1. **Inflight offload check**: Requests with active G1→G2 transfers are
    ///    excluded via `connector.can_evict()`. Evicting these would corrupt transfers.
    ///
    /// 2. **G2 coverage scoring**: Among safe candidates, prefer requests with higher
    ///    G2 block coverage via `connector.get_eviction_score()`. These can resume
    ///    faster with less prefill.
    ///
    /// 3. **Block boundary alignment** (future): Prefer requests at block boundaries
    ///    via `connector.get_block_boundary_info()`. Continuing to a boundary costs
    ///    zero extra resources.
    ///
    /// # Block Deallocation Pattern
    ///
    /// Preemption follows a different pattern than request completion:
    ///
    /// 1. **Blocks are freed immediately** via RAII when `victim.preempt()` clears
    ///    the block_state
    /// 2. **The connector is NOT notified** of the preemption (by design)
    /// 3. **num_computed_tokens is reset to 0** - the request will recompute from scratch
    ///
    /// # Connector Interaction: SAFE (by design)
    ///
    /// Preemption is safe without connector notification because:
    ///
    /// 1. **Async loads are protected**: Requests actively loading external KV data are
    ///    in the waiting queue (status `WAITING_FOR_REMOTE_KVS`), not running. Only
    ///    running requests can be preempted.
    ///
    /// 2. **Inflight offloads are checked**: When a connector is present, we check
    ///    `can_evict()` before selecting a victim. Requests with inflight offloads
    ///    are skipped.
    ///
    /// 3. **Recompute from scratch**: Preempted requests restart with `num_computed_tokens = 0`,
    ///    so any partially computed data is discarded anyway.
    ///
    /// See `STATE_TRANSITIONS.md` for the complete eviction behavior documentation.
    fn try_preempt(&mut self, blocks_needed: usize) -> bool {
        let mut freed_blocks = 0;

        while freed_blocks < blocks_needed {
            // Collect running requests for policy evaluation
            let running_refs: Vec<&SchedulerRequest> =
                self.running.iter().map(|(_, r)| r).collect();

            if running_refs.is_empty() {
                return false;
            }

            // Select victim with connector-aware filtering
            let victim_id =
                match self.select_eviction_victim(&running_refs, blocks_needed - freed_blocks) {
                    Some(id) => id,
                    None => return false,
                };

            // Preempt the victim
            // NOTE: Blocks are freed immediately via RAII. The connector is NOT notified.
            // This is safe because we've already checked can_evict() above.
            if let Some(mut victim) = self.running.remove(&victim_id) {
                // Remove from global projection state if enabled
                if let Some(proj) = &mut self.projector {
                    proj.remove_request(&victim_id);
                }

                // Count blocks before clearing (RAII will return them to pools)
                let victim_blocks = victim.block_state.total_blocks();
                freed_blocks += victim_blocks;

                // preempt() clears block_state - blocks return to pools via RAII Drop
                // This also resets num_computed_tokens to 0
                victim.preempt();

                // resume() transitions status from Preempted -> Waiting and sets
                // resumed_from_preemption flag for special handling on next schedule
                victim.resume();

                // Bump priority to avoid repeated eviction of the same request
                victim.mark_restarted();

                // Put at front of waiting queue (higher priority for rescheduling)
                self.waiting.push_front(victim);
            } else {
                return false;
            }
        }

        true
    }

    /// Select a victim for eviction, considering connector constraints.
    ///
    /// When no connector is present, this delegates directly to the scheduling policy's
    /// `select_victim()` method.
    ///
    /// When a connector is present, this method:
    /// 1. Filters out requests with inflight offloads (`can_evict()`)
    /// 2. Scores remaining candidates by G2 coverage (`get_eviction_score()`)
    /// 3. Delegates final selection to the policy with the filtered candidates
    ///
    /// # Returns
    ///
    /// The request ID of the selected victim, or `None` if no victim can be selected.
    fn select_eviction_victim(
        &self,
        running_refs: &[&SchedulerRequest],
        blocks_needed: usize,
    ) -> Option<String> {
        // If no connector, use policy directly
        let Some(shim) = &self.connector_shim else {
            // SAFETY: policy is always initialized by new() or build()
            return self
                .policy
                .as_ref()
                .expect("policy always initialized")
                .select_victim(running_refs, blocks_needed)
                .map(|id| id.to_string());
        };

        // Filter candidates by eviction safety (no inflight offloads)
        let safe_candidates: Vec<&SchedulerRequest> = running_refs
            .iter()
            .filter(|req| {
                let request_id = req.request_id();
                let can_evict = shim.can_evict(request_id);
                if !can_evict {
                    tracing::debug!(
                        request_id,
                        "Skipping eviction candidate - has inflight offloads"
                    );
                }
                can_evict
            })
            .copied()
            .collect();

        if safe_candidates.is_empty() {
            tracing::warn!(
                "No eviction candidates available - all running requests have inflight offloads"
            );
            return None;
        }

        // Score candidates by G2 coverage
        // Prefer candidates with higher G2 coverage (faster resume)
        let mut scored_candidates: Vec<(&SchedulerRequest, f32)> = safe_candidates
            .iter()
            .map(|req| {
                let score = shim
                    .get_eviction_score(req.request_id())
                    .map(|s| s.coverage_ratio)
                    .unwrap_or(0.0);
                (*req, score)
            })
            .collect();

        // Sort by G2 coverage (highest first = best candidates for eviction)
        scored_candidates
            .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // If all candidates have zero G2 coverage, fall back to policy
        // Otherwise, prefer the candidate with highest G2 coverage
        let best_score = scored_candidates.first().map(|(_, s)| *s).unwrap_or(0.0);
        if best_score > 0.0 {
            // Use the candidate with highest G2 coverage
            let best_candidate = scored_candidates.first().map(|(req, _)| *req)?;
            Some(best_candidate.request_id().to_string())
        } else {
            // Fall back to policy for selection among equally-scored candidates
            let candidate_refs: Vec<&SchedulerRequest> = safe_candidates.iter().copied().collect();
            // SAFETY: policy is always initialized by new() or build()
            self.policy
                .as_ref()
                .expect("policy always initialized")
                .select_victim(&candidate_refs, blocks_needed)
                .map(|id| id.to_string())
        }
    }

    /// Update state after model output is received.
    ///
    /// This should be called after each forward pass to update computed tokens,
    /// register blocks, and extend token sequences with generated output.
    ///
    /// # What This Method Does
    ///
    /// 1. **Block Registration**: Transitions pending blocks to registered state
    ///    for all running requests. This happens after the forward pass computes
    ///    KV cache data for the pending blocks.
    ///
    /// 2. **Token Sync**: Extends token sequences with newly generated output
    ///    tokens, maintaining block hash consistency.
    ///
    /// 3. **Token Count Updates**: Updates computed token counts and applies
    ///    forward pass completion state.
    ///
    /// 4. **Projection Cleanup**: Removes finished requests from projection state.
    ///
    /// # Connector Signal Processing
    ///
    /// **Note**: Connector signal processing (`finished_recving`, `finished_sending`)
    /// is handled separately by [`update_connector_signals()`](Self::update_connector_signals).
    /// Call that method after this one if the model execution returned connector
    /// output signals.
    ///
    /// # Usage
    ///
    /// ```ignore
    /// // After forward pass:
    /// scheduler.update_from_output(&finished_request_ids, &output_tokens);
    ///
    /// // If connector signals were returned:
    /// scheduler.update_connector_signals(finished_sending, finished_recving);
    /// ```
    ///
    /// See `STATE_TRANSITIONS.md` for the complete flow.
    pub fn update_from_output(
        &mut self,
        finished_ids: &[String],
        output_tokens: &HashMap<String, Vec<u32>>,
    ) {
        // First, register completed blocks for ALL running requests.
        // This handles both prefill and decode phases correctly:
        //
        // - Prefill: All prompt tokens exist in TokenBlockSequence from the start,
        //   but KV data wasn't computed until the forward pass. After prefill,
        //   we need to register the pending blocks that now have KV cache data.
        //
        // - Decode: New output tokens may complete a block. We extend the token
        //   sequence first, then register any newly completed blocks.
        //
        // The key insight: num_computed_tokens tells us how many tokens have KV data.
        // computed_blocks = num_computed_tokens / block_size should equal num_registered.
        // If registered < computed_blocks, we have pending blocks that need registration.

        // Process all running requests (not just those in output_tokens)
        for (_, request) in self.running.iter_mut() {
            let request_id = request.request_id().to_string();

            // Get the TokenBlocks from the sequence (these have sequence hashes)
            let token_blocks_all = request.token_sequence.blocks();

            // Calculate how many blocks should now have KV data
            // The forward pass computed KV for all tokens up to total_known_tokens
            let tokens_with_kv = request.total_known_tokens();
            let block_size = self.config.block_size;
            let computed_blocks = tokens_with_kv / block_size;

            // How many blocks are already registered?
            let registered_blocks = request.block_state.num_registered();
            let pending_blocks = request.block_state.num_unassigned();

            // Calculate blocks that should be registered but aren't
            let blocks_to_register = computed_blocks
                .saturating_sub(registered_blocks)
                .min(pending_blocks);

            if blocks_to_register > 0 {
                // Get the TokenBlocks for the blocks we're about to register
                // These start at index `registered_blocks` and we take `blocks_to_register`
                let token_blocks_for_registration: Vec<_> = token_blocks_all
                    [registered_blocks..registered_blocks + blocks_to_register]
                    .to_vec();

                // Log sequence hashes being registered (for debugging prefix caching)
                let seq_hashes_for_registration: Vec<_> = token_blocks_for_registration
                    .iter()
                    .map(|tb| tb.kvbm_sequence_hash())
                    .collect();
                tracing::debug!(
                    request_id = %request_id,
                    blocks_to_register,
                    seq_hashes = ?seq_hashes_for_registration,
                    "Registering blocks with sequence hashes"
                );

                // Use transition_with to convert pending MutableBlocks to registered ImmutableBlocks
                // The closure captures kv_cache to perform the registration
                let kv_cache = &self.kv_cache;
                let result: Result<usize, String> =
                    request
                        .block_state
                        .transition_with(blocks_to_register, |mutable_blocks| {
                            match kv_cache.complete_and_register(
                                mutable_blocks,
                                token_blocks_for_registration,
                            ) {
                                Ok(immutable_blocks) => Ok(immutable_blocks),
                                Err(returned_blocks) => {
                                    Err((returned_blocks, "Failed to register blocks".to_string()))
                                }
                            }
                        });

                match result {
                    Ok(num_registered) => {
                        tracing::debug!(
                            request_id = %request_id,
                            registered = num_registered,
                            total_registered = request.block_state.num_assigned(),
                            remaining_pending = request.block_state.num_unassigned(),
                            computed_blocks,
                            tokens_with_kv,
                            "Registered blocks after forward pass"
                        );
                    }
                    Err(e) => {
                        tracing::error!(
                            request_id = %request_id,
                            error = %e,
                            "Failed to register blocks"
                        );
                    }
                }
            }
        }

        // Handle finished requests
        // Register any remaining blocks before removing the request
        for request_id in finished_ids {
            if let Some(mut request) = self.running.remove(request_id) {
                // Remove from global projection state if enabled
                if let Some(proj) = &mut self.projector {
                    proj.remove_request(request_id);
                }

                // Notify connector of request finish before freeing blocks.
                // If connector has inflight offloads, hold blocks until finished_sending.
                if let Some(shim) = &self.connector_shim {
                    let finished_status = shim.request_finished(request_id);
                    match finished_status {
                        FinishedStatus::Pending => {
                            // Connector has inflight offloads - blocks must be held.
                            // Move to pending_block_free collection until finished_sending arrives.
                            tracing::debug!(
                                request_id = %request_id,
                                blocks = request.block_state.num_assigned(),
                                "Request has pending offloads, holding blocks until offload completes"
                            );
                            request.finish(RequestStatus::FinishedStopped);
                            self.pending_block_free
                                .insert(request_id.to_string(), request);
                            continue; // Skip normal drop - blocks held until update_connector_output
                        }
                        FinishedStatus::Finished | FinishedStatus::UntrackedRequest => {
                            tracing::debug!(
                                request_id = %request_id,
                                ?finished_status,
                                "Connector cleared request for block release"
                            );
                        }
                    }
                }

                request.finish(RequestStatus::FinishedStopped);
                // Request is dropped here, RAII frees blocks
            }
        }

        // Update running requests with output tokens (decode phase)
        for (request_id, tokens) in output_tokens {
            if let Some(request) = self.running.get_mut(request_id) {
                // IMPORTANT: Capture computed tokens BEFORE extending the sequence.
                // The model has computed KV for all tokens that existed before this output.
                let tokens_before_extension = request.total_known_tokens();

                // Extend the token sequence with new output tokens
                if let Err(e) = request.extend_tokens(tokens) {
                    tracing::error!(
                        request_id = %request_id,
                        error = %e,
                        "Failed to extend token sequence"
                    );
                    continue;
                }

                // Update token counts
                request.add_output_tokens(tokens.len());
                request.apply_forward_pass_completion(tokens_before_extension);
            }
        }

        // -------------------------------------------------------------------------
        // Connector Signal Processing
        // -------------------------------------------------------------------------
        // NOTE: Connector output signals (finished_recving, finished_sending) are
        // processed separately by `update_connector_signals()`. This separation
        // keeps model output processing distinct from async transfer coordination.
        //
        // See: update_connector_signals() for handling of:
        // - finished_recving: Completed async KV loads (G2→G1)
        // - finished_sending: Completed offloads (G1→G2)
        // -------------------------------------------------------------------------

        // -------------------------------------------------------------------------
        // Projection Cleanup for Finished Requests
        // -------------------------------------------------------------------------
        // Note: GlobalProjectionState maintains schedules based on worst-case bounds,
        // not actual tokens generated. We only need to clean up finished requests.
        // The `output_tokens` are not used for projection updates since schedules
        // are deterministic based on configuration, not actual generation.
        let _ = output_tokens; // Acknowledge unused parameter
        if let Some(projector) = &mut self.projector {
            for request_id in finished_ids {
                projector.remove_request(request_id);
            }
        }
    }

    /// Process connector output signals from workers.
    ///
    /// Called after model execution with signals indicating completed transfers:
    /// - `finished_recving`: Async KV load complete (G2→G1), requests can now be scheduled
    /// - `finished_sending`: Offload complete (G1→G2), blocks can now be freed
    ///
    /// This is a separate method from `update_from_output` to keep concerns separate:
    /// - `update_from_output`: Processes model output (tokens, block registration)
    /// - `update_connector_signals`: Processes async transfer completion signals
    ///
    /// # Usage
    ///
    /// ```ignore
    /// // After model execution returns connector output:
    /// scheduler.update_connector_signals(
    ///     connector_output.finished_sending,
    ///     connector_output.finished_recving,
    /// );
    /// ```
    pub fn update_connector_signals(
        &mut self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) {
        // Process completed onboarding (async KV loads from G2→G1)
        // Mark these requests as complete so schedule() can move them to waiting
        for request_id in &finished_recving {
            if let Some(request) = self.onboarding.get_mut(request_id) {
                request.set_onboarding_status(OnboardingStatus::Complete);
                tracing::debug!(
                    request_id = %request_id,
                    "Marked onboarding complete, ready for scheduling"
                );
            }
        }

        // Process completed offloads (G1→G2) - free held blocks
        // These were requests that finished but held blocks for offload completion
        for request_id in &finished_sending {
            if let Some(request) = self.pending_block_free.remove(request_id) {
                tracing::debug!(
                    request_id = %request_id,
                    blocks = request.block_state.num_assigned(),
                    "Offload complete, freeing held blocks"
                );
                drop(request); // RAII frees blocks
            }
        }

        // Delegate to connector shim to update leader state
        if let Some(shim) = &self.connector_shim {
            if let Err(e) = shim.update_connector_output(finished_sending, finished_recving) {
                tracing::error!(error = %e, "Failed to update connector output");
            }
        }
    }

    // =========================================================================
    // Async Onboarding Methods
    // =========================================================================

    /// Process completed onboarding requests (async KV loads from G2→G1).
    ///
    /// Called at the start of each scheduling iteration to move requests that
    /// have completed their async KV load back to the waiting queue. These
    /// requests get scheduling priority since they've already allocated blocks
    /// and are ready to run.
    fn process_completed_onboarding(&mut self) {
        // Find all requests with completed onboarding
        let completed: Vec<String> = self
            .onboarding
            .iter()
            .filter(|(_, req)| req.is_onboarding_complete())
            .map(|(id, _)| id.clone())
            .collect();

        if completed.is_empty() {
            return;
        }

        for request_id in completed {
            if let Some(mut request) = self.onboarding.remove(&request_id) {
                // Reset onboarding status
                request.set_onboarding_status(OnboardingStatus::None);
                // Set status to waiting
                request.set_status(RequestStatus::Waiting);

                tracing::info!(
                    request_id = %request_id,
                    computed_tokens = request.num_computed_tokens,
                    "Onboarding complete, moved to waiting queue"
                );

                // Add to front of waiting queue for priority scheduling
                self.waiting.push_front(request);
            }
        }
    }

    // =========================================================================
    // Projection System Methods
    // =========================================================================

    /// Update projections at the start of each scheduling iteration.
    ///
    /// Called at the start of each scheduling iteration when projection is enabled.
    /// This advances the iteration counter and recomputes choke points.
    fn update_projections(&mut self) {
        if let Some(projector) = &mut self.projector {
            projector.advance_iteration();
        }
    }

    /// Check if we should evaluate waiting requests this iteration.
    ///
    /// Returns false when backfilling isn't possible (active multi-chunk prefill
    /// that hasn't reached its final chunk yet).
    fn should_evaluate_waiting(&mut self) -> bool {
        match &mut self.projector {
            Some(proj) => self.iteration >= proj.next_iteration_for_new_requests(),
            None => true,
        }
    }

    /// Check if we should evaluate paused requests for resume this iteration.
    ///
    /// Returns false when there's no headroom to resume any paused request
    /// without creating or worsening a choke point.
    fn should_evaluate_paused(&self) -> bool {
        match &self.projector {
            Some(proj) => proj.has_headroom_for_resume(),
            None => true,
        }
    }

    /// Process proactive evictions based on choke point predictions.
    ///
    /// When a choke point is detected in the lookahead window, this method
    /// identifies requests eligible for pause/eviction and transitions them
    /// to the appropriate state.
    fn process_proactive_evictions(&mut self) {
        let Some(projector) = &self.projector else {
            return;
        };

        // Check if we have any choke points
        let Some(choke_point) = projector.next_choke_point().cloned() else {
            return;
        };

        // Get eviction candidates from projector
        let candidates: Vec<String> = projector
            .recommend_pause_candidates(choke_point.deficit.max(0) as usize)
            .into_iter()
            .map(|s| s.to_string())
            .collect();

        // Process candidates - pause them for now
        // Future: Could plan for eviction instead if connector supports priority offload
        for request_id in candidates {
            if let Some(request) = self.running.remove(&request_id) {
                // Remove from global projection state if enabled
                // Paused requests don't grow, so they shouldn't be in projection
                if let Some(proj) = &mut self.projector {
                    proj.remove_request(&request_id);
                }

                tracing::debug!(
                    request_id = %request_id,
                    iteration = self.iteration,
                    choke_point_iteration = choke_point.iteration,
                    deficit = choke_point.deficit,
                    "Proactively pausing request due to predicted choke point"
                );
                self.paused.pause(request);
            }
        }
    }

    /// Try to resume paused requests if space is available.
    ///
    /// Called BEFORE scheduling new requests to prioritize resuming paused
    /// requests that have already made progress. Requests are resumed in LIFO order.
    fn try_resume_paused(
        &mut self,
        output: &mut SchedulerOutput,
        num_scheduled_tokens: &mut HashMap<String, usize>,
    ) {
        while !self.paused.is_empty() {
            // Check if we have headroom to resume
            let available_blocks = self.kv_cache.free_blocks();
            if available_blocks == 0 {
                break;
            }

            // Check budget limits
            let total_scheduled: usize = num_scheduled_tokens.values().sum();
            if total_scheduled >= self.config.max_num_batched_tokens {
                break;
            }
            if self.running.len() >= self.config.max_num_seqs {
                break;
            }

            // Get the most recently paused request and check if we can resume it
            let resume_candidate: Option<(String, usize, usize)> = {
                let last_id = self.paused.request_ids_by_pause_order().last().cloned();

                if let Some(request_id) = last_id {
                    if let Some(paused_request) = self.paused.get(&request_id) {
                        let blocks_needed =
                            paused_request.num_new_blocks_needed(self.config.block_size);
                        let blocks_held = paused_request.block_state.total_blocks();
                        if blocks_needed <= available_blocks {
                            Some((request_id, blocks_held, blocks_needed))
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            };

            let Some((request_id, blocks_held, blocks_needed)) = resume_candidate else {
                break;
            };

            // Resume the request
            let Some(mut request) = self.paused.resume_by_id(&request_id) else {
                break;
            };

            // Allocate any additional blocks needed
            let new_block_ids = if blocks_needed > 0 {
                if let Some(new_blocks) = self.kv_cache.allocate(blocks_needed) {
                    let ids: Vec<_> = new_blocks.iter().map(|b| b.block_id()).collect();
                    request.add_pending_blocks(new_blocks);
                    ids
                } else {
                    // Can't allocate - put back and stop
                    self.paused.pause(request);
                    break;
                }
            } else {
                vec![]
            };

            let tokens_to_compute = request.tokens_to_compute();
            request.status = RequestStatus::Running;

            // Get all tokens for resumed request
            let all_tokens = Some(request.all_tokens_for_resume());

            // Record in output as resumed cached request
            output.add_cached_request(
                request.request_id().to_string(),
                true, // resumed = true
                vec![],
                all_tokens,
                new_block_ids,
                request.num_computed_tokens,
                request.num_output_tokens,
            );

            num_scheduled_tokens.insert(request.request_id().to_string(), tokens_to_compute);

            tracing::debug!(
                request_id = %request.request_id(),
                blocks_held,
                blocks_needed,
                tokens_to_compute,
                iteration = self.iteration,
                "Resumed paused request"
            );

            // Add to global projection state if enabled
            // Paused requests are NOT restored (they never lost their blocks)
            if let Some(proj) = &mut self.projector {
                proj.add_request(&request, false);
            }

            self.running.insert(request);
        }
    }

    /// Get the number of paused requests.
    pub fn num_paused(&self) -> usize {
        self.paused.len()
    }

    /// Get the total blocks held by paused requests.
    pub fn paused_blocks(&self) -> usize {
        self.paused.total_held_blocks()
    }

    /// Check if the projection system detected any choke points.
    pub fn has_choke_points(&self) -> bool {
        self.projector
            .as_ref()
            .map(|p| p.has_choke_points())
            .unwrap_or(false)
    }

    /// Get the nearest choke point if any.
    pub fn nearest_choke_point(&self) -> Option<&super::projection::ChokePoint> {
        self.projector.as_ref().and_then(|p| p.next_choke_point())
    }

    // =========================================================================
    // Backfill and Reservation Methods
    // =========================================================================

    /// Check if backfill prefill is allowed based on current running requests.
    ///
    /// Backfill is only allowed when:
    /// - There is no active chunked prefill, OR
    /// - The active chunked prefill is on its final pass
    ///
    /// This ensures we complete one request's prefill before starting another,
    /// except for the final chunk where we can backfill with remaining capacity.
    fn can_backfill_prefill(&self) -> bool {
        // Find any request that is actively prefilling
        for (_, request) in self.running.iter() {
            if let Some(remaining) = request.remaining_prefill() {
                // This request is still prefilling
                let chunk_size = self.config.effective_prefill_chunk_size();
                // Only allow backfill if this is the final chunk
                if remaining > chunk_size {
                    return false;
                }
            }
        }
        // No active multi-chunk prefill, or all prefills are on final chunk
        true
    }

    /// Compute the worst-case block reservation needed for the next forward pass.
    ///
    /// This reservation ensures we can always complete the next pass without allocation
    /// failure. The formula accounts for:
    ///
    /// 1. **Requests completing a block**: 1 block per request that will have a
    ///    complete block after the next pass (current partial + 1 token >= block_size)
    ///
    /// 2. **Chunked prefill continuation**: If there's an active chunked prefill
    ///    that will continue, reserve blocks for the next chunk
    ///
    /// 3. **Backfill blocks**: If backfill is allowed and there are pending requests,
    ///    reserve blocks for the first chunk of promoted requests
    ///
    /// # Returns
    ///
    /// The number of blocks that should be reserved for the next pass.
    pub fn compute_next_pass_reservation(&self) -> usize {
        let mut reservation = 0;
        let block_size = self.config.block_size;

        // 1. One block for each request that will complete a block after next pass
        for (_, request) in self.running.iter() {
            let current_tokens = request.total_known_tokens();
            let tokens_in_partial_block = current_tokens % block_size;

            // If adding 1 token (decode) completes a block, reserve 1 block
            // For prefilling requests, this is more complex - handled below
            if !request.is_prefilling() && tokens_in_partial_block + 1 >= block_size {
                reservation += 1;
            }
        }

        // 2. Blocks for active chunked prefill continuing
        let chunk_size = self.config.effective_prefill_chunk_size();
        for (_, request) in self.running.iter() {
            if let Some(remaining) = request.remaining_prefill() {
                if remaining > chunk_size {
                    // Prefill will continue - reserve blocks for next chunk
                    reservation += chunk_size.div_ceil(block_size);
                    break; // Only one request can be actively prefilling
                }
            }
        }

        // 3. Blocks for promoted pending requests (backfill)
        // Only if backfill is allowed (primary prefill is on final pass)
        if self.can_backfill_prefill() {
            if let Some(pending) = self.waiting.peek() {
                let prompt_tokens = pending.original_prompt_len();
                // Estimate blocks needed for first chunk of new request
                let first_chunk = prompt_tokens.min(chunk_size);
                reservation += first_chunk.div_ceil(block_size);
            }
        }

        reservation
    }

    /// Get the current block reservation for the next pass.
    ///
    /// This is a convenience method that can be called after schedule()
    /// to get the reservation value for logging or pre-allocation.
    pub fn next_pass_block_reservation(&self) -> usize {
        self.compute_next_pass_reservation()
    }
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler")
            .field("iteration", &self.iteration)
            .field("waiting", &self.waiting.len())
            .field("running", &self.running.len())
            .field("paused", &self.paused.len())
            .field("kv_cache", &self.kv_cache)
            .field("has_shared_state", &self.shared_state.is_some())
            .field("has_connector", &self.connector_shim.is_some())
            .field("projection_enabled", &self.projector.is_some())
            .field("planned_evictions", &self.planned_evictions.len())
            .finish()
    }
}
