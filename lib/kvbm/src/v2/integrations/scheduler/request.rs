// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request state and lifecycle management.
//!
//! # State Machine
//!
//! Requests follow a defined lifecycle through the scheduler:
//!
//! ```text
//!                    ┌──────────────────────────────────────────────────────┐
//!                    │                                                      │
//!                    ▼                                                      │
//! [New] ──► Waiting ──► Running ──► Finished*                               │
//!               ▲          │                                                │
//!               │          │──► Paused ──────────────────────┐              │
//!               │          │       │                         │              │
//!               │          │       └─► PlannedForEviction ───┤              │
//!               │          │                                 ▼              │
//!               │          ▼                             Preempted ─────────┤
//!               └──── Preempted ────────────────────────────────────────────┘
//! ```
//!
//! # Block Ownership Invariants
//!
//! | Status              | Has Blocks | Blocks Freed By |
//! |---------------------|------------|-----------------|
//! | Waiting             | No         | N/A             |
//! | Running             | Yes        | preempt() or finish() |
//! | Paused              | Yes*       | Progressive release or evict() |
//! | PlannedForEviction  | Yes        | After G2 offload completes |
//! | Preempted           | No         | Already freed by preempt() |
//! | Finished*           | No*        | finish() or delayed by connector |
//!
//! *Note: Paused requests may progressively release blocks that are already
//! offloaded to G2, lending them to other requests while retaining some blocks.
//!
//! *Note: With connector integration, finished requests may temporarily hold
//! blocks until the connector signals `finished_sending`. See `STATE_TRANSITIONS.md`.
//!
//! # Connector Interaction
//!
//! The state machine in this module does NOT directly interact with the connector.
//! The scheduler is responsible for coordinating with the connector before calling
//! state transition methods. See `core.rs` for connector interaction patterns.

use super::kv_cache::RequestBlockState;
use crate::v2::BlockId;
use crate::v2::integrations::common::Request;
use crate::v2::logical::blocks::{ImmutableBlock, MutableBlock};
use crate::{G1, KvbmSequenceHashProvider};
use dynamo_tokens::{TokenBlock, TokenBlockSequence};

/// Status of a request in the scheduler.
///
/// # State Transition Rules
///
/// Valid transitions:
/// - `Waiting` -> `Running` (scheduled)
/// - `Waiting` -> `WaitingForRemoteKvs` (async KV load started)
/// - `WaitingForRemoteKvs` -> `Waiting` (async KV load completed)
/// - `Running` -> `Paused` (proactive pause before memory pressure)
/// - `Running` -> `PlannedForEviction` (priority G2 offload started)
/// - `Running` -> `Preempted` (memory pressure, blocks freed)
/// - `Running` -> `Finished*` (completed, blocks freed)
/// - `Paused` -> `Running` (space freed, can resume)
/// - `Paused` -> `PlannedForEviction` (need to evict)
/// - `Paused` -> `Preempted` (urgent eviction, skip offload)
/// - `PlannedForEviction` -> `Preempted` (offload complete, blocks freed)
/// - `Preempted` -> `Waiting` (resumed for rescheduling)
///
/// Invalid transitions (will panic in debug builds):
/// - `Waiting` -> `Preempted` (must be Running first)
/// - `Finished*` -> any state (terminal)
/// - `Preempted` -> `Running` (must go through Waiting)
/// - `WaitingForRemoteKvs` -> `Running` (must complete load first)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// Request is waiting to be scheduled.
    /// Invariant: `block_state` is empty (no blocks allocated).
    Waiting,

    /// Request is waiting for external KV cache data to be loaded asynchronously.
    ///
    /// # KV Connector Integration
    ///
    /// This state is entered when the connector reports external KV cache hits
    /// that need to be loaded asynchronously (inter-pass mode). The request
    /// stays in this state until the connector signals `finished_recving`.
    ///
    /// Memory for the blocks has been allocated, but the KV data is still being
    /// transferred from external storage (G2/G3 or remote nodes).
    ///
    /// # Invariants
    /// - Blocks have been allocated (matched_blocks + new blocks)
    /// - `num_computed_tokens` reflects the expected external tokens
    /// - Connector is actively loading data
    ///
    /// # Transitions
    /// - `WaitingForRemoteKvs` -> `Waiting` (load completed successfully)
    /// - `WaitingForRemoteKvs` -> `Waiting` with reset (load failed, need recompute)
    WaitingForRemoteKvs,

    /// Request is currently running (scheduled for this iteration).
    /// Invariant: `block_state` contains allocated blocks.
    Running,

    /// Request is paused, holding blocks but not scheduled for execution.
    ///
    /// # Proactive Pause System
    ///
    /// A request enters `Paused` when the projection system predicts a future
    /// choke point and the request is eligible for pause (met minimum progress).
    /// Paused requests hold their blocks but don't consume scheduling tokens.
    ///
    /// # Block Lending
    ///
    /// Paused requests can progressively release blocks that are already
    /// offloaded to G2 (or that we're willing to recompute). This allows
    /// other requests to use these blocks while the paused request retains
    /// enough state for efficient resumption.
    ///
    /// # Invariants
    /// - `block_state` contains some or all allocated blocks
    /// - Some blocks may have been "lent" to other requests
    /// - `num_computed_tokens` reflects actual computed state
    /// - Request has met minimum progress guarantee (eviction eligible)
    ///
    /// # Transitions
    /// - `Paused` -> `Running` (space freed, can resume with full blocks)
    /// - `Paused` -> `PlannedForEviction` (need to fully evict, start priority offload)
    /// - `Paused` -> `Preempted` (urgent eviction, skip offload)
    Paused,

    /// Request is planned for eviction with priority G2 offload in progress.
    ///
    /// # Priority Offload System
    ///
    /// When a request must be evicted but has blocks not yet in G2, we first
    /// request priority offload from the connector. The request stays in this
    /// state until all blocks are offloaded, then transitions to Preempted.
    ///
    /// # Invariants
    /// - `block_state` contains all allocated blocks (no lending)
    /// - Connector has priority offload request for remaining blocks
    /// - `PlannedEviction` entry exists in `PlannedEvictionTracker`
    ///
    /// # Transitions
    /// - `PlannedForEviction` -> `Preempted` (all blocks offloaded, can evict)
    PlannedForEviction,

    /// Request was preempted due to memory pressure.
    /// Invariant: `block_state` is empty (blocks were freed during preemption).
    /// Invariant: `num_computed_tokens == 0` (must recompute from scratch).
    Preempted,

    /// Request finished normally (hit stop token or max tokens).
    /// Invariant: `block_state` is empty (blocks freed, possibly after delay).
    FinishedStopped,

    /// Request was aborted (cancelled by user or error).
    /// Invariant: `block_state` is empty (blocks freed, possibly after delay).
    FinishedAborted,

    /// Request finished due to reaching length limit.
    /// Invariant: `block_state` is empty (blocks freed, possibly after delay).
    FinishedLengthCapped,
}

impl RequestStatus {
    /// Returns true if the request is in a finished state.
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            RequestStatus::FinishedStopped
                | RequestStatus::FinishedAborted
                | RequestStatus::FinishedLengthCapped
        )
    }

    /// Returns true if the request can be scheduled.
    pub fn can_schedule(&self) -> bool {
        matches!(self, RequestStatus::Waiting | RequestStatus::Preempted)
    }

    /// Returns true if the request can be paused.
    ///
    /// A request can be paused when:
    /// - It's currently `Running` (not in any other state)
    ///
    /// Note: The scheduler should also check eviction eligibility
    /// (met minimum progress guarantee) before pausing.
    pub fn can_pause(&self) -> bool {
        matches!(self, RequestStatus::Running)
    }

    /// Returns true if the request is in a paused state.
    ///
    /// Both `Paused` and `PlannedForEviction` are considered paused
    /// because they hold blocks but are not actively scheduled.
    pub fn is_paused(&self) -> bool {
        matches!(
            self,
            RequestStatus::Paused | RequestStatus::PlannedForEviction
        )
    }

    /// Returns true if the request currently holds blocks.
    ///
    /// This is used to determine if blocks need to be freed when
    /// transitioning states.
    pub fn holds_blocks(&self) -> bool {
        matches!(
            self,
            RequestStatus::Running
                | RequestStatus::Paused
                | RequestStatus::PlannedForEviction
                | RequestStatus::WaitingForRemoteKvs
        )
    }

    /// Returns true if the request can be resumed from pause.
    pub fn can_resume_from_pause(&self) -> bool {
        matches!(self, RequestStatus::Paused)
    }
}

/// Status of async KV onboarding for a request.
///
/// Used when a request is in the `onboarding` collection, waiting for
/// external KV cache data to be loaded asynchronously (inter-pass mode).
///
/// # Flow
///
/// ```text
/// schedule_waiting():
///   connector returns load_kv_async=true
///   → allocate blocks
///   → set onboarding_status = Loading
///   → move to onboarding collection
///
/// update_connector_signals(finished_recving):
///   → set onboarding_status = Complete
///
/// schedule():
///   → check onboarding collection for Complete status
///   → move completed requests to waiting queue
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnboardingStatus {
    /// No async onboarding in progress.
    #[default]
    None,

    /// Async KV transfer from external storage is in progress.
    ///
    /// The request is in the `onboarding` collection, holding blocks.
    /// Waiting for `finished_recving` signal from connector.
    Loading,

    /// Async KV transfer completed successfully.
    ///
    /// The request is ready to be moved from `onboarding` to `waiting`.
    Complete,
}

/// Internal scheduler representation of a request.
///
/// This struct tracks the block allocations for a request using RAII guards.
/// The `block_state` holds both pending (mutable) and registered (immutable)
/// blocks, managing their lifecycle automatically.
///
/// # Block Lifecycle
///
/// ```text
/// Allocation:   schedule() -> kv_cache.allocate() -> add_pending_blocks()
///                                                           │
///                                                           ▼
/// Forward Pass:                                     [pending blocks]
///                                                           │
///                                                           ▼
/// Registration: complete_and_register() ----------> [registered blocks]
///                                                           │
///                           ┌───────────────────────────────┴───────────────────────────────┐
///                           │                                                               │
/// Preemption:        preempt() clears block_state                                           │
///                    (blocks return to reset pool via RAII)                                 │
///                                                                                           │
/// Completion:                                                                         finish()
///                                                                                           │
///                                                                    ┌──────────────────────┴──────────────────────┐
///                                                                    │                                             │
///                                                       No connector delay                              Connector delay
///                                                                    │                                             │
///                                                                    ▼                                             ▼
///                                                       block_state.clear()                         Blocks held until
///                                                       (immediate RAII drop)                       finished_sending signal
/// ```
///
/// # Connector Interaction Warning
///
/// This struct does NOT track connector state. When a request finishes, the scheduler
/// must check with the connector before allowing blocks to be freed. If the connector
/// is actively offloading, blocks must be held until `finished_sending` is signaled.
///
/// The `finish()` method unconditionally clears blocks. For proper connector integration,
/// the scheduler should:
/// 1. Check `connector.request_finished()` before calling `finish()`
/// 2. If delay is needed, hold a reference to blocks elsewhere
/// 3. Only call `finish()` after `finished_sending` is received
pub struct SchedulerRequest {
    /// The original request data.
    pub request: Request,

    /// Current status of the request.
    pub status: RequestStatus,

    /// RAII block state for this request.
    ///
    /// Contains both pending blocks (allocated but not yet filled with token data)
    /// and registered blocks (completed and in the cache).
    ///
    /// # Invariants
    ///
    /// - Empty when `status` is `Waiting`, `Preempted`, or any `Finished*` state
    /// - Non-empty when `status` is `Running` (after first allocation)
    ///
    /// # RAII Behavior
    ///
    /// When blocks are dropped (via `clear()` or struct drop), they automatically
    /// return to the appropriate pool in the BlockManager:
    /// - `MutableBlock<G1>` -> reset pool
    /// - `ImmutableBlock<G1>` -> inactive pool (if cached) or reset pool
    pub block_state: RequestBlockState,

    /// Token sequence for tracking tokens and computing block hashes.
    ///
    /// Initialized with prompt tokens when the request is created, and extended
    /// with output tokens as they are generated. The TBS computes sequence hashes
    /// for each complete block, which are used for prefix caching and deduplication.
    ///
    /// # Block Alignment
    ///
    /// The blocks in `token_sequence.blocks()[i]` correspond to `block_state.registered[i]`.
    /// When blocks become complete in the TBS, they are registered with the block manager.
    ///
    /// # Preemption Behavior
    ///
    /// The token sequence is NOT cleared on preemption - we keep the token history.
    /// Only `block_state` is cleared and `num_computed_tokens` is reset.
    pub token_sequence: TokenBlockSequence,

    /// Number of tokens that have been computed (KV cache filled).
    ///
    /// # Invariants
    ///
    /// - Reset to 0 on preemption (request must recompute from scratch)
    /// - Monotonically increases during normal execution
    /// - May be set from external sources (connector's cached tokens)
    pub num_computed_tokens: usize,

    /// Number of output tokens generated so far.
    pub num_output_tokens: usize,

    /// Number of tokens found in prefix cache (local G1) during scheduling.
    ///
    /// # Prefix Caching
    ///
    /// This is set when `get_computed_blocks()` finds cached blocks in G1.
    /// It represents the longest prefix of this request's tokens that were
    /// already in the cache from previous requests with the same prefix.
    ///
    /// A value of -1 (represented as `isize::MAX` in usize) means the request
    /// hasn't been checked for prefix cache hits yet.
    ///
    /// # Usage
    /// - Set during `schedule_waiting()` after prefix cache lookup
    /// - Used for metrics and to avoid redundant lookups
    /// - Not reset on preemption (prefix cache state is independent)
    pub num_cached_tokens: isize,

    /// Whether this request was just resumed from preemption.
    /// Reset to false after being scheduled once.
    ///
    /// When true, the scheduler sends `all_token_ids` to workers since they
    /// may have lost track of this request's state during preemption.
    pub resumed_from_preemption: bool,

    /// Status of async KV onboarding (inter-pass mode).
    ///
    /// Used when the request is in the scheduler's `onboarding` collection.
    /// - `None`: Not in async onboarding
    /// - `Loading`: Waiting for external KV data
    /// - `Complete`: Ready to move to waiting queue
    pub onboarding_status: OnboardingStatus,
}

impl SchedulerRequest {
    /// Create a new scheduler request from a request.
    ///
    /// Initializes the token sequence with the prompt tokens and the given block size.
    /// The salt from the request is used for deterministic hash computation.
    ///
    /// # Arguments
    /// * `request` - The original request with prompt tokens
    /// * `block_size` - Block size in tokens (for TokenBlockSequence)
    pub fn new(request: Request, block_size: usize) -> Self {
        // Initialize TokenBlockSequence with prompt tokens
        let token_sequence = TokenBlockSequence::new(
            request.tokens.clone(),
            block_size as u32,
            Some(request.salt_hash),
        );

        Self {
            request,
            status: RequestStatus::Waiting,
            block_state: RequestBlockState::new(),
            token_sequence,
            num_computed_tokens: 0,
            num_output_tokens: 0,
            num_cached_tokens: -1, // Not yet checked for prefix cache hits
            resumed_from_preemption: false,
            onboarding_status: OnboardingStatus::None,
        }
    }

    /// Get the request ID.
    pub fn request_id(&self) -> &str {
        &self.request.request_id
    }

    // =========================================================================
    // Token Counting API (Unified for fresh and resumed requests)
    // =========================================================================
    //
    // These methods provide a consistent interface for token counting that works
    // correctly for both fresh requests and resumed requests (after preemption).
    //
    // Key insight: A resumed request is conceptually a new request whose "prompt"
    // is the full sequence up to the eviction point. The TokenBlockSequence tracks
    // all known tokens and is the source of truth.

    /// Get the total known tokens (prompt + generated output).
    ///
    /// This is the authoritative count from `TokenBlockSequence`.
    /// - Fresh requests: equals initial prompt length + output tokens
    /// - Resumed requests: equals full sequence up to eviction + any new output
    ///
    /// Use this instead of the old `total_tokens()` which was ambiguous.
    #[inline]
    pub fn total_known_tokens(&self) -> usize {
        self.token_sequence.total_tokens()
    }

    /// Get the number of tokens that need to be computed (prefilled or decoded).
    ///
    /// This is the sequence length that needs KV cache, minus what's already computed.
    /// Works correctly for both fresh requests and resumed requests.
    #[inline]
    pub fn tokens_to_compute(&self) -> usize {
        self.total_known_tokens()
            .saturating_sub(self.num_computed_tokens)
    }

    /// Check if the request is currently prefilling.
    ///
    /// A request is prefilling if it hasn't computed all known tokens yet.
    /// This works for both:
    /// - Fresh requests: computing initial prompt
    /// - Resumed requests: recomputing full sequence up to eviction point
    #[inline]
    pub fn is_prefilling(&self) -> bool {
        self.num_computed_tokens < self.total_known_tokens()
    }

    /// Get the remaining tokens to prefill.
    ///
    /// Returns `Some(count)` if prefilling, `None` if in decode phase.
    /// Works correctly for both fresh and resumed requests.
    #[inline]
    pub fn remaining_prefill(&self) -> Option<usize> {
        let total = self.total_known_tokens();
        if self.num_computed_tokens < total {
            Some(total - self.num_computed_tokens)
        } else {
            None
        }
    }

    /// Get the original prompt length (immutable, for prefix cache).
    ///
    /// This is ONLY for prefix cache hash comparison - it's the original
    /// input sequence that might have cached blocks from other requests.
    ///
    /// **Do NOT use this for prefill tracking** - use `total_known_tokens()`
    /// or `remaining_prefill()` instead.
    #[inline]
    pub fn original_prompt_len(&self) -> usize {
        self.request.tokens.len()
    }

    /// Get the maximum tokens this request can reach.
    ///
    /// Computed as: `original_prompt + max_output_tokens`, capped at `max_seq_len`.
    pub fn max_total_tokens(&self, max_seq_len: usize) -> usize {
        let max_output = self
            .request
            .max_tokens
            .unwrap_or(max_seq_len.saturating_sub(self.original_prompt_len()));
        self.original_prompt_len() + max_output
    }

    /// Get the remaining output tokens until max_tokens limit.
    #[inline]
    pub fn remaining_output_capacity(&self) -> usize {
        if let Some(max) = self.request.max_tokens {
            max.saturating_sub(self.num_output_tokens)
        } else {
            usize::MAX
        }
    }

    // =========================================================================
    // Computed Tokens Management
    // =========================================================================

    /// Apply cache lookup results during scheduling.
    ///
    /// Called during `schedule_waiting()` when prefix cache and external cache
    /// matches are found. Sets the initial computed token count.
    ///
    /// # Arguments
    /// * `num_local_cached` - Tokens from local prefix cache matches
    /// * `num_external_cached` - Tokens from external cache (G2/G3/remote)
    pub fn apply_cache_matches(&mut self, num_local_cached: usize, num_external_cached: usize) {
        self.num_computed_tokens = num_local_cached + num_external_cached;
    }

    /// Apply forward pass completion.
    ///
    /// Called after `update_from_output()` when the model has computed KV cache
    /// for the scheduled tokens. Updates computed tokens to reflect that all
    /// tokens before the new output now have KV cache.
    ///
    /// # Arguments
    /// * `tokens_before_new_output` - Total tokens before new output was added
    ///   (typically `total_known_tokens()` captured before `add_output_tokens()`)
    pub fn apply_forward_pass_completion(&mut self, tokens_before_new_output: usize) {
        debug_assert!(
            tokens_before_new_output >= self.num_computed_tokens,
            "Forward pass cannot decrease computed tokens: {} -> {}",
            self.num_computed_tokens,
            tokens_before_new_output
        );
        self.num_computed_tokens = tokens_before_new_output;
    }

    // =========================================================================
    // Block counting
    // =========================================================================

    /// Get the number of blocks required for the current token count.
    pub fn num_blocks_required(&self, block_size: usize) -> usize {
        (self.total_known_tokens() + block_size - 1) / block_size
    }

    /// Get the number of new blocks needed (beyond what's already allocated).
    pub fn num_new_blocks_needed(&self, block_size: usize) -> usize {
        self.num_blocks_required(block_size)
            .saturating_sub(self.block_state.total_blocks())
    }

    /// Check if the request has reached its maximum token limit.
    pub fn is_at_max_tokens(&self) -> bool {
        if let Some(max) = self.request.max_tokens {
            self.num_output_tokens >= max
        } else {
            false
        }
    }

    /// Get the block IDs allocated to this request.
    ///
    /// Returns all block IDs (both pending and registered).
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.block_state.all_block_ids()
    }

    /// Add pending (mutable) blocks to this request.
    pub fn add_pending_blocks(&mut self, blocks: Vec<MutableBlock<G1>>) {
        self.block_state.add_pending(blocks);
    }

    /// Add registered (immutable) blocks to this request.
    pub fn add_registered_blocks(&mut self, blocks: Vec<ImmutableBlock<G1>>) {
        self.block_state.add_registered(blocks);
    }

    /// Take pending blocks out of this request.
    ///
    /// Used when transitioning blocks to registered after a forward pass.
    pub fn take_pending_blocks(&mut self) -> Vec<MutableBlock<G1>> {
        self.block_state.take_pending()
    }

    /// Transition the request to running state.
    pub fn start_running(&mut self) {
        self.status = RequestStatus::Running;
    }

    /// Preempt the request, releasing all blocks.
    ///
    /// All RAII blocks are dropped, returning them to the appropriate pools.
    ///
    /// # Block Deallocation
    ///
    /// Blocks are freed **immediately** via RAII. Unlike `finish()`, preemption
    /// does NOT coordinate with the connector. This matches vLLM's behavior where:
    ///
    /// 1. Preempted requests are not offloading (offload only happens on completion)
    /// 2. Async KV loads happen in `WAITING_FOR_REMOTE_KVS` state (not preemptable)
    /// 3. The request will recompute from scratch anyway
    ///
    /// # State Reset
    ///
    /// - `status` -> `Preempted`
    /// - `block_state` -> cleared (blocks dropped)
    /// - `num_computed_tokens` -> 0 (must recompute all tokens)
    ///
    /// # Connector Warning
    ///
    /// If the connector implementation supports streaming offload during execution
    /// (not just on completion), this method may race with inflight transfers.
    /// In such cases, check `has_inflight_offloads()` on the connector slot before
    /// calling this method.
    pub fn preempt(&mut self) {
        self.status = RequestStatus::Preempted;
        // Clear blocks - RAII returns them to pools immediately.
        // NOTE: The connector is NOT notified. This is intentional per vLLM design.
        self.block_state.clear();
        // Reset computed tokens since blocks are freed and data is lost.
        // The request must recompute from scratch when rescheduled.
        self.num_computed_tokens = 0;
    }

    /// Resume the request from preemption.
    ///
    /// Transitions the request from `Preempted` back to `Waiting` for rescheduling.
    /// Sets `resumed_from_preemption` flag to signal the scheduler to send full
    /// token state to workers (since workers may have lost track during preemption).
    ///
    /// # State Invariants
    ///
    /// - Requires: `status == Preempted`
    /// - Requires: `block_state` is empty (cleared during preempt)
    /// - Requires: `num_computed_tokens == 0` (reset during preempt)
    ///
    /// # Panics
    ///
    /// Debug-asserts that status is `Preempted`. Calling on a non-preempted
    /// request indicates a bug in the scheduler state machine.
    pub fn resume(&mut self) {
        debug_assert_eq!(self.status, RequestStatus::Preempted);
        self.status = RequestStatus::Waiting;
        self.resumed_from_preemption = true;
    }

    /// Mark the request as restarted, bumping its priority to avoid repeated eviction.
    ///
    /// This should be called when a request is resumed from preemption. It increments
    /// the restart counter and bumps the request priority so that this request is
    /// less likely to be evicted again in the future.
    ///
    /// # Priority Bumping
    ///
    /// Each restart adds 10 to the effective priority:
    /// - First restart: priority becomes 10
    /// - Second restart: priority becomes 20
    /// - etc.
    ///
    /// This ensures repeatedly-evicted requests eventually become high enough priority
    /// to complete without further preemption.
    pub fn mark_restarted(&mut self) {
        self.request.mark_restarted();
    }

    /// Get the remaining tokens until completion.
    ///
    /// Returns the number of output tokens remaining before this request reaches
    /// its max_tokens limit. If no max_tokens is set, returns usize::MAX.
    pub fn remaining_output_tokens(&self) -> usize {
        if let Some(max) = self.request.max_tokens {
            max.saturating_sub(self.num_output_tokens)
        } else {
            usize::MAX
        }
    }

    /// Pause the request, keeping blocks allocated but not scheduling.
    ///
    /// Transitions from `Running` to `Paused`. The request keeps its blocks
    /// but is removed from active scheduling. Paused requests can later:
    /// - Resume to `Running` if space becomes available
    /// - Progressively release blocks that are in G2
    /// - Transition to `PlannedForEviction` if eviction is needed
    ///
    /// # State Invariants
    ///
    /// - Requires: `status == Running`
    /// - Preserves: `block_state` (blocks are retained)
    /// - Preserves: `num_computed_tokens` (can resume without recompute)
    ///
    /// # Panics
    ///
    /// Debug-asserts that status is `Running`. Pausing a non-running
    /// request indicates a bug.
    pub fn pause(&mut self) {
        debug_assert_eq!(self.status, RequestStatus::Running);
        self.status = RequestStatus::Paused;
    }

    /// Resume the request from pause.
    ///
    /// Transitions from `Paused` back to `Running`. The request resumes
    /// execution with its retained blocks and computed tokens.
    ///
    /// # Note on Lent Blocks
    ///
    /// If the request had lent blocks while paused, those blocks must be
    /// reclaimed before calling this method. The scheduler is responsible
    /// for ensuring all blocks are available.
    ///
    /// # State Invariants
    ///
    /// - Requires: `status == Paused`
    /// - Requires: All lent blocks have been returned
    ///
    /// # Panics
    ///
    /// Debug-asserts that status is `Paused`. Resuming a non-paused
    /// request indicates a bug.
    pub fn resume_from_pause(&mut self) {
        debug_assert_eq!(self.status, RequestStatus::Paused);
        self.status = RequestStatus::Running;
    }

    /// Mark the request for planned eviction.
    ///
    /// Transitions from `Paused` or `Running` to `PlannedForEviction`.
    /// The scheduler should request priority G2 offload for any blocks
    /// not yet in G2, then evict when offload completes.
    ///
    /// # State Invariants
    ///
    /// - Requires: `status == Paused` or `status == Running`
    /// - Preserves: `block_state` (blocks held until offload completes)
    ///
    /// # Panics
    ///
    /// Debug-asserts valid source state.
    pub fn plan_for_eviction(&mut self) {
        debug_assert!(
            self.status == RequestStatus::Paused || self.status == RequestStatus::Running,
            "Invalid state for plan_for_eviction: {:?}",
            self.status
        );
        self.status = RequestStatus::PlannedForEviction;
    }

    /// Finish the request with the given status.
    ///
    /// All RAII blocks are dropped, returning them to the appropriate pools.
    ///
    /// # Block Deallocation
    ///
    /// This method **immediately** frees all blocks. However, with connector
    /// integration, blocks may need to be held for ongoing offload operations.
    ///
    /// # Connector Integration (TODO)
    ///
    /// The proper flow with connector integration should be:
    ///
    /// ```text
    /// 1. Scheduler calls connector.request_finished(request_id, block_ids)
    /// 2. Connector returns (delay_free_blocks, kv_xfer_params)
    /// 3. If delay_free_blocks == false:
    ///    - Call finish() immediately (current behavior)
    /// 4. If delay_free_blocks == true:
    ///    - Hold blocks elsewhere (e.g., a pending_free map)
    ///    - DO NOT call finish() yet
    ///    - Wait for finished_sending signal from connector
    ///    - Then call finish() to release blocks
    /// ```
    ///
    /// Currently, this method is called unconditionally, which may cause race
    /// conditions with connector offload operations. The scheduler must
    /// coordinate with the connector before calling this method.
    ///
    /// # Panics
    ///
    /// Debug-asserts that `status.is_finished()` is true. Passing a non-finished
    /// status indicates a bug.
    pub fn finish(&mut self, status: RequestStatus) {
        debug_assert!(status.is_finished());
        self.status = status;
        // Clear blocks - RAII returns them to pools immediately.
        // WARNING: If connector has active offload operations, this may race
        // with those transfers. Scheduler must check connector.request_finished()
        // before calling this method.
        self.block_state.clear();
    }

    /// Set the request status.
    ///
    /// Use this for status transitions during scheduling (e.g., Waiting → Running).
    /// For completing requests, use [`finish()`](Self::finish) instead.
    pub fn set_status(&mut self, status: RequestStatus) {
        self.status = status;
    }

    /// Add output tokens after a forward pass.
    pub fn add_output_tokens(&mut self, num_tokens: usize) {
        self.num_output_tokens += num_tokens;
    }

    /// Clear the resumed flag (called after scheduling).
    pub fn clear_resumed_flag(&mut self) {
        self.resumed_from_preemption = false;
    }

    // =========================================================================
    // Token sequence methods for block hash computation
    // =========================================================================

    /// Extend the token sequence with new output tokens.
    ///
    /// This updates the internal TokenBlockSequence with the newly generated tokens.
    /// As tokens accumulate, new complete blocks are formed in the sequence.
    ///
    /// # Arguments
    /// * `tokens` - Output tokens to add to the sequence
    ///
    /// # Returns
    /// * `Ok(())` on success
    /// * `Err` if extending the sequence fails
    pub fn extend_tokens(&mut self, tokens: &[u32]) -> Result<(), anyhow::Error> {
        let tokens = dynamo_tokens::Tokens::from(tokens.to_vec());
        self.token_sequence
            .extend(tokens)
            .map_err(|e| anyhow::anyhow!("Failed to extend tokens: {}", e))?;
        Ok(())
    }

    /// Get the number of complete blocks in the token sequence.
    ///
    /// A block is complete when it has exactly `block_size` tokens.
    /// Partial blocks (with fewer tokens) are not counted.
    pub fn num_complete_blocks(&self) -> usize {
        self.token_sequence.blocks().len()
    }

    /// Get complete TokenBlocks starting from the given index.
    ///
    /// This is used to get newly complete blocks that need to be registered.
    ///
    /// # Arguments
    /// * `start_idx` - Starting block index (typically the number of already-registered blocks)
    ///
    /// # Returns
    /// Vector of TokenBlocks from `start_idx` to the end of complete blocks
    pub fn get_token_blocks(&self, start_idx: usize) -> Vec<TokenBlock> {
        self.token_sequence.blocks()[start_idx..].to_vec()
    }

    // =========================================================================
    // Prefix caching methods
    // =========================================================================

    /// Get sequence hashes for all complete blocks in this request.
    ///
    /// These hashes are used for prefix cache lookup in G1 (and potentially
    /// G2/G3 via the connector).
    ///
    /// # Returns
    /// Vector of sequence hashes, one per complete block in the token sequence.
    /// The hashes are computed using `kvbm_sequence_hash()` which includes
    /// position information to differentiate blocks at different positions.
    pub fn get_sequence_hashes(&self) -> Vec<crate::v2::SequenceHash> {
        self.token_sequence
            .blocks()
            .iter()
            .map(|b| b.kvbm_sequence_hash())
            .collect()
    }

    /// Get all tokens (prompt + output) for resumption.
    ///
    /// Returns the full token sequence for resumed requests that need
    /// to send all tokens to workers. This uses `TokenBlockSequence::tokens_at()`
    /// to efficiently retrieve all tokens.
    pub fn all_tokens_for_resume(&self) -> Vec<u32> {
        let total = self.token_sequence.total_tokens();
        self.token_sequence.tokens_at(0..total).into()
    }

    /// Set the number of cached tokens found in prefix cache.
    ///
    /// # Arguments
    /// * `num_tokens` - Number of tokens found in local G1 prefix cache
    pub fn set_num_cached_tokens(&mut self, num_tokens: usize) {
        self.num_cached_tokens = num_tokens as isize;
    }

    /// Check if prefix cache has been checked for this request.
    pub fn has_checked_prefix_cache(&self) -> bool {
        self.num_cached_tokens >= 0
    }

    // =========================================================================
    // Onboarding status methods
    // =========================================================================

    /// Get the current onboarding status.
    pub fn onboarding_status(&self) -> OnboardingStatus {
        self.onboarding_status
    }

    /// Set the onboarding status.
    pub fn set_onboarding_status(&mut self, status: OnboardingStatus) {
        self.onboarding_status = status;
    }

    /// Check if onboarding has completed.
    pub fn is_onboarding_complete(&self) -> bool {
        self.onboarding_status == OnboardingStatus::Complete
    }
}

impl std::fmt::Debug for SchedulerRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRequest")
            .field("request_id", &self.request.request_id)
            .field("status", &self.status)
            .field("block_state", &self.block_state)
            .field("token_sequence_blocks", &self.token_sequence.blocks().len())
            .field("token_sequence_tokens", &self.token_sequence.total_tokens())
            .field("num_computed_tokens", &self.num_computed_tokens)
            .field("num_output_tokens", &self.num_output_tokens)
            .field("num_cached_tokens", &self.num_cached_tokens)
            .field("resumed_from_preemption", &self.resumed_from_preemption)
            .finish()
    }
}
