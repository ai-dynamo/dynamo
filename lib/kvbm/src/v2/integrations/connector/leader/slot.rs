// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::ops::Range;

use dynamo_tokens::TokenBlockSequence;

use super::Request;
use crate::distributed::leader::FindMatchesResult;
use crate::distributed::offload::TransferHandle;
use crate::v2::{BlockId, SequenceHash};

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during state transitions.
#[derive(Debug, Clone, thiserror::Error)]
pub enum StateTransitionError {
    #[error("Invalid transition from {from} to {to}")]
    InvalidTransition {
        from: &'static str,
        to: &'static str,
    },
    #[error("Slot is marked for deletion; no new transactions allowed")]
    MarkedForDeletion,
}

// ============================================================================
// State Data Structs
// ============================================================================

/// Data associated with onboarding operations (both PreparingToOnboard and Onboarding states).
///
/// This struct holds all the state needed for finding and loading external KV cache blocks.
/// The `session_id` is `None` while preparing (searching/staging) and becomes `Some` when
/// actively onboarding.
pub struct OnboardingState {
    /// The active find session for discovering external blocks.
    pub find_session: FindMatchesResult,
}

/// Data associated with offloading operations.
///
/// This struct holds all the state needed for offloading KV cache blocks to remote storage.
/// The presence of this state indicates we're in the scheduler-output-driven phase,
/// not necessarily actively transferring.
#[derive(Debug, Default)]
pub struct OffloadingState {
    /// Mapping from external BlockId (from vLLM) to SequenceHash for blocks we've processed.
    /// The count of entries indicates how many blocks have been mapped; the next token_block
    /// index to evaluate is `block_mappings.len()`.
    pub block_mappings: HashMap<BlockId, SequenceHash>,

    /// Transfer handles for inflight offload operations.
    /// We keep appending handles and don't evaluate completion until request_finished.
    pub handles: Vec<TransferHandle>,
}

/// Wrapper enum for state data that can be recovered from error states.
///
/// When a transaction enters the `Error` state, the original state data is preserved
/// in this enum so that cleanup/recovery operations can access it.
pub enum ActiveStateData {
    Onboarding(OnboardingState),
    Offloading(OffloadingState),
}

// ============================================================================
// Transaction State Enum (with embedded data)
// ============================================================================

/// The current state of a transaction being issued on behalf of a request.
///
/// This enum uses associated data to ensure that state-specific information is only
/// accessible when in the appropriate state, preventing invalid access patterns.
pub enum TransactionState {
    /// No active onboarding or offloading.
    Inactive,

    /// The slot is preparing to onboard blocks from remote storage.
    /// This state is active while searching for and staging blocks.
    PreparingToOnboard(OnboardingState),

    /// The slot is actively onboarding blocks from remote to worker memory.
    Onboarding(OnboardingState),

    /// The slot is actively offloading blocks from worker memory to remote storage.
    Offloading(OffloadingState),

    /// An error occurred during transaction processing.
    /// The original state data is preserved for recovery/debugging.
    Error(ActiveStateData),
}

impl TransactionState {
    /// Returns the name of the current state for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            TransactionState::Inactive => "Inactive",
            TransactionState::PreparingToOnboard(_) => "PreparingToOnboard",
            TransactionState::Onboarding(_) => "Onboarding",
            TransactionState::Offloading(_) => "Offloading",
            TransactionState::Error(_) => "Error",
        }
    }

    /// Returns true if the state is `Inactive`.
    pub fn is_inactive(&self) -> bool {
        matches!(self, TransactionState::Inactive)
    }
}

// ============================================================================
// Slot Lifecycle State
// ============================================================================

/// Lifecycle state of the slot itself (separate from transaction state).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotState {
    /// The slot is active and can be used to process transactions.
    Active,

    /// The slot is marked for deletion but waiting for outstanding transactions.
    /// No new transactions will be accepted in this state.
    MarkedForDeletion,

    /// Indicates that the workers should be notified to finish the request.
    /// This is the final state before the slot is removed.
    NotifyWorkersToFinish,

    /// Workers have been notified, each will report when it has finished using
    /// any resources that were allocated to the request.
    ///
    /// Request IDs arriving at the [`super::ConnectorLeader::update_connector_output`]
    /// expect the slot to be in this state.
    #[allow(dead_code)] // Used when offloading is fully implemented
    AwaitCompletion,
}

// ============================================================================
// Private State Machine
// ============================================================================

/// Private state machine that encapsulates all state-related logic.
///
/// This struct enforces valid state transitions through its API and prevents
/// direct field access from outside this module.
struct SlotStateMachine {
    slot_state: SlotState,
    txn_state: TransactionState,
}

impl SlotStateMachine {
    /// Create a new state machine in the initial state.
    fn new() -> Self {
        Self {
            slot_state: SlotState::Active,
            txn_state: TransactionState::Inactive,
        }
    }

    /// Get the current transaction state.
    fn txn_state(&self) -> &TransactionState {
        &self.txn_state
    }

    /// Check if the slot is marked for deletion (not Active).
    fn is_marked_for_deletion(&self) -> bool {
        !matches!(self.slot_state, SlotState::Active)
    }

    // ------------------------------------------------------------------------
    // Private helper: txn_to_inactive
    // ------------------------------------------------------------------------

    /// Internal helper to transition to Inactive state.
    ///
    /// This handles the side effect of transitioning `slot_state` from
    /// `MarkedForDeletion` to `NotifyWorkersToFinish` when appropriate.
    fn txn_to_inactive(&mut self) {
        self.txn_state = TransactionState::Inactive;
        if matches!(self.slot_state, SlotState::MarkedForDeletion) {
            self.slot_state = SlotState::NotifyWorkersToFinish;
        }
    }

    // ------------------------------------------------------------------------
    // Transaction State Methods (txn_*)
    // ------------------------------------------------------------------------

    /// Begin preparing to onboard blocks from remote storage.
    ///
    /// Only valid from `Inactive` state. Takes ownership of the onboarding state.
    fn txn_prepare_to_onboard(
        &mut self,
        state: OnboardingState,
    ) -> Result<(), StateTransitionError> {
        if self.is_marked_for_deletion() {
            return Err(StateTransitionError::MarkedForDeletion);
        }

        match &self.txn_state {
            TransactionState::Inactive => {
                self.txn_state = TransactionState::PreparingToOnboard(state);
                Ok(())
            }
            other => Err(StateTransitionError::InvalidTransition {
                from: other.name(),
                to: "PreparingToOnboard",
            }),
        }
    }

    /// Transition from preparing to actively onboarding.
    ///
    /// Only valid from `PreparingToOnboard` state. The existing state data is moved.
    fn txn_start_onboarding(&mut self) -> Result<(), StateTransitionError> {
        if self.is_marked_for_deletion() {
            return Err(StateTransitionError::MarkedForDeletion);
        }

        let current = std::mem::replace(&mut self.txn_state, TransactionState::Inactive);

        match current {
            TransactionState::PreparingToOnboard(state) => {
                self.txn_state = TransactionState::Onboarding(state);
                Ok(())
            }
            other => {
                // Restore the state if transition is invalid
                self.txn_state = other;
                Err(StateTransitionError::InvalidTransition {
                    from: self.txn_state.name(),
                    to: "Onboarding",
                })
            }
        }
    }

    /// Take the onboarding state, transitioning to Inactive.
    ///
    /// Only valid from `Onboarding` state. Returns the state data to the caller.
    fn txn_take_onboarding(&mut self) -> Result<OnboardingState, StateTransitionError> {
        let current = std::mem::replace(&mut self.txn_state, TransactionState::Inactive);

        match current {
            TransactionState::Onboarding(state) => {
                self.txn_to_inactive();
                Ok(state)
            }
            other => {
                self.txn_state = other;
                Err(StateTransitionError::InvalidTransition {
                    from: self.txn_state.name(),
                    to: "Inactive (via take_onboarding)",
                })
            }
        }
    }

    /// Begin offloading blocks to remote storage.
    ///
    /// Only valid from `Inactive` state. Takes ownership of the offloading state.
    fn txn_start_offloading(&mut self, state: OffloadingState) -> Result<(), StateTransitionError> {
        if self.is_marked_for_deletion() {
            return Err(StateTransitionError::MarkedForDeletion);
        }

        match &self.txn_state {
            TransactionState::Inactive => {
                self.txn_state = TransactionState::Offloading(state);
                Ok(())
            }
            other => Err(StateTransitionError::InvalidTransition {
                from: other.name(),
                to: "Offloading",
            }),
        }
    }

    /// Take the offloading state, transitioning to Inactive.
    ///
    /// Only valid from `Offloading` state. Returns the state data to the caller.
    fn txn_take_offloading(&mut self) -> Result<OffloadingState, StateTransitionError> {
        let current = std::mem::replace(&mut self.txn_state, TransactionState::Inactive);

        match current {
            TransactionState::Offloading(state) => {
                self.txn_to_inactive();
                Ok(state)
            }
            other => {
                self.txn_state = other;
                Err(StateTransitionError::InvalidTransition {
                    from: self.txn_state.name(),
                    to: "Inactive (via take_offloading)",
                })
            }
        }
    }

    /// Transition to error state, preserving the current state data.
    ///
    /// Valid from any state with associated data. For Inactive state, this is a no-op
    /// since there's no data to preserve.
    fn txn_to_error(&mut self) {
        let current = std::mem::replace(&mut self.txn_state, TransactionState::Inactive);

        match current {
            TransactionState::Inactive => {
                // No data to preserve; stay in a recoverable state
                // We could also transition to Error with no data, but Inactive is cleaner
            }
            TransactionState::PreparingToOnboard(state) => {
                self.txn_state = TransactionState::Error(ActiveStateData::Onboarding(state));
            }
            TransactionState::Onboarding(state) => {
                self.txn_state = TransactionState::Error(ActiveStateData::Onboarding(state));
            }
            TransactionState::Offloading(state) => {
                self.txn_state = TransactionState::Error(ActiveStateData::Offloading(state));
            }
            TransactionState::Error(data) => {
                // Already in error state; restore it
                self.txn_state = TransactionState::Error(data);
            }
        }
    }

    /// Take the error state data, transitioning to Inactive.
    ///
    /// Only valid from `Error` state. Returns the preserved state data to the caller.
    fn txn_take_error(&mut self) -> Result<ActiveStateData, StateTransitionError> {
        let current = std::mem::replace(&mut self.txn_state, TransactionState::Inactive);

        match current {
            TransactionState::Error(data) => {
                self.txn_to_inactive();
                Ok(data)
            }
            other => {
                self.txn_state = other;
                Err(StateTransitionError::InvalidTransition {
                    from: self.txn_state.name(),
                    to: "Inactive (via take_error)",
                })
            }
        }
    }

    // ------------------------------------------------------------------------
    // Slot Lifecycle Methods (slot_*)
    // ------------------------------------------------------------------------

    /// Mark the slot as finished.
    ///
    /// If the transaction is inactive, returns `Finished` indicating the slot can be removed.
    /// Otherwise, returns `Pending` and the slot will be cleaned up when the transaction completes.
    fn slot_mark_finished(&mut self) -> FinishedStatus {
        if self.slot_state == SlotState::Active {
            self.slot_state = SlotState::MarkedForDeletion;
        }

        if self.txn_state.is_inactive() {
            FinishedStatus::Finished
        } else {
            FinishedStatus::Pending
        }
    }

    #[allow(dead_code)] // Used when offloading is fully implemented
    fn slot_mark_workers_notified(&mut self) {
        self.slot_state = SlotState::AwaitCompletion;
    }

    // ------------------------------------------------------------------------
    // Accessor methods for state-specific data
    // ------------------------------------------------------------------------

    /// Get a reference to the onboarding state if in PreparingToOnboard or Onboarding.
    fn onboarding_state(&self) -> Option<&OnboardingState> {
        match &self.txn_state {
            TransactionState::PreparingToOnboard(state) => Some(state),
            TransactionState::Onboarding(state) => Some(state),
            _ => None,
        }
    }

    /// Get a mutable reference to the onboarding state if in PreparingToOnboard or Onboarding.
    fn onboarding_state_mut(&mut self) -> Option<&mut OnboardingState> {
        match &mut self.txn_state {
            TransactionState::PreparingToOnboard(state) => Some(state),
            TransactionState::Onboarding(state) => Some(state),
            _ => None,
        }
    }

    /// Check if there's an active find session (in PreparingToOnboard or Onboarding).
    fn has_onboarding_state(&self) -> bool {
        self.onboarding_state().is_some()
    }

    /// Get a reference to the offloading state if in Offloading.
    fn offloading_state(&self) -> Option<&OffloadingState> {
        match &self.txn_state {
            TransactionState::Offloading(state) => Some(state),
            _ => None,
        }
    }

    /// Get a mutable reference to the offloading state if in Offloading.
    fn offloading_state_mut(&mut self) -> Option<&mut OffloadingState> {
        match &mut self.txn_state {
            TransactionState::Offloading(state) => Some(state),
            _ => None,
        }
    }

    /// Check if there's an active offloading state.
    #[allow(dead_code)] // Used when offloading is fully implemented
    fn has_offloading_state(&self) -> bool {
        self.offloading_state().is_some()
    }
}

// ============================================================================
// Public Types
// ============================================================================

/// Return value for the [`RequestSlot::slot_mark_finished`] method.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinishedStatus {
    /// The slot is in inactive state, so the request is finished and can be deleted.
    Finished,

    /// The slot has an active transaction; we must await completion.
    Pending,

    /// The request is not tracked by the leader. There is no slot for the request.
    UntrackedRequest,
}

/// Outcome of checking for matched tokens - used as guard pattern
/// to ensure state transitions are handled on all return paths.
pub enum MatchCheckOutcome {
    /// Still searching/staging - stay in PreparingToOnboard
    InProgress,
    /// No match possible (not enough tokens, or search found 0) - transition to Inactive
    NoMatch,
    /// Found matches - transition to Onboarding
    Found { matched_tokens: usize },
}

// ============================================================================
// RequestSlot
// ============================================================================

/// A slot representing an active request with its associated state.
///
/// The state machine is private and can only be manipulated through validated methods.
pub struct RequestSlot {
    request: Request,

    /// The sequence of tokens organized by blocks. This will grow as tokens are decoded.
    pub(crate) sequence: TokenBlockSequence,

    pub(crate) block_matches: BlockAssignments,

    /// Private state machine - not directly accessible.
    state: SlotStateMachine,

    /// The number of token blocks that have been evaluated by our offloading policies.
    evaluated_tokens: usize,

    /// Whether we've stopped evaluating new blocks for offload.
    /// Set when: block_ids exceed token_blocks OR request was paused/resumed/evicted.
    /// This is a slot-level flag that persists across phases.
    finished_evaluating: bool,

    /// If `get_num_new_matched_tokens` is called again, we should reset the state of the slot.
    match_requires_reset: bool,
}

#[derive(Default)]
pub struct BlockAssignments {
    /// The blocks that have been aligned to the sequence.
    pub assigned_blocks: Vec<(SequenceHash, BlockId)>,

    pub unassigned_blocks: Vec<BlockId>,
    // /// The range of blocks with device matches. Matched by the scheduler. Provides by `get_num_new_matched_tokens`.
    // pub device_matches: Option<Range<usize>>,

    // /// The range of blocks with host matches. Matched by the connector, match performed during `get_num_new_matched_tokens`.
    // pub external_matches: Option<Range<usize>>,

    // /// The range of blocks with prefill tokens. Essentially the difference between the ISL and the total matches.
    // pub prefill_tokens: Option<Range<usize>>,

    // /// The range of blocks with decode tokens. Essentially the difference between the ISL and the total matches.
    // /// The tail block with partial prefill is considered to be a decode block.
    // pub decode_tokens: Option<Range<usize>>,
}

impl RequestSlot {
    /// Assign physical block_ids to logical sequence hashes.
    ///
    /// Block IDs are paired with sequence hashes by skipping already assigned logical blocks,
    /// then applying new blocks in the order provided. Any extra block_ids (beyond what can
    /// be paired with sequence hashes) are assigned to `unassigned_blocks`.
    ///
    /// # Returns
    /// The range of indices into `assigned_blocks` for the newly assigned blocks.
    #[tracing::instrument(level = "debug", skip(self), ret)]
    pub fn apply_new_blocks(&mut self, block_ids: Vec<BlockId>) -> Range<usize> {
        tracing::debug!(
            "applying {} new blocks; assigned_blocks_count: {}; unassigned_blocks_count: {}; token_block_count: {}",
            block_ids.len(),
            self.block_matches.assigned_blocks.len(),
            self.block_matches.unassigned_blocks.len(),
            self.sequence.blocks().len()
        );
        let start_idx = self.block_matches.assigned_blocks.len();

        // first apply unassigned blocks
        self.block_matches.unassigned_blocks.extend(block_ids);
        let block_ids = std::mem::take(&mut self.block_matches.unassigned_blocks);

        let mut block_ids = block_ids;
        let mut drain = block_ids.drain(0..);
        let newly_assigned_blocks = self
            .sequence
            .blocks()
            .iter()
            .skip(start_idx)
            .zip(&mut drain)
            .map(|(b, id)| (b.positional_sequence_hash(), id))
            .collect::<Vec<_>>();

        self.block_matches
            .assigned_blocks
            .extend(newly_assigned_blocks);
        self.block_matches.unassigned_blocks.extend(drain);

        let end_idx = self.block_matches.assigned_blocks.len();

        tracing::debug!(
            "after applying new blocks: assigned_blocks_count: {}; unassigned_blocks_count: {}; token_block_count: {}",
            self.block_matches.assigned_blocks.len(),
            self.block_matches.unassigned_blocks.len(),
            self.sequence.blocks().len()
        );

        start_idx..end_idx
    }

    /// Filter the block_ids to only include those that are not already known (assigned or unassigned).
    ///
    /// It is expected that `all_block_ids` is in order and at the first miss, all remaining block_ids
    /// should be returned. This will be validated.
    ///
    /// The method validates that the prefix of `all_block_ids` matches:
    /// 1. First, the already assigned block IDs (in order)
    /// 2. Then, the unassigned block IDs (in order)
    ///
    /// If there's a mismatch, this indicates a bug and will panic.
    ///
    /// # Arguments
    /// * `all_block_ids` - The complete list of block IDs from the scheduler, in order.
    ///
    /// # Returns
    /// The block IDs that are not yet known (the suffix after assigned + unassigned).
    ///
    /// # Panics
    /// Panics if the prefix of `all_block_ids` doesn't match the assigned and unassigned block IDs in order.
    pub fn filter_block_ids(&self, all_block_ids: Vec<BlockId>) -> Vec<BlockId> {
        let num_assigned = self.block_matches.assigned_blocks.len();
        let num_unassigned = self.block_matches.unassigned_blocks.len();
        let num_known = num_assigned + num_unassigned;

        // If no blocks are known, return all block_ids
        if num_known == 0 {
            return all_block_ids;
        }

        // Validate that we have enough block_ids
        assert!(
            all_block_ids.len() >= num_known,
            "all_block_ids length ({}) is less than number of known blocks (assigned={} + unassigned={})",
            all_block_ids.len(),
            num_assigned,
            num_unassigned
        );

        // Validate that the prefix matches assigned blocks
        for (i, ((_hash, assigned_id), provided_id)) in self
            .block_matches
            .assigned_blocks
            .iter()
            .zip(all_block_ids.iter())
            .enumerate()
        {
            assert_eq!(
                *assigned_id, *provided_id,
                "Assigned block ID mismatch at index {}: assigned={}, provided={}",
                i, assigned_id, provided_id
            );
        }

        // Validate that the next portion matches unassigned blocks
        for (i, (unassigned_id, provided_id)) in self
            .block_matches
            .unassigned_blocks
            .iter()
            .zip(all_block_ids.iter().skip(num_assigned))
            .enumerate()
        {
            assert_eq!(
                *unassigned_id, *provided_id,
                "Unassigned block ID mismatch at index {}: unassigned={}, provided={}",
                i, unassigned_id, provided_id
            );
        }

        // Return the suffix (block_ids not yet known)
        all_block_ids.into_iter().skip(num_known).collect()
    }

    pub fn get_next_block_mappings(
        &self,
        num_scheduled_tokens: usize,
    ) -> Vec<(BlockId, SequenceHash)> {
        let evaluated_blocks = self.evaluated_blocks();
        let num_blocks_after_evaluation =
            (self.evaluated_tokens + num_scheduled_tokens) / self.block_size();
        let new_blocks_to_evaluate = num_blocks_after_evaluation - evaluated_blocks;

        self.block_matches
            .assigned_blocks
            .iter()
            .skip(evaluated_blocks)
            .take(new_blocks_to_evaluate)
            .map(|(hash, block_id)| (*block_id, *hash))
            .collect::<Vec<_>>()
    }
}

impl RequestSlot {
    /// Create a new RequestSlot for the given request.
    pub fn new(request: Request, block_size: usize) -> Result<Self, anyhow::Error> {
        let sequence = TokenBlockSequence::new(
            request.tokens.clone(),
            block_size as u32,
            Some(request.salt_hash),
        );
        Ok(Self {
            request,
            sequence,
            block_matches: BlockAssignments::default(),
            state: SlotStateMachine::new(),
            evaluated_tokens: 0,
            finished_evaluating: false,
            match_requires_reset: false,
        })
    }

    // ------------------------------------------------------------------------
    // Basic accessors
    // ------------------------------------------------------------------------

    pub fn request_id(&self) -> &str {
        &self.request.request_id
    }

    /// Get the current transaction state (read-only).
    pub fn txn_state(&self) -> &TransactionState {
        self.state.txn_state()
    }

    /// Check if the slot is marked for deletion.
    pub fn is_marked_for_deletion(&self) -> bool {
        self.state.is_marked_for_deletion()
    }

    /// Get the block size from the token sequence.
    pub fn block_size(&self) -> usize {
        self.sequence.block_size()
    }

    pub fn all_sequence_hashes(&self) -> Vec<SequenceHash> {
        self.sequence
            .blocks()
            .iter()
            .map(|b| b.positional_sequence_hash())
            .collect()
    }

    /// Check if we've stopped evaluating new blocks for offload.
    ///
    /// This is set when block_ids exceed token_blocks OR request was paused/resumed/evicted.
    pub fn is_finished_evaluating(&self) -> bool {
        self.finished_evaluating
    }

    /// Mark that we've stopped evaluating new blocks for offload.
    ///
    /// Called when:
    /// - block_ids from vLLM exceed our token knowledge
    /// - request was paused/resumed/evicted
    pub fn mark_finished_evaluating(&mut self) {
        self.finished_evaluating = true;
    }

    pub fn match_requires_reset(&self) -> bool {
        self.match_requires_reset
    }

    pub fn set_match_requires_reset(&mut self, requires_reset: bool) {
        self.match_requires_reset = requires_reset;
    }

    pub fn advance_evaluated_tokens(&mut self, num_tokens: usize) {
        self.evaluated_tokens = self.evaluated_tokens.saturating_add(num_tokens);
    }

    pub fn evaluated_blocks(&self) -> usize {
        self.evaluated_tokens / self.block_size()
    }

    // ------------------------------------------------------------------------
    // Transaction State Methods (txn_*)
    // ------------------------------------------------------------------------

    /// Begin preparing to onboard blocks from remote storage.
    ///
    /// Creates an `OnboardingState` with the given find session and computed tokens count.
    /// Only valid when in `Inactive` state and slot is not marked for deletion.
    pub fn txn_prepare_to_onboard(
        &mut self,
        find_session: FindMatchesResult,
    ) -> Result<(), StateTransitionError> {
        let state = OnboardingState { find_session };
        self.evaluated_tokens = 0;
        self.state.txn_prepare_to_onboard(state)
    }

    /// Transition from PreparingToOnboard to Onboarding.
    ///
    /// Only valid when in `PreparingToOnboard` state.
    pub fn txn_start_onboarding(&mut self) -> Result<(), StateTransitionError> {
        self.state.txn_start_onboarding()
    }

    /// Take the onboarding state, transitioning to Inactive.
    ///
    /// Only valid when in `Onboarding` state.
    /// Returns the `OnboardingState` containing the session ID and find session.
    pub fn txn_take_onboarding(&mut self) -> Result<OnboardingState, StateTransitionError> {
        self.state.txn_take_onboarding()
    }

    /// Begin offloading blocks to remote storage.
    ///
    /// Only valid when in `Inactive` state and slot is not marked for deletion.
    pub fn txn_start_offloading(&mut self) -> Result<(), StateTransitionError> {
        self.state.txn_start_offloading(OffloadingState::default())
    }

    /// Take the offloading state, transitioning to Inactive.
    ///
    /// Only valid when in `Offloading` state.
    /// Returns the `OffloadingState` containing the session ID.
    pub fn txn_take_offloading(&mut self) -> Result<OffloadingState, StateTransitionError> {
        self.state.txn_take_offloading()
    }

    /// Transition to error state, preserving current state data for recovery.
    pub fn txn_to_error(&mut self) {
        self.state.txn_to_error()
    }

    /// Take the error state data, transitioning to Inactive.
    ///
    /// Only valid when in `Error` state.
    /// Returns the preserved state data for cleanup.
    pub fn txn_take_error(&mut self) -> Result<ActiveStateData, StateTransitionError> {
        self.state.txn_take_error()
    }

    // ------------------------------------------------------------------------
    // Slot Lifecycle Methods (slot_*)
    // ------------------------------------------------------------------------

    /// Mark the slot as finished.
    ///
    /// Returns `Finished` if the slot can be immediately removed,
    /// or `Pending` if we must wait for an active transaction to complete.
    pub fn slot_mark_finished(&mut self) -> FinishedStatus {
        self.state.slot_mark_finished()
    }

    // ------------------------------------------------------------------------
    // Find Session Accessors
    // ------------------------------------------------------------------------

    /// Check if there's an active find session.
    pub fn has_onboarding_state(&self) -> bool {
        self.state.has_onboarding_state()
    }

    /// Get a reference to the find session, if in PreparingToOnboard or Onboarding.
    pub fn onboarding_state(&self) -> Option<&OnboardingState> {
        self.state.onboarding_state()
    }

    pub fn onboarding_state_mut(&mut self) -> Option<&mut OnboardingState> {
        self.state.onboarding_state_mut()
    }

    pub fn offloading_state(&self) -> Option<&OffloadingState> {
        self.state.offloading_state()
    }

    pub fn offloading_state_mut(&mut self) -> Option<&mut OffloadingState> {
        self.state.offloading_state_mut()
    }

    pub fn get_or_create_offloading_state(&mut self) -> &mut OffloadingState {
        if self.state.offloading_state().is_none() {
            self.state
                .txn_start_offloading(OffloadingState::default())
                .expect("get_or_create_offloading_state called in invalid state");
        }

        self.state
            .offloading_state_mut()
            .expect("offloading state must exist after creation")
    }

    /// Finalize a match check by transitioning state and returning the vLLM-compatible tuple.
    ///
    /// This is the single exit point for state transitions from PreparingToOnboard.
    /// The connector is best effort; errors result in continuing without external KV cache.
    pub fn finalize_match_check(
        &mut self,
        outcome: Result<MatchCheckOutcome, anyhow::Error>,
    ) -> Result<(Option<usize>, bool), anyhow::Error> {
        // Verify we're in the expected state
        // if !matches!(self.txn_state(), TransactionState::PreparingToOnboard(_)) {
        //     return Err(anyhow::anyhow!(
        //         "finalize_match_check called in unexpected state: {}",
        //         self.txn_state().name()
        //     ));
        // }

        match outcome {
            Ok(MatchCheckOutcome::InProgress) => {
                // Stay in PreparingToOnboard
                Ok((None, false))
            }
            Ok(MatchCheckOutcome::NoMatch) => {
                // Take the state and discard it (transition to Inactive)
                let _ = self.state.txn_take_onboarding();
                // Note: txn_take_onboarding expects Onboarding state, so we need a different approach
                // We need to directly transition from PreparingToOnboard to Inactive
                self.state.txn_to_inactive();
                Ok((Some(0), false))
            }
            Ok(MatchCheckOutcome::Found { matched_tokens }) => {
                if matched_tokens > 0 {
                    tracing::debug!(
                        "Found {} matched tokens for request ID: {}",
                        matched_tokens,
                        self.request_id()
                    );
                } else {
                    // No matches - go back to Inactive
                    self.state.txn_to_inactive();
                }
                Ok((Some(matched_tokens), matched_tokens > 0))
            }
            Err(e) => {
                tracing::warn!("Error processing match check: {}", e);
                tracing::warn!(
                    "{}",
                    concat!(
                        "This will not effect the ability to process the request; however, it may ",
                        "indicate some logical errors and possible mismatched version of the connector and the application."
                    )
                );
                self.state.txn_to_error();
                Err(e)
            }
        }
    }

    // ------------------------------------------------------------------------
    // Offloading Methods
    // ------------------------------------------------------------------------

    /// Record block mappings and store transfer handle for offloading.
    ///
    /// # Arguments
    /// * `block_mappings` - Pairs of (BlockId, SequenceHash) to record
    /// * `handle` - The transfer handle for this offload batch
    pub fn record_offload(
        &mut self,
        block_mappings: Vec<(BlockId, SequenceHash)>,
        handle: TransferHandle,
    ) -> Result<(), anyhow::Error> {
        // The state must be active, not marked for deletion, and the txn_state
        // must be inactive or offloading
        if self.is_marked_for_deletion() {
            return Err(anyhow::anyhow!("Slot is marked for deletion"));
        }

        if !matches!(
            self.txn_state(),
            TransactionState::Inactive | TransactionState::Offloading(_)
        ) {
            return Err(anyhow::anyhow!("Invalid transaction state"));
        }

        // Create or get the offloading state
        let offloading_state = self.get_or_create_offloading_state();

        // Add all block mappings
        for (block_id, sequence_hash) in block_mappings {
            offloading_state
                .block_mappings
                .insert(block_id, sequence_hash);
        }

        // Store the transfer handle
        offloading_state.handles.push(handle);

        Ok(())
    }

    /// Get the number of blocks that have been mapped for offloading.
    ///
    /// This indicates the next token_block index to evaluate.
    pub fn mapped_block_count(&self) -> usize {
        self.state
            .offloading_state()
            .map(|s| s.block_mappings.len())
            .unwrap_or(0)
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(test)]
    mod apply_new_blocks_tests {
        use super::*;

        const TEST_BLOCK_SIZE: usize = 4;

        /// Helper to create a RequestSlot with a given number of complete blocks and optional partial.
        fn create_test_slot(num_complete_blocks: usize, partial_tokens: usize) -> RequestSlot {
            let total_tokens = num_complete_blocks * TEST_BLOCK_SIZE + partial_tokens;
            let tokens: Vec<u32> = (0..total_tokens as u32).collect();

            let request = Request::new(
                "test-request",
                tokens,
                None, // lora_name
                None, // salt
                None, // max_tokens
            );

            RequestSlot::new(request, TEST_BLOCK_SIZE).expect("Failed to create RequestSlot")
        }

        /// Helper to get the expected sequence hashes from a slot.
        fn get_expected_hashes(slot: &RequestSlot) -> Vec<SequenceHash> {
            slot.sequence
                .blocks()
                .iter()
                .map(|b| b.positional_sequence_hash())
                .collect()
        }

        // =========================================================================
        // Test Cases: Aligned sequences (no partial block)
        // =========================================================================

        #[test]
        fn test_aligned_0_blocks_0_block_ids() {
            // 0 complete blocks, 0 block_ids
            let mut slot = create_test_slot(0, 0);
            let block_ids: Vec<BlockId> = vec![];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..0);
            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_aligned_1_block_0_block_ids() {
            // 1 complete block, 0 block_ids
            let mut slot = create_test_slot(1, 0);
            let block_ids: Vec<BlockId> = vec![];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..0);
            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_aligned_1_block_1_block_id() {
            // 1 complete block, 1 block_id - exact match
            let mut slot = create_test_slot(1, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![100];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..1);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 1);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_aligned_1_block_2_block_ids() {
            // 1 complete block, 2 block_ids - excess goes to unassigned
            let mut slot = create_test_slot(1, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![100, 200];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..1);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 1);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(slot.block_matches.unassigned_blocks, vec![200]);
        }

        #[test]
        fn test_aligned_3_blocks_3_block_ids() {
            // 3 complete blocks, 3 block_ids - exact match
            let mut slot = create_test_slot(3, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![100, 200, 300];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..3);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[2],
                (expected_hashes[2], 300)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_aligned_3_blocks_1_block_id() {
            // 3 complete blocks, 1 block_id - partial assignment
            let mut slot = create_test_slot(3, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![100];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..1);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 1);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_aligned_3_blocks_5_block_ids() {
            // 3 complete blocks, 5 block_ids - excess goes to unassigned
            let mut slot = create_test_slot(3, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![100, 200, 300, 400, 500];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..3);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[2],
                (expected_hashes[2], 300)
            );
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400, 500]);
        }

        // =========================================================================
        // Test Cases: Sequences with partial (dangling) block
        // =========================================================================

        #[test]
        fn test_partial_0_complete_2_partial_0_block_ids() {
            // 0 complete blocks + 2 partial tokens, 0 block_ids
            // TokenBlockSequence only counts complete blocks
            let mut slot = create_test_slot(0, 2);
            let block_ids: Vec<BlockId> = vec![];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..0);
            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_partial_2_complete_1_partial_2_block_ids() {
            // 2 complete blocks + 1 partial token, 2 block_ids - exact match for complete blocks
            let mut slot = create_test_slot(2, 1);
            let expected_hashes = get_expected_hashes(&slot);
            assert_eq!(expected_hashes.len(), 2); // Only complete blocks have hashes
            let block_ids: Vec<BlockId> = vec![100, 200];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_partial_2_complete_3_partial_4_block_ids() {
            // 2 complete blocks + 3 partial tokens, 4 block_ids - excess goes to unassigned
            let mut slot = create_test_slot(2, 3);
            let expected_hashes = get_expected_hashes(&slot);
            assert_eq!(expected_hashes.len(), 2);
            let block_ids: Vec<BlockId> = vec![100, 200, 300, 400];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);
        }

        #[test]
        fn test_partial_3_complete_2_partial_1_block_id() {
            // 3 complete blocks + 2 partial tokens, 1 block_id - partial assignment
            let mut slot = create_test_slot(3, 2);
            let expected_hashes = get_expected_hashes(&slot);
            assert_eq!(expected_hashes.len(), 3);
            let block_ids: Vec<BlockId> = vec![100];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..1);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 1);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        // =========================================================================
        // Test Cases: Multiple calls to apply_new_blocks (incremental assignment)
        // =========================================================================

        #[test]
        fn test_incremental_assignment_aligned() {
            // 4 complete blocks, apply 2 block_ids, then 2 more
            let mut slot = create_test_slot(4, 0);
            let expected_hashes = get_expected_hashes(&slot);

            // First call: assign first 2 blocks
            let block_ids_1: Vec<BlockId> = vec![100, 200];
            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(
                slot.block_matches.assigned_blocks[0],
                (expected_hashes[0], 100)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );

            // Second call: assign next 2 blocks
            let block_ids_2: Vec<BlockId> = vec![300, 400];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range_2, 2..4);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 4);
            assert_eq!(
                slot.block_matches.assigned_blocks[2],
                (expected_hashes[2], 300)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[3],
                (expected_hashes[3], 400)
            );
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_incremental_assignment_with_excess() {
            // 3 complete blocks, apply 2 block_ids, then 3 more (1 excess)
            let mut slot = create_test_slot(3, 0);
            let expected_hashes = get_expected_hashes(&slot);

            // First call: assign first 2 blocks
            let block_ids_1: Vec<BlockId> = vec![100, 200];
            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);

            // Second call: try to assign 3 more, but only 1 block remaining
            let block_ids_2: Vec<BlockId> = vec![300, 400, 500];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range_2, 2..3);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert_eq!(
                slot.block_matches.assigned_blocks[2],
                (expected_hashes[2], 300)
            );
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400, 500]);
        }

        #[test]
        fn test_incremental_assignment_partial_then_excess() {
            // 2 complete + 1 partial, apply 1, then 3 (2 excess)
            let mut slot = create_test_slot(2, 1);
            let expected_hashes = get_expected_hashes(&slot);
            assert_eq!(expected_hashes.len(), 2);

            // First call: assign 1 block
            let block_ids_1: Vec<BlockId> = vec![100];
            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..1);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 1);

            // Second call: assign 3 more, but only 1 complete block remaining
            let block_ids_2: Vec<BlockId> = vec![200, 300, 400];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range_2, 1..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(
                slot.block_matches.assigned_blocks[1],
                (expected_hashes[1], 200)
            );
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);
        }

        #[test]
        fn test_all_blocks_already_assigned_extra_goes_to_unassigned() {
            // 2 complete blocks, assign both, then try to add more
            let mut slot = create_test_slot(2, 0);
            let _expected_hashes = get_expected_hashes(&slot);

            // First call: assign all blocks
            let block_ids_1: Vec<BlockId> = vec![100, 200];
            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);

            // Second call: all go to unassigned since all blocks are assigned
            let block_ids_2: Vec<BlockId> = vec![300, 400];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range_2, 2..2); // Empty range - no new assignments
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);
        }

        // =========================================================================
        // Test Cases: Edge cases
        // =========================================================================

        #[test]
        fn test_empty_slot_receives_block_ids() {
            // 0 blocks, but receive block_ids - all go to unassigned
            let mut slot = create_test_slot(0, 0);
            let block_ids: Vec<BlockId> = vec![100, 200, 300];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..0);
            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert_eq!(slot.block_matches.unassigned_blocks, vec![100, 200, 300]);
        }

        #[test]
        fn test_only_partial_tokens_receives_block_ids() {
            // Only partial tokens (no complete blocks), receive block_ids
            let mut slot = create_test_slot(0, 3); // 3 tokens, block_size=4, so no complete block
            let block_ids: Vec<BlockId> = vec![100, 200];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..0);
            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert_eq!(slot.block_matches.unassigned_blocks, vec![100, 200]);
        }

        #[test]
        fn test_large_sequence_exact_match() {
            // 10 complete blocks, 10 block_ids
            let mut slot = create_test_slot(10, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = (0..10).map(|i| (i + 1) * 100).collect();

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..10);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 10);
            for i in 0..10 {
                assert_eq!(
                    slot.block_matches.assigned_blocks[i],
                    (expected_hashes[i], (i + 1) * 100)
                );
            }
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_verify_hash_block_id_pairing_order() {
            // Verify that hashes and block_ids are paired in correct order
            let mut slot = create_test_slot(5, 0);
            let expected_hashes = get_expected_hashes(&slot);
            let block_ids: Vec<BlockId> = vec![999, 888, 777, 666, 555];

            let range = slot.apply_new_blocks(block_ids);

            assert_eq!(range, 0..5);
            // Verify each (hash, block_id) pair is in the correct order
            assert_eq!(slot.block_matches.assigned_blocks[0].1, 999);
            assert_eq!(slot.block_matches.assigned_blocks[1].1, 888);
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 777);
            assert_eq!(slot.block_matches.assigned_blocks[3].1, 666);
            assert_eq!(slot.block_matches.assigned_blocks[4].1, 555);

            // And hashes match expected sequence order
            for i in 0..5 {
                assert_eq!(slot.block_matches.assigned_blocks[i].0, expected_hashes[i]);
            }
        }

        // =========================================================================
        // Cartesian product test: various (num_blocks, partial_tokens, num_block_ids)
        // =========================================================================

        #[test]
        fn test_cartesian_product_combinations() {
            // Test matrix:
            // num_complete_blocks: [0, 1, 3, 5]
            // partial_tokens: [0, 1, 3] (3 is block_size-1)
            // num_block_ids: [0, fewer, exact, more]

            let num_blocks_options = [0, 1, 3, 5];
            let partial_options = [0, 1, 3];

            for &num_blocks in &num_blocks_options {
                for &partial in &partial_options {
                    let slot = create_test_slot(num_blocks, partial);
                    let expected_hashes = get_expected_hashes(&slot);
                    let available_blocks = expected_hashes.len();

                    // Test with 0 block_ids
                    {
                        let mut slot = create_test_slot(num_blocks, partial);
                        let range = slot.apply_new_blocks(vec![]);
                        assert_eq!(range, 0..0);
                        assert!(slot.block_matches.assigned_blocks.is_empty());
                        assert!(slot.block_matches.unassigned_blocks.is_empty());
                    }

                    // Test with fewer block_ids than available blocks (if available > 0)
                    if available_blocks > 1 {
                        let mut slot = create_test_slot(num_blocks, partial);
                        let expected_hashes = get_expected_hashes(&slot);
                        let fewer = available_blocks / 2;
                        let block_ids: Vec<BlockId> = (0..fewer).collect();
                        let range = slot.apply_new_blocks(block_ids);

                        assert_eq!(range, 0..fewer);
                        assert_eq!(slot.block_matches.assigned_blocks.len(), fewer);
                        assert!(slot.block_matches.unassigned_blocks.is_empty());

                        for i in 0..fewer {
                            assert_eq!(slot.block_matches.assigned_blocks[i].0, expected_hashes[i]);
                            assert_eq!(slot.block_matches.assigned_blocks[i].1, i);
                        }
                    }

                    // Test with exact number of block_ids
                    if available_blocks > 0 {
                        let mut slot = create_test_slot(num_blocks, partial);
                        let expected_hashes = get_expected_hashes(&slot);
                        let block_ids: Vec<BlockId> = (0..available_blocks).collect();
                        let range = slot.apply_new_blocks(block_ids);

                        assert_eq!(range, 0..available_blocks);
                        assert_eq!(slot.block_matches.assigned_blocks.len(), available_blocks);
                        assert!(slot.block_matches.unassigned_blocks.is_empty());

                        for i in 0..available_blocks {
                            assert_eq!(slot.block_matches.assigned_blocks[i].0, expected_hashes[i]);
                            assert_eq!(slot.block_matches.assigned_blocks[i].1, i);
                        }
                    }

                    // Test with more block_ids than available blocks
                    {
                        let mut slot = create_test_slot(num_blocks, partial);
                        let expected_hashes = get_expected_hashes(&slot);
                        let excess = 3;
                        let total_ids = available_blocks + excess;
                        let block_ids: Vec<BlockId> = (0..total_ids).collect();
                        let range = slot.apply_new_blocks(block_ids);

                        assert_eq!(range, 0..available_blocks);
                        assert_eq!(slot.block_matches.assigned_blocks.len(), available_blocks);
                        assert_eq!(slot.block_matches.unassigned_blocks.len(), excess);

                        for i in 0..available_blocks {
                            assert_eq!(slot.block_matches.assigned_blocks[i].0, expected_hashes[i]);
                            assert_eq!(slot.block_matches.assigned_blocks[i].1, i);
                        }

                        let expected_unassigned: Vec<BlockId> =
                            (available_blocks..total_ids).collect();
                        assert_eq!(slot.block_matches.unassigned_blocks, expected_unassigned);
                    }
                }
            }
        }

        // =========================================================================
        // Test Cases: Previously unassigned blocks feature
        // =========================================================================

        #[test]
        fn test_unassigned_blocks_applied_before_new_blocks() {
            // Create slot with 5 blocks, apply 7 block_ids (2 excess)
            let mut slot = create_test_slot(5, 0);
            let _expected_hashes = get_expected_hashes(&slot);
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400, 500, 600, 700];

            let range_1 = slot.apply_new_blocks(block_ids_1);

            // First 5 should be assigned, 2 unassigned
            assert_eq!(range_1, 0..5);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 5);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![600, 700]);

            // Now add more tokens to create 2 more complete blocks
            let new_tokens: Vec<u32> = (20..28).collect(); // 8 more tokens = 2 blocks
            for token in new_tokens {
                slot.sequence.append(token).unwrap();
            }
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 7); // Now 7 blocks total

            // Apply new blocks - the unassigned blocks (600, 700) should be applied first
            let block_ids_2: Vec<BlockId> = vec![800, 900];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            // Range should be 5..7 (the 2 new blocks that got assigned)
            assert_eq!(range_2, 5..7);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 7);

            // Verify the previously unassigned blocks (600, 700) were assigned to blocks 5, 6
            assert_eq!(
                slot.block_matches.assigned_blocks[5],
                (expected_hashes_after[5], 600)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[6],
                (expected_hashes_after[6], 700)
            );

            // New blocks (800, 900) should be unassigned since there was no room
            assert_eq!(slot.block_matches.unassigned_blocks, vec![800, 900]);
        }

        #[test]
        fn test_unassigned_blocks_with_new_blocks_all_assigned() {
            // Create slot with 3 blocks, apply 4 block_ids (1 excess)
            let mut slot = create_test_slot(3, 0);
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400];

            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..3);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400]);

            // Add 3 more blocks worth of tokens
            for token in 12..24 {
                slot.sequence.append(token).unwrap();
            }
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 6); // Now 6 blocks total

            // Apply 2 new blocks - unassigned block (400) + new blocks should all fit
            let block_ids_2: Vec<BlockId> = vec![500, 600];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            // All 3 blocks (1 old unassigned + 2 new) should be assigned
            assert_eq!(range_2, 3..6);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 6);

            // Verify unassigned block (400) was assigned to block 3
            assert_eq!(
                slot.block_matches.assigned_blocks[3],
                (expected_hashes_after[3], 400)
            );
            // Verify new blocks assigned to blocks 4, 5
            assert_eq!(
                slot.block_matches.assigned_blocks[4],
                (expected_hashes_after[4], 500)
            );
            assert_eq!(
                slot.block_matches.assigned_blocks[5],
                (expected_hashes_after[5], 600)
            );

            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_unassigned_blocks_no_new_space() {
            // Create slot with 2 blocks, apply 4 block_ids (2 excess)
            let mut slot = create_test_slot(2, 0);
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400];

            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // Apply new blocks without adding more token blocks
            let block_ids_2: Vec<BlockId> = vec![500, 600];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            // No new assignments since no new complete blocks
            assert_eq!(range_2, 2..2); // Empty range
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);

            // All blocks (old unassigned + new) should still be unassigned
            assert_eq!(
                slot.block_matches.unassigned_blocks,
                vec![300, 400, 500, 600]
            );
        }

        #[test]
        fn test_unassigned_blocks_partial_space() {
            // Create slot with 3 blocks, apply 5 block_ids (2 excess)
            let mut slot = create_test_slot(3, 0);
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400, 500];

            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..3);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400, 500]);

            // Add 1 more block worth of tokens
            for token in 12..16 {
                slot.sequence.append(token).unwrap();
            }
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 4); // Now 4 blocks total

            // Apply 3 new blocks - only 1 spot available, should take first unassigned
            let block_ids_2: Vec<BlockId> = vec![600, 700, 800];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            // Only 1 block can be assigned (from the 5 total: 2 old unassigned + 3 new)
            assert_eq!(range_2, 3..4);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 4);

            // First old unassigned block (400) should get assigned
            assert_eq!(
                slot.block_matches.assigned_blocks[3],
                (expected_hashes_after[3], 400)
            );

            // Rest should be unassigned in order: second old unassigned, then new ones
            assert_eq!(
                slot.block_matches.unassigned_blocks,
                vec![500, 600, 700, 800]
            );
        }

        #[test]
        fn test_multiple_rounds_of_unassigned_accumulation() {
            // Test that unassigned blocks accumulate correctly over multiple calls
            let mut slot = create_test_slot(2, 0);

            // Round 1: 2 blocks assigned, 2 unassigned
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400];
            let range_1 = slot.apply_new_blocks(block_ids_1);
            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // Round 2: No new space, add 2 more to unassigned
            let block_ids_2: Vec<BlockId> = vec![500, 600];
            let range_2 = slot.apply_new_blocks(block_ids_2);
            assert_eq!(range_2, 2..2); // Empty
            assert_eq!(
                slot.block_matches.unassigned_blocks,
                vec![300, 400, 500, 600]
            );

            // Round 3: Still no space, add 1 more
            let block_ids_3: Vec<BlockId> = vec![700];
            let range_3 = slot.apply_new_blocks(block_ids_3);
            assert_eq!(range_3, 2..2); // Empty
            assert_eq!(
                slot.block_matches.unassigned_blocks,
                vec![300, 400, 500, 600, 700]
            );

            // Now add space for 3 more blocks
            for token in 8..20 {
                slot.sequence.append(token).unwrap();
            }
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 5); // Now 5 blocks total

            // Apply with no new blocks - should assign first 3 from unassigned
            let block_ids_4: Vec<BlockId> = vec![];
            let range_4 = slot.apply_new_blocks(block_ids_4);

            assert_eq!(range_4, 2..5);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 5);

            // First 3 unassigned (300, 400, 500) should be assigned
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 300);
            assert_eq!(slot.block_matches.assigned_blocks[3].1, 400);
            assert_eq!(slot.block_matches.assigned_blocks[4].1, 500);

            // Last 2 should still be unassigned
            assert_eq!(slot.block_matches.unassigned_blocks, vec![600, 700]);
        }

        #[test]
        fn test_unassigned_blocks_ordering_preserved() {
            // Verify that the order of unassigned blocks is preserved (FIFO)
            let mut slot = create_test_slot(1, 0);

            // Create 5 excess blocks
            let block_ids_1: Vec<BlockId> = vec![10, 20, 30, 40, 50, 60];
            slot.apply_new_blocks(block_ids_1);

            // 10 should be assigned, rest unassigned in order
            assert_eq!(slot.block_matches.assigned_blocks[0].1, 10);
            assert_eq!(
                slot.block_matches.unassigned_blocks,
                vec![20, 30, 40, 50, 60]
            );

            // Add 2 more blocks of space
            for token in 4..12 {
                slot.sequence.append(token).unwrap();
            }

            // Apply empty list - should assign first 2 from unassigned (20, 30)
            slot.apply_new_blocks(vec![]);
            assert_eq!(slot.block_matches.assigned_blocks[1].1, 20);
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 30);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![40, 50, 60]);

            // Add 1 new block ID
            for token in 12..16 {
                slot.sequence.append(token).unwrap();
            }

            let block_ids_2: Vec<BlockId> = vec![70];
            slot.apply_new_blocks(block_ids_2);

            // Should assign 40 (first from old unassigned), not 70 (new)
            assert_eq!(slot.block_matches.assigned_blocks[3].1, 40);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![50, 60, 70]);
        }

        #[test]
        fn test_unassigned_blocks_with_partial_token_block() {
            // Test with partial blocks to ensure logic still works
            let mut slot = create_test_slot(2, 2); // 2 complete + 2 partial tokens
            let expected_hashes = get_expected_hashes(&slot);
            assert_eq!(expected_hashes.len(), 2);

            // Apply 4 block_ids - 2 assigned, 2 unassigned
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400];
            let range_1 = slot.apply_new_blocks(block_ids_1);

            assert_eq!(range_1, 0..2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // Add 2 more tokens to complete the partial block
            slot.sequence.append(10).unwrap();
            slot.sequence.append(11).unwrap();
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 3); // Now 3 complete blocks

            // Apply 1 new block - unassigned 300 should be applied first
            let block_ids_2: Vec<BlockId> = vec![500];
            let range_2 = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range_2, 2..3);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 300);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400, 500]);
        }

        #[test]
        fn test_unassigned_blocks_exactly_fill_new_space() {
            // Test when unassigned blocks exactly fill new available space
            let mut slot = create_test_slot(2, 0);

            // Apply 5 block_ids - 2 assigned, 3 unassigned
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300, 400, 500];
            slot.apply_new_blocks(block_ids_1);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400, 500]);

            // Add exactly 3 more blocks of space
            for token in 8..20 {
                slot.sequence.append(token).unwrap();
            }
            let expected_hashes_after = get_expected_hashes(&slot);
            assert_eq!(expected_hashes_after.len(), 5);

            // Apply no new blocks - unassigned should exactly fill space
            let range = slot.apply_new_blocks(vec![]);

            assert_eq!(range, 2..5);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 5);
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 300);
            assert_eq!(slot.block_matches.assigned_blocks[3].1, 400);
            assert_eq!(slot.block_matches.assigned_blocks[4].1, 500);
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        #[test]
        fn test_empty_unassigned_with_new_blocks() {
            // Test that normal behavior works when there are no previous unassigned blocks
            let mut slot = create_test_slot(3, 0);
            let _expected_hashes = get_expected_hashes(&slot);

            // Apply exactly the right number of blocks
            let block_ids_1: Vec<BlockId> = vec![100, 200, 300];
            slot.apply_new_blocks(block_ids_1);
            assert!(slot.block_matches.unassigned_blocks.is_empty());

            // Add more space
            for token in 12..16 {
                slot.sequence.append(token).unwrap();
            }
            let _expected_hashes_after = get_expected_hashes(&slot);

            // Apply new blocks with no previous unassigned
            let block_ids_2: Vec<BlockId> = vec![400];
            let range = slot.apply_new_blocks(block_ids_2);

            assert_eq!(range, 3..4);
            assert_eq!(slot.block_matches.assigned_blocks[3].1, 400);
            assert!(slot.block_matches.unassigned_blocks.is_empty());
        }

        // =========================================================================
        // Test Cases: filter_block_ids
        // =========================================================================

        #[test]
        fn test_filter_block_ids_no_assigned_blocks() {
            // No blocks assigned, should return all block_ids
            let slot = create_test_slot(3, 0);
            let all_block_ids: Vec<BlockId> = vec![100, 200, 300];

            let filtered = slot.filter_block_ids(all_block_ids.clone());

            assert_eq!(filtered, all_block_ids);
        }

        #[test]
        fn test_filter_block_ids_all_already_assigned() {
            // All block_ids are already assigned, should return empty
            let mut slot = create_test_slot(3, 0);
            slot.apply_new_blocks(vec![100, 200, 300]);

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert!(filtered.is_empty());
        }

        #[test]
        fn test_filter_block_ids_partial_assigned() {
            // Some block_ids are assigned, should return the rest
            let mut slot = create_test_slot(5, 0);
            slot.apply_new_blocks(vec![100, 200]);

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300, 400, 500];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert_eq!(filtered, vec![300, 400, 500]);
        }

        #[test]
        fn test_filter_block_ids_single_assigned() {
            // One block assigned, should return the rest
            let mut slot = create_test_slot(4, 0);
            slot.apply_new_blocks(vec![100]);

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300, 400];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert_eq!(filtered, vec![200, 300, 400]);
        }

        #[test]
        fn test_filter_block_ids_exact_match() {
            // all_block_ids exactly matches assigned blocks
            let mut slot = create_test_slot(2, 0);
            slot.apply_new_blocks(vec![100, 200]);

            let all_block_ids: Vec<BlockId> = vec![100, 200];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert!(filtered.is_empty());
        }

        #[test]
        fn test_filter_block_ids_empty_input() {
            // Empty input with no assigned blocks
            let slot = create_test_slot(3, 0);
            let all_block_ids: Vec<BlockId> = vec![];

            let filtered = slot.filter_block_ids(all_block_ids);

            assert!(filtered.is_empty());
        }

        #[test]
        fn test_filter_block_ids_many_new_blocks() {
            // Few assigned, many new
            let mut slot = create_test_slot(10, 0);
            slot.apply_new_blocks(vec![10, 20]);

            let all_block_ids: Vec<BlockId> = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert_eq!(filtered, vec![30, 40, 50, 60, 70, 80, 90, 100]);
        }

        #[test]
        #[should_panic(expected = "Assigned block ID mismatch")]
        fn test_filter_block_ids_mismatch_at_start() {
            // First block_id doesn't match assigned
            let mut slot = create_test_slot(3, 0);
            slot.apply_new_blocks(vec![100, 200]);

            let all_block_ids: Vec<BlockId> = vec![999, 200, 300]; // 999 != 100
            let _ = slot.filter_block_ids(all_block_ids);
        }

        #[test]
        #[should_panic(expected = "Assigned block ID mismatch")]
        fn test_filter_block_ids_mismatch_at_middle() {
            // Middle block_id doesn't match assigned
            let mut slot = create_test_slot(4, 0);
            slot.apply_new_blocks(vec![100, 200, 300]);

            let all_block_ids: Vec<BlockId> = vec![100, 999, 300, 400]; // 999 != 200
            let _ = slot.filter_block_ids(all_block_ids);
        }

        #[test]
        #[should_panic(expected = "all_block_ids length")]
        fn test_filter_block_ids_too_few_provided() {
            // Fewer block_ids provided than assigned
            let mut slot = create_test_slot(3, 0);
            slot.apply_new_blocks(vec![100, 200, 300]);

            let all_block_ids: Vec<BlockId> = vec![100, 200]; // Missing 300
            let _ = slot.filter_block_ids(all_block_ids);
        }

        #[test]
        fn test_filter_block_ids_with_unassigned_blocks() {
            // Test that unassigned_blocks ARE filtered out
            let mut slot = create_test_slot(2, 0);
            // This will assign 2 blocks and put 2 in unassigned
            slot.apply_new_blocks(vec![100, 200, 300, 400]);

            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // Filter should consider both assigned AND unassigned blocks
            let all_block_ids: Vec<BlockId> = vec![100, 200, 300, 400, 500, 600];
            let filtered = slot.filter_block_ids(all_block_ids);

            // Should return everything after assigned (100, 200) AND unassigned (300, 400)
            assert_eq!(filtered, vec![500, 600]);
        }

        #[test]
        fn test_filter_block_ids_only_unassigned() {
            // Test with only unassigned blocks (no assigned blocks can be assigned)
            let mut slot = create_test_slot(0, 0); // No token blocks
            // All will go to unassigned since there are no token blocks
            slot.apply_new_blocks(vec![100, 200, 300]);

            assert!(slot.block_matches.assigned_blocks.is_empty());
            assert_eq!(slot.block_matches.unassigned_blocks, vec![100, 200, 300]);

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300, 400, 500];
            let filtered = slot.filter_block_ids(all_block_ids);

            // Should return everything after the unassigned blocks
            assert_eq!(filtered, vec![400, 500]);
        }

        #[test]
        #[should_panic(expected = "Unassigned block ID mismatch")]
        fn test_filter_block_ids_unassigned_mismatch() {
            // Test that unassigned block mismatch panics
            let mut slot = create_test_slot(2, 0);
            slot.apply_new_blocks(vec![100, 200, 300, 400]);

            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // 999 doesn't match unassigned block 300
            let all_block_ids: Vec<BlockId> = vec![100, 200, 999, 400, 500];
            let _ = slot.filter_block_ids(all_block_ids);
        }

        #[test]
        #[should_panic(expected = "all_block_ids length")]
        fn test_filter_block_ids_too_few_with_unassigned() {
            // Fewer block_ids provided than assigned + unassigned
            let mut slot = create_test_slot(2, 0);
            slot.apply_new_blocks(vec![100, 200, 300, 400]);

            // Only providing assigned blocks, missing unassigned
            let all_block_ids: Vec<BlockId> = vec![100, 200];
            let _ = slot.filter_block_ids(all_block_ids);
        }

        #[test]
        fn test_filter_block_ids_after_incremental_assignment() {
            // Test filtering after multiple apply_new_blocks calls
            let mut slot = create_test_slot(5, 0);

            // First assignment
            slot.apply_new_blocks(vec![100, 200]);

            // Verify filter works
            let filtered1 = slot.filter_block_ids(vec![100, 200, 300, 400, 500]);
            assert_eq!(filtered1, vec![300, 400, 500]);

            // Second assignment
            slot.apply_new_blocks(vec![300]);

            // Verify filter works again
            let filtered2 = slot.filter_block_ids(vec![100, 200, 300, 400, 500]);
            assert_eq!(filtered2, vec![400, 500]);
        }

        #[test]
        fn test_filter_block_ids_after_incremental_with_unassigned() {
            // Test filtering after multiple calls where unassigned accumulate
            let mut slot = create_test_slot(2, 0);

            // First: 2 assigned, 2 unassigned
            slot.apply_new_blocks(vec![100, 200, 300, 400]);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 2);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![300, 400]);

            // Filter should skip all 4
            let filtered1 = slot.filter_block_ids(vec![100, 200, 300, 400, 500, 600]);
            assert_eq!(filtered1, vec![500, 600]);

            // Add more tokens to create space for 1 more block
            for token in 8..12 {
                slot.sequence.append(token).unwrap();
            }

            // Second call: unassigned 300 gets assigned, 400 stays unassigned, 500 new unassigned
            slot.apply_new_blocks(vec![500]);
            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert_eq!(slot.block_matches.assigned_blocks[2].1, 300);
            assert_eq!(slot.block_matches.unassigned_blocks, vec![400, 500]);

            // Now filter should skip assigned (100, 200, 300) + unassigned (400, 500)
            let filtered2 = slot.filter_block_ids(vec![100, 200, 300, 400, 500, 600, 700]);
            assert_eq!(filtered2, vec![600, 700]);
        }

        #[test]
        fn test_filter_block_ids_returns_owned_vec() {
            // Verify that the returned vec is independent
            let mut slot = create_test_slot(3, 0);
            slot.apply_new_blocks(vec![100]);

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert_eq!(filtered.len(), 2);
            assert_eq!(filtered[0], 200);
            assert_eq!(filtered[1], 300);
        }

        #[test]
        fn test_filter_block_ids_empty_unassigned() {
            // Verify behavior when unassigned is empty
            let mut slot = create_test_slot(3, 0);
            slot.apply_new_blocks(vec![100, 200, 300]); // Exactly fills, no unassigned

            assert_eq!(slot.block_matches.assigned_blocks.len(), 3);
            assert!(slot.block_matches.unassigned_blocks.is_empty());

            let all_block_ids: Vec<BlockId> = vec![100, 200, 300, 400, 500];
            let filtered = slot.filter_block_ids(all_block_ids);

            assert_eq!(filtered, vec![400, 500]);
        }
    }
}
