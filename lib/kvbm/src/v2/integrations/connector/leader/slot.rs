// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_tokens::TokenBlockSequence;

use super::Request;
use crate::distributed::leader::FindMatchesResult;
use crate::distributed::offload::{TransferHandle, TransferId};
use crate::v2::SequenceHash;

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
#[derive(Debug)]
pub struct OffloadingState {
    /// The sequence hashes of the blocks that have been sent to the offload engine.
    pub sequence_hashes: HashSet<SequenceHash>,

    /// The transfer handle for the offload operation.
    pub handles: HashMap<TransferId, TransferHandle>,
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

    /// Private state machine - not directly accessible.
    state: SlotStateMachine,

    /// The number of token blocks that have been evaluated by our offloading policies.
    evaluated_blocks: usize,
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
            state: SlotStateMachine::new(),
            evaluated_blocks: 0,
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
        self.evaluated_blocks = 0;
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
        let state = OffloadingState {
            sequence_hashes: HashSet::new(),
            handles: HashMap::new(),
        };
        self.state.txn_start_offloading(state)
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
            let state = OffloadingState {
                sequence_hashes: HashSet::new(),
                handles: HashMap::new(),
            };
            self.state
                .txn_start_offloading(state)
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
        if !matches!(self.txn_state(), TransactionState::PreparingToOnboard(_)) {
            return Err(anyhow::anyhow!(
                "finalize_match_check called in unexpected state: {}",
                self.txn_state().name()
            ));
        }

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
                    // Transition to Onboarding
                    self.state.txn_start_onboarding()?;
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

    pub fn offload_blocks(
        &mut self,
        sequence_hashes: &[SequenceHash],
        handle: TransferHandle,
    ) -> Result<(), anyhow::Error> {
        // the state must be active, not marked for deletion, and the txn_state
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

        // create or get the offloading state
        let mut offloading_state = self.get_or_create_offloading_state();

        // check that the sequence hashes provides are not already in the set
        // add the sequence hashes to the set
        // add the transfer handle to the hashmap

        unimplemented!()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

// #[cfg(test)]
// mod tests {
//     use super::*;

//     // ------------------------------------------------------------------------
//     // Test Helpers
//     // ------------------------------------------------------------------------

//     /// Create a test OnboardingState without needing real FindMatchesResult.
//     /// This uses a mock ReadyResult with no blocks.
//     fn create_test_onboarding_state() -> OnboardingState {
//         // Create a minimal ReadyResult for testing
//         let ready_result = crate::distributed::leader::ReadyResult::new(vec![]);
//         OnboardingState {
//             find_session: FindMatchesResult::Ready(ready_result),
//         }
//     }

//     // /// Create a test OnboardingState with a session ID set.
//     // fn create_test_onboarding_state_with_session() -> OnboardingState {
//     //     let ready_result = crate::distributed::leader::ReadyResult::new(vec![]);
//     //     OnboardingState {
//     //         find_session: FindMatchesResult::Ready(ready_result),
//     //     }
//     // }

//     /// Create a test OffloadingState.
//     fn create_test_offloading_state(session_id: SessionId) -> OffloadingState {
//         OffloadingState {}
//     }

//     // ------------------------------------------------------------------------
//     // SlotStateMachine Tests - Valid Transitions
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_state_machine_initial_state() {
//         let sm = SlotStateMachine::new();
//         assert!(sm.txn_state().is_inactive());
//         assert!(!sm.is_marked_for_deletion());
//         assert!(!sm.has_find_session());
//     }

//     #[test]
//     fn test_txn_prepare_to_onboard_from_inactive() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();

//         let result = sm.txn_prepare_to_onboard(state);
//         assert!(result.is_ok());
//         assert!(matches!(
//             sm.txn_state(),
//             TransactionState::PreparingToOnboard(_)
//         ));
//         assert!(sm.has_onboarding_state());
//     }

//     #[test]
//     fn test_txn_start_onboarding_from_preparing() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();
//         sm.txn_prepare_to_onboard(state).unwrap();

//         let result = sm.txn_start_onboarding();
//         assert!(result.is_ok());
//         assert!(matches!(sm.txn_state(), TransactionState::Onboarding(_)));
//     }

//     #[test]
//     fn test_txn_take_onboarding_from_onboarding() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let result = sm.txn_take_onboarding();
//         assert!(result.is_ok());
//         let onboarding_state = result.unwrap();
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_txn_start_offloading_from_inactive() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());

//         let result = sm.txn_start_offloading(state);
//         assert!(result.is_ok());
//         assert!(matches!(sm.txn_state(), TransactionState::Offloading(_)));
//     }

//     #[test]
//     fn test_txn_take_offloading_from_offloading() {
//         let mut sm = SlotStateMachine::new();
//         let session_id = SessionId::new_v4();
//         let state = create_test_offloading_state(session_id);
//         sm.txn_start_offloading(state).unwrap();

//         let result = sm.txn_take_offloading();
//         assert!(result.is_ok());
//         let offloading_state = result.unwrap();
//         assert_eq!(offloading_state.session_id, session_id);
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_txn_to_error_from_preparing_to_onboard() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();
//         sm.txn_prepare_to_onboard(state).unwrap();

//         sm.txn_to_error();
//         assert!(matches!(sm.txn_state(), TransactionState::Error(_)));
//     }

//     #[test]
//     fn test_txn_to_error_from_onboarding() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         sm.txn_to_error();
//         assert!(matches!(sm.txn_state(), TransactionState::Error(_)));
//     }

//     #[test]
//     fn test_txn_to_error_from_offloading() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();

//         sm.txn_to_error();
//         assert!(matches!(sm.txn_state(), TransactionState::Error(_)));
//     }

//     #[test]
//     fn test_txn_to_error_from_inactive_is_noop() {
//         let mut sm = SlotStateMachine::new();
//         sm.txn_to_error();
//         // From Inactive, txn_to_error is a no-op (stays Inactive since no data to preserve)
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_txn_take_error_returns_onboarding_data() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state();
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_to_error();

//         let result = sm.txn_take_error();
//         assert!(result.is_ok());
//         match result.unwrap() {
//             ActiveStateData::Onboarding(state) => {}
//             ActiveStateData::Offloading(_) => panic!("Expected Onboarding data"),
//         }
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_txn_take_error_returns_offloading_data() {
//         let mut sm = SlotStateMachine::new();
//         let session_id = SessionId::new_v4();
//         let state = create_test_offloading_state(session_id);
//         sm.txn_start_offloading(state).unwrap();
//         sm.txn_to_error();

//         let result = sm.txn_take_error();
//         assert!(result.is_ok());
//         match result.unwrap() {
//             ActiveStateData::Offloading(state) => {
//                 assert_eq!(state.session_id, session_id);
//             }
//             ActiveStateData::Onboarding(_) => panic!("Expected Offloading data"),
//         }
//         assert!(sm.txn_state().is_inactive());
//     }

//     // ------------------------------------------------------------------------
//     // SlotStateMachine Tests - Invalid Transitions
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_txn_prepare_to_onboard_from_preparing_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state1 = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state1).unwrap();

//         let state2 = create_test_onboarding_state(200);
//         let result = sm.txn_prepare_to_onboard(state2);
//         assert!(result.is_err());
//         match result.unwrap_err() {
//             StateTransitionError::InvalidTransition { from, to } => {
//                 assert_eq!(from, "PreparingToOnboard");
//                 assert_eq!(to, "PreparingToOnboard");
//             }
//             _ => panic!("Expected InvalidTransition error"),
//         }
//     }

//     #[test]
//     fn test_txn_prepare_to_onboard_from_onboarding_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let state2 = create_test_onboarding_state(200);
//         let result = sm.txn_prepare_to_onboard(state2);
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_prepare_to_onboard_from_offloading_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();

//         let state2 = create_test_onboarding_state(200);
//         let result = sm.txn_prepare_to_onboard(state2);
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_start_onboarding_from_inactive_fails() {
//         let mut sm = SlotStateMachine::new();
//         let result = sm.txn_start_onboarding();
//         assert!(result.is_err());
//         match result.unwrap_err() {
//             StateTransitionError::InvalidTransition { from, to } => {
//                 assert_eq!(from, "Inactive");
//                 assert_eq!(to, "Onboarding");
//             }
//             _ => panic!("Expected InvalidTransition error"),
//         }
//     }

//     #[test]
//     fn test_txn_start_onboarding_from_onboarding_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let result = sm.txn_start_onboarding();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_onboarding_from_inactive_fails() {
//         let mut sm = SlotStateMachine::new();
//         let result = sm.txn_take_onboarding();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_onboarding_from_preparing_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();

//         let result = sm.txn_take_onboarding();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_onboarding_from_offloading_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();

//         let result = sm.txn_take_onboarding();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_start_offloading_from_preparing_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();

//         let offload_state = create_test_offloading_state(SessionId::new_v4());
//         let result = sm.txn_start_offloading(offload_state);
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_start_offloading_from_onboarding_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let offload_state = create_test_offloading_state(SessionId::new_v4());
//         let result = sm.txn_start_offloading(offload_state);
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_offloading_from_inactive_fails() {
//         let mut sm = SlotStateMachine::new();
//         let result = sm.txn_take_offloading();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_offloading_from_onboarding_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let result = sm.txn_take_offloading();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_error_from_inactive_fails() {
//         let mut sm = SlotStateMachine::new();
//         let result = sm.txn_take_error();
//         assert!(result.is_err());
//     }

//     #[test]
//     fn test_txn_take_error_from_onboarding_fails() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let result = sm.txn_take_error();
//         assert!(result.is_err());
//     }

//     // ------------------------------------------------------------------------
//     // SlotStateMachine Tests - Marked for Deletion
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_marked_for_deletion_blocks_new_transactions() {
//         let mut sm = SlotStateMachine::new();
//         sm.slot_mark_finished();

//         let state = create_test_onboarding_state(100);
//         let result = sm.txn_prepare_to_onboard(state);
//         assert!(result.is_err());
//         assert!(matches!(
//             result.unwrap_err(),
//             StateTransitionError::MarkedForDeletion
//         ));
//     }

//     #[test]
//     fn test_marked_for_deletion_blocks_offloading() {
//         let mut sm = SlotStateMachine::new();
//         sm.slot_mark_finished();

//         let state = create_test_offloading_state(SessionId::new_v4());
//         let result = sm.txn_start_offloading(state);
//         assert!(result.is_err());
//         assert!(matches!(
//             result.unwrap_err(),
//             StateTransitionError::MarkedForDeletion
//         ));
//     }

//     #[test]
//     fn test_slot_mark_finished_while_inactive_returns_finished() {
//         let mut sm = SlotStateMachine::new();
//         let status = sm.slot_mark_finished();
//         assert_eq!(status, FinishedStatus::Finished);
//     }

//     #[test]
//     fn test_slot_mark_finished_while_onboarding_returns_pending() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let status = sm.slot_mark_finished();
//         assert_eq!(status, FinishedStatus::Pending);
//         assert!(sm.is_marked_for_deletion());
//     }

//     #[test]
//     fn test_slot_mark_finished_while_offloading_returns_pending() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();

//         let status = sm.slot_mark_finished();
//         assert_eq!(status, FinishedStatus::Pending);
//         assert!(sm.is_marked_for_deletion());
//     }

//     // ------------------------------------------------------------------------
//     // SlotStateMachine Tests - Slot State Transitions on txn_to_inactive
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_txn_to_inactive_transitions_slot_state_when_marked() {
//         let mut sm = SlotStateMachine::new();

//         // Start onboarding
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         // Mark for deletion while onboarding
//         sm.slot_mark_finished();
//         assert!(matches!(sm.slot_state, SlotState::MarkedForDeletion));

//         // Take onboarding (calls txn_to_inactive internally)
//         sm.txn_take_onboarding().unwrap();

//         // Slot state should transition to NotifyWorkersToFinish
//         assert!(matches!(sm.slot_state, SlotState::NotifyWorkersToFinish));
//     }

//     #[test]
//     fn test_txn_to_inactive_does_not_change_active_slot_state() {
//         let mut sm = SlotStateMachine::new();

//         // Start and complete onboarding without marking for deletion
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();
//         sm.txn_take_onboarding().unwrap();

//         // Slot state should still be Active
//         assert!(matches!(sm.slot_state, SlotState::Active));
//     }

//     // ------------------------------------------------------------------------
//     // Onboarding State Accessor Tests
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_onboarding_state_accessor_in_preparing() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(150);
//         sm.txn_prepare_to_onboard(state).unwrap();

//         let state_ref = sm.onboarding_state();
//         assert!(state_ref.is_some());
//         assert_eq!(state_ref.unwrap().num_computed_tokens, 150);
//     }

//     #[test]
//     fn test_onboarding_state_accessor_in_onboarding() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(150);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         let state_ref = sm.onboarding_state();
//         assert!(state_ref.is_some());
//         assert_eq!(state_ref.unwrap().num_computed_tokens, 150);
//     }

//     #[test]
//     fn test_onboarding_state_accessor_in_inactive() {
//         let sm = SlotStateMachine::new();
//         assert!(sm.onboarding_state().is_none());
//     }

//     #[test]
//     fn test_onboarding_state_accessor_in_offloading() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();

//         assert!(sm.onboarding_state().is_none());
//     }

//     #[test]
//     fn test_onboarding_state_mut_can_modify_session_id() {
//         let mut sm = SlotStateMachine::new();
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();

//         let session_id = SessionId::new_v4();
//         if let Some(state) = sm.onboarding_state_mut() {
//             state.session_id = Some(session_id);
//         }

//         let state_ref = sm.onboarding_state().unwrap();
//         assert_eq!(state_ref.session_id, Some(session_id));
//     }

//     // ------------------------------------------------------------------------
//     // Full Lifecycle Tests
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_full_onboarding_lifecycle() {
//         let mut sm = SlotStateMachine::new();
//         let session_id = SessionId::new_v4();

//         // 1. Start preparing
//         let state = create_test_onboarding_state(100);
//         assert!(sm.txn_prepare_to_onboard(state).is_ok());
//         assert!(matches!(
//             sm.txn_state(),
//             TransactionState::PreparingToOnboard(_)
//         ));

//         // 2. Set session ID
//         sm.onboarding_state_mut().unwrap().session_id = Some(session_id);

//         // 3. Transition to onboarding
//         assert!(sm.txn_start_onboarding().is_ok());
//         assert!(matches!(sm.txn_state(), TransactionState::Onboarding(_)));

//         // 4. Complete onboarding
//         let result = sm.txn_take_onboarding();
//         assert!(result.is_ok());
//         let state = result.unwrap();
//         assert_eq!(state.session_id, Some(session_id));
//         assert_eq!(state.num_computed_tokens, 100);
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_full_offloading_lifecycle() {
//         let mut sm = SlotStateMachine::new();
//         let session_id = SessionId::new_v4();

//         // 1. Start offloading
//         let state = create_test_offloading_state(session_id);
//         assert!(sm.txn_start_offloading(state).is_ok());
//         assert!(matches!(sm.txn_state(), TransactionState::Offloading(_)));

//         // 2. Complete offloading
//         let result = sm.txn_take_offloading();
//         assert!(result.is_ok());
//         let state = result.unwrap();
//         assert_eq!(state.session_id, session_id);
//         assert!(sm.txn_state().is_inactive());
//     }

//     #[test]
//     fn test_onboarding_to_error_recovery() {
//         let mut sm = SlotStateMachine::new();

//         // Start onboarding
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();

//         // Error occurs
//         sm.txn_to_error();
//         assert!(matches!(sm.txn_state(), TransactionState::Error(_)));

//         // Recover
//         let result = sm.txn_take_error();
//         assert!(result.is_ok());
//         match result.unwrap() {
//             ActiveStateData::Onboarding(state) => {
//                 assert_eq!(state.num_computed_tokens, 100);
//             }
//             _ => panic!("Expected Onboarding data"),
//         }
//         assert!(sm.txn_state().is_inactive());

//         // Can start new transaction
//         let state = create_test_onboarding_state(200);
//         assert!(sm.txn_prepare_to_onboard(state).is_ok());
//     }

//     #[test]
//     fn test_multiple_sequential_transactions() {
//         let mut sm = SlotStateMachine::new();

//         // First onboarding
//         let state = create_test_onboarding_state(100);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();
//         sm.txn_take_onboarding().unwrap();

//         // Offloading
//         let state = create_test_offloading_state(SessionId::new_v4());
//         sm.txn_start_offloading(state).unwrap();
//         sm.txn_take_offloading().unwrap();

//         // Second onboarding
//         let state = create_test_onboarding_state(200);
//         sm.txn_prepare_to_onboard(state).unwrap();
//         sm.txn_start_onboarding().unwrap();
//         let result = sm.txn_take_onboarding().unwrap();
//         assert_eq!(result.num_computed_tokens, 200);
//     }

//     // ------------------------------------------------------------------------
//     // TransactionState Name Tests
//     // ------------------------------------------------------------------------

//     #[test]
//     fn test_transaction_state_names() {
//         assert_eq!(TransactionState::Inactive.name(), "Inactive");

//         let onboarding = create_test_onboarding_state(0);
//         assert_eq!(
//             TransactionState::PreparingToOnboard(onboarding).name(),
//             "PreparingToOnboard"
//         );

//         let onboarding = create_test_onboarding_state(0);
//         assert_eq!(
//             TransactionState::Onboarding(onboarding).name(),
//             "Onboarding"
//         );

//         let offloading = create_test_offloading_state(SessionId::new_v4());
//         assert_eq!(
//             TransactionState::Offloading(offloading).name(),
//             "Offloading"
//         );

//         let onboarding = create_test_onboarding_state(0);
//         assert_eq!(
//             TransactionState::Error(ActiveStateData::Onboarding(onboarding)).name(),
//             "Error"
//         );
//     }
// }
