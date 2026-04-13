// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State machine for the GPU Memory Service.
//!
//! Port of `lib/gpu_memory_service/server/locking.py` (`GMSLocalFSM`).
//!
//! State is DERIVED from actual connection tracking:
//! - `rw_session`: The active RW session (or None)
//! - `ro_sessions`: Set of active RO sessions
//! - `committed`: Whether allocations have been committed
//!
//! ```text
//! EMPTY ──RW_CONNECT──► RW ──RW_COMMIT──► COMMITTED
//!   ▲                    │                   │
//!   │                    │                   │
//!   └───RW_ABORT─────────┘                   │
//!                                            ▼
//! COMMITTED ◄──RO_DISCONNECT (last)── RO ◄──RO_CONNECT
//!               │                      ▲
//!               │                      │
//!               └──RO_CONNECT──────────┘
//!               └──RO_DISCONNECT───┘ (not last)
//! ```

use std::collections::HashSet;

use crate::error::{GmsError, GmsResult};

/// Server state — derived from connection objects, not stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ServerState {
    /// No connections, no committed data.
    Empty,
    /// A read-write session is active.
    Rw,
    /// Data has been committed, no active connections.
    Committed,
    /// One or more read-only sessions are active.
    Ro,
}

impl std::fmt::Display for ServerState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ServerState::Empty => write!(f, "EMPTY"),
            ServerState::Rw => write!(f, "RW"),
            ServerState::Committed => write!(f, "COMMITTED"),
            ServerState::Ro => write!(f, "RO"),
        }
    }
}

/// Events that trigger state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateEvent {
    /// A read-write client connects.
    RwConnect,
    /// The read-write client commits.
    RwCommit,
    /// The read-write client disconnects without committing.
    RwAbort,
    /// A read-only client connects.
    RoConnect,
    /// A read-only client disconnects.
    RoDisconnect,
}

/// Explicit state machine for the GPU Memory Service.
///
/// State is derived from the actual session tracking, not stored separately.
/// All mutations happen through validated transitions.
pub struct StateMachine {
    /// Active RW session ID (at most one).
    rw_session: Option<u64>,
    /// Set of active RO session IDs.
    ro_sessions: HashSet<u64>,
    /// Whether data has been committed.
    committed: bool,
}

impl StateMachine {
    /// Create a new state machine in the Empty state.
    pub fn new() -> Self {
        Self {
            rw_session: None,
            ro_sessions: HashSet::new(),
            committed: false,
        }
    }

    /// Derive the current state from session tracking.
    pub fn state(&self) -> ServerState {
        if self.rw_session.is_some() {
            ServerState::Rw
        } else if !self.ro_sessions.is_empty() {
            ServerState::Ro
        } else if self.committed {
            ServerState::Committed
        } else {
            ServerState::Empty
        }
    }

    /// Whether data has been committed.
    pub fn committed(&self) -> bool {
        self.committed
    }

    /// Whether an RW session is active.
    pub fn has_rw(&self) -> bool {
        self.rw_session.is_some()
    }

    /// Number of active RO sessions.
    pub fn ro_count(&self) -> usize {
        self.ro_sessions.len()
    }

    /// The active RW session ID, if any.
    pub fn rw_session(&self) -> Option<u64> {
        self.rw_session
    }

    /// Check if RW lock can be acquired now.
    ///
    /// RW requires: no current RW holder AND no RO holders.
    pub fn can_acquire_rw(&self) -> bool {
        self.rw_session.is_none() && self.ro_sessions.is_empty()
    }

    /// Check if RO lock can be acquired now.
    ///
    /// RO requires: no RW holder AND no waiting writers AND committed.
    pub fn can_acquire_ro(&self, waiting_writers: usize) -> bool {
        self.rw_session.is_none() && waiting_writers == 0 && self.committed
    }

    /// Execute a state transition.
    ///
    /// Validates the transition against the current state and applies it.
    /// Returns the new state after the transition.
    pub fn transition(&mut self, event: StateEvent, session_id: u64) -> GmsResult<ServerState> {
        let from_state = self.state();

        // Validate and apply the transition
        match (from_state, event) {
            // EMPTY or COMMITTED → RW (writer connects)
            (ServerState::Empty | ServerState::Committed, StateEvent::RwConnect) => {
                self.rw_session = Some(session_id);
                self.committed = false; // Invalidate on RW connect
            }

            // RW → COMMITTED (writer commits)
            (ServerState::Rw, StateEvent::RwCommit) => {
                if self.rw_session != Some(session_id) {
                    return Err(GmsError::InvalidTransition { from: from_state, event });
                }
                self.committed = true;
                self.rw_session = None;
            }

            // RW → EMPTY (writer aborts)
            (ServerState::Rw, StateEvent::RwAbort) => {
                if self.rw_session != Some(session_id) {
                    return Err(GmsError::InvalidTransition { from: from_state, event });
                }
                self.rw_session = None;
                // committed is already false — it was reset to false on RwConnect.
                // We don't explicitly reset it here because the invariant is maintained
                // by RwConnect always setting committed = false.
            }

            // COMMITTED or RO → RO (reader connects)
            (ServerState::Committed | ServerState::Ro, StateEvent::RoConnect) => {
                self.ro_sessions.insert(session_id);
            }

            // RO → RO or COMMITTED (reader disconnects)
            (ServerState::Ro, StateEvent::RoDisconnect) => {
                self.ro_sessions.remove(&session_id);
                // If last reader, state transitions to COMMITTED automatically
                // via the derived state logic
            }

            // Invalid transition
            _ => {
                return Err(GmsError::InvalidTransition {
                    from: from_state,
                    event,
                });
            }
        }

        let to_state = self.state();
        tracing::info!(
            "State transition: {from_state} --{event:?}--> {to_state} (session={session_id})"
        );
        Ok(to_state)
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sm = StateMachine::new();
        assert_eq!(sm.state(), ServerState::Empty);
        assert!(!sm.committed());
        assert!(!sm.has_rw());
        assert_eq!(sm.ro_count(), 0);
    }

    #[test]
    fn test_rw_connect_from_empty() {
        let mut sm = StateMachine::new();
        let state = sm.transition(StateEvent::RwConnect, 1).unwrap();
        assert_eq!(state, ServerState::Rw);
        assert!(sm.has_rw());
    }

    #[test]
    fn test_rw_commit() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        let state = sm.transition(StateEvent::RwCommit, 1).unwrap();
        assert_eq!(state, ServerState::Committed);
        assert!(sm.committed());
        assert!(!sm.has_rw());
    }

    #[test]
    fn test_rw_abort() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        let state = sm.transition(StateEvent::RwAbort, 1).unwrap();
        assert_eq!(state, ServerState::Empty);
        assert!(!sm.committed());
        assert!(!sm.has_rw());
    }

    #[test]
    fn test_ro_connect_from_committed() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        sm.transition(StateEvent::RwCommit, 1).unwrap();
        let state = sm.transition(StateEvent::RoConnect, 2).unwrap();
        assert_eq!(state, ServerState::Ro);
        assert_eq!(sm.ro_count(), 1);
    }

    #[test]
    fn test_multiple_ro_sessions() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        sm.transition(StateEvent::RwCommit, 1).unwrap();
        sm.transition(StateEvent::RoConnect, 2).unwrap();
        sm.transition(StateEvent::RoConnect, 3).unwrap();
        assert_eq!(sm.ro_count(), 2);
        assert_eq!(sm.state(), ServerState::Ro);

        // Disconnect one reader
        sm.transition(StateEvent::RoDisconnect, 2).unwrap();
        assert_eq!(sm.ro_count(), 1);
        assert_eq!(sm.state(), ServerState::Ro);

        // Disconnect last reader
        sm.transition(StateEvent::RoDisconnect, 3).unwrap();
        assert_eq!(sm.ro_count(), 0);
        assert_eq!(sm.state(), ServerState::Committed);
    }

    #[test]
    fn test_rw_connect_from_committed() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        sm.transition(StateEvent::RwCommit, 1).unwrap();
        assert_eq!(sm.state(), ServerState::Committed);

        // Reconnect as RW (explicit reload)
        let state = sm.transition(StateEvent::RwConnect, 2).unwrap();
        assert_eq!(state, ServerState::Rw);
        assert!(!sm.committed()); // committed reset on RW connect
    }

    #[test]
    fn test_invalid_rw_connect_from_rw() {
        let mut sm = StateMachine::new();
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        let result = sm.transition(StateEvent::RwConnect, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_ro_connect_from_empty() {
        let mut sm = StateMachine::new();
        let result = sm.transition(StateEvent::RoConnect, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_commit_from_empty() {
        let mut sm = StateMachine::new();
        let result = sm.transition(StateEvent::RwCommit, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_can_acquire_rw() {
        let mut sm = StateMachine::new();
        assert!(sm.can_acquire_rw());

        sm.transition(StateEvent::RwConnect, 1).unwrap();
        assert!(!sm.can_acquire_rw());

        sm.transition(StateEvent::RwCommit, 1).unwrap();
        assert!(sm.can_acquire_rw());

        sm.transition(StateEvent::RoConnect, 2).unwrap();
        assert!(!sm.can_acquire_rw()); // RO holders block RW
    }

    #[test]
    fn test_can_acquire_ro() {
        let mut sm = StateMachine::new();
        assert!(!sm.can_acquire_ro(0)); // Not committed

        sm.transition(StateEvent::RwConnect, 1).unwrap();
        assert!(!sm.can_acquire_ro(0)); // RW active

        sm.transition(StateEvent::RwCommit, 1).unwrap();
        assert!(sm.can_acquire_ro(0)); // Committed, no RW, no waiting writers

        assert!(!sm.can_acquire_ro(1)); // Waiting writers block RO
    }

    #[test]
    fn test_full_lifecycle() {
        let mut sm = StateMachine::new();

        // Writer loads model
        sm.transition(StateEvent::RwConnect, 1).unwrap();
        sm.transition(StateEvent::RwCommit, 1).unwrap();

        // Multiple readers connect
        sm.transition(StateEvent::RoConnect, 2).unwrap();
        sm.transition(StateEvent::RoConnect, 3).unwrap();
        sm.transition(StateEvent::RoConnect, 4).unwrap();
        assert_eq!(sm.state(), ServerState::Ro);

        // Readers disconnect
        sm.transition(StateEvent::RoDisconnect, 2).unwrap();
        sm.transition(StateEvent::RoDisconnect, 3).unwrap();
        sm.transition(StateEvent::RoDisconnect, 4).unwrap();
        assert_eq!(sm.state(), ServerState::Committed);

        // New writer (model update)
        sm.transition(StateEvent::RwConnect, 5).unwrap();
        assert_eq!(sm.state(), ServerState::Rw);
        sm.transition(StateEvent::RwCommit, 5).unwrap();
        assert_eq!(sm.state(), ServerState::Committed);
    }
}
