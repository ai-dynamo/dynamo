// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Session state types for the unified session model.
//!
//! This module provides the core state machine types:
//! - [`ControlRole`]: Whether this session is controller, controllee, or neutral
//! - [`AttachmentState`]: Whether a peer is attached
//! - [`SessionPhase`]: The current operational phase of the session

use serde::{Deserialize, Serialize};

use crate::v2::InstanceId;

/// Control role in a session relationship.
///
/// Sessions can dynamically transition between roles:
/// - Start as `Neutral` (independent, can initiate in either direction)
/// - Become `Controller` when issuing commands to a peer
/// - Become `Controllee` when executing commands from a peer
///
/// Control can be transferred bidirectionally via `YieldControl`/`AcquireControl`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ControlRole {
    /// Independent - can initiate control in either direction.
    /// This is the initial state and the state after yielding control.
    #[default]
    Neutral,
    /// Currently controlling a peer session (issues commands).
    Controller,
    /// Currently being controlled by a peer session (executes commands).
    Controllee,
}

impl ControlRole {
    /// Check if this role can issue control commands.
    pub fn can_command(&self) -> bool {
        matches!(self, ControlRole::Controller)
    }

    /// Check if this role should respond to control commands.
    pub fn responds_to_commands(&self) -> bool {
        matches!(self, ControlRole::Controllee)
    }

    /// Check if this role is neutral (can transition either way).
    pub fn is_neutral(&self) -> bool {
        matches!(self, ControlRole::Neutral)
    }

    /// Get the opposite role.
    ///
    /// - `Controller` ↔ `Controllee`
    /// - `Neutral` → `Neutral` (no opposite)
    pub fn opposite(&self) -> ControlRole {
        match self {
            ControlRole::Controller => ControlRole::Controllee,
            ControlRole::Controllee => ControlRole::Controller,
            ControlRole::Neutral => ControlRole::Neutral,
        }
    }
}

/// Attachment state - whether a peer is connected.
///
/// Valid state combinations:
/// - `Neutral + Unattached`: Initial state, waiting for connection
/// - `Neutral + Attached`: Post-yield state, peer still connected
/// - `Controllee + Unattached`: Waiting for controller to attach
/// - `Controllee + Attached`: Being actively controlled
/// - `Controller + Attached`: Actively controlling (Controller + Unattached is invalid)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AttachmentState {
    /// No peer attached.
    #[default]
    Unattached,
    /// Peer attached with the given instance ID.
    Attached { peer: InstanceId },
}

impl AttachmentState {
    /// Check if a peer is attached.
    pub fn is_attached(&self) -> bool {
        matches!(self, AttachmentState::Attached { .. })
    }

    /// Get the attached peer's instance ID if attached.
    pub fn peer(&self) -> Option<InstanceId> {
        match self {
            AttachmentState::Attached { peer } => Some(*peer),
            AttachmentState::Unattached => None,
        }
    }
}

/// Operational phase of a session.
///
/// Represents the lifecycle of block operations within a session:
/// 1. `Searching` - Initial discovery/search phase
/// 2. `Holding` - Blocks found and held, no staging yet
/// 3. `Staging` - Transfer in progress (G3→G2, G4→G2, etc.)
/// 4. `Ready` - All blocks in target tier, ready for transfer
/// 5. `Complete` - Session completed successfully
/// 6. `Failed` - Session failed or cancelled
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SessionPhase {
    /// Initial search/discovery phase.
    #[default]
    Searching,
    /// Blocks found and held, no staging started.
    Holding,
    /// Transfer/staging in progress (any direction).
    Staging,
    /// All blocks in target tier, ready for RDMA pull.
    Ready,
    /// Session completed successfully.
    Complete,
    /// Session failed or was cancelled.
    Failed,
}

impl SessionPhase {
    /// Check if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, SessionPhase::Complete | SessionPhase::Failed)
    }

    /// Check if the session is active (not terminal).
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }

    /// Check if blocks are ready for transfer.
    pub fn is_ready(&self) -> bool {
        matches!(self, SessionPhase::Ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_control_role_transitions() {
        let role = ControlRole::Neutral;
        assert!(role.is_neutral());
        assert!(!role.can_command());
        assert!(!role.responds_to_commands());

        let role = ControlRole::Controller;
        assert!(!role.is_neutral());
        assert!(role.can_command());
        assert!(!role.responds_to_commands());

        let role = ControlRole::Controllee;
        assert!(!role.is_neutral());
        assert!(!role.can_command());
        assert!(role.responds_to_commands());
    }

    #[test]
    fn test_attachment_state() {
        let state = AttachmentState::Unattached;
        assert!(!state.is_attached());
        assert!(state.peer().is_none());

        let peer_id = InstanceId::new_v4();
        let state = AttachmentState::Attached { peer: peer_id };
        assert!(state.is_attached());
        assert_eq!(state.peer(), Some(peer_id));
    }

    #[test]
    fn test_session_phase() {
        assert!(!SessionPhase::Searching.is_terminal());
        assert!(!SessionPhase::Holding.is_terminal());
        assert!(!SessionPhase::Staging.is_terminal());
        assert!(!SessionPhase::Ready.is_terminal());
        assert!(SessionPhase::Complete.is_terminal());
        assert!(SessionPhase::Failed.is_terminal());

        assert!(SessionPhase::Ready.is_ready());
        assert!(!SessionPhase::Staging.is_ready());
    }
}
