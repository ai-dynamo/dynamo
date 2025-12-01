// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Action event recording for connector leader operations.
//!
//! Captures discrete actions that modify leader or slot state, providing
//! a linear audit log for debugging, testing, and observability.

use std::sync::{Arc, Mutex};

use uuid::Uuid;

use crate::v2::integrations::connector::slot::{SlotState, TransferDirection};

/// Discrete action that modifies leader or slot state.
#[derive(Debug, Clone)]
pub enum ConnectorAction {
    // === Slot Lifecycle ===
    SlotCreated {
        request_id: String,
        num_tokens: usize,
        iteration: u64,
    },
    SlotRemoved {
        request_id: String,
    },

    // === Matching ===
    BlocksMatched {
        request_id: String,
        num_g2: usize,
        num_g3: usize,
        total_tokens: usize,
    },

    // === Allocation ===
    BlocksAllocated {
        request_id: String,
        num_blocks: usize,
        purpose: AllocPurpose,
    },

    // === Operations ===
    OperationRecorded {
        request_id: String,
        operation_id: Uuid,
        direction: TransferDirection,
        num_blocks: usize,
    },
    OperationCompleted {
        request_id: String,
        operation_id: Uuid,
    },

    // === Transfers ===
    OnboardingEnqueued {
        request_id: String,
        operation_id: Uuid,
        num_blocks: usize,
    },
    OnboardingFinished {
        request_id: String,
    },
    OffloadingFinished {
        request_id: String,
    },

    // === Deletion ===
    SlotQueuedForDeletion {
        request_id: String,
        iteration: u64,
    },

    // === Metadata Build ===
    MetadataBuildStarted {
        iteration: u64,
        num_new: usize,
        num_cached: usize,
        num_onboarding: usize,
    },
    MetadataBuildCompleted {
        iteration: u64,
        bytes_len: usize,
    },

    // === Tracking ===
    RequestMarkedOnboarding {
        request_id: String,
    },
    RequestMarkedUnscheduled {
        request_id: String,
        iteration: u64,
    },

    // === State Transitions ===
    StateTransition {
        request_id: String,
        from: SlotState,
        to: SlotState,
    },
}

/// Purpose of block allocation.
#[derive(Debug, Clone, Copy)]
pub enum AllocPurpose {
    Onboarding,
    Prefill,
}

/// Trait for collecting connector actions.
///
/// Implementations can record actions to a log, emit to external systems,
/// or provide no-op behavior for production.
pub trait ActionCollector: Send + Sync {
    fn record_action(&self, action: ConnectorAction);
}

/// Type alias for shared action collector handles.
pub type ActionCollectorRef = Arc<dyn ActionCollector>;

/// Recording action collector that stores all actions in memory.
///
/// Useful for testing and debugging to validate action sequences.
#[derive(Default)]
pub struct RecordingActionCollector {
    actions: Mutex<Vec<ConnectorAction>>,
}

impl RecordingActionCollector {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn actions(&self) -> Vec<ConnectorAction> {
        self.actions.lock().unwrap().clone()
    }

    pub fn clear(&self) {
        self.actions.lock().unwrap().clear();
    }
}

impl ActionCollector for RecordingActionCollector {
    fn record_action(&self, action: ConnectorAction) {
        self.actions.lock().unwrap().push(action);
    }
}

/// No-op action collector for production builds.
#[derive(Debug, Default)]
pub struct NoopActionCollector;

impl NoopActionCollector {
    pub fn shared() -> ActionCollectorRef {
        Arc::new(Self)
    }
}

impl ActionCollector for NoopActionCollector {
    fn record_action(&self, _action: ConnectorAction) {
        // No-op
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recording_action_collector() {
        let collector = RecordingActionCollector::new();

        collector.record_action(ConnectorAction::SlotCreated {
            request_id: "test-1".to_string(),
            num_tokens: 256,
            iteration: 0,
        });

        collector.record_action(ConnectorAction::BlocksMatched {
            request_id: "test-1".to_string(),
            num_g2: 3,
            num_g3: 0,
            total_tokens: 48,
        });

        let actions = collector.actions();
        assert_eq!(actions.len(), 2);

        // Verify order preserved
        match &actions[0] {
            ConnectorAction::SlotCreated { request_id, .. } => {
                assert_eq!(request_id, "test-1");
            }
            _ => panic!("Expected SlotCreated"),
        }

        match &actions[1] {
            ConnectorAction::BlocksMatched {
                request_id,
                total_tokens,
                ..
            } => {
                assert_eq!(request_id, "test-1");
                assert_eq!(*total_tokens, 48);
            }
            _ => panic!("Expected BlocksMatched"),
        }
    }

    #[test]
    fn test_clear_actions() {
        let collector = RecordingActionCollector::new();
        collector.record_action(ConnectorAction::SlotCreated {
            request_id: "test".to_string(),
            num_tokens: 100,
            iteration: 0,
        });

        assert_eq!(collector.actions().len(), 1);

        collector.clear();
        assert_eq!(collector.actions().len(), 0);
    }
}
