// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! G4 State Machine for remote storage operations.
//!
//! Provides formal state tracking for G4 (object storage) transfers,
//! ensuring correct state transitions and enabling proper failure recovery.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// G4 Onboard State Machine
///
/// Tracks the lifecycle of a G4 onboard operation (Remote → Device).
#[derive(Debug, Clone)]
pub struct G4OnboardStateMachine {
    state: G4OnboardState,
    request_id: String,
    operation_id: Option<Uuid>,
    num_blocks: usize,
    started_at: Option<Instant>,
    hashes: Vec<u64>,
    attempted_hashes: Option<Vec<(u64, u32)>>,
    skip_on_retry: bool,
}

/// States for G4 onboard operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum G4OnboardState {
    /// No G4 operation in progress
    Idle,
    /// Querying registry for matching hashes
    Lookup,
    /// Hashes found, waiting for trigger
    Staged,
    /// Transfer request sent, executing on worker
    Transferring,
    /// Awaiting structured response from worker
    AwaitingResponse,
    /// Transfer completed successfully
    Complete,
    /// Transfer failed
    Failed,
    /// Recovered from failure, ready for retry
    Recovered,
}

/// Events that trigger state transitions
#[derive(Debug, Clone)]
pub enum G4OnboardEvent {
    /// Start lookup for matching hashes
    StartLookup,
    /// Registry returned matching hashes
    RegistryHit {
        hashes: Vec<u64>,
        positions: Vec<u32>,
    },
    /// Registry returned no matches
    RegistryMiss,
    /// Trigger the onboard transfer
    TriggerTransfer { operation_id: Uuid },
    /// ZMQ request sent to worker
    RequestSent,
    /// Received success response from worker
    ResponseSuccess,
    /// Received failure response from worker
    ResponseFailure {
        error: String,
        failed_blocks: Vec<usize>,
    },
    /// Timed out waiting for response
    Timeout { elapsed: Duration },
    /// vLLM rescheduled the request
    Reschedule,
    /// Registry eviction complete, ready for retry
    EvictionComplete,
    /// Prefill scheduled, reset for next request
    Reset,
}

/// Result of a state transition
#[derive(Debug, Clone)]
pub struct G4TransitionResult {
    /// New state after transition
    pub new_state: G4OnboardState,
    /// Actions to take as a result of this transition
    pub actions: Vec<G4Action>,
}

/// Actions to take after a state transition
#[derive(Debug, Clone)]
pub enum G4Action {
    /// Query the registry for matching hashes
    QueryRegistry { hashes: Vec<u64> },
    /// Send RemoteTransferRequest to worker
    SendTransferRequest { operation_id: Uuid },
    /// Send failure notification (for timeout)
    SendFailureNotification {
        operation_id: Uuid,
        error: String,
        failed_blocks: Vec<usize>,
    },
    /// Evict stale hashes from registry
    EvictFromRegistry { hashes: Vec<(u64, u32)> },
    /// Mark blocks as failed (for vLLM)
    MarkBlocksFailed { block_ids: Vec<usize> },
    /// Log state transition
    Log { message: String },
}

/// Error for invalid state transitions
#[derive(Debug, Clone)]
pub struct G4StateError {
    pub from_state: G4OnboardState,
    pub event: String,
    pub message: String,
}

impl std::fmt::Display for G4StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Invalid g4 state transition: {:?} + {} - {}",
            self.from_state, self.event, self.message
        )
    }
}

impl std::error::Error for G4StateError {}

impl G4OnboardStateMachine {
    /// Create a new state machine for a request
    pub fn new(request_id: String) -> Self {
        Self {
            state: G4OnboardState::Idle,
            request_id,
            operation_id: None,
            num_blocks: 0,
            started_at: None,
            hashes: Vec::new(),
            attempted_hashes: None,
            skip_on_retry: false,
        }
    }

    /// Get current state
    pub fn state(&self) -> G4OnboardState {
        self.state
    }

    /// Get request ID
    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Get operation ID (if transfer started)
    pub fn operation_id(&self) -> Option<Uuid> {
        self.operation_id
    }

    /// Check if G4 should be skipped on retry
    pub fn skip_on_retry(&self) -> bool {
        self.skip_on_retry
    }

    /// Get hashes that were attempted (for registry eviction)
    pub fn attempted_hashes(&self) -> Option<&[(u64, u32)]> {
        self.attempted_hashes.as_deref()
    }

    /// Take attempted hashes (consumes them)
    pub fn take_attempted_hashes(&mut self) -> Option<Vec<(u64, u32)>> {
        self.attempted_hashes.take()
    }

    /// Get elapsed time since transfer started
    pub fn elapsed(&self) -> Option<Duration> {
        self.started_at.map(|t| t.elapsed())
    }

    /// Process an event and transition state
    pub fn transition(
        &mut self,
        event: G4OnboardEvent,
    ) -> Result<G4TransitionResult, G4StateError> {
        use G4OnboardEvent::*;
        use G4OnboardState::*;

        let (new_state, actions) = match (&self.state, event) {
            // Idle -> Lookup
            (Idle, StartLookup) => (
                Lookup,
                vec![G4Action::Log {
                    message: format!("[{}] g4 lookup started", self.request_id),
                }],
            ),

            // Lookup -> Staged (registry hit)
            (Lookup, RegistryHit { hashes, positions }) => {
                self.num_blocks = hashes.len();
                self.hashes = hashes.clone();
                self.attempted_hashes = Some(
                    hashes
                        .iter()
                        .copied()
                        .zip(positions.iter().copied())
                        .collect(),
                );
                (
                    Staged,
                    vec![G4Action::Log {
                        message: format!(
                            "[{}] g4 registry hit: {} blocks staged",
                            self.request_id, self.num_blocks
                        ),
                    }],
                )
            }

            // Lookup -> Idle (registry miss)
            (Lookup, RegistryMiss) => (
                Idle,
                vec![G4Action::Log {
                    message: format!("[{}] g4 registry miss", self.request_id),
                }],
            ),

            // Staged -> Transferring
            (Staged, TriggerTransfer { operation_id }) => {
                self.operation_id = Some(operation_id);
                self.started_at = Some(Instant::now());
                (
                    Transferring,
                    vec![
                        G4Action::SendTransferRequest { operation_id },
                        G4Action::Log {
                            message: format!(
                                "[{}] g4 transfer triggered: {} blocks, op={}",
                                self.request_id, self.num_blocks, operation_id
                            ),
                        },
                    ],
                )
            }

            // Transferring -> AwaitingResponse
            (Transferring, RequestSent) => (
                AwaitingResponse,
                vec![G4Action::Log {
                    message: format!("[{}] g4 request sent, awaiting response", self.request_id),
                }],
            ),

            // AwaitingResponse -> Complete (success)
            (AwaitingResponse, ResponseSuccess) => {
                let elapsed = self.elapsed().unwrap_or_default();
                // Clear attempted hashes on success (no eviction needed)
                self.attempted_hashes = None;
                (
                    Complete,
                    vec![G4Action::Log {
                        message: format!(
                            "[{}] g4 transfer complete: {} blocks in {:?}",
                            self.request_id, self.num_blocks, elapsed
                        ),
                    }],
                )
            }

            // AwaitingResponse -> Failed (failure response)
            (
                AwaitingResponse,
                ResponseFailure {
                    error,
                    failed_blocks,
                },
            ) => {
                let elapsed = self.elapsed().unwrap_or_default();
                (
                    Failed,
                    vec![
                        G4Action::MarkBlocksFailed {
                            block_ids: failed_blocks,
                        },
                        G4Action::Log {
                            message: format!(
                                "[{}] g4 transfer failed after {:?}: {}",
                                self.request_id, elapsed, error
                            ),
                        },
                    ],
                )
            }

            // AwaitingResponse -> Failed (timeout)
            (AwaitingResponse, Timeout { elapsed }) => {
                let operation_id = self.operation_id.unwrap_or_else(Uuid::nil);
                let failed_blocks: Vec<usize> = (0..self.num_blocks).collect();
                (
                    Failed,
                    vec![
                        G4Action::SendFailureNotification {
                            operation_id,
                            error: format!("g4 transfer timed out after {:?}", elapsed),
                            failed_blocks: failed_blocks.clone(),
                        },
                        G4Action::MarkBlocksFailed {
                            block_ids: failed_blocks,
                        },
                        G4Action::Log {
                            message: format!(
                                "[{}] g4 transfer timed out after {:?}",
                                self.request_id, elapsed
                            ),
                        },
                    ],
                )
            }

            // Failed -> Recovered (vLLM rescheduled)
            (Failed, Reschedule) => {
                self.skip_on_retry = true;
                let mut actions = vec![];

                // Evict stale hashes from registry
                if let Some(hashes) = self.attempted_hashes.as_ref() {
                    actions.push(G4Action::EvictFromRegistry {
                        hashes: hashes.clone(),
                    });
                }

                actions.push(G4Action::Log {
                    message: format!("[{}] g4 recovery: skip_on_retry=true", self.request_id),
                });

                (Recovered, actions)
            }

            // Recovered -> Idle (eviction complete)
            (Recovered, EvictionComplete) => {
                self.attempted_hashes = None;
                (
                    Idle,
                    vec![G4Action::Log {
                        message: format!(
                            "[{}] g4 eviction complete, ready for retry",
                            self.request_id
                        ),
                    }],
                )
            }

            // Complete -> Idle (reset for next request)
            (Complete, Reset) | (Idle, Reset) => {
                self.reset_state();
                (Idle, vec![])
            }

            // Invalid transition
            (state, event) => {
                return Err(G4StateError {
                    from_state: *state,
                    event: format!("{:?}", event),
                    message: "Invalid state transition".to_string(),
                });
            }
        };

        self.state = new_state;

        Ok(G4TransitionResult { new_state, actions })
    }

    /// Reset internal state (but preserve skip_on_retry)
    fn reset_state(&mut self) {
        self.operation_id = None;
        self.num_blocks = 0;
        self.started_at = None;
        self.hashes.clear();
        // Note: skip_on_retry is preserved across resets
    }

    /// Check if currently in a transfer operation
    pub fn is_transferring(&self) -> bool {
        matches!(
            self.state,
            G4OnboardState::Transferring | G4OnboardState::AwaitingResponse
        )
    }

    /// Check if failed and needs recovery
    pub fn needs_recovery(&self) -> bool {
        matches!(self.state, G4OnboardState::Failed)
    }

    /// Force transition to Failed state (for external error handling)
    pub fn force_fail(&mut self, error: &str) {
        tracing::warn!(
            request_id = %self.request_id,
            current_state = ?self.state,
            error = %error,
            "Forcing g4 state to Failed"
        );
        self.state = G4OnboardState::Failed;
    }
}

// ============================================================================
// G4 Offload State Machine
// ============================================================================

/// G4 Offload State Machine
///
/// Tracks the lifecycle of a G4 offload operation (Device → Host → Remote).
#[derive(Debug, Clone)]
pub struct G4OffloadStateMachine {
    state: G4OffloadState,
    request_id: String,
    operation_id: Option<Uuid>,
    num_blocks: usize,
    started_at: Option<Instant>,
    hashes: Vec<(u64, u32)>,
}

/// States for G4 offload operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum G4OffloadState {
    /// No offload in progress
    Idle,
    /// Device to Host transfer in progress
    D2H,
    /// D2H complete, H2O pending (checking backpressure)
    H2OPending,
    /// Host to Object transfer in progress
    H2OActive,
    /// Awaiting ZMQ response from worker
    AwaitingResponse,
    /// Registering hashes in distributed registry
    Registering,
    /// Offload completed successfully (data confirmed in object storage)
    Complete,
    /// Offload failed (confirmed failure OR timeout - don't register!)
    Failed,
}

/// Events for offload state transitions
#[derive(Debug, Clone)]
pub enum G4OffloadEvent {
    /// Start D2H transfer
    StartD2H { num_blocks: usize },
    /// D2H transfer completed
    D2HComplete { hashes: Vec<(u64, u32)> },
    /// Backpressure check passed, can proceed with H2O
    H2OPermitAcquired { operation_id: Uuid },
    /// Backpressure - skip H2O
    H2OSkipped,
    /// ZMQ request sent, awaiting response
    RequestSent,
    /// H2O transfer completed successfully (confirmed by worker response)
    H2OComplete,
    /// H2O transfer failed (worker reported failure OR timeout - no registry update!)
    H2OFailed { error: String },
    /// Registry registration complete
    Registered,
    /// Reset state
    Reset,
}

impl G4OffloadStateMachine {
    /// Create a new offload state machine
    pub fn new(request_id: String) -> Self {
        Self {
            state: G4OffloadState::Idle,
            request_id,
            operation_id: None,
            num_blocks: 0,
            started_at: None,
            hashes: Vec::new(),
        }
    }

    /// Get current state
    pub fn state(&self) -> G4OffloadState {
        self.state
    }

    pub fn request_id(&self) -> &str {
        &self.request_id
    }

    /// Process an event and transition state
    pub fn transition(&mut self, event: G4OffloadEvent) -> Result<G4OffloadState, G4StateError> {
        use G4OffloadEvent::*;
        use G4OffloadState::*;

        let new_state = match (&self.state, event) {
            (Idle, StartD2H { num_blocks }) => {
                self.num_blocks = num_blocks;
                self.started_at = Some(Instant::now());
                D2H
            }

            (D2H, D2HComplete { hashes }) => {
                self.hashes = hashes;
                H2OPending
            }

            (H2OPending, H2OPermitAcquired { operation_id }) => {
                self.operation_id = Some(operation_id);
                H2OActive
            }

            (H2OPending, H2OSkipped) => {
                // D2H blocks stay in host, no H2O
                Complete
            }

            (H2OActive, RequestSent) => AwaitingResponse,

            (H2OActive, H2OComplete) | (AwaitingResponse, H2OComplete) => {
                // Worker confirmed success - safe to register
                Registering
            }

            (H2OActive, H2OFailed { .. }) | (AwaitingResponse, H2OFailed { .. }) => {
                // Worker reported failure OR timeout - DO NOT register!
                // Data may be orphaned in object storage, but that's safer than
                // having registry entries pointing to non-existent data.
                Failed
            }

            (Registering, Registered) => Complete,

            (Complete, Reset) | (Failed, Reset) | (Idle, Reset) => {
                self.operation_id = None;
                self.num_blocks = 0;
                self.started_at = None;
                self.hashes.clear();
                Idle
            }

            (state, event) => {
                return Err(G4StateError {
                    from_state: match state {
                        Idle => G4OnboardState::Idle,
                        D2H => G4OnboardState::Transferring,
                        H2OPending => G4OnboardState::AwaitingResponse,
                        H2OActive => G4OnboardState::Transferring,
                        AwaitingResponse => G4OnboardState::AwaitingResponse,
                        Registering => G4OnboardState::AwaitingResponse,
                        Complete => G4OnboardState::Complete,
                        Failed => G4OnboardState::Failed,
                    },
                    event: format!("{:?}", event),
                    message: "Invalid offload state transition".to_string(),
                });
            }
        };

        self.state = new_state;
        Ok(new_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onboard_happy_path() {
        let mut sm = G4OnboardStateMachine::new("test-req".to_string());
        assert_eq!(sm.state(), G4OnboardState::Idle);

        // Idle -> Lookup
        let result = sm.transition(G4OnboardEvent::StartLookup).unwrap();
        assert_eq!(result.new_state, G4OnboardState::Lookup);

        // Lookup -> Staged
        let result = sm
            .transition(G4OnboardEvent::RegistryHit {
                hashes: vec![1, 2, 3],
                positions: vec![0, 1, 2],
            })
            .unwrap();
        assert_eq!(result.new_state, G4OnboardState::Staged);
        assert!(sm.attempted_hashes().is_some());

        // Staged -> Transferring
        let op_id = Uuid::new_v4();
        let result = sm
            .transition(G4OnboardEvent::TriggerTransfer {
                operation_id: op_id,
            })
            .unwrap();
        assert_eq!(result.new_state, G4OnboardState::Transferring);
        assert_eq!(sm.operation_id(), Some(op_id));

        // Transferring -> AwaitingResponse
        let result = sm.transition(G4OnboardEvent::RequestSent).unwrap();
        assert_eq!(result.new_state, G4OnboardState::AwaitingResponse);

        // AwaitingResponse -> Complete
        let result = sm.transition(G4OnboardEvent::ResponseSuccess).unwrap();
        assert_eq!(result.new_state, G4OnboardState::Complete);
        assert!(sm.attempted_hashes().is_none()); // Cleared on success
    }

    #[test]
    fn test_onboard_failure_recovery() {
        let mut sm = G4OnboardStateMachine::new("test-req".to_string());

        // Setup: Idle -> Lookup -> Staged -> Transferring -> AwaitingResponse
        sm.transition(G4OnboardEvent::StartLookup).unwrap();
        sm.transition(G4OnboardEvent::RegistryHit {
            hashes: vec![1, 2],
            positions: vec![0, 1],
        })
        .unwrap();
        sm.transition(G4OnboardEvent::TriggerTransfer {
            operation_id: Uuid::new_v4(),
        })
        .unwrap();
        sm.transition(G4OnboardEvent::RequestSent).unwrap();

        // AwaitingResponse -> Failed (failure response)
        let result = sm
            .transition(G4OnboardEvent::ResponseFailure {
                error: "test error".to_string(),
                failed_blocks: vec![0, 1],
            })
            .unwrap();
        assert_eq!(result.new_state, G4OnboardState::Failed);
        assert!(
            result
                .actions
                .iter()
                .any(|a| matches!(a, G4Action::MarkBlocksFailed { .. }))
        );

        // Failed -> Recovered (reschedule)
        let result = sm.transition(G4OnboardEvent::Reschedule).unwrap();
        assert_eq!(result.new_state, G4OnboardState::Recovered);
        assert!(sm.skip_on_retry());
        assert!(
            result
                .actions
                .iter()
                .any(|a| matches!(a, G4Action::EvictFromRegistry { .. }))
        );

        // Recovered -> Idle
        let result = sm.transition(G4OnboardEvent::EvictionComplete).unwrap();
        assert_eq!(result.new_state, G4OnboardState::Idle);
        assert!(sm.skip_on_retry()); // Still true after recovery
    }

    #[test]
    fn test_onboard_timeout() {
        let mut sm = G4OnboardStateMachine::new("test-req".to_string());

        // Setup to AwaitingResponse
        sm.transition(G4OnboardEvent::StartLookup).unwrap();
        sm.transition(G4OnboardEvent::RegistryHit {
            hashes: vec![1],
            positions: vec![0],
        })
        .unwrap();
        sm.transition(G4OnboardEvent::TriggerTransfer {
            operation_id: Uuid::new_v4(),
        })
        .unwrap();
        sm.transition(G4OnboardEvent::RequestSent).unwrap();

        // Timeout
        let result = sm
            .transition(G4OnboardEvent::Timeout {
                elapsed: Duration::from_secs(30),
            })
            .unwrap();
        assert_eq!(result.new_state, G4OnboardState::Failed);
        assert!(
            result
                .actions
                .iter()
                .any(|a| matches!(a, G4Action::SendFailureNotification { .. }))
        );
    }

    #[test]
    fn test_invalid_transition() {
        let mut sm = G4OnboardStateMachine::new("test-req".to_string());

        // Can't go from Idle directly to Complete
        let result = sm.transition(G4OnboardEvent::ResponseSuccess);
        assert!(result.is_err());
    }

    #[test]
    fn test_offload_happy_path() {
        let mut sm = G4OffloadStateMachine::new("test-req".to_string());
        assert_eq!(sm.state(), G4OffloadState::Idle);

        // Start D2H
        sm.transition(G4OffloadEvent::StartD2H { num_blocks: 4 })
            .unwrap();
        assert_eq!(sm.state(), G4OffloadState::D2H);

        // D2H complete
        sm.transition(G4OffloadEvent::D2HComplete {
            hashes: vec![(1, 0), (2, 1)],
        })
        .unwrap();
        assert_eq!(sm.state(), G4OffloadState::H2OPending);

        // H2O permit acquired
        sm.transition(G4OffloadEvent::H2OPermitAcquired {
            operation_id: Uuid::new_v4(),
        })
        .unwrap();
        assert_eq!(sm.state(), G4OffloadState::H2OActive);

        // H2O complete
        sm.transition(G4OffloadEvent::H2OComplete).unwrap();
        assert_eq!(sm.state(), G4OffloadState::Registering);

        // Registered
        sm.transition(G4OffloadEvent::Registered).unwrap();
        assert_eq!(sm.state(), G4OffloadState::Complete);
    }

    #[test]
    fn test_offload_backpressure() {
        let mut sm = G4OffloadStateMachine::new("test-req".to_string());

        sm.transition(G4OffloadEvent::StartD2H { num_blocks: 4 })
            .unwrap();
        sm.transition(G4OffloadEvent::D2HComplete {
            hashes: vec![(1, 0)],
        })
        .unwrap();

        // Backpressure - skip H2O
        sm.transition(G4OffloadEvent::H2OSkipped).unwrap();
        assert_eq!(sm.state(), G4OffloadState::Complete);
    }
}
