// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cancellation protocol for offload transfers.
//!
//! The cancellation protocol ensures clean release of all blocks with confirmation
//! that no outstanding operations remain:
//!
//! 1. `cancel()` called → sets `CancelState::Requested`
//! 2. Each stage checks at safe points (between items, not during ops)
//! 3. If in-flight ops: `CancelState::Draining` → wait for completion
//! 4. Drop all `ImmutableBlock` guards → blocks released
//! 5. `CancelState::Confirmed` → `CancelConfirmation` resolves

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use tokio::sync::watch;

/// State of a cancellation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelState {
    /// Transfer is active, not cancelled
    Active,
    /// Cancel requested, waiting for checkpoint
    Requested,
    /// Draining in-flight operations
    Draining {
        /// Number of in-flight operations remaining
        in_flight: usize,
    },
    /// All operations complete, blocks released, confirmed
    Confirmed,
}

impl CancelState {
    /// Check if cancellation has been requested (including draining/confirmed states).
    pub fn is_cancelled(&self) -> bool {
        !matches!(self, CancelState::Active)
    }

    /// Check if we're in the draining phase.
    pub fn is_draining(&self) -> bool {
        matches!(self, CancelState::Draining { .. })
    }

    /// Check if cancellation is fully confirmed.
    pub fn is_confirmed(&self) -> bool {
        matches!(self, CancelState::Confirmed)
    }
}

/// Token for requesting and tracking cancellation.
///
/// The token is shared between the `TransferHandle` (user-facing) and the
/// pipeline stages (internal). When `request()` is called, stages will
/// check at safe points and transition through draining to confirmed.
#[derive(Clone)]
pub struct CancellationToken {
    /// Sender for cancellation requests
    request_tx: Arc<watch::Sender<bool>>,
    /// Receiver for cancel state updates
    state_rx: watch::Receiver<CancelState>,
}

impl CancellationToken {
    /// Create a new cancellation token pair.
    ///
    /// Returns `(token, state_tx)` where:
    /// - `token`: Clone and give to TransferHandle for user access
    /// - `state_tx`: Keep in pipeline for updating state
    pub fn new() -> (Self, CancelStateUpdater) {
        let (request_tx, request_rx) = watch::channel(false);
        let (state_tx, state_rx) = watch::channel(CancelState::Active);

        let token = CancellationToken {
            request_tx: Arc::new(request_tx),
            state_rx,
        };

        let updater = CancelStateUpdater {
            request_rx,
            state_tx,
        };

        (token, updater)
    }

    /// Request cancellation.
    ///
    /// This signals all pipeline stages to stop processing at the next safe point.
    /// Returns immediately - use `wait_confirmed()` to await full confirmation.
    pub fn request(&self) {
        let _ = self.request_tx.send(true);
    }

    /// Check if cancellation has been requested.
    pub fn is_requested(&self) -> bool {
        *self.request_tx.borrow()
    }

    /// Get the current cancellation state.
    pub fn state(&self) -> CancelState {
        *self.state_rx.borrow()
    }

    /// Check if cancellation is fully confirmed.
    pub fn is_confirmed(&self) -> bool {
        self.state().is_confirmed()
    }

    /// Create a future that resolves when cancellation is confirmed.
    ///
    /// This is the primary way to await clean release of all blocks.
    pub fn wait_confirmed(&self) -> CancelConfirmation {
        CancelConfirmation {
            state_rx: self.state_rx.clone(),
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        CancellationToken::new().0
    }
}

/// Internal updater for cancellation state.
///
/// Held by pipeline stages to update state and check for cancel requests.
pub struct CancelStateUpdater {
    /// Receiver for cancellation requests
    request_rx: watch::Receiver<bool>,
    /// Sender for state updates
    state_tx: watch::Sender<CancelState>,
}

impl CancelStateUpdater {
    /// Check if cancellation has been requested.
    pub fn is_requested(&self) -> bool {
        *self.request_rx.borrow()
    }

    /// Wait for a cancellation request (async).
    pub async fn wait_for_request(&mut self) {
        while !*self.request_rx.borrow() {
            if self.request_rx.changed().await.is_err() {
                // Channel closed, treat as cancelled
                break;
            }
        }
    }

    /// Get the current state.
    pub fn state(&self) -> CancelState {
        *self.state_tx.borrow()
    }

    /// Transition to Requested state.
    pub fn set_requested(&self) {
        let _ = self.state_tx.send(CancelState::Requested);
    }

    /// Transition to Draining state with count of in-flight operations.
    pub fn set_draining(&self, in_flight: usize) {
        let _ = self.state_tx.send(CancelState::Draining { in_flight });
    }

    /// Update the in-flight count during draining.
    pub fn update_draining(&self, in_flight: usize) {
        if in_flight == 0 {
            self.set_confirmed();
        } else {
            let _ = self.state_tx.send(CancelState::Draining { in_flight });
        }
    }

    /// Transition to Confirmed state (all blocks released).
    pub fn set_confirmed(&self) {
        let _ = self.state_tx.send(CancelState::Confirmed);
    }

    /// Subscribe to state changes.
    pub fn subscribe(&self) -> watch::Receiver<CancelState> {
        self.state_tx.subscribe()
    }
}

/// Future that resolves when cancellation is fully confirmed.
///
/// Obtained via `CancellationToken::wait_confirmed()` or `TransferHandle::cancel()`.
pub struct CancelConfirmation {
    state_rx: watch::Receiver<CancelState>,
}

impl CancelConfirmation {
    /// Wait for confirmation (async).
    ///
    /// This is the recommended way to await cancellation confirmation.
    pub async fn wait(mut self) {
        loop {
            // Check current state
            if self.state_rx.borrow().is_confirmed() {
                return;
            }

            // Wait for state change
            if self.state_rx.changed().await.is_err() {
                // Channel closed, treat as confirmed
                return;
            }
        }
    }
}

impl Future for CancelConfirmation {
    type Output = ();

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        // For manual polling, we need to store additional state.
        // Use the wait() method for simpler async/await usage.
        // This implementation checks once and requires re-polling.
        let this = self.get_mut();

        // Check current state
        if this.state_rx.borrow().is_confirmed() {
            return Poll::Ready(());
        }

        // Register waker for future notification
        // Note: watch::Receiver doesn't have a poll interface, so we
        // need to check periodically. For most use cases, use the
        // wait() method which is async and works properly.
        cx.waker().wake_by_ref();
        Poll::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_state_transitions() {
        let state = CancelState::Active;
        assert!(!state.is_cancelled());
        assert!(!state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Requested;
        assert!(state.is_cancelled());
        assert!(!state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Draining { in_flight: 5 };
        assert!(state.is_cancelled());
        assert!(state.is_draining());
        assert!(!state.is_confirmed());

        let state = CancelState::Confirmed;
        assert!(state.is_cancelled());
        assert!(!state.is_draining());
        assert!(state.is_confirmed());
    }

    #[test]
    fn test_cancellation_token_request() {
        let (token, _updater) = CancellationToken::new();

        assert!(!token.is_requested());
        assert_eq!(token.state(), CancelState::Active);

        token.request();

        assert!(token.is_requested());
    }

    #[test]
    fn test_cancellation_updater_state() {
        let (token, updater) = CancellationToken::new();

        assert_eq!(token.state(), CancelState::Active);

        updater.set_requested();
        assert_eq!(token.state(), CancelState::Requested);

        updater.set_draining(3);
        assert_eq!(token.state(), CancelState::Draining { in_flight: 3 });

        updater.update_draining(1);
        assert_eq!(token.state(), CancelState::Draining { in_flight: 1 });

        updater.update_draining(0);
        assert_eq!(token.state(), CancelState::Confirmed);
    }

    #[tokio::test]
    async fn test_cancel_confirmation_immediate() {
        let (token, updater) = CancellationToken::new();

        // Set confirmed before waiting
        updater.set_confirmed();

        // Should resolve immediately
        token.wait_confirmed().wait().await;
        assert!(token.is_confirmed());
    }

    #[tokio::test]
    async fn test_cancel_confirmation_delayed() {
        let (token, updater) = CancellationToken::new();

        let confirmation = token.wait_confirmed();

        // Spawn task to confirm after short delay
        let updater_clone = updater.state_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            let _ = updater_clone.send(CancelState::Confirmed);
        });

        // Wait for confirmation
        tokio::time::timeout(tokio::time::Duration::from_millis(100), confirmation.wait())
            .await
            .expect("Should complete within timeout");

        assert!(token.is_confirmed());
    }
}
