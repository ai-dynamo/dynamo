// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async transfer coordination for the connector leader.
//!
//! This module provides a background task that tracks in-flight transfers and
//! broadcasts completion messages to workers when transfers finish.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────┐   submit_onboard/offload   ┌────────────────────┐
//! │ ConnectorLeader  │ ──────────────────────────►│ TransferCoordTask  │
//! │                  │                            │                    │
//! │(scheduler thread)│                            │ (tokio task)       │
//! └──────────────────┘                            └────────────────────┘
//!                                                          │
//!                                                          │ await notification
//!                                                          │ then broadcast
//!                                                          ▼
//!                                                 ┌────────────────────┐
//!                                                 │ TransferCoordinator│
//!                                                 │ notify_*_complete()│
//!                                                 └────────────────────┘
//! ```

use std::sync::Arc;

use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::physical::transfer::context::TransferCompleteNotification;

use super::super::TransferCoordinator;

/// Type of transfer completion.
#[derive(Debug, Clone, Copy)]
pub enum TransferKind {
    Onboard,
    Offload,
}

/// Message sent to the coordination task when a transfer is submitted.
pub struct PendingTransfer {
    pub request_id: String,
    pub kind: TransferKind,
    pub notification: TransferCompleteNotification,
}

/// Handle to submit transfers to the coordination task.
#[derive(Clone)]
pub struct TransferCoordHandle {
    tx: mpsc::UnboundedSender<PendingTransfer>,
}

impl TransferCoordHandle {
    /// Submit an onboarding transfer for tracking.
    ///
    /// The coordination task will await the notification and call
    /// `notify_onboard_complete` on the coordinator when done.
    pub fn submit_onboard(
        &self,
        request_id: String,
        notification: TransferCompleteNotification,
    ) -> Result<(), mpsc::error::SendError<PendingTransfer>> {
        self.tx.send(PendingTransfer {
            request_id,
            kind: TransferKind::Onboard,
            notification,
        })
    }

    /// Submit an offloading transfer for tracking.
    ///
    /// The coordination task will await the notification and call
    /// `notify_offload_complete` on the coordinator when done.
    pub fn submit_offload(
        &self,
        request_id: String,
        notification: TransferCompleteNotification,
    ) -> Result<(), mpsc::error::SendError<PendingTransfer>> {
        self.tx.send(PendingTransfer {
            request_id,
            kind: TransferKind::Offload,
            notification,
        })
    }
}

/// Spawns the transfer coordination task.
///
/// Returns a handle to submit transfers and a join handle for the task.
///
/// # Arguments
/// * `coordinator` - The transfer coordinator for broadcasting completion
/// * `cancellation` - Token to cancel the task
pub fn spawn_coordination_task(
    coordinator: Arc<dyn TransferCoordinator>,
    cancellation: CancellationToken,
) -> (TransferCoordHandle, tokio::task::JoinHandle<()>) {
    let (tx, rx) = mpsc::unbounded_channel();
    let handle = TransferCoordHandle { tx };

    let task = tokio::spawn(coordination_task(coordinator, rx, cancellation));

    (handle, task)
}

/// The main coordination task loop.
///
/// Receives pending transfers and spawns a sub-task for each one to:
/// 1. Await the transfer notification
/// 2. Call the appropriate notify_*_complete method
async fn coordination_task(
    coordinator: Arc<dyn TransferCoordinator>,
    mut rx: mpsc::UnboundedReceiver<PendingTransfer>,
    cancellation: CancellationToken,
) {
    loop {
        tokio::select! {
            biased;

            _ = cancellation.cancelled() => {
                tracing::debug!("Transfer coordination task cancelled");
                break;
            }

            maybe_pending = rx.recv() => {
                match maybe_pending {
                    Some(pending) => {
                        // Clone coordinator for the sub-task
                        let coord = coordinator.clone();

                        // Spawn a sub-task to await this specific transfer
                        tokio::spawn(handle_pending_transfer(coord, pending));
                    }
                    None => {
                        // Channel closed, task is shutting down
                        tracing::debug!("Transfer coordination channel closed");
                        break;
                    }
                }
            }
        }
    }
}

/// Handle a single pending transfer.
///
/// Awaits the notification and broadcasts completion to workers.
async fn handle_pending_transfer(
    coordinator: Arc<dyn TransferCoordinator>,
    pending: PendingTransfer,
) {
    let request_id = pending.request_id;
    let kind = pending.kind;

    // Check if this is an immediate completion (no await needed)
    if !pending.notification.could_yield() {
        // Already complete, broadcast immediately
        broadcast_completion(&coordinator, &request_id, kind);
        return;
    }

    // Await the transfer notification
    match pending.notification.into_future().await {
        Ok(()) => {
            tracing::debug!(
                request_id = %request_id,
                kind = ?kind,
                "Transfer completed, broadcasting to workers"
            );
            broadcast_completion(&coordinator, &request_id, kind);
        }
        Err(e) => {
            tracing::error!(
                request_id = %request_id,
                kind = ?kind,
                error = %e,
                "Transfer failed"
            );
            // TODO: Consider error handling strategy - should we still notify workers?
            // For now, we don't broadcast on failure to avoid workers thinking it succeeded.
        }
    }
}

/// Broadcast transfer completion to workers.
fn broadcast_completion(
    coordinator: &Arc<dyn TransferCoordinator>,
    request_id: &str,
    kind: TransferKind,
) {
    let result = match kind {
        TransferKind::Onboard => coordinator.notify_onboard_complete(request_id),
        TransferKind::Offload => coordinator.notify_offload_complete(request_id),
    };

    if let Err(e) = result {
        tracing::error!(
            request_id = %request_id,
            kind = ?kind,
            error = %e,
            "Failed to broadcast transfer completion to workers"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::integrations::connector::MockCoordinator;
    use std::time::Duration;

    #[tokio::test]
    async fn test_coordination_task_immediate_completion() {
        let coordinator = Arc::new(MockCoordinator::new());
        let cancellation = CancellationToken::new();

        let (handle, task) = spawn_coordination_task(coordinator, cancellation.clone());

        // Submit an immediately-complete onboard transfer
        let notification = TransferCompleteNotification::completed();
        handle
            .submit_onboard("test-req-1".to_string(), notification)
            .unwrap();

        // Give the task time to process
        tokio::time::sleep(Duration::from_millis(10)).await;

        // Cancel and wait for cleanup
        cancellation.cancel();
        let _ = task.await;
    }

    #[tokio::test]
    async fn test_coordination_task_cancellation() {
        let coordinator = Arc::new(MockCoordinator::new());
        let cancellation = CancellationToken::new();

        let (_handle, task) = spawn_coordination_task(coordinator, cancellation.clone());

        // Cancel immediately
        cancellation.cancel();

        // Task should complete cleanly
        let _ = task.await;
    }
}
