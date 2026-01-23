// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification system for G4 remote transfers.
//!
use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use nixl_sys::{Agent as NixlAgent, XferRequest, XferStatus};
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use tracing::warn;
use uuid::Uuid;

/// Trait for checking if a transfer operation has completed.
pub trait CompletionChecker: Send {
    /// Returns true if the transfer is complete, false if still pending.
    fn is_complete(&self) -> Result<bool>;
}

/// Completion checker that polls NIXL transfer status.
pub struct NixlStatusChecker {
    agent: NixlAgent,
    xfer_req: XferRequest,
}

impl NixlStatusChecker {
    pub fn new(agent: NixlAgent, xfer_req: XferRequest) -> Self {
        Self { agent, xfer_req }
    }
}

impl CompletionChecker for NixlStatusChecker {
    fn is_complete(&self) -> Result<bool> {
        match self.agent.get_xfer_status(&self.xfer_req) {
            Ok(status) => Ok(matches!(status, XferStatus::Success)),
            Err(e) => Err(anyhow!("NIXL transfer status check failed: {}", e)),
        }
    }
}

/// Registration message for a transfer to be polled.
pub struct RegisterTransferNotification<C: CompletionChecker = NixlStatusChecker> {
    pub uuid: Uuid,
    pub checker: C,
    pub done: oneshot::Sender<Result<()>>,
}

/// Error returned when transfer notification registration fails.
///
/// When this error is returned, the caller must fall back to inline polling
/// to check transfer completion status manually.
#[derive(Debug, Clone)]
pub enum RegistrationError {
    /// The notification handler is not available (not initialized or disabled).
    HandlerNotAvailable,
    /// The notification channel is full or closed.
    ChannelUnavailable,
}

impl std::fmt::Display for RegistrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistrationError::HandlerNotAvailable => {
                write!(f, "Notification handler not available")
            }
            RegistrationError::ChannelUnavailable => {
                write!(f, "Notification channel full or closed")
            }
        }
    }
}

impl std::error::Error for RegistrationError {}

/// Notification handle for an in-progress transfer.
///
/// This object can be awaited to block until the transfer completes.
/// The transfer is tracked by a background handler that polls for completion.
pub struct TransferCompleteNotification {
    status: oneshot::Receiver<Result<()>>,
}

impl TransferCompleteNotification {
    /// Create a new notification with the given receiver.
    pub fn new(status: oneshot::Receiver<Result<()>>) -> Self {
        Self { status }
    }

    /// Create a notification that is already completed (for synchronous transfers).
    ///
    /// This is for transfers that complete immediately without needing background
    /// polling, such as memcpy operations.
    pub fn completed() -> Self {
        let (tx, rx) = oneshot::channel();
        let _ = tx.send(Ok(()));
        Self { status: rx }
    }

    /// Wait for the transfer to complete (blocking).
    pub fn wait(self) -> Result<()> {
        self.status
            .blocking_recv()
            .map_err(|_| anyhow!("Transfer handler dropped before completion"))?
    }
}

impl std::future::Future for TransferCompleteNotification {
    type Output = Result<()>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        use std::pin::Pin;
        Pin::new(&mut self.status).poll(cx).map(|result| {
            result
                .map_err(|_| anyhow!("Transfer handler dropped before completion"))
                .and_then(|r| r)
        })
    }
}

/// Tracking struct for outstanding transfers.
struct OutstandingTransfer<C: CompletionChecker> {
    checker: C,
    done: oneshot::Sender<Result<()>>,
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
}

/// Helper function to check if a transfer should be warned about.
fn check_and_warn_slow_transfer(
    uuid: &Uuid,
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
) -> Option<Instant> {
    let elapsed = arrived_at.elapsed();
    if elapsed > Duration::from_secs(60) {
        let should_warn = last_warned_at
            .map(|last| last.elapsed() > Duration::from_secs(30))
            .unwrap_or(true);

        if should_warn {
            warn!(
                uuid = %uuid,
                elapsed_secs = elapsed.as_secs(),
                "Transfer has been pending for over 1 minute"
            );
            return Some(Instant::now());
        }
    }
    last_warned_at
}

/// Background task that polls all outstanding transfers.
///
/// This runs in a loop, checking all registered transfers every 1ms.
/// Much more efficient than per-transfer 100μs polling.
pub async fn process_transfer_notifications<C: CompletionChecker>(
    mut rx: mpsc::Receiver<RegisterTransferNotification<C>>,
) {
    let mut outstanding: HashMap<Uuid, OutstandingTransfer<C>> = HashMap::new();
    let mut check_interval = interval(Duration::from_millis(1));

    loop {
        tokio::select! {
            // Handle new transfer requests
            notification = rx.recv() => {
                match notification {
                    Some(notif) => {
                        outstanding.insert(notif.uuid, OutstandingTransfer {
                            checker: notif.checker,
                            done: notif.done,
                            arrived_at: Instant::now(),
                            last_warned_at: None,
                        });
                    }
                    None => {
                        // Channel closed, finish processing outstanding transfers then exit
                        break;
                    }
                }
            }

            // Periodically check status of outstanding transfers
            _ = check_interval.tick(), if !outstanding.is_empty() => {
                let mut completed = Vec::new();

                for (uuid, transfer) in outstanding.iter_mut() {
                    match transfer.checker.is_complete() {
                        Ok(true) => {
                            completed.push((*uuid, Ok(())));
                        }
                        Ok(false) => {
                            transfer.last_warned_at = check_and_warn_slow_transfer(
                                uuid,
                                transfer.arrived_at,
                                transfer.last_warned_at,
                            );
                        }
                        Err(e) => {
                            warn!(
                                uuid = %uuid,
                                error = %e,
                                "Transfer status check failed"
                            );
                            completed.push((*uuid, Err(e)));
                        }
                    }
                }

                // Remove completed transfers and signal completion
                for (uuid, result) in completed {
                    if let Some(transfer) = outstanding.remove(&uuid) {
                        let _ = transfer.done.send(result);
                    }
                }
            }
        }
    }

    // Channel closed, but we may still have outstanding transfers
    // Continue processing them until all are complete
    while !outstanding.is_empty() {
        check_interval.tick().await;

        let mut completed = Vec::new();

        for (uuid, transfer) in outstanding.iter() {
            match transfer.checker.is_complete() {
                Ok(true) => {
                    completed.push((*uuid, Ok(())));
                }
                Ok(false) => {
                    // Still pending
                }
                Err(e) => {
                    warn!(
                        uuid = %uuid,
                        error = %e,
                        "Transfer status check failed during shutdown"
                    );
                    completed.push((*uuid, Err(e)));
                }
            }
        }

        for (uuid, result) in completed {
            if let Some(transfer) = outstanding.remove(&uuid) {
                let _ = transfer.done.send(result);
            }
        }
    }
}

/// Sender type for registering NIXL transfer notifications.
pub type NixlNotificationSender = mpsc::Sender<RegisterTransferNotification<NixlStatusChecker>>;

/// Spawn the notification handler task and return the sender.
pub fn spawn_notification_handler(handle: &tokio::runtime::Handle) -> NixlNotificationSender {
    let (tx, rx) = mpsc::channel(256);
    handle.spawn(process_transfer_notifications(rx));
    tx
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockChecker {
        complete: std::sync::atomic::AtomicBool,
    }

    impl MockChecker {
        fn new(complete: bool) -> Self {
            Self {
                complete: std::sync::atomic::AtomicBool::new(complete),
            }
        }

        fn set_complete(&self) {
            self.complete
                .store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl CompletionChecker for MockChecker {
        fn is_complete(&self) -> Result<bool> {
            Ok(self.complete.load(std::sync::atomic::Ordering::SeqCst))
        }
    }

    impl CompletionChecker for std::sync::Arc<MockChecker> {
        fn is_complete(&self) -> Result<bool> {
            Ok(self.complete.load(std::sync::atomic::Ordering::SeqCst))
        }
    }

    #[tokio::test]
    async fn test_immediate_completion() {
        let (tx, rx) = mpsc::channel(16);
        let handler = tokio::spawn(process_transfer_notifications(rx));

        let (done_tx, done_rx) = oneshot::channel();
        let notification = RegisterTransferNotification {
            uuid: Uuid::new_v4(),
            checker: MockChecker::new(true), // Already complete
            done: done_tx,
        };

        tx.send(notification).await.unwrap();

        // Should complete quickly
        let result = tokio::time::timeout(Duration::from_millis(100), done_rx).await;
        assert!(result.is_ok());
        assert!(result.unwrap().unwrap().is_ok());

        drop(tx);
        handler.await.unwrap();
    }

    #[tokio::test]
    async fn test_delayed_completion() {
        let (tx, rx) = mpsc::channel(16);
        let handler = tokio::spawn(process_transfer_notifications(rx));

        let checker = std::sync::Arc::new(MockChecker::new(false));
        let checker_for_notification = checker.clone();
        let checker_for_task = checker.clone();

        let (done_tx, _done_rx) = oneshot::channel();
        let notification = RegisterTransferNotification {
            uuid: Uuid::new_v4(),
            checker: checker_for_notification,
            done: done_tx,
        };

        tx.send(notification).await.unwrap();

        // Complete after a short delay
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(10)).await;
            checker_for_task.set_complete();
        });

        // Give time for completion
        tokio::time::sleep(Duration::from_millis(50)).await;

        drop(tx);
        handler.await.unwrap();
    }

    #[tokio::test]
    async fn test_notification_already_completed() {
        let notification = TransferCompleteNotification::completed();
        let result = notification.await;
        assert!(result.is_ok());
    }
}
