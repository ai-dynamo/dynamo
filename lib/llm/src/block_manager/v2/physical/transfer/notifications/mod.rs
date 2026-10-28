// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification system.
//!
//! This module provides abstractions for waiting on transfer completions using different
//! mechanisms: polling-based (NIXL status, CUDA events) and event-based (NIXL notifications).

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use tracing::warn;
use uuid::Uuid;

pub mod cuda_event;
pub mod nixl_events;
pub mod nixl_status;
pub mod notification;

pub use cuda_event::CudaEventChecker;
pub use nixl_events::{RegisterNixlNotification, process_nixl_notification_events};
pub use nixl_status::NixlStatusChecker;
pub use notification::TransferCompleteNotification;

/// Trait for checking if a transfer operation has completed.
/// Supports polling-based completion checks (NIXL status, CUDA events).
pub trait CompletionChecker: Send {
    /// Returns true if the transfer is complete, false if still pending.
    fn is_complete(&self) -> Result<bool>;
}

/// Registration message for polling-based transfer completion.
pub struct RegisterPollingNotification<C: CompletionChecker> {
    pub uuid: Uuid,
    pub checker: C,
    pub done: oneshot::Sender<Result<()>>,
}

/// Tracking struct for outstanding polling-based transfers.
struct OutstandingPollingTransfer<C: CompletionChecker> {
    checker: C,
    done: oneshot::Sender<Result<()>>,
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
}

/// How long a transfer may stay pending before it is reported as slow.
const SLOW_TRANSFER_WARN_AFTER: Duration = Duration::from_secs(60);
/// Minimum time between two slow-transfer warnings for the same transfer.
const SLOW_TRANSFER_WARN_REPEAT: Duration = Duration::from_secs(30);

/// Returns how long the transfer has been pending when a slow-transfer warning is due.
///
/// Shared by the completion handlers. Each handler emits the warning itself so
/// that the event keeps that handler's module as its tracing target.
fn slow_transfer_warning_due(
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
) -> Option<Duration> {
    let elapsed = arrived_at.elapsed();
    let should_warn = elapsed > SLOW_TRANSFER_WARN_AFTER
        && last_warned_at
            .map(|last| last.elapsed() > SLOW_TRANSFER_WARN_REPEAT)
            .unwrap_or(true);
    should_warn.then_some(elapsed)
}

/// Helper function to check if a transfer should be warned about and log the warning.
/// Returns the new last_warned_at time if a warning was issued.
fn check_and_warn_slow_transfer(
    uuid: &Uuid,
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
) -> Option<Instant> {
    let Some(elapsed) = slow_transfer_warning_due(arrived_at, last_warned_at) else {
        return last_warned_at;
    };
    warn!(
        uuid = %uuid,
        elapsed_secs = elapsed.as_secs(),
        "Transfer has been pending for over 1 minute"
    );
    Some(Instant::now())
}

/// Generic polling-based transfer completion handler.
/// Works with any CompletionChecker implementation (NIXL status, CUDA events, etc.)
pub async fn process_polling_notifications<C: CompletionChecker>(
    mut rx: mpsc::Receiver<RegisterPollingNotification<C>>,
) {
    let mut outstanding: HashMap<Uuid, OutstandingPollingTransfer<C>> = HashMap::new();
    let mut check_interval = interval(Duration::from_millis(1));

    loop {
        tokio::select! {
            // Handle new transfer requests
            notification = rx.recv() => {
                match notification {
                    Some(notif) => {
                        outstanding.insert(notif.uuid, OutstandingPollingTransfer {
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
                    // Check transfer status
                    match transfer.checker.is_complete() {
                        Ok(true) => {
                            // Transfer complete - mark for removal
                            completed.push((*uuid, Ok(())));
                        }
                        Ok(false) => {
                            // Transfer still in progress - check if we should warn
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
                        // Signal completion (ignore if receiver dropped)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn ago(secs: u64) -> Instant {
        Instant::now()
            .checked_sub(Duration::from_secs(secs))
            .expect("monotonic clock is too close to its start for this test")
    }

    /// Warn after 60 s pending, then at most once per 30 s.
    #[test]
    fn slow_transfer_warning_waits_and_throttles() {
        assert!(slow_transfer_warning_due(ago(59), None).is_none());
        assert!(slow_transfer_warning_due(ago(61), None).is_some());
        assert!(slow_transfer_warning_due(ago(61), Some(ago(29))).is_none());
        assert!(slow_transfer_warning_due(ago(61), Some(ago(31))).is_some());
    }
}
