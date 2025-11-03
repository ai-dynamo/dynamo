// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/// Helper function to check if a transfer should be warned about and log the warning.
/// Returns the new last_warned_at time if a warning was issued.
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
