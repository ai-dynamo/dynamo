// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer completion notification system.
//!
//! This module provides abstractions for waiting on transfer completions using different
//! mechanisms: polling-based (NIXL status, CUDA events) and event-based (NIXL notifications).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{error, warn};
use uuid::Uuid;
use velo::{EventHandle, EventManager};

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

/// Per-worker telemetry for a single NIXL transfer, threaded from the
/// planner's post site ([`execute_planner_nixl_transfer`]) through
/// [`RegisterPollingNotification`] to the completion poller so the
/// `worker_xfer_complete` audit event is emitted in ONE place with the full
/// control / post / completion picture. Each TP rank is its own process with
/// its own poller, so one event is emitted per rank per transfer.
///
/// [`execute_planner_nixl_transfer`]: crate::transfer::executor
pub struct XferTelemetry {
    /// Velo messenger per-process id (== `ctx.worker_id()`). Opaque, but
    /// distinct per TP rank — group the bench's events by this.
    pub worker_id: u64,
    /// "Read" (pull) or "Write" (push).
    pub op: &'static str,
    /// Local CUDA device ordinal of the destination (for a Read/pull this is
    /// the rank's own GPU) — maps rank → GPU → GPU-local NIC in the report.
    pub device_id: u64,
    /// Logical KV blocks in this transfer (`dst_block_ids.len()`).
    pub num_blocks: usize,
    /// Coalesced NIXL descriptor count (`ops.len()`) — may be < num_blocks.
    pub num_descs: usize,
    /// Bytes actually moved by THIS rank — the per-rank shard (sum of op sizes).
    pub bytes: usize,
    /// Control/setup time: descriptor build + `create_xfer_req` (µs).
    pub ctrl_us: u64,
    /// `post_xfer_req` submit cost (µs).
    pub post_us: u64,
    /// Captured just after the post returns; `completion_us = .elapsed()` is
    /// measured by the poller when the transfer is observed complete.
    pub submitted_at: Instant,
}

impl XferTelemetry {
    /// Emit the per-worker `worker_xfer_complete` kvbm_audit event. Called from
    /// the polling handler on async completion, or inline for a synchronous
    /// completion. `completion_us` is the post→complete wall time.
    pub fn emit_complete(&self, success: bool) {
        let completion_us = self.submitted_at.elapsed().as_micros() as u64;
        tracing::info!(
            target: "kvbm_audit",
            event = "worker_xfer_complete",
            worker_id = self.worker_id,
            op = self.op,
            device_id = self.device_id,
            num_blocks = self.num_blocks,
            num_descs = self.num_descs,
            bytes = self.bytes,
            ctrl_us = self.ctrl_us,
            post_us = self.post_us,
            completion_us = completion_us,
            success = success,
        );
    }
}

/// Registration message for polling-based transfer completion.
pub struct RegisterPollingNotification<C: CompletionChecker> {
    pub uuid: Uuid,
    pub checker: C,
    pub event_handle: EventHandle,
    /// Worker-side timing for the `worker_xfer_complete` event, or `None` for
    /// transfers we don't instrument (CUDA events, staged legs).
    pub telemetry: Option<XferTelemetry>,
}

/// Tracking struct for outstanding polling-based transfers.
struct OutstandingPollingTransfer<C: CompletionChecker> {
    checker: C,
    event_handle: EventHandle,
    arrived_at: Instant,
    last_warned_at: Option<Instant>,
    telemetry: Option<XferTelemetry>,
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
    system: Arc<EventManager>,
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
                            event_handle: notif.event_handle,
                            arrived_at: Instant::now(),
                            last_warned_at: None,
                            telemetry: notif.telemetry,
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
                        // Per-worker RDMA telemetry: emit on the transition to
                        // complete (this site fires exactly once per transfer).
                        if let Some(tel) = &transfer.telemetry {
                            tel.emit_complete(result.is_ok());
                        }
                        // Signal completion via Velo event system
                        match result {
                            Ok(()) => {
                                if let Err(e) = system.trigger(transfer.event_handle) {
                                    error!(
                                        uuid = %uuid,
                                        error = %e,
                                        "Failed to trigger completion event"
                                    );
                                }
                            }
                            Err(e) => {
                                if let Err(err) = system.poison(transfer.event_handle, e.to_string()) {
                                    error!(
                                        uuid = %uuid,
                                        error = %err,
                                        "Failed to poison completion event"
                                    );
                                }
                            }
                        }
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

        for (uuid, transfer) in outstanding.iter_mut() {
            match transfer.checker.is_complete() {
                Ok(true) => completed.push((*uuid, Ok(()))),
                Ok(false) => {}
                Err(e) => completed.push((*uuid, Err(e))),
            }
        }

        for (uuid, result) in completed {
            if let Some(transfer) = outstanding.remove(&uuid) {
                // Per-worker RDMA telemetry (post-channel-close drain path).
                if let Some(tel) = &transfer.telemetry {
                    tel.emit_complete(result.is_ok());
                }
                match result {
                    Ok(()) => {
                        let _ = system.trigger(transfer.event_handle);
                    }
                    Err(e) => {
                        let _ = system.poison(transfer.event_handle, e.to_string());
                    }
                }
            }
        }
    }
}
