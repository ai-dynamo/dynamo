// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL notification-based completion handler.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use nixl_sys::{Agent as NixlAgent, NotificationMap, XferRequest};
use tokio::sync::{mpsc, oneshot};
use tokio::time::interval;
use tracing::warn;
use uuid::Uuid;

/// Registration message for NIXL notification-based transfer completion.
pub struct RegisterNixlNotification {
    pub uuid: Uuid,
    pub xfer_req: XferRequest,
    pub done: oneshot::Sender<Result<()>>,
}

/// Tracking struct for outstanding NIXL notification transfers.
struct OutstandingTransfer {
    #[allow(dead_code)] // Kept for potential future cleanup or debugging
    xfer_req: XferRequest,
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

/// NIXL notification-based transfer completion handler.
/// Fetches notifications in batches and matches them against outstanding transfers.
pub async fn process_nixl_notification_events(
    agent: NixlAgent,
    mut rx: mpsc::Receiver<RegisterNixlNotification>,
) {
    let mut outstanding: HashMap<Uuid, OutstandingTransfer> = HashMap::new();
    let mut check_interval = interval(Duration::from_millis(1));

    loop {
        tokio::select! {
            // Handle new transfer requests
            notification = rx.recv() => {
                match notification {
                    Some(notif) => {
                        outstanding.insert(notif.uuid, OutstandingTransfer {
                            xfer_req: notif.xfer_req,
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

            // Periodically fetch and process notifications
            _ = check_interval.tick(), if !outstanding.is_empty() => {
                // Create notification map inside this branch to avoid Send issues
                let mut notif_map = match NotificationMap::new() {
                    Ok(map) => map,
                    Err(e) => {
                        warn!(error = %e, "Failed to create notification map");
                        continue;
                    }
                };

                // Fetch all pending notifications
                if let Err(e) = agent.get_notifications(&mut notif_map, None) {
                    warn!(error = %e, "Failed to fetch NIXL notifications");
                    continue;
                }

                // Process notifications and match against outstanding transfers
                let notifications = match notif_map.take_notifs() {
                    Ok(notifs) => notifs,
                    Err(e) => {
                        warn!(error = %e, "Failed to extract notifications from map");
                        continue;
                    }
                };

                let mut completed = Vec::new();

                // Iterate through all notifications
                for (_agent_name, notif_strings) in notifications {
                    for notif_str in notif_strings {
                        // Try to parse notification as UUID
                        // NOTE: This assumes notifications contain UUIDs.
                        // The actual format may be different and may need adjustment.
                        if let Ok(notif_uuid) = Uuid::parse_str(&notif_str) {
                            if outstanding.contains_key(&notif_uuid) {
                                completed.push(notif_uuid);
                            } else {
                                // Notification arrived before we started waiting for it
                                // This is the race condition we need to handle
                                warn!(
                                    uuid = %notif_uuid,
                                    "Received notification for transfer not in outstanding map (early arrival)"
                                );
                            }
                        }
                    }
                }

                // Check for slow transfers and update warnings
                for (uuid, transfer) in outstanding.iter_mut() {
                    if !completed.contains(uuid) {
                        transfer.last_warned_at = check_and_warn_slow_transfer(
                            uuid,
                            transfer.arrived_at,
                            transfer.last_warned_at,
                        );
                    }
                }

                // Remove completed transfers and signal completion
                for uuid in completed {
                    if let Some(transfer) = outstanding.remove(&uuid) {
                        let _ = transfer.done.send(Ok(()));
                    }
                }
            }
        }
    }

    // Channel closed, but we may still have outstanding transfers
    // Continue processing them until all are complete
    while !outstanding.is_empty() {
        check_interval.tick().await;

        let mut notif_map = match NotificationMap::new() {
            Ok(map) => map,
            Err(_) => continue,
        };

        if let Ok(()) = agent.get_notifications(&mut notif_map, None)
            && let Ok(notifications) = notif_map.take_notifs()
        {
            let mut completed = Vec::new();

            for (_agent_name, notif_strings) in notifications {
                for notif_str in notif_strings {
                    if let Ok(notif_uuid) = Uuid::parse_str(&notif_str)
                        && outstanding.contains_key(&notif_uuid)
                    {
                        completed.push(notif_uuid);
                    }
                }
            }

            for uuid in completed {
                if let Some(transfer) = outstanding.remove(&uuid) {
                    let _ = transfer.done.send(Ok(()));
                }
            }
        }
    }
}
