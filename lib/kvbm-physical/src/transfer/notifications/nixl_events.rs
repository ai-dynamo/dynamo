// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL notification-based completion handler.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_memory::nixl::{Agent as NixlAgent, NotificationMap, XferRequest};
use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{error, warn};
use uuid::Uuid;
use velo_events::{EventHandle, EventManager};

/// Registration message for NIXL notification-based transfer completion.
pub struct RegisterNixlNotification {
    pub uuid: Uuid,
    pub xfer_req: XferRequest,
    pub event_handle: EventHandle,
}

// ── Abstractions for testability ────────────────────────────────────

/// Trait abstracting notification polling (wraps NixlAgent + NotificationMap).
trait NotificationSource: Send {
    fn poll_notifications(&self) -> anyhow::Result<HashMap<String, Vec<String>>>;
}

/// Production implementation backed by a real NIXL agent.
struct NixlNotificationSource {
    agent: NixlAgent,
}

impl NotificationSource for NixlNotificationSource {
    fn poll_notifications(&self) -> anyhow::Result<HashMap<String, Vec<String>>> {
        let mut notif_map = NotificationMap::new()?;
        self.agent.get_notifications(&mut notif_map, None)?;
        Ok(notif_map.take_notifs()?)
    }
}

/// Trait for registration messages, abstracting away `XferRequest` for testability.
///
/// `Payload` is kept alive in the outstanding map for the duration of the transfer
/// (e.g. `XferRequest` must not be dropped while the NIXL transfer is in flight).
trait Registration: Send {
    type Payload: Send;
    fn decompose(self) -> (Uuid, EventHandle, Self::Payload);
}

impl Registration for RegisterNixlNotification {
    type Payload = XferRequest;
    fn decompose(self) -> (Uuid, EventHandle, XferRequest) {
        (self.uuid, self.event_handle, self.xfer_req)
    }
}

// ── Internal types ──────────────────────────────────────────────────

/// Tracking struct for outstanding transfers.
struct OutstandingTransfer<T> {
    #[allow(dead_code)] // Kept alive for the duration of the transfer
    _payload: T,
    event_handle: EventHandle,
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

// ── Core processing loop ────────────────────────────────────────────

/// Generic notification event loop, parameterized over the notification source
/// and registration message type for testability.
async fn process_events_core<S: NotificationSource, R: Registration>(
    source: S,
    mut rx: mpsc::Receiver<R>,
    system: Arc<EventManager>,
) {
    let mut outstanding: HashMap<Uuid, OutstandingTransfer<R::Payload>> = HashMap::new();
    let mut early_arrivals: HashSet<Uuid> = HashSet::new();
    let mut check_interval = interval(Duration::from_millis(1));

    loop {
        tokio::select! {
            // Handle new transfer requests
            notification = rx.recv() => {
                match notification {
                    Some(reg) => {
                        let (uuid, event_handle, payload) = reg.decompose();
                        if early_arrivals.remove(&uuid) {
                            // Notification arrived before registration — complete immediately.
                            // Payload is dropped here; the transfer is already done.
                            drop(payload);
                            if let Err(e) = system.trigger(event_handle) {
                                error!(
                                    uuid = %uuid,
                                    error = %e,
                                    "Failed to trigger completion event for early arrival"
                                );
                            }
                        } else {
                            outstanding.insert(uuid, OutstandingTransfer {
                                _payload: payload,
                                event_handle,
                                arrived_at: Instant::now(),
                                last_warned_at: None,
                            });
                        }
                    }
                    None => {
                        // Channel closed, finish processing outstanding transfers then exit
                        break;
                    }
                }
            }

            // Periodically fetch and process notifications
            _ = check_interval.tick(), if !outstanding.is_empty() => {
                let notifications = match source.poll_notifications() {
                    Ok(notifs) => notifs,
                    Err(e) => {
                        warn!(error = %e, "Failed to fetch notifications");
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
                                early_arrivals.insert(notif_uuid);
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
                    if let Some(transfer) = outstanding.remove(&uuid)
                        && let Err(e) = system.trigger(transfer.event_handle) {
                            error!(
                                uuid = %uuid,
                                error = %e,
                                "Failed to trigger completion event"
                            );
                        }
                }
            }
        }
    }

    // Channel closed, but we may still have outstanding transfers
    // Continue processing them until all are complete
    while !outstanding.is_empty() {
        check_interval.tick().await;

        if let Ok(notifications) = source.poll_notifications() {
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
                    let _ = system.trigger(transfer.event_handle);
                }
            }
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────

/// NIXL notification-based transfer completion handler.
/// Fetches notifications in batches and matches them against outstanding transfers.
pub async fn process_nixl_notification_events(
    agent: NixlAgent,
    rx: mpsc::Receiver<RegisterNixlNotification>,
    system: Arc<EventManager>,
) {
    process_events_core(NixlNotificationSource { agent }, rx, system).await
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use tokio::task::yield_now;
    use velo_events::EventStatus;

    // ── Mock notification source ────────────────────────────────────

    type NotificationQueue = Arc<Mutex<VecDeque<anyhow::Result<HashMap<String, Vec<String>>>>>>;

    struct MockNotificationSource {
        queue: NotificationQueue,
    }

    struct MockControl {
        queue: NotificationQueue,
    }

    fn mock_source() -> (MockNotificationSource, MockControl) {
        let queue = Arc::new(Mutex::new(VecDeque::new()));
        (
            MockNotificationSource {
                queue: queue.clone(),
            },
            MockControl { queue },
        )
    }

    impl NotificationSource for MockNotificationSource {
        fn poll_notifications(&self) -> anyhow::Result<HashMap<String, Vec<String>>> {
            self.queue
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or(Ok(HashMap::new()))
        }
    }

    impl MockControl {
        fn push_notification(&self, uuid: Uuid) {
            let mut map = HashMap::new();
            map.insert("test_agent".to_string(), vec![uuid.to_string()]);
            self.queue.lock().unwrap().push_back(Ok(map));
        }

        fn push_error(&self) {
            self.queue
                .lock()
                .unwrap()
                .push_back(Err(anyhow::anyhow!("mock poll error")));
        }
    }

    // ── Test registration type (no XferRequest needed) ──────────────

    struct TestRegistration {
        uuid: Uuid,
        event_handle: EventHandle,
    }

    impl Registration for TestRegistration {
        type Payload = ();
        fn decompose(self) -> (Uuid, EventHandle, ()) {
            (self.uuid, self.event_handle, ())
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────

    /// Advance paused tokio time and yield so spawned tasks can process.
    async fn tick() {
        tokio::time::advance(Duration::from_millis(2)).await;
        yield_now().await;
    }

    // ── check_and_warn_slow_transfer unit tests ─────────────────────

    #[test]
    fn warn_under_threshold_returns_none() {
        let result = check_and_warn_slow_transfer(
            &Uuid::new_v4(),
            Instant::now() - Duration::from_secs(30),
            None,
        );
        assert!(result.is_none());
    }

    #[test]
    fn warn_over_threshold_first_time() {
        let before = Instant::now();
        let result = check_and_warn_slow_transfer(
            &Uuid::new_v4(),
            Instant::now() - Duration::from_secs(61),
            None,
        );
        let after = Instant::now();
        let t = result.expect("should have warned");
        assert!(t >= before && t <= after);
    }

    #[test]
    fn warn_throttled_within_30s() {
        let last = Instant::now() - Duration::from_secs(10);
        let result = check_and_warn_slow_transfer(
            &Uuid::new_v4(),
            Instant::now() - Duration::from_secs(90),
            Some(last),
        );
        assert_eq!(result, Some(last));
    }

    #[test]
    fn warn_throttle_expired_after_30s() {
        let before = Instant::now();
        let result = check_and_warn_slow_transfer(
            &Uuid::new_v4(),
            Instant::now() - Duration::from_secs(120),
            Some(Instant::now() - Duration::from_secs(35)),
        );
        let t = result.expect("should have re-warned");
        assert!(t >= before);
    }

    // ── Integration tests for process_events_core ───────────────────

    #[tokio::test(start_paused = true)]
    async fn normal_flow_register_then_notify() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid = Uuid::new_v4();
        let event = system.new_event()?;
        let handle = event.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register transfer
        tx.send(TestRegistration {
            uuid,
            event_handle: handle,
        })
        .await?;
        yield_now().await;

        // Deliver notification
        control.push_notification(uuid);
        tick().await;

        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn early_arrival_notify_before_register() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid_early = Uuid::new_v4();
        let event_early = system.new_event()?;
        let handle_early = event_early.into_handle();

        // Filler transfer to make the tick branch fire
        let uuid_filler = Uuid::new_v4();
        let event_filler = system.new_event()?;
        let handle_filler = event_filler.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register filler so outstanding is non-empty
        tx.send(TestRegistration {
            uuid: uuid_filler,
            event_handle: handle_filler,
        })
        .await?;
        yield_now().await;

        // Notification arrives for uuid_early before it's registered (early arrival)
        control.push_notification(uuid_early);
        tick().await;

        // Now register uuid_early — should complete immediately from early_arrivals
        tx.send(TestRegistration {
            uuid: uuid_early,
            event_handle: handle_early,
        })
        .await?;
        yield_now().await;

        assert_eq!(system.poll(handle_early)?, EventStatus::Ready);

        // Cleanup filler
        control.push_notification(uuid_filler);
        tick().await;

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn multiple_transfers_complete_independently() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid_a = Uuid::new_v4();
        let event_a = system.new_event()?;
        let handle_a = event_a.into_handle();

        let uuid_b = Uuid::new_v4();
        let event_b = system.new_event()?;
        let handle_b = event_b.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register both
        tx.send(TestRegistration {
            uuid: uuid_a,
            event_handle: handle_a,
        })
        .await?;
        tx.send(TestRegistration {
            uuid: uuid_b,
            event_handle: handle_b,
        })
        .await?;
        yield_now().await;

        // Complete B only
        control.push_notification(uuid_b);
        tick().await;

        assert_eq!(system.poll(handle_b)?, EventStatus::Ready);
        assert_eq!(system.poll(handle_a)?, EventStatus::Pending);

        // Complete A
        control.push_notification(uuid_a);
        tick().await;

        assert_eq!(system.poll(handle_a)?, EventStatus::Ready);

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn mixed_ordering_early_and_normal() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid_early = Uuid::new_v4();
        let event_early = system.new_event()?;
        let handle_early = event_early.into_handle();

        let uuid_normal = Uuid::new_v4();
        let event_normal = system.new_event()?;
        let handle_normal = event_normal.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register normal first (makes outstanding non-empty for tick)
        tx.send(TestRegistration {
            uuid: uuid_normal,
            event_handle: handle_normal,
        })
        .await?;
        yield_now().await;

        // Early arrival for uuid_early
        control.push_notification(uuid_early);
        tick().await;

        // Register uuid_early — triggers immediately
        tx.send(TestRegistration {
            uuid: uuid_early,
            event_handle: handle_early,
        })
        .await?;
        yield_now().await;

        assert_eq!(system.poll(handle_early)?, EventStatus::Ready);
        assert_eq!(system.poll(handle_normal)?, EventStatus::Pending);

        // Complete normal
        control.push_notification(uuid_normal);
        tick().await;

        assert_eq!(system.poll(handle_normal)?, EventStatus::Ready);

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn channel_close_drains_outstanding() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid = Uuid::new_v4();
        let event = system.new_event()?;
        let handle = event.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register
        tx.send(TestRegistration {
            uuid,
            event_handle: handle,
        })
        .await?;
        yield_now().await;

        // Close channel — enters drain loop
        drop(tx);
        yield_now().await;

        // Deliver notification during drain
        control.push_notification(uuid);
        tick().await;

        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn duplicate_notification_is_harmless() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid = Uuid::new_v4();
        let event = system.new_event()?;
        let handle = event.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register
        tx.send(TestRegistration {
            uuid,
            event_handle: handle,
        })
        .await?;
        yield_now().await;

        // Deliver same notification twice (two separate polls)
        control.push_notification(uuid);
        tick().await;

        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        // Second notification — UUID no longer in outstanding, goes to early_arrivals
        // This should not panic or cause issues
        control.push_notification(uuid);

        // Need another outstanding transfer for the tick to fire
        let uuid2 = Uuid::new_v4();
        let event2 = system.new_event()?;
        let handle2 = event2.into_handle();
        tx.send(TestRegistration {
            uuid: uuid2,
            event_handle: handle2,
        })
        .await?;
        yield_now().await;

        tick().await;

        // Original event still Ready, no double-trigger issues
        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        // Cleanup
        control.push_notification(uuid2);
        tick().await;

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn poll_error_does_not_crash_loop() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid = Uuid::new_v4();
        let event = system.new_event()?;
        let handle = event.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register
        tx.send(TestRegistration {
            uuid,
            event_handle: handle,
        })
        .await?;
        yield_now().await;

        // Inject a poll error
        control.push_error();
        tick().await;

        // Event should still be pending (error was swallowed, not fatal)
        assert_eq!(system.poll(handle)?, EventStatus::Pending);

        // Now deliver the real notification — handler recovered
        control.push_notification(uuid);
        tick().await;

        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        drop(tx);
        task.await?;
        Ok(())
    }

    #[tokio::test(start_paused = true)]
    async fn unknown_notification_in_drain_is_ignored() -> anyhow::Result<()> {
        let (source, control) = mock_source();
        let (tx, rx) = mpsc::channel(16);
        let system = Arc::new(EventManager::local());

        let uuid = Uuid::new_v4();
        let event = system.new_event()?;
        let handle = event.into_handle();

        let task = tokio::spawn(process_events_core(source, rx, system.clone()));

        // Register
        tx.send(TestRegistration {
            uuid,
            event_handle: handle,
        })
        .await?;
        yield_now().await;

        // Close channel to enter drain
        drop(tx);
        yield_now().await;

        // Deliver both our UUID and an unknown one — should not panic
        let mut map = HashMap::new();
        map.insert(
            "test_agent".to_string(),
            vec![uuid.to_string(), Uuid::new_v4().to_string()],
        );
        control.queue.lock().unwrap().push_back(Ok(map));
        tick().await;

        assert_eq!(system.poll(handle)?, EventStatus::Ready);

        task.await?;
        Ok(())
    }
}
