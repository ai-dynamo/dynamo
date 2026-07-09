// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};

use anyhow::Result;
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, StorageTier, WorkerId};

use crate::common::protocols::{
    ForwardPassSnapshot, FpmPublisher, KvCacheEventSink, KvEventPublishers, RawKvEvent,
    RawKvEventSink,
};

/// Captures router-ready events for offline replay and scheduler tests.
///
/// This path converts raw KV events into `RouterEvent`s immediately because the
/// caller only needs worker-tagged router events, not the original token-id
/// payloads used by the live publisher path.
#[derive(Clone, Default)]
pub(crate) struct CapturedRouterEventBuffer {
    events: Arc<Mutex<Vec<RouterEvent>>>,
}

impl CapturedRouterEventBuffer {
    pub(crate) fn push(&self, event: RouterEvent) {
        self.events.lock().unwrap().push(event);
    }

    pub(crate) fn drain(&self) -> Vec<RouterEvent> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

/// Sink implementation that records `RouterEvent`s into
/// `CapturedRouterEventBuffer`.
#[derive(Clone)]
struct RouterEventCaptureSink {
    worker_id: WorkerId,
    buffer: CapturedRouterEventBuffer,
}

impl KvCacheEventSink for RouterEventCaptureSink {
    fn publish(&self, event: KvCacheEvent) -> Result<()> {
        self.buffer.push(RouterEvent::new(self.worker_id, event));
        Ok(())
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> Result<()> {
        self.buffer.push(RouterEvent::with_storage_tier(
            self.worker_id,
            event,
            storage_tier,
        ));
        Ok(())
    }
}

/// Returns the capture buffer plus a sink handle that can be passed into a
/// scheduler core for offline replay or tests.
pub(crate) fn capture_router_event_sink(
    worker_id: WorkerId,
) -> (CapturedRouterEventBuffer, Arc<dyn KvCacheEventSink>) {
    let buffer = CapturedRouterEventBuffer::default();
    let sink: Arc<dyn KvCacheEventSink> = Arc::new(RouterEventCaptureSink {
        worker_id,
        buffer: buffer.clone(),
    });
    (buffer, sink)
}

/// Raw KV event payload buffered by the live scheduler so it can forward the
/// event to the real publisher sink at the correct pass phase.
#[derive(Debug, Clone)]
pub(crate) struct DeferredKvPublish {
    pub(crate) event: KvCacheEvent,
    pub(crate) block_token_ids: Option<Vec<Vec<u32>>>,
    pub(crate) storage_tier: StorageTier,
}

/// Captures raw KV publishes for the live `python -m dynamo.mocker` and online
/// replay paths.
///
/// Unlike `CapturedRouterEventBuffer`, this keeps `block_token_ids` so delayed
/// forwarding still works for sinks like ZMQ publishers that need the original
/// token-id payloads.
#[derive(Clone, Default)]
pub(crate) struct DeferredKvPublishBuffer {
    events: Option<Arc<Mutex<Vec<DeferredKvPublish>>>>,
}

impl DeferredKvPublishBuffer {
    fn enabled() -> Self {
        Self {
            events: Some(Arc::new(Mutex::new(Vec::new()))),
        }
    }

    pub(crate) fn push(
        &self,
        event: KvCacheEvent,
        block_token_ids: Option<Vec<Vec<u32>>>,
        storage_tier: StorageTier,
    ) {
        let Some(events) = self.events.as_ref() else {
            return;
        };
        events.lock().unwrap().push(DeferredKvPublish {
            event,
            block_token_ids,
            storage_tier,
        });
    }

    pub(crate) fn drain(&self) -> Vec<DeferredKvPublish> {
        self.events
            .as_ref()
            .map(|events| std::mem::take(&mut *events.lock().unwrap()))
            .unwrap_or_default()
    }
}

/// Sink implementation that records raw KV publishes into
/// `DeferredKvPublishBuffer` instead of forwarding them immediately.
#[derive(Clone, Default)]
struct DeferredKvEventSink {
    buffer: DeferredKvPublishBuffer,
}

impl KvCacheEventSink for DeferredKvEventSink {
    fn publish(&self, event: KvCacheEvent) -> Result<()> {
        self.buffer.push(event, None, StorageTier::Device);
        Ok(())
    }

    fn publish_with_storage_tier(
        &self,
        event: KvCacheEvent,
        storage_tier: StorageTier,
    ) -> Result<()> {
        self.buffer.push(event, None, storage_tier);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct DeferredRawKvEventSink {
    buffer: DeferredKvPublishBuffer,
}

impl RawKvEventSink for DeferredRawKvEventSink {
    fn publish(&self, event: RawKvEvent) -> Result<()> {
        let Some(events) = self.buffer.events.as_ref() else {
            return Ok(());
        };
        let mut events = events.lock().unwrap();
        if let Some(last) = events.last_mut()
            && last.event.event_id == event.event.event_id
            && last.event.dp_rank == event.event.dp_rank
            && last.storage_tier == event.storage_tier
        {
            last.block_token_ids = event.block_token_ids;
            return Ok(());
        }

        events.push(DeferredKvPublish {
            event: event.event,
            block_token_ids: event.block_token_ids,
            storage_tier: event.storage_tier,
        });
        Ok(())
    }
}

/// Returns the deferred-publish buffer plus a sink handle that can be passed
/// into the live scheduler core while `live.rs` retains control over when the
/// buffered events are forwarded to the real sink.
pub(crate) fn capture_deferred_kv_publish_sink(
    enabled: bool,
    capture_raw: bool,
) -> (DeferredKvPublishBuffer, KvEventPublishers) {
    if !enabled {
        return (
            DeferredKvPublishBuffer::default(),
            KvEventPublishers::default(),
        );
    }
    let buffer = DeferredKvPublishBuffer::enabled();
    let event_sink: Arc<dyn KvCacheEventSink> = Arc::new(DeferredKvEventSink {
        buffer: buffer.clone(),
    });
    let raw_sink = capture_raw.then(|| {
        Arc::new(DeferredRawKvEventSink {
            buffer: buffer.clone(),
        }) as Arc<dyn RawKvEventSink>
    });
    (buffer, KvEventPublishers::new(Some(event_sink), raw_sink))
}

/// Forwards buffered live-scheduler KV events to the real sink once the pass
/// reaches the configured visibility point.
fn forward_deferred_kv_events(
    sinks: &KvEventPublishers,
    events: Vec<DeferredKvPublish>,
) -> DeferredKvForwardingSummary {
    let mut summary = DeferredKvForwardingSummary {
        attempted: events.len(),
        ..Default::default()
    };
    for event in events {
        if let Err(error) = sinks.publish_with_storage_tier(
            event.event,
            event.block_token_ids.as_deref(),
            event.storage_tier,
        ) {
            summary.failed += 1;
            summary.first_error.get_or_insert_with(|| error.to_string());
        }
    }
    summary
}

#[derive(Debug, Default, PartialEq, Eq)]
struct DeferredKvForwardingSummary {
    attempted: usize,
    failed: usize,
    first_error: Option<String>,
}

/// Cross-drain logging state owned by the live-effects boundary.
///
/// A closed channel cannot recover while the publisher bundle remains alive,
/// so repeating the same terminal warning for every scheduler pass adds no
/// diagnostic value. Event delivery is still attempted on later drains so a
/// second, open sink in the bundle keeps receiving best-effort updates.
#[derive(Default)]
pub(crate) struct DeferredKvForwardingLogState {
    terminal_failure_logged: AtomicBool,
}

impl DeferredKvForwardingLogState {
    fn should_log_failure(&self, terminal: bool) -> bool {
        !terminal || !self.terminal_failure_logged.swap(true, Ordering::Relaxed)
    }
}

/// Forwards buffered live-scheduler KV events to the real sink once the pass
/// reaches the configured visibility point.
pub(crate) fn publish_deferred_kv_events(
    sinks: &KvEventPublishers,
    events: Vec<DeferredKvPublish>,
    log_state: &DeferredKvForwardingLogState,
) {
    let summary = forward_deferred_kv_events(sinks, events);
    let Some(first_error) = summary.first_error.as_deref() else {
        return;
    };
    let terminal = sinks.has_closed_sink();
    if !log_state.should_log_failure(terminal) {
        return;
    }
    tracing::warn!(
        terminal,
        total_events = summary.attempted,
        failed_events = summary.failed,
        successful_events = summary.attempted - summary.failed,
        suppressed_error_logs = summary.failed.saturating_sub(1),
        error = %first_error,
        "Failed to forward buffered KV event batch"
    );
}

/// Forwards buffered FPM snapshots to the real sink once the pass reaches
/// the configured visibility point.
pub(crate) fn publish_deferred_fpm(sink: &FpmPublisher, snapshots: Vec<ForwardPassSnapshot>) {
    for snapshot in snapshots {
        if let Err(error) = sink.publish(snapshot) {
            tracing::warn!("Failed to forward buffered FPM snapshot: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::bail;
    use dynamo_kv_router::protocols::KvCacheEventData;

    use super::*;

    struct SelectiveFailureSink {
        attempts: AtomicUsize,
        fail_through_event_id: u64,
        closed: bool,
    }

    impl KvCacheEventSink for SelectiveFailureSink {
        fn publish(&self, event: KvCacheEvent) -> Result<()> {
            self.attempts.fetch_add(1, Ordering::Relaxed);
            if event.event_id <= self.fail_through_event_id {
                bail!("event sink channel closed at event {}", event.event_id);
            }
            Ok(())
        }

        fn is_closed(&self) -> bool {
            self.closed
        }
    }

    fn deferred_event(event_id: u64) -> DeferredKvPublish {
        DeferredKvPublish {
            event: KvCacheEvent {
                event_id,
                data: KvCacheEventData::Cleared,
                dp_rank: 0,
            },
            block_token_ids: None,
            storage_tier: StorageTier::Device,
        }
    }

    #[test]
    fn forwarding_summarizes_closed_sink_failures_per_batch() {
        let sink = Arc::new(SelectiveFailureSink {
            attempts: AtomicUsize::new(0),
            fail_through_event_id: u64::MAX,
            closed: true,
        });
        let publishers = KvEventPublishers::new(Some(sink.clone()), None);
        let events = (1..=64).map(deferred_event).collect();

        let summary = forward_deferred_kv_events(&publishers, events);

        assert_eq!(sink.attempts.load(Ordering::Relaxed), 64);
        assert_eq!(summary.attempted, 64);
        assert_eq!(summary.failed, 64);
        assert_eq!(
            summary.first_error.as_deref(),
            Some("event sink channel closed at event 1")
        );
    }

    #[test]
    fn forwarding_continues_after_an_event_error() {
        let sink = Arc::new(SelectiveFailureSink {
            attempts: AtomicUsize::new(0),
            fail_through_event_id: 2,
            closed: false,
        });
        let publishers = KvEventPublishers::new(Some(sink.clone()), None);
        let events = (1..=4).map(deferred_event).collect();

        let summary = forward_deferred_kv_events(&publishers, events);

        assert_eq!(sink.attempts.load(Ordering::Relaxed), 4);
        assert_eq!(summary.attempted, 4);
        assert_eq!(summary.failed, 2);
    }

    #[test]
    fn terminal_failures_log_once_but_transient_failures_remain_visible() {
        let state = DeferredKvForwardingLogState::default();

        assert!(state.should_log_failure(true));
        assert!(!state.should_log_failure(true));
        assert!(!state.should_log_failure(true));
        assert!(state.should_log_failure(false));
        assert!(state.should_log_failure(false));
    }

    #[test]
    fn closed_sink_is_still_attempted_after_terminal_warning_is_suppressed() {
        let sink = Arc::new(SelectiveFailureSink {
            attempts: AtomicUsize::new(0),
            fail_through_event_id: u64::MAX,
            closed: true,
        });
        let publishers = KvEventPublishers::new(Some(sink.clone()), None);
        let state = DeferredKvForwardingLogState::default();

        publish_deferred_kv_events(
            &publishers,
            vec![deferred_event(1), deferred_event(2)],
            &state,
        );
        publish_deferred_kv_events(
            &publishers,
            vec![deferred_event(3), deferred_event(4)],
            &state,
        );

        assert_eq!(sink.attempts.load(Ordering::Relaxed), 4);
        assert!(state.terminal_failure_logged.load(Ordering::Relaxed));
    }
}
