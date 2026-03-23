// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

use anyhow::Result;
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, WorkerId};

use crate::common::protocols::{KvCacheEventSink, KvEventPublishers, RawKvEvent, RawKvEventSink};

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
}

/// Captures raw KV publishes for the live `python -m dynamo.mocker` and online
/// replay paths.
///
/// Unlike `CapturedRouterEventBuffer`, this keeps `block_token_ids` so delayed
/// forwarding still works for sinks like ZMQ publishers that need the original
/// token-id payloads.
#[derive(Clone, Default)]
pub(crate) struct DeferredKvPublishBuffer {
    events: Arc<Mutex<Vec<DeferredKvPublish>>>,
}

impl DeferredKvPublishBuffer {
    pub(crate) fn push(&self, event: KvCacheEvent, block_token_ids: Option<Vec<Vec<u32>>>) {
        self.events.lock().unwrap().push(DeferredKvPublish {
            event,
            block_token_ids,
        });
    }

    pub(crate) fn drain(&self) -> Vec<DeferredKvPublish> {
        std::mem::take(&mut *self.events.lock().unwrap())
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
        self.buffer.push(event, None);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct DeferredRawKvEventSink {
    buffer: DeferredKvPublishBuffer,
}

impl RawKvEventSink for DeferredRawKvEventSink {
    fn publish(&self, event: RawKvEvent) -> Result<()> {
        let mut events = self.buffer.events.lock().unwrap();
        if let Some(last) = events.last_mut()
            && last.event.event_id == event.event.event_id
            && last.event.dp_rank == event.event.dp_rank
        {
            last.block_token_ids = event.block_token_ids;
            return Ok(());
        }

        events.push(DeferredKvPublish {
            event: event.event,
            block_token_ids: event.block_token_ids,
        });
        Ok(())
    }
}

/// Returns the deferred-publish buffer plus a sink handle that can be passed
/// into the live scheduler core while `live.rs` retains control over when the
/// buffered events are forwarded to the real sink.
pub(crate) fn capture_deferred_kv_publish_sink(
    capture_raw: bool,
) -> (DeferredKvPublishBuffer, KvEventPublishers) {
    let buffer = DeferredKvPublishBuffer::default();
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
pub(crate) fn publish_deferred_kv_events(
    sinks: &KvEventPublishers,
    events: Vec<DeferredKvPublish>,
) {
    for event in events {
        if let Err(error) = sinks.publish(event.event, event.block_token_ids.as_deref()) {
            tracing::warn!("Failed to forward buffered KV event: {error}");
        }
    }
}
