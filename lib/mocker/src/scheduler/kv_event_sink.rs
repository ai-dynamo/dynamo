// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, Mutex};

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
pub(crate) fn publish_deferred_kv_events(
    sinks: &KvEventPublishers,
    events: Vec<DeferredKvPublish>,
) {
    if events.is_empty() {
        return;
    }

    let raw_events: Vec<RawKvEvent> = events
        .into_iter()
        .map(|event| RawKvEvent {
            event: event.event,
            block_token_ids: event.block_token_ids,
            storage_tier: event.storage_tier,
        })
        .collect();

    let normal_events = raw_events
        .iter()
        .map(|event| (event.event.clone(), event.storage_tier))
        .collect();

    if let Err(error) = sinks.publish_event_sink_batch_only(normal_events) {
        tracing::warn!("Failed to forward buffered KV event batch: {error}");
    }

    if let Err(error) = sinks.publish_raw_batch(raw_events) {
        tracing::warn!("Failed to forward buffered raw KV event batch: {error}");
    }
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
    use std::sync::{Arc, Mutex};

    use dynamo_kv_router::protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheStoreData,
        KvCacheStoredBlockData, LocalBlockHash,
    };
    use dynamo_kv_router::zmq_wire::decode_event_batch;

    use super::*;
    use crate::services::zmq_events::encode_event_batch;

    #[derive(Default)]
    struct CapturingSink {
        normal_batches: Mutex<Vec<Vec<(KvCacheEvent, StorageTier)>>>,
        raw_batches: Mutex<Vec<Vec<RawKvEvent>>>,
    }

    impl KvCacheEventSink for CapturingSink {
        fn publish(&self, event: KvCacheEvent) -> Result<()> {
            self.publish_batch_with_storage_tiers(vec![(event, StorageTier::Device)])
        }

        fn publish_batch_with_storage_tiers(
            &self,
            events: Vec<(KvCacheEvent, StorageTier)>,
        ) -> Result<()> {
            self.normal_batches.lock().unwrap().push(events);
            Ok(())
        }
    }

    impl RawKvEventSink for CapturingSink {
        fn publish(&self, event: RawKvEvent) -> Result<()> {
            self.publish_batch(vec![event])
        }

        fn publish_batch(&self, events: Vec<RawKvEvent>) -> Result<()> {
            self.raw_batches.lock().unwrap().push(events);
            Ok(())
        }
    }

    #[test]
    fn deferred_visibility_boundary_emits_one_normal_and_raw_batch() {
        let sink = Arc::new(CapturingSink::default());
        let sinks = KvEventPublishers::new(Some(sink.clone()), Some(sink.clone()));
        let stored_data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash: None,
            start_position: None,
            blocks: vec![KvCacheStoredBlockData {
                block_hash: ExternalSequenceBlockHash(1),
                tokens_hash: LocalBlockHash(1),
                mm_extra_info: None,
            }],
        });

        publish_deferred_kv_events(&sinks, Vec::new());
        assert!(sink.normal_batches.lock().unwrap().is_empty());
        assert!(sink.raw_batches.lock().unwrap().is_empty());

        publish_deferred_kv_events(
            &sinks,
            [1, 2]
                .into_iter()
                .map(|event_id| DeferredKvPublish {
                    event: KvCacheEvent {
                        event_id,
                        data: stored_data.clone(),
                        dp_rank: 0,
                    },
                    block_token_ids: Some(vec![vec![event_id as u32; 4]]),
                    storage_tier: StorageTier::Device,
                })
                .collect(),
        );

        let normal_batches = sink.normal_batches.lock().unwrap();
        assert_eq!(normal_batches.len(), 1);
        assert_eq!(
            normal_batches[0]
                .iter()
                .map(|(event, _)| event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );

        let raw_batches = sink.raw_batches.lock().unwrap();
        assert_eq!(raw_batches.len(), 1);
        assert_eq!(
            raw_batches[0]
                .iter()
                .map(|event| event.event.event_id)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        let payload = encode_event_batch(&raw_batches[0], 4, 3)
            .unwrap()
            .expect("stored events should produce a payload");
        let batch =
            decode_event_batch(&payload).expect("payload should use the native wire format");

        assert_eq!(batch.data_parallel_rank, Some(3));
        assert_eq!(batch.events.len(), 2);
        assert_eq!(batch.events[0].event_type_label(), "stored");
        assert_eq!(batch.events[1].event_type_label(), "stored");
    }
}
