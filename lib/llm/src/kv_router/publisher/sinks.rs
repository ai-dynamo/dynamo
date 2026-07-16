// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEvent, KvCacheEventData, RouterEvent, StorageTier};
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

/// Bound normal event-plane batches while always allowing one complete event.
///
/// The default NATS Core `max_payload` is 1 MiB. Production-shaped batches at
/// this cap, including one multimodal object with one offset per stored block,
/// remain below that limit in the wire-size regression tests. Multimodal
/// metadata is not intrinsically bounded, so an exceptional batch can still
/// exceed a deployment's configured transport limit and fail under the
/// existing best-effort publication semantics.
pub(super) const MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS: usize = 8_192;

pub(super) trait RouterEventBatchSink: Send + Sync {
    fn publish_events(&self, events: &[RouterEvent]) -> impl Future<Output = Result<()>> + Send;
}

#[derive(Default)]
struct PublishFailures {
    publishes: usize,
    events: usize,
    first_error: Option<anyhow::Error>,
}

impl PublishFailures {
    fn record(&mut self, event_count: usize, error: anyhow::Error) {
        self.publishes += 1;
        self.events += event_count;
        if self.first_error.is_none() {
            self.first_error = Some(error);
        }
    }

    fn into_result(self) -> Result<()> {
        let Some(first_error) = self.first_error else {
            return Ok(());
        };
        let summary = format!(
            "{} publish attempt(s) failed; {} event(s) dropped; first error: {first_error}",
            self.publishes, self.events
        );
        Err(first_error.context(summary))
    }
}

impl<P: RouterEventSink + Send + Sync> RouterEventBatchSink for P {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut failures = PublishFailures::default();
        for event in events {
            if let Err(error) = self.publish_event(event).await {
                tracing::error!(
                    worker_id = event.worker_id,
                    event_id = event.event.event_id,
                    error = %error,
                    "Failed to publish KV event"
                );
                failures.record(1, error);
            }
        }
        failures.into_result()
    }
}

impl RouterEventBatchSink for EventPlanePublisher {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut failures = PublishFailures::default();
        for batch in event_plane_event_batches(events, MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS) {
            if let Err(error) = self.0.publish(&batch).await {
                let first_event_id = batch.first().map(|event| event.event.event_id);
                let last_event_id = batch.last().map(|event| event.event.event_id);
                tracing::error!(
                    transport = ?self.0.transport_kind(),
                    event_count = batch.len(),
                    ?first_event_id,
                    ?last_event_id,
                    error = %error,
                    "Failed to publish KV event batch"
                );
                failures.record(batch.len(), error);
            }
        }
        failures.into_result()
    }
}

/// Partition ordered events at event boundaries without exceeding the block cap.
/// A single event larger than the cap is always emitted intact.
pub(super) fn event_plane_event_batches(
    events: &[RouterEvent],
    max_blocks: usize,
) -> impl Iterator<Item = &[RouterEvent]> {
    let mut batch_start = 0;

    std::iter::from_fn(move || {
        if batch_start == events.len() {
            return None;
        }

        let mut batch_end = batch_start;
        let mut batch_blocks = 0usize;
        while let Some(event) = events.get(batch_end) {
            let event_blocks = match &event.event.data {
                KvCacheEventData::Stored(data) => data.blocks.len(),
                KvCacheEventData::Removed(data) => data.block_hashes.len(),
                KvCacheEventData::Cleared => 0,
            };
            if batch_end > batch_start && batch_blocks.saturating_add(event_blocks) > max_blocks {
                break;
            }
            batch_blocks = batch_blocks.saturating_add(event_blocks);
            batch_end += 1;
        }

        let batch = &events[batch_start..batch_end];
        batch_start = batch_end;
        Some(batch)
    })
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

pub(super) async fn emit(
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
    output: &mut Vec<RouterEvent>,
) {
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    output.push(router_event);
}
