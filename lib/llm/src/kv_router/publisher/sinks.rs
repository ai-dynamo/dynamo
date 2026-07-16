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

pub(super) const MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS: usize = 8_192;

pub(super) trait RouterEventBatchSink: Send + Sync {
    fn publish_events(&self, events: &[RouterEvent]) -> impl Future<Output = Result<()>> + Send;
}

impl<P: RouterEventSink + Send + Sync> RouterEventBatchSink for P {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut first_error = None;
        for event in events {
            if let Err(error) = self.publish_event(event).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }
}

impl RouterEventBatchSink for EventPlanePublisher {
    async fn publish_events(&self, events: &[RouterEvent]) -> Result<()> {
        let mut first_error = None;
        for batch in event_plane_event_batches(events, MAX_EVENT_PLANE_KV_EVENT_BATCH_BLOCKS) {
            let payload = encode_event_plane_batch(batch)?;
            if let Err(error) = self.0.publish_bytes(payload).await
                && first_error.is_none()
            {
                first_error = Some(error);
            }
        }
        first_error.map_or(Ok(()), Err)
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

/// Encode one complete ordered event-plane batch.
pub(super) fn encode_event_plane_batch(events: &[RouterEvent]) -> Result<Vec<u8>> {
    Ok(rmp_serde::to_vec_named(events)?)
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
