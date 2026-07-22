// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::future::Future;
use std::sync::Arc;

use anyhow::Result;

use dynamo_kv_router::RouterEventSink;
use dynamo_kv_router::indexer::LocalKvIndexer;
use dynamo_kv_router::protocols::{KvCacheEvent, RouterEvent, StorageTier};
use dynamo_runtime::transports::event_plane::EventPublisher;
use dynamo_runtime::transports::nats::NatsQueue;

use crate::kv_router::KV_EVENT_SUBJECT;

pub(super) struct EventPlanePublisher(pub(super) EventPublisher);

impl RouterEventSink for EventPlanePublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        self.0.publish(event)
    }
}

pub(super) struct JetStreamPublisher(pub(super) NatsQueue);

impl RouterEventSink for JetStreamPublisher {
    fn publish_event(&self, event: &RouterEvent) -> impl Future<Output = Result<()>> + Send {
        NatsQueue::publish_event(&self.0, KV_EVENT_SUBJECT, event)
    }
}

pub(super) async fn emit<P: RouterEventSink>(
    publisher: &P,
    local_indexer: &Option<Arc<LocalKvIndexer>>,
    worker_id: u64,
    storage_tier: StorageTier,
    event: KvCacheEvent,
) {
    let router_event = RouterEvent::with_storage_tier(worker_id, event, storage_tier);
    if let Some(indexer) = local_indexer
        && let Err(e) = indexer.apply_event_with_buffer(router_event.clone()).await
    {
        tracing::warn!(worker_id, error = %e, "Failed to apply event to local indexer");
    }
    if let Err(e) = publisher.publish_event(&router_event).await {
        tracing::error!(worker_id, error = %e, "Failed to publish event");
    }
}
